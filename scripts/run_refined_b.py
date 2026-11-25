"""Replay cached A/D with the refined B; labels used only after prediction."""
import argparse
from dataclasses import replace
import json
from pathlib import Path
import time

from src.evaluation.collection import check_annotations,summarize
from scripts.run_complementary import digest
from src.intent.formal_evaluation import read_csv,score_clip
from src.intent.persistent import PersistentIntentStateMachine
from src.intent.semantic_priority import SemanticPriorityIntentStateMachine,SemanticBranchIntentStateMachine
from src.intent.fusion import FusionConfig
from src.intent.rest_aware import RestAwareIntentStateMachine
from src.speech.presentation_language import ContextualPresentationLanguage
from src.speech.bridge import read_events_jsonl,SpeechEvidenceBridge
from src.intent.audio_timeline import AudioTimeline


def collection(name):
    dataset=Path('evaluation')/name
    rows=read_csv(dataset/'dataset_manifest.csv')
    if name in {'fruit','books'}:
        rows=[r for r in rows if r['source_collection']==('fruit_original' if name=='fruit' else 'books_original')]
    else:check_annotations(dataset, verify_media=False)
    raw=Path({'fruit':'outputs/cache/fruit','books':'outputs/cache/books','cups':'outputs/cache/cups','drinks':'outputs/cache/drinks'}[name])
    return dataset,rows,raw


def replay(per,events,visual,fusion,memory,*,profile='persistent'):
    if profile not in {'persistent','semantic-priority','semantic-priority-v1','semantic-priority-rest'}:raise ValueError('Unknown B profile')
    lang=ContextualPresentationLanguage();observations=[]
    for e in sorted(events,key=lambda e:(e.end_ms,e.start_ms)):
        observations.append(replace(e,available_at_ms=e.end_ms,semantics=lang.analyze_event(e)))
    observations.sort(key=lambda e:e.available_at_ms)
    streams={};traces=[]
    for mode in ['visual-only','showintent']:
        cls=({'semantic-priority-rest':RestAwareIntentStateMachine,'semantic-priority':SemanticPriorityIntentStateMachine,'semantic-priority-v1':SemanticBranchIntentStateMachine}.get(profile,PersistentIntentStateMachine)
             if mode=='showintent' else PersistentIntentStateMachine)
        engine=cls(**visual,fusion_config=fusion if mode=='showintent' else None,hand_memory_ms=memory)
        bridge=SpeechEvidenceBridge(observations,enabled=True)
        values=[]
        for p in per:
            values.append(engine.update(p,speech_evidence=bridge.evidence_for(p)) if mode=='showintent' else engine.update(p))
            if mode=='showintent':traces.append(dict(engine.fusion.diagnostics,intent=values[-1]))
        streams[mode]=values
    audio=AudioTimeline(events,timing='content');audio.language=ContextualPresentationLanguage()
    streams['audio-only']=[audio.update(p['timestamp_ms']) for p in per]
    return streams,observations,traces


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--collections',nargs='+',choices=['cups','drinks','fruit','books'],default=['cups','drinks','fruit','books']);p.add_argument('--hand-memory-ms',type=int,default=3000);p.add_argument('--profile',choices=['semantic-priority-rest'],default='semantic-priority-rest');a=p.parse_args()
    if a.out.exists() or not a.out.resolve().is_relative_to(Path('outputs').resolve()):p.error('Choose a fresh directory under outputs')
    if len(set(a.collections))!=len(a.collections) or a.hand_memory_ms<=0:p.error('Invalid collection/memory')
    visual=json.loads(Path('configs/presentation_intent.json').read_text());fusion_path=Path('configs/semantic_priority_fusion.json' if a.profile.startswith('semantic-priority') else 'configs/stable_fusion.json');fconfig=json.loads(fusion_path.read_text());fusion=FusionConfig(**fconfig)
    a.out.mkdir(parents=True);reports={};hashes={};start=time.perf_counter()
    for name in a.collections:
        dataset,rows,raw=collection(name);reports[name]={}
        labels=read_csv(dataset/'annotations.csv');hashes[str(dataset/'annotations.csv')]=digest(dataset/'annotations.csv')
        for row in rows:
            c=row['clip_id'];perpath=Path(row['perception_path']);asrpath=raw/'speech_raw'/f'speech_events_{c}.jsonl'
            per=[json.loads(l) for l in perpath.read_text().splitlines() if l.strip()];events=read_events_jsonl(asrpath)
            streams,observations,traces=replay(per,events,visual,fusion,a.hand_memory_ms,profile=a.profile)
            dest=a.out/name/c;dest.mkdir(parents=True)
            for method,values in streams.items():
                (dest/f'{method}.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
                (dest/f'intent_{method}.jsonl').symlink_to(method+'.jsonl')
            (dest/'speech.jsonl').write_text(''.join(json.dumps(v.to_dict(),ensure_ascii=False)+'\n' for v in observations))
            (dest/'evidence.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in traces))
            scores={}
            # Annotation windows are read only by the evaluator, after all outputs.
            for method,values in streams.items():
                normalized=[dict(timestamp_ms=v['timestamp_ms'],state=v['state'],target_id='TEMPORAL_ONLY' if v['state']=='SHOWING' else None) for v in values]
                score=score_clip([dict(timestamp_ms=v['timestamp_ms'],objects=[]) for v in per],normalized,[r for r in labels if r['clip_id']==c],[])
                scores[method]={k:score[k] for k in ['temporal_events','positive_ms','negative_ms','showing_ms','false_showing_ms','matched_boundary_errors','predicted_episodes']}
            reports[name][c]=scores
            hashes[str(perpath)]=digest(perpath);hashes[str(asrpath)]=digest(asrpath)
            print(name,c,scores['showintent']['temporal_events'],flush=True)
    summary={name:{m:summarize([r[m] for r in clips.values()]) for m in ['visual-only','showintent','audio-only']} for name,clips in reports.items()}
    new=[r['showintent'] for name,clips in reports.items() if name in {'cups','drinks'} for r in clips.values()]
    errors=sum(r['temporal_events']['fp']+r['temporal_events']['fn'] for r in new)
    acceptance=dict(scope='New 8 clips: binary temporal FP+FN <= 1; physical identity still requires review',all_new_clips_present=all(n in reports and len(reports[n])==4 for n in ['cups','drinks']),event_errors=errors,passed=all(n in reports and len(reports[n])==4 for n in ['cups','drinks']) and errors<=1)
    version={'semantic-priority-rest':'semantic_priority_B_v3_rest_aware_candidate','semantic-priority':'semantic_priority_B_v2_visual_guard','semantic-priority-v1':'semantic_priority_B_v1','persistent':'persistent_B_v1'}[a.profile]
    report=dict(version=version,profile=a.profile,min_audio_lead_ms=200 if a.profile in {'semantic-priority','semantic-priority-rest'} else None,scope='DEVELOPMENT_REGRESSION_AFTER_VIEWING_DATA_NOT_NEW_HOLDOUT',timing='RETROSPECTIVE_CONTENT',visual_config=visual,fusion_config=fconfig,hand_memory_ms=a.hand_memory_ms,elapsed_seconds=time.perf_counter()-start,input_sha256=hashes,source_sha256={str(x):digest(x) for folder in ['src/intent','src/speech'] for x in Path(folder).glob('*.py')},summary=summary,acceptance=acceptance,clips=reports)
    (a.out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(dict(summary=summary,acceptance=acceptance),ensure_ascii=False,indent=2))


if __name__=='__main__':main()

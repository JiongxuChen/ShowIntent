"""Replay the accepted offline B using the portable scene cache."""
import argparse,json
from pathlib import Path
from dataclasses import replace
from src.intent.formal_evaluation import read_csv,score_clip
from src.evaluation.collection import summarize
from src.intent.delayed_adaptation import RecoveryOnlyIntentStateMachine
from src.intent.streaming import SparseVisualIntentStateMachine
from src.intent.audio_timeline import AudioTimeline
from src.intent.fusion import FusionConfig
from src.speech.bridge import read_events_jsonl,SpeechEvidenceBridge
from src.speech.presentation_language import ContextualPresentationLanguage

def collection(name):
 dataset=Path('evaluation')/name
 return dataset,read_csv(dataset/'dataset_manifest.csv'),Path('outputs/cache')/name

def read(path):return [json.loads(l) for l in Path(path).read_text().splitlines() if l.strip()]
def normalized_score(per,values,labels):
 states=[dict(timestamp_ms=v['timestamp_ms'],state=v['state'],target_id='TEMPORAL_ONLY' if v['state']=='SHOWING' else None) for v in values]
 s=score_clip([dict(timestamp_ms=p['timestamp_ms'],objects=[]) for p in per],states,labels,[])
 return {k:s[k] for k in ['temporal_events','positive_ms','negative_ms','showing_ms','false_showing_ms','matched_boundary_errors','predicted_episodes']}
def offline_streams(per,events,config,fconfig):
 language=ContextualPresentationLanguage();observations=[replace(e,available_at_ms=e.end_ms,semantics=language.analyze_event(e)) for e in sorted(events,key=lambda e:(e.end_ms,e.start_ms))]
 bridge=SpeechEvidenceBridge(observations,enabled=True);fused=RecoveryOnlyIntentStateMachine(**config,fusion_config=FusionConfig(**fconfig),hand_memory_ms=3000);visual=SparseVisualIntentStateMachine(**config,hand_memory_ms=3000);audio=AudioTimeline(events,timing='content');audio.language=ContextualPresentationLanguage()
 return {'showintent':[fused.update(p,speech_evidence=bridge.evidence_for(p)) for p in per],'visual-only':[visual.update(p) for p in per],'audio-only':[audio.update(p['timestamp_ms']) for p in per]}
def replay(per,events,visual,fusion,memory,*,profile='semantic-priority-recovery'):
 if profile!='semantic-priority-recovery':raise ValueError('Only the accepted B profile is supported')
 from dataclasses import asdict
 return offline_streams(per,events,visual,asdict(fusion)),events,[]
def process(perpath,speech,timing,labels,out):
 per=read(perpath)
 if [p['timestamp_ms'] for p in per]!=json.loads(Path(timing).read_text())['timestamp_ms']:raise ValueError('Incomplete or shifted perception')
 streams=offline_streams(per,read_events_jsonl(speech),json.loads(Path('configs/presentation_intent.json').read_text()),json.loads(Path('configs/semantic_priority_fusion.json').read_text()));out.mkdir(parents=True)
 for m,rows in streams.items():(out/(m+'.jsonl')).write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in rows))
 scores={m:normalized_score(per,rows,labels) for m,rows in streams.items()};(out/'evaluation.json').write_text(json.dumps(scores,indent=2)+'\n');return scores

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--collections',nargs='+',choices=['fruit','books','cups','drinks'],default=['fruit','books','cups','drinks']);p.add_argument('--clips',nargs='+');a=p.parse_args()
 if a.out.exists():p.error('Choose a new output directory')
 report={}
 for g in a.collections:
  dataset,rows,cache=collection(g);labels=read_csv(dataset/'annotations.csv')
  for r in rows:
   c=r['clip_id']
   if a.clips and c not in a.clips:continue
   report[g+'/'+c]=process(cache/'perception'/('perception_'+c+'.jsonl'),cache/'speech_raw'/('speech_events_'+c+'.jsonl'),cache/'timestamps'/(c+'.json'),[r for r in labels if r['clip_id']==c],a.out/g/c)
 if not report:p.error('No matching clips')
 (a.out/'evaluation.json').write_text(json.dumps(dict(clips=report,summary={mode:summarize([v[mode] for v in report.values()]) for mode in ['showintent','visual-only','audio-only']}),indent=2)+'\n');print('Completed',len(report),'clips')
if __name__=='__main__':main()

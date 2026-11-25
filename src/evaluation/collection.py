"""Score the new cup collection, explicitly retaining pending assistant-label status."""
import argparse
import csv
import json
from pathlib import Path

from scripts.run_complementary import digest
from src.intent.audio_timeline import AudioTimeline
from src.intent.formal_evaluation import read_csv, score_clip, prf
from src.speech.bridge import read_events_jsonl
from src.perception.timing import validate_timestamps


def check_timebase(packets, saved, row, annotation_pts=None):
    """Keep exact prediction/PTS alignment; accept verified round-vs-floor annotation clocks."""
    c=row['clip_id'];validate_timestamps(saved)
    if [p['timestamp_ms'] for p in packets]!=saved:
        raise ValueError('Incomplete or shifted perception: '+c)
    declared=[int(row['eval_start_ms']),int(row['eval_end_ms'])]
    actual=[saved[0],saved[-1]]
    record=dict(annotation_bounds_ms=declared,perception_bounds_ms=actual,
                endpoint_delta_ms=[a-b for a,b in zip(declared,actual)],rounding_verified=False)
    if declared==actual:return record
    # Only accept a matching full-frame annotation clock, not an arbitrary tolerance
    # on prediction timestamps or a shortened/shifted perception stream.
    if annotation_pts is None:
        raise ValueError('Annotation/perception bounds differ without original annotation PTS: '+c)
    validate_timestamps(annotation_pts)
    if (len(annotation_pts)!=len(saved) or declared!=[annotation_pts[0],annotation_pts[-1]]
            or any(a-b not in (0,1) for a,b in zip(annotation_pts,saved))):
        raise ValueError('Annotation PTS mismatch beyond round-versus-floor milliseconds: '+c)
    record.update(rounding_verified=True,annotation_frames=len(annotation_pts),
                  rule='Verified full-frame round-vs-floor PTS; frozen labels unchanged; last perception packet not extrapolated')
    return record


def check_annotations(dataset, *, verify_media=True):
    manifest = read_csv(dataset/'dataset_manifest.csv')
    labels = read_csv(dataset/'annotations.csv')
    record = json.loads((dataset/'annotation_record.json').read_text())
    if digest(dataset/'annotations.csv') != record['annotations_sha256']:
        raise ValueError('Annotations changed after freeze; version and record the revision before scoring')
    if set(r['clip_id'] for r in labels) != set(r['clip_id'] for r in manifest):
        raise ValueError('Annotation and dataset clips differ')
    for row in manifest:
        clip = row['clip_id']
        if verify_media and digest(Path(row['source_path'])) != record['video_sha256'][clip]:
            raise ValueError('Original video changed: '+clip)
        cursor = int(row['eval_start_ms'])
        events = sorted((r for r in labels if r['clip_id']==clip), key=lambda r:int(r['start_ms']))
        if len({r['event_id'] for r in events}) != len(events):
            raise ValueError('Duplicate annotation IDs')
        for r in events:
            a,b = int(r['start_ms']), int(r['end_ms'])
            if a != cursor or b <= a or r['label'] not in {'SHOWING','NON_SHOWING'}:
                raise ValueError('Invalid annotation partition: '+clip)
            if bool(r['physical_target_id']) != (r['label']=='SHOWING'):
                raise ValueError('Invalid annotation target: '+clip)
            if not r['annotator'] or r['review_status'] not in {'PENDING','REVIEWED'}:
                raise ValueError('Missing annotation provenance')
            cursor = b
        if cursor != int(row['eval_end_ms']):
            raise ValueError('Incomplete annotation partition: '+clip)
    return manifest, labels


def summarize(scores):
    counts = {k:sum(s['temporal_events'][k] for s in scores) for k in ['tp','fp','fn']}
    tp,fp,fn = (counts[k] for k in ['tp','fp','fn'])
    pos = sum(s['positive_ms'] for s in scores)
    neg = sum(s['negative_ms'] for s in scores)
    false = sum(s['false_showing_ms'] for s in scores)
    errors = [e for s in scores for e in s['matched_boundary_errors']]
    return dict(events=prf(tp,tp+fp,tp+fn), temporal_coverage=sum(s['showing_ms']-s['false_showing_ms'] for s in scores)/pos if pos else None,
                false_seconds=false/1000, false_seconds_per_negative_minute=60*false/neg if neg else None,
                boundary_mae_ms={kind:sum(abs(e[kind+'_error_ms']) for e in errors if e[kind+'_error_ms'] is not None)/sum(e[kind+'_error_ms'] is not None for e in errors) if any(e[kind+'_error_ms'] is not None for e in errors) else None for kind in ['start','end']})


def evaluate(dataset, run, out):
    manifest, labels = check_annotations(dataset)
    if out.exists():
        raise ValueError('Evaluation output exists; choose a new file')
    report = dict(status='PROVISIONAL_ASSISTANT_LABELS_PENDING_USER_REVIEW',
                  scope='BINARY_TEMPORAL_NOT_INSTANCE_ACCURACY', timing='RETROSPECTIVE_CONTENT_NOT_REALTIME',
                  test_independence='New scene before tuning; participant/session independence unverified',
                  annotations_sha256=digest(dataset/'annotations.csv'), physical_metrics=None, clips={}, input_sha256={})
    report['evaluator_sha256']=digest(Path(__file__))
    report['audio_baseline_sha256']=digest(Path('src/intent/audio_timeline.py')) if Path('src/intent/audio_timeline.py').is_file() else None
    report['timebase_checks']={}
    annotation_record=json.loads((dataset/'annotation_record.json').read_text())
    audio_outputs={}
    mappings = []
    for row in manifest:
        c = row['clip_id']
        per = run/'perception'/f'perception_{c}.jsonl'
        packets = [json.loads(s) for s in per.read_text().splitlines() if s.strip()]
        source = Path(json.loads((run/'run_config.json').read_text())['original_run'])
        saved = json.loads((source/'timestamps'/f'{c}.json').read_text())['timestamp_ms']
        annotation_pts=None
        if annotation_record.get('sheet_manifest'):
            path=Path(annotation_record['sheet_manifest']).parent/c/'timestamps.json'
            if path.is_file():annotation_pts=json.loads(path.read_text())['timestamp_ms']
        report['timebase_checks'][c]=check_timebase(packets,saved,row,annotation_pts)
        raw_path = source/'speech_raw'/f'speech_events_{c}.jsonl'
        audio = AudioTimeline(read_events_jsonl(raw_path), timing='content')
        streams = {'audio-only':[audio.update(t) for t in saved]}
        audio_outputs[c]=streams['audio-only']
        for method in ['visual-only','showintent','pickup-baseline']:
            streams[method] = [json.loads(s) for s in (run/'results'/c/f'intent_{method}.jsonl').read_text().splitlines() if s.strip()]
        report['clips'][c] = {}
        target_ids = set()
        for method, values in streams.items():
            normalized = [dict(timestamp_ms=v['timestamp_ms'],state=v['state'],target_id='TEMPORAL_ONLY' if v['state']=='SHOWING' else None) for v in values]
            clock = [dict(timestamp_ms=t,objects=[]) for t in saved]
            score = score_clip(clock, normalized, [r for r in labels if r['clip_id']==c], [])
            report['clips'][c][method] = {k:score[k] for k in ['temporal_events','positive_ms','negative_ms','showing_ms','false_showing_ms','matched_boundary_errors','predicted_episodes']}
            if method in {'visual-only','showintent'}:
                target_ids.update(v['target_id'] for v in values if v['state']=='SHOWING')
        for track in sorted(target_ids):
            times = [p['timestamp_ms'] for p in packets if any(o['track_id']==track for o in p['objects'])]
            if times:
                mappings.append(dict(clip_id=c,track_id=track,start_ms=times[0],end_ms=min(saved[-1],times[-1]+1),physical_target_id='',review_status='PENDING',annotator='',reviewer='',notes='Verify against original video; split if identity changes'))
        report['input_sha256'][c] = dict(perception=digest(per),asr=digest(raw_path),predictions={m:digest(run/'results'/c/f'intent_{m}.jsonl') for m in ['visual-only','showintent','pickup-baseline']})
    report['summary'] = {m:summarize([r[m] for r in report['clips'].values()]) for m in ['visual-only','audio-only','showintent','pickup-baseline']}
    report['negative_clips'] = {c:{m:dict(false_seconds=r['false_showing_ms']/1000,triggered=r['showing_ms']>0) for m,r in v.items()} for c,v in report['clips'].items() if not v['showintent']['positive_ms']}
    out.parent.mkdir(parents=True,exist_ok=True)
    mapping_path=out.parent/'track_mapping_pending.csv'
    if mapping_path.exists():
        raise ValueError('Mapping output exists; choose a fresh evaluation directory')
    with mapping_path.open('w',newline='') as f:
        fields=['clip_id','track_id','start_ms','end_ms','physical_target_id','review_status','annotator','reviewer','notes']
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(mappings)
    for c,values in audio_outputs.items():
        dest=out.parent/'audio_only'/f'{c}.jsonl';dest.parent.mkdir(exist_ok=True)
        dest.write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
    out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))

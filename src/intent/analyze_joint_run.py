"""Audit completed joint outputs without running ASR or intent inference."""

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path
from statistics import median

from .evaluate_review import score_intervals
from .diagnose import showing_segments


def read_jsonl(path):
    return [json.loads(line) for line in path.read_text(encoding='utf-8').splitlines() if line.strip()]


def analyze_run(run_dir, perception_dir, review_path, *, replay_dir=None):
    config_path=run_dir/'run_config.json'
    config=json.loads(config_path.read_text(encoding='utf-8'))
    with review_path.open(encoding='utf-8') as stream:
        reviews=list(csv.DictReader(stream))
    replay_dir = replay_dir if replay_dir is not None else run_dir/'results'
    clips=[];queue=[];scored={};total_packets=0
    for clip in config['clips']:
        directory=replay_dir/clip
        report=json.loads((directory/'report.json').read_text(encoding='utf-8'))
        source=perception_dir/f'perception_{clip}.jsonl'
        speech_path=run_dir/'speech'/f'speech_events_{clip}.jsonl'
        if hashlib.sha256(source.read_bytes()).hexdigest()!=report['perception_sha256']:
            raise ValueError(f'{clip}: perception hash differs from recorded input')
        if hashlib.sha256(speech_path.read_bytes()).hexdigest()!=report['speech_sha256']:
            raise ValueError(f'{clip}: speech hash differs from recorded input')
        packets=read_jsonl(source);events=read_jsonl(speech_path)
        trace=read_jsonl(directory/'evidence_showintent.jsonl')
        meta=json.loads(speech_path.with_suffix('.meta.json').read_text(encoding='utf-8'))
        if meta['event_count']!=len(events):raise ValueError(f'{clip}: incomplete ASR events')
        timestamps=[p['timestamp_ms'] for p in packets]
        if [r['timestamp_ms'] for r in trace]!=timestamps:
            raise ValueError(f'{clip}: evidence/perception alignment error')
        if any(r['speech'].get('source_available_at_ms') is not None
               and r['speech']['source_available_at_ms']>r['timestamp_ms'] for r in trace):
            raise ValueError(f'{clip}: future speech in evidence trace')
        methods={};streams={};class_map={o['track_id']:o['class_name'] for p in packets for o in p['objects']}
        for mode in ('visual-only','showintent','pickup-baseline'):
            intents=read_jsonl(directory/f'intent_{mode}.jsonl')
            if [i['timestamp_ms'] for i in intents]!=timestamps:
                raise ValueError(f'{clip}: {mode} alignment error')
            streams[mode]=intents
            segments=showing_segments(intents)
            methods[mode]={'states':dict(Counter(i['state'] for i in intents)),
                           'segments':[{**row,'class_name':class_map.get(row['target_id'])} for row in segments]}
            scored.setdefault(mode,[]).extend(score_intervals(packets,intents,[r for r in reviews if r['clip_id']==clip]))
        for n,(visual,joint) in enumerate(zip(streams['visual-only'],streams['showintent'])):
            if n+1==len(timestamps):break
            vs,js=visual['state']=='SHOWING',joint['state']=='SHOWING'
            reason=('TARGET_ID_DISAGREEMENT' if vs and js and visual['target_id']!=joint['target_id'] else
                    'ADDITIONAL_JOINT_SHOWING' if js and not vs else 'JOINT_NOT_SHOWING' if vs and not js else None)
            if reason is None:continue
            key=(clip,reason,visual['target_id'] if vs else None,joint['target_id'] if js else None)
            speech=trace[n]['speech']
            source_key=tuple(speech.get(k) for k in ('source_start_ms','source_end_ms','source_available_at_ms'))
            key+=source_key
            if queue and queue[-1]['_key']==key and queue[-1]['end_ms']==timestamps[n]:
                queue[-1]['end_ms']=timestamps[n+1]
                continue
            event=next((e for e in events if (e['start_ms'],e['end_ms'],e['available_at_ms'])==source_key),{})
            queue.append(dict(_key=key,clip_id=clip,start_ms=timestamps[n],end_ms=timestamps[n+1],reason=reason,
                              visual_target=visual['target_id'] if vs else '',joint_target=joint['target_id'] if js else '',
                              visual_class=class_map.get(visual['target_id'],'') if vs else '',
                              joint_class=class_map.get(joint['target_id'],'') if js else '',
                              source_start_ms=source_key[0],source_end_ms=source_key[1],available_at_ms=source_key[2],
                              transcript=event.get('text',''),review_status='PENDING',notes=''))
        delays=[e['available_at_ms']-e['end_ms'] for e in events]
        clips.append(dict(clip_id=clip,packet_count=len(packets),asr_event_count=len(events),
                          asr_mode=meta['mode'],asr_compute_rtf=meta['compute_rtf'],
                          median_event_delay_ms=median(delays) if delays else None,
                          events_available_after_clip=sum(e['available_at_ms']>timestamps[-1] for e in events),
                          explicit_presentation_events=sum('PRESENTATION_LANGUAGE' in e['cue_types'] for e in events),
                          speech_reason_counts=dict(Counter(c for r in trace for c in r['speech']['reason_codes'])),
                          methods=methods))
        total_packets+=len(packets)
    summary={}
    for mode,rows in scored.items():
        positives=[r for r in rows if r['label']=='SHOWING'];negatives=[r for r in rows if r['label']=='NON_SHOWING']
        positive_ms=sum(r['covered_ms'] for r in positives)
        summary[mode]=dict(evaluation_status='PROVISIONAL_CORE_REVIEW' if rows else 'NO_MATCHING_REVIEW_INTERVALS',
                          review_interval_count=len(rows),positive_core_ms=positive_ms,
                          correct_class_coverage=sum(r['correct_class_showing_ms'] for r in positives)/positive_ms if positive_ms else None,
                          wrong_class_ms=sum(r['wrong_class_showing_ms'] for r in positives),
                          negative_core_ms=sum(r['covered_ms'] for r in negatives),
                          showing_in_negative_cores_ms=sum(r['showing_ms'] for r in negatives))
    result=dict(scope='audit_of_completed_development_run_not_independent_accuracy',review_status='NEEDS_HUMAN_REVIEW',
                run_config_sha256=hashlib.sha256(config_path.read_bytes()).hexdigest(),
                review_sha256=hashlib.sha256(review_path.read_bytes()).hexdigest(),
                current_source_differences=[p for p,h in config['source_sha256'].items()
                                            if not Path(p).is_file() or hashlib.sha256(Path(p).read_bytes()).hexdigest()!=h],
                replay_directory=str(replay_dir),
                total_packets=total_packets,clip_count=len(clips),summary=summary,clips=clips)
    return result,[{k:v for k,v in row.items() if k!='_key'} for row in queue]


def main():
    parser=argparse.ArgumentParser(description='分析已完成的视听结果，不运行 ASR 或 B 推理')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--replay-dir',type=Path,help='Optional B-only replay directory; speech still comes from --run-dir')
    parser.add_argument('--perception-dir',type=Path,default=Path('outputs/perception_real'))
    parser.add_argument('--review',type=Path,default=Path('evaluation/development_review.csv'))
    parser.add_argument('--out',type=Path,required=True,help='New audit directory')
    args=parser.parse_args()
    if args.out.exists():parser.error('audit output already exists; choose a new directory')
    result,queue=analyze_run(args.run_dir,args.perception_dir,args.review,replay_dir=args.replay_dir)
    args.out.mkdir(parents=True)
    (args.out/'audit.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    with (args.out/'review_queue.csv').open('w',encoding='utf-8',newline='') as stream:
        fields=list(queue[0]) if queue else ['clip_id','start_ms','end_ms','reason','review_status']
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(queue)
    print(json.dumps({'clips':result['clip_count'],'packets':result['total_packets'],'summary':result['summary'],
                      'review_windows':len(queue)},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

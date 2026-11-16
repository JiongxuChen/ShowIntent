"""Time-weighted checks on provisional, visually reviewed development cores.

This is neither independent evaluation nor exact event-boundary/instance truth.
Labels are consumed only after intent inference.
"""

from __future__ import annotations

import argparse
import csv
import json
import hashlib
from pathlib import Path


def score_intervals(perception: list[dict], intents: list[dict], intervals: list[dict]) -> list[dict]:
    if len(perception)!=len(intents) or any(p['timestamp_ms']!=i['timestamp_ms'] for p,i in zip(perception,intents)):
        raise ValueError('unaligned perception / intent packets')
    results=[]
    for row in intervals:
        classes={}
        start,end=int(row['start_ms']),int(row['end_ms'])
        if end<=start or row['label'] not in ('SHOWING','NON_SHOWING'):
            raise ValueError('invalid review interval')
        allowed=set(row.get('allowed_classes','').split('|'))-{''}
        if row['label']=='SHOWING' and not allowed:
            raise ValueError('showing interval needs allowed classes')
        covered=showing=correct=wrong=0
        for n,(p,it) in enumerate(zip(perception,intents)):
            for o in p['objects']:classes[o['track_id']]=o['class_name']
            nxt=perception[n+1]['timestamp_ms'] if n+1<len(perception) else p['timestamp_ms']
            dt=max(0,min(end,nxt)-max(start,p['timestamp_ms']))
            covered+=dt
            if it['state']=='SHOWING':
                showing+=dt
                if classes.get(it['target_id']) in allowed:correct+=dt
                elif allowed:wrong+=dt
        results.append({**row,'covered_ms':covered,'showing_ms':showing,
                        'correct_class_showing_ms':correct,'wrong_class_showing_ms':wrong,
                        'correct_class_coverage':correct/covered if covered and allowed else None})
    return results


def main() -> None:
    parser=argparse.ArgumentParser(description='仅评分已抽帧复核的开发区间，非正式事件 F1')
    parser.add_argument('--perception-dir',type=Path,required=True)
    parser.add_argument('--replay-dir',type=Path,required=True)
    parser.add_argument('--review',type=Path,default=Path('evaluation/development_review.csv'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    with args.review.open(encoding='utf-8') as f:reviews=list(csv.DictReader(f))
    def load(path):return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    all_results={}
    for clip in sorted({r['clip_id'] for r in reviews}):
        perception=load(args.perception_dir/f'perception_{clip}.jsonl')
        for path in sorted((args.replay_dir/clip).glob('intent_*.jsonl')):
            mode=path.stem.removeprefix('intent_')
            all_results.setdefault(mode,[]).extend(score_intervals(perception,load(path),[r for r in reviews if r['clip_id']==clip]))
    summaries={}
    for mode,rows in all_results.items():
        positives=[r for r in rows if r['label']=='SHOWING'];negatives=[r for r in rows if r['label']=='NON_SHOWING']
        positive_ms=sum(r['covered_ms'] for r in positives)
        summaries[mode]={'positive_core_ms':positive_ms,
                         'correct_class_positive_coverage':sum(r['correct_class_showing_ms'] for r in positives)/positive_ms,
                         'wrong_class_ms':sum(r['wrong_class_showing_ms'] for r in positives),
                         'negative_core_ms':sum(r['covered_ms'] for r in negatives),
                         'false_showing_ms_in_negative_cores':sum(r['showing_ms'] for r in negatives),
                         'positive_cores_with_correct_trigger':sum(r['correct_class_showing_ms']>0 for r in positives),
                         'positive_core_count':len(positives)}
    result={'scope':'provisional_development_core_review_not_event_F1_or_instance_accuracy',
            'review_status':'NEEDS_HUMAN_REVIEW',
            'review_sha256':hashlib.sha256(args.review.read_bytes()).hexdigest(),
            'evaluator_sha256':hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            'replay_directory':str(args.replay_dir.resolve()),
            'summary':summaries,'intervals':all_results}
    args.out.parent.mkdir(parents=True,exist_ok=True)
    args.out.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(summaries,ensure_ascii=False,indent=2))


if __name__=='__main__':main()

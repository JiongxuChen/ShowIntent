"""Compare peer ablations; extra predictions are review candidates, not wins."""

import argparse
import csv
import json
from pathlib import Path
from collections import Counter

from .evaluate_review import score_intervals


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def episodes(rows):
    # Adjacent SHOWING ID fragments are counted as one continuous interval.
    result=[]
    for i,row in enumerate(rows[:-1]):
        if row['state']!='SHOWING':continue
        end=rows[i+1]['timestamp_ms']
        if result and result[-1]['end_ms']==row['timestamp_ms']:
            result[-1]['end_ms']=end
            if row['target_id'] not in result[-1]['target_ids']:result[-1]['target_ids'].append(row['target_id'])
        else:
            result.append(dict(start_ms=row['timestamp_ms'],end_ms=end,target_ids=[row['target_id']]))
    return result


def analyze(run_dir, *, review=None):
    config=json.loads((run_dir/'run_config.json').read_text())
    reviews=list(csv.DictReader(review.open())) if review and review.is_file() else []
    clips=[];queue=[]
    for clip in config['clips']:
        directory=run_dir/'results'/clip
        packets=read(run_dir/'perception'/f'perception_{clip}.jsonl')
        streams={m:read(directory/f'intent_{m}.jsonl') for m in
                 ('visual-only','speech-with-localization','showintent','pickup-baseline')}
        if (directory/'intent_visual-policy-only.jsonl').exists():
            streams['visual-policy-only']=read(directory/'intent_visual-policy-only.jsonl')
        timestamps=[p['timestamp_ms'] for p in packets]
        if any([r['timestamp_ms'] for r in rows]!=timestamps for rows in streams.values()):
            raise ValueError('unaligned comparison')
        trace=read(directory/'evidence_showintent.jsonl')
        if [r['timestamp_ms'] for r in trace]!=timestamps:raise ValueError('unaligned trace')
        if any(r['speech'].get('source_available_at_ms',0) is not None and
               r['speech']['source_available_at_ms']>r['timestamp_ms'] for r in trace):
            raise ValueError('future speech in comparison')
        methods={}
        for method,rows in streams.items():
            cores=score_intervals(packets,rows,[r for r in reviews if r['clip_id']==clip])
            positives=[r for r in cores if r['label']=='SHOWING']
            negatives=[r for r in cores if r['label']=='NON_SHOWING']
            positive=sum(r['covered_ms'] for r in positives)
            methods[method]=dict(episodes=episodes(rows),
                showing_ms=sum(timestamps[i+1]-timestamps[i] for i,r in enumerate(rows[:-1]) if r['state']=='SHOWING'),
                selected_core_status='PROVISIONAL_NOT_ACCURACY' if cores else 'NO_MATCHING_REVIEW_INTERVALS',
                selected_correct_class_coverage=sum(r['correct_class_showing_ms'] for r in positives)/positive if positive else None,
                selected_negative_showing_ms=sum(r['showing_ms'] for r in negatives),
                selected_wrong_class_ms=sum(r['wrong_class_showing_ms'] for r in positives))
        counts=Counter()
        for i in range(len(timestamps)-1):
            t,end=timestamps[i],timestamps[i+1];dt=end-t
            v=streams['visual-only'][i];s=streams['speech-with-localization'][i];j=streams['showintent'][i]
            vs,ss,js=(r['state']=='SHOWING' for r in (v,s,j))
            reason=None
            if js and not vs and not ss:reason='JOINT_ONLY_NEEDS_REVIEW'
            elif js and not vs:reason='LANGUAGE_ADDED_NEEDS_REVIEW'
            elif vs and not js:reason='WITHHELD_VISUAL_NEEDS_REVIEW'
            elif vs and js and v['target_id']!=j['target_id']:reason='TARGET_DISAGREEMENT'
            view=trace[i]['candidates'].get(j['target_id'],{})
            if (js and 0<view.get('visual',0)<1 and 0<view.get('speech',0)<1
                    and not view.get('visual_full',view.get('visual',0)>=1)):
                counts['partial_evidence_joint_showing_ms']+=dt
            if 'visual-policy-only' in streams:
                policy=streams['visual-policy-only'][i]
                ps=policy['state']=='SHOWING'
                if js and not ps:counts['added_vs_same_visual_policy_ms']+=dt
                if ps and not js:counts['withheld_vs_same_visual_policy_ms']+=dt
            if reason:
                counts[reason]+=dt
                key=(clip,reason,v['target_id'],s['target_id'],j['target_id'])
                if queue and queue[-1]['_key']==key and queue[-1]['end_ms']==t:
                    queue[-1]['end_ms']=end
                else:
                    queue.append(dict(_key=key,clip_id=clip,start_ms=t,end_ms=end,reason=reason,
                                      visual_target=v['target_id'],speech_target=s['target_id'],joint_target=j['target_id'],
                                      review_status='PENDING',notes=''))
        clips.append(dict(clip_id=clip,methods=methods,diagnostic_durations_ms=dict(counts),
                          grounding_counts=dict(Counter(r['grounding_status'] for r in trace))))
    return dict(status='DIAGNOSTICS_NOT_PROOF_OF_IMPROVEMENT',clips=clips,
                explanation='Joint-only durations need independent event/instance review; they are not accuracy or benefit.'),[
                    {k:v for k,v in r.items() if k!='_key'} for r in queue]


def main():
    parser=argparse.ArgumentParser(description='Compare visual, speech with localization, joint, and pickup outputs')
    parser.add_argument('--run-dir',type=Path,required=True)
    parser.add_argument('--review',type=Path,default=Path('evaluation/development_review.csv'))
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):parser.error('analysis output exists')
    result,queue=analyze(args.run_dir,review=args.review)
    args.out.mkdir(parents=True,exist_ok=True)
    (args.out/'complementarity.json').write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n')
    fields=list(queue[0]) if queue else ['clip_id','start_ms','end_ms','reason','review_status','notes']
    with (args.out/'review_windows.csv').open('w',newline='',encoding='utf-8') as stream:
        writer=csv.DictWriter(stream,fieldnames=fields);writer.writeheader();writer.writerows(queue)
    print(f'Wrote {len(result["clips"])} clip comparisons; {len(queue)} windows require review.')


if __name__=='__main__':main()

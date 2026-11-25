"""Validate independent labels and score fixed B predictions, no inference."""
import argparse,json,hashlib,statistics
from pathlib import Path
from src.intent.formal_evaluation import read_csv,validate_labels,score_clip


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,default=Path('evaluation/fruit'));p.add_argument('--check-only',action='store_true')
    p.add_argument('--clips',nargs='+');p.add_argument('--method',default='showintent',choices=['showintent','visual-only','pickup-baseline','visual-policy-only'])
    p.add_argument('--event-iou',type=float,default=.3);p.add_argument('--temporal-only',action='store_true');p.add_argument('--out',type=Path,required=True);args=p.parse_args()
    if args.out.exists():p.error('choose a new output file')
    if not 0<args.event_iou<=1:p.error('event IoU must be in (0,1]')
    files={name:args.dataset/name for name in ['dataset_manifest.csv','annotations.csv','track_mapping.csv','split_manifest.csv','protocol.json']}
    protocol=json.loads(files['protocol.json'].read_text())
    if args.event_iou!=protocol['event_iou_threshold'] or protocol['physical_match_min_correct_overlap_fraction']!=.5:
        p.error('parameters differ from protocol; version the protocol and evaluator before scoring')
    manifest=read_csv(files['dataset_manifest.csv']);clips=args.clips or [r['clip_id'] for r in manifest]
    if len(set(clips))!=len(clips) or set(clips)-{r['clip_id'] for r in manifest}:p.error('invalid clip subset')
    manifest=[r for r in manifest if r['clip_id'] in clips]
    annotations=[r for r in read_csv(files['annotations.csv']) if r['clip_id'] in clips]
    mappings=[r for r in read_csv(files['track_mapping.csv']) if r['clip_id'] in clips]
    # Check split leakage against the FULL dataset even when scoring a subset.
    allmanifest=read_csv(files['dataset_manifest.csv']);splits=read_csv(files['split_manifest.csv'])
    errors=validate_labels(annotations,mappings,manifest,splits)
    if not args.temporal_only and any(r.get("class_name")=="fruit_group" for r in annotations):
        errors.append("D01: simultaneous fruit group requires a versioned multi-target instance protocol; single-target scoring is blocked (temporal-only remains available after review)")
    for name in ['annotations.csv','track_mapping.csv','split_manifest.csv']:
        if any(r['clip_id'] not in {m['clip_id'] for m in allmanifest} for r in read_csv(files[name])):
            errors.append(f'{name}: contains unknown clip IDs')
    leakage=validate_labels([],[],allmanifest,splits)
    errors.extend(e for e in leakage if 'leakage' in e or 'development data' in e)
    report=dict(status='NOT_READY' if errors else 'READY',errors=sorted(set(errors)),
        split_grouping='recording_session',scope='human_reviewed_development_regression' if all(r['used_for_development']=='true' for r in manifest) else 'independence_requires_verified_split',
        dataset_designation={r['clip_id']:r['split'] for r in splits if r['clip_id'] in clips},test_independence='books previously used for development; test by explicit user designation',method=args.method,metric_scope='temporal_only_without_verified_instance_mapping' if args.temporal_only else 'temporal_and_physical',event_iou=args.event_iou,physical_overlap_fraction=.5,annotation_sha256={k:hashlib.sha256(v.read_bytes()).hexdigest() for k,v in files.items()},
        evaluator_sha256=hashlib.sha256(Path('src/intent/formal_evaluation.py').read_bytes()).hexdigest())
    if not args.check_only and not errors:
        results={};hashes={}
        for row in manifest:
            c=row['clip_id'];per=Path(row['perception_path']);pred=Path(row['result_root'])/'results'/c/f'intent_{args.method}.jsonl'
            if hashlib.sha256(per.read_bytes()).hexdigest()!=row['perception_sha256']:p.error(f'{c}: perception changed')
            packets=[json.loads(l) for l in per.read_text().splitlines() if l.strip()];intents=[json.loads(l) for l in pred.read_text().splitlines() if l.strip()]
            results[c]=score_clip(packets,intents,[r for r in annotations if r['clip_id']==c],[r for r in mappings if r['clip_id']==c],args.event_iou)
            hashes[str(pred)]=hashlib.sha256(pred.read_bytes()).hexdigest()
        if not args.temporal_only and any(r['physical_events'] is None for r in results.values()):
            report['status']='INCOMPLETE_INSTANCE_MAPPING'
            report['errors'].append('Reviewed mapping missing for predicted SHOWING; no physical-event score is claimed')
        else:report['status']='SCORED'
        if args.temporal_only:
            for result in results.values():
                result['physical_events']=None
                result['physical_metric_status']='NOT_EVALUATED_TEMPORAL_ONLY'
                for field in ['correct_target_ms','wrong_target_ms','correct_target_coverage']:
                    result[field]=None
        report.update(clips=results,prediction_sha256=hashes)
        def aggregate(key):
            if any(r[key] is None for r in results.values()):return None
            totals={k:sum(r[key][k] for r in results.values()) for k in ['tp','fp','fn']};t=totals
            totals.update(precision=t['tp']/(t['tp']+t['fp']) if t['tp']+t['fp'] else None,
                recall=t['tp']/(t['tp']+t['fn']) if t['tp']+t['fn'] else None,
                f1=2*t['tp']/(2*t['tp']+t['fp']+t['fn']) if 2*t['tp']+t['fp']+t['fn'] else None)
            return totals
        report['micro_summary']=dict(temporal_events=aggregate('temporal_events'),physical_events=aggregate('physical_events'),
            false_showing_ms=sum(r['false_showing_ms'] for r in results.values()),negative_ms=sum(r['negative_ms'] for r in results.values()),
            correct_target_ms=None if args.temporal_only else sum(r['correct_target_ms'] for r in results.values()),positive_ms=sum(r['positive_ms'] for r in results.values()),
            wrong_target_ms=None if args.temporal_only else sum(r['wrong_target_ms'] for r in results.values()),unmapped_target_ms=sum(r['unmapped_target_ms'] for r in results.values()),
            extra_fragments=sum(r['extra_fragments'] for r in results.values()),missing_visible_target_ms=sum(r['missing_visible_target_ms'] for r in results.values()))
    if 'micro_summary' in report:
        summary=report['micro_summary']
        summary['correct_target_coverage']=summary['correct_target_ms']/summary['positive_ms'] if summary['positive_ms'] and summary['correct_target_ms'] is not None else None
        summary['temporal_showing_coverage']=sum(r['showing_ms']-r['false_showing_ms'] for r in results.values())/summary['positive_ms'] if summary['positive_ms'] else None
        summary['false_seconds_per_negative_minute']=60*summary['false_showing_ms']/summary['negative_ms'] if summary['negative_ms'] else None
        for boundary in ['start','end']:
            values=[e[boundary+'_error_ms'] for r in results.values() for e in r['matched_boundary_errors'] if e[boundary+'_error_ms'] is not None]
            summary[boundary+'_mae_ms']=statistics.mean(abs(v) for v in values) if values else None
            summary[boundary+'_signed_mean_ms']=statistics.mean(values) if values else None
    args.out.parent.mkdir(parents=True,exist_ok=True);args.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(f'{report["status"]}: {len(clips)} clips; {len(report["errors"])} issues. See {args.out}',flush=True)

    if not args.check_only and report['status']!='SCORED':raise SystemExit(2)


if __name__=='__main__':main()

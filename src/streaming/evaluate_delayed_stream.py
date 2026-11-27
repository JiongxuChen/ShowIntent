"""Evaluate actual playback states against frozen labels and a common offline horizon."""
import argparse
import bisect
import json
from pathlib import Path

from src.intent.formal_evaluation import read_csv,score_clip
from scripts.run_complementary import digest
from src.evaluation.collection import summarize
from src.evaluation.registry import annotation_rows,extra_books,result_dir,book_addition


def read_rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def hold(rows,times,stale_ms=None):
    if not rows or any(a['timestamp_ms']>=b['timestamp_ms'] for a,b in zip(rows,rows[1:])):
        raise ValueError('Missing or nonmonotonic state timeline')
    clock=[r['timestamp_ms'] for r in rows];out=[]
    for t in times:
        i=bisect.bisect_right(clock,t)-1
        row=rows[i] if i>=0 and (stale_ms is None or t-clock[i]<=stale_ms) else dict(state='NONE',target_id=None)
        out.append(dict(row,timestamp_ms=t))
    return out


def evaluate(run,dataset,clip,reference=Path('outputs/results'),collection='drinks',allow_legacy=False):
    metrics=json.loads((run/'metrics.json').read_text())
    if metrics['status']!='finished' or metrics['forced_worker_shutdown'] or metrics['worker_exitcodes']!=[0,0]:
        raise ValueError('Failed stream cannot enter quality summary')
    manifest=read_csv(dataset/'dataset_manifest.csv')
    if Path(dataset)==Path('evaluation'):manifest+=extra_books()[0]
    row=next(r for r in manifest if r['clip_id']==clip)
    labels=[r for r in annotation_rows(dataset) if r['clip_id']==clip]
    video=Path(metrics['configuration']['video']);side=video.with_suffix('.source.json')
    provenance=False
    if side.exists():
        info=json.loads(side.read_text())
        if info.get('version')!=2 or info['source_sha256']!=digest(Path(row['source_path'])) or info['proxy_sha256']!=digest(video):
            raise ValueError('Proxy source provenance mismatch')
        provenance=True
    elif not allow_legacy:raise ValueError('Proxy has no source sidecar; prepare a new proxy or explicitly allow legacy diagnosis')
    playback=read_rows(run/'playback-intent.jsonl')
    if len(playback)!=metrics['output_frames'] or metrics['input_frames']!=metrics['output_frames']:
        raise ValueError('Incomplete playback frame log')
    if len(playback)<2:raise ValueError('Too few frames')
    offline={m:read_rows(result_dir(reference,collection,clip)/(m+'.jsonl')) for m in ['showintent','visual-only']}
    step=round(1000/metrics.get('input_fps',15))
    left=max(int(row['eval_start_ms']),playback[0]['timestamp_ms'],*(v[0]['timestamp_ms'] for v in offline.values()))
    right=min(int(row['eval_end_ms']),playback[-1]['timestamp_ms']+step,*(v[-1]['timestamp_ms'] for v in offline.values()))
    if right<=left:raise ValueError('No common evaluation horizon')
    times=sorted({left,right,*[v['timestamp_ms'] for v in playback if left<v['timestamp_ms']<right]})
    cropped=[dict(r,start_ms=max(left,int(r['start_ms'])),end_ms=min(right,int(r['end_ms']))) for r in labels if int(r['end_ms'])>left and int(r['start_ms'])<right]
    if sum(r['end_ms']-r['start_ms'] for r in cropped)!=right-left:raise ValueError('Label partition does not cover common horizon')
    visual_path=run/'playback-visual.jsonl';derived=not visual_path.exists()
    visual=read_rows(visual_path) if not derived else hold(read_rows(run/'visual-only.jsonl'),[r['timestamp_ms'] for r in playback],500)
    if [v['timestamp_ms'] for v in visual]!=[v['timestamp_ms'] for v in playback]:raise ValueError('Playback visual alignment mismatch')
    streams={'stream_av':hold(playback,times),'stream_visual':hold(visual,times),**{'offline_'+m:hold(v,times) for m,v in offline.items()}}
    scores={}
    for name,values in streams.items():
        normalized=[dict(timestamp_ms=v['timestamp_ms'],state=v['state'],target_id='TEMPORAL_ONLY' if v['state']=='SHOWING' else None) for v in values]
        score=score_clip([dict(timestamp_ms=t,objects=[]) for t in times],normalized,cropped,[])
        scores[name]={k:score[k] for k in ['temporal_events','positive_ms','negative_ms','showing_ms','false_showing_ms','matched_boundary_errors','predicted_episodes']}
    window_path=run/'speech-windows.jsonl'
    windows=read_rows(window_path) if window_path.exists() else []
    speech_runtime={}
    if windows:
        total_audio=sum(w['end_ms']-w['start_ms'] for w in windows)
        total_compute=sum(w['compute_ms'] for w in windows)
        speech_runtime=dict(received_windows=len(windows),compute_rtf=total_compute/total_audio if total_audio else None,mean_window_compute_ms=total_compute/len(windows),last_received_audio_end_ms=windows[-1]['end_ms'],last_window_ready_ms=windows[-1]['wall_ready_ms'],max_audio_result_lag_ms=max(w['wall_ready_ms']-w['end_ms'] for w in windows),note='Playback-received windows only; late results discarded after output are excluded')
    compute_path=run/'vision-compute.jsonl'
    compute=read_rows(compute_path) if compute_path.exists() else []
    vision_runtime={}
    if compute:
        costs=sorted(r['compute_ms'] for r in compute)
        vision_runtime=dict(recorded_updates=len(compute),mean_update_ms=sum(costs)/len(costs),p95_update_ms=costs[min(len(costs)-1,int(.95*len(costs)))],full_model_calls=sum(r['full_model_calls'] for r in compute),crop_model_calls=sum(r['crop_model_calls'] for r in compute),crop_images=sum(r['crop_images'] for r in compute),note='Playback-received diagnostics; model call count differs from number of inferred images')
    timing=read_rows(run/'timing.jsonl');lags=sorted(max(0,r['written_at_ms']-r['deadline_ms']) for r in timing)
    evidence_files=[dataset/'annotations.csv',dataset/'dataset_manifest.csv',run/'playback-intent.jsonl',run/'metrics.json']
    if clip=='E04' and Path(dataset)==Path('evaluation'):
        evidence_files += [book_addition()/f for f in ['annotations.csv','dataset_manifest.csv','annotation_record.json']]
    return dict(clip_id=clip,collection=collection,scope='BINARY_TEMPORAL_PLAYBACK_B_STATE_COMMON_HORIZON_NOT_CROP_OR_INSTANCE_ACCURACY',proxy_provenance_verified=provenance,legacy_visual_derived=derived,horizon_ms=[left,right],full_label_horizon=[int(row['eval_start_ms']),int(row['eval_end_ms'])],partial=left!=int(row['eval_start_ms']) or right!=int(row['eval_end_ms']),annotation_review=sorted({r['review_status'] for r in cropped}),books_test_caveat='Books E01-E03 previously used in development; E04 new same-scene negative, independence unverified',input_sha256={str(p):digest(p) for p in evidence_files},scores=scores,summary={m:summarize([s]) for m,s in scores.items()},runtime=dict(metrics,speech_runtime=speech_runtime,vision_runtime=vision_runtime,playback_lateness_p50_ms=lags[len(lags)//2],playback_lateness_p95_ms=lags[min(len(lags)-1,int(.95*len(lags)))],playback_lateness_over_100ms_frames=sum(x>100 for x in lags)))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--dataset',type=Path,required=True)
    p.add_argument('--clip',required=True);p.add_argument('--collection',choices=['fruit','books','cups','drinks'],required=True)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--allow-legacy-proxy',action='store_true')
    a=p.parse_args()
    if a.out.exists():p.error('Choose a fresh output file')
    report=evaluate(a.run,a.dataset,a.clip,collection=a.collection,allow_legacy=a.allow_legacy_proxy)
    a.out.parent.mkdir(parents=True,exist_ok=True);a.out.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(report['summary'],ensure_ascii=False,indent=2))


if __name__=='__main__':main()

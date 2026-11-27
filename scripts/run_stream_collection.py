"""User-launched paced inference suite. Keeps every failed run and supports resume."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
import uuid

from src.intent.formal_evaluation import read_csv
from scripts.run_complementary import digest
from src.evaluation.collection import summarize
from src.streaming.evaluate_delayed_stream import evaluate
from src.evaluation.registry import extra_books

PILOT={'fruit':['A02','B03'],'books':None,'cups':['F02','F04'],'drinks':['G03','G04']}


def selected(suite,collections):
    for name in collections:
        dataset=Path('evaluation')/name
        rows=read_csv(dataset/'dataset_manifest.csv')
        if name in {'fruit','books'}:rows=[r for r in rows if r['source_collection']==name.replace('fruit','fruit_original').replace('books','books_original')]
        if name=='books':rows+=extra_books()[0]
        if suite=='pilot':
            wanted=PILOT[name] if PILOT[name] is not None else [rows[0]['clip_id']]
            lookup={r['clip_id']:r for r in rows}
            rows=[lookup[c] for c in wanted if c in lookup]
            if len(rows)!=len(wanted):raise ValueError('Missing pilot clips: '+name)
        for row in rows:yield name,dataset,row


def exclude_tasks(tasks,excluded):
    if len(set(excluded))!=len(excluded):raise ValueError('Duplicate excluded clips')
    known={r['clip_id'] for _,_,r in tasks}
    if set(excluded)-known:raise ValueError('Excluded clips not in selection: '+str(sorted(set(excluded)-known)))
    return [(n,d,r) for n,d,r in tasks if r['clip_id'] not in set(excluded)]


def audio_preview(run,proxy):
    from src.streaming.prepare_stream_proxy import resolve_ffmpeg
    target=run/'preview_with_audio.mp4'
    if target.exists():return
    temporary=run/('.preview_'+uuid.uuid4().hex+'.mp4')
    try:
        subprocess.run([resolve_ffmpeg(),'-nostdin','-n','-loglevel','error','-i',str(run/'silent.mp4'),'-i',str(proxy),'-map','0:v:0','-map','1:a:0?','-c:v','copy','-c:a','aac','-shortest',str(temporary)],check=True)
        temporary.replace(target)
    finally:
        temporary.unlink(missing_ok=True)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--out',type=Path,required=True);p.add_argument('--suite',choices=['pilot','all'],default='pilot')
    p.add_argument('--collections',nargs='+',choices=list(PILOT),default=list(PILOT))
    p.add_argument('--exclude-clips',nargs='*',default=[])
    p.add_argument('--clips',nargs='+',help='Current clip IDs to include; excludes are applied afterwards')
    p.add_argument('--reference',type=Path,default=Path('outputs/results'),help='Offline reference for common-horizon evaluation')
    p.add_argument('--audio-preview',action='store_true',help='Save preview with source audio, if present')
    p.add_argument('--resume',action='store_true');p.add_argument('--retry-failed',action='store_true');p.add_argument('--dry-run',action='store_true')
    p.add_argument('--delay',type=float,default=6);p.add_argument('--detect-fps',type=float,default=10);p.add_argument('--imgsz',type=int,default=1280)
    p.add_argument('--audio-window',type=float,default=4,help='Completed audio window in seconds; default 4 validated on drinks pilot')
    p.add_argument('--vision-profile',choices=['enhanced'],default='enhanced')
    p.add_argument('--asr',type=Path,default=Path('models/faster-whisper-base'))
    p.add_argument('--asr-device',choices=['cpu','cuda'],default='cpu')
    p.add_argument('--asr-beam-size',type=int,default=1)
    p.add_argument('--intent-profile',choices=['adapted'],default='adapted')
    p.add_argument('--height',type=int,default=1280);p.add_argument('--fps',type=int,default=15)
    a=p.parse_args()
    try:
        tasks=list(selected(a.suite,a.collections))
        if a.clips:
            known={r['clip_id'] for _,_,r in tasks}
            if len(a.clips)!=len(set(a.clips)) or set(a.clips)-known:raise ValueError('Invalid included clips')
            tasks=[t for t in tasks if t[2]['clip_id'] in a.clips]
        tasks=exclude_tasks(tasks,a.exclude_clips)
    except ValueError as exc:p.error(str(exc))
    if not tasks:p.error('No clips remain')
    if len(set(a.collections))!=len(a.collections):p.error('Duplicate collections')
    if a.dry_run:
        print(json.dumps([dict(collection=n,clip=r['clip_id'],source=r['source_path']) for n,d,r in tasks],ensure_ascii=False,indent=2));return
    if a.delay<=0 or a.detect_fps<=0 or a.imgsz<128 or a.height<128 or a.fps<5 or a.audio_window<1:p.error('Invalid timing/size')
    if a.out.exists() and not a.resume:p.error('Use a new output directory or --resume')
    if not (a.reference/'evaluation.json').is_file():p.error('Offline reference evaluation.json missing')
    sources=[Path('scripts')/x for x in ['run_delayed_stream.py','stream_workers.py','stream_perception.py','stream_timeline.py','stream_process.py','stream_shutdown.py','evaluate_delayed_stream.py','prepare_stream_proxy.py','run_stream_collection.py']]
    sources += list(Path('src/perception').glob('*.py'))
    sources += list(Path('src/intent').glob('*.py'))+list(Path('src/speech').glob('*.py'))
    sources += [Path('configs/presentation_intent.json'),Path('configs/semantic_priority_fusion.json')]
    plan=dict(version=4,reference=str(a.reference),reference_sha256=digest(a.reference/'evaluation.json'),exclude_clips=a.exclude_clips,include_clips=a.clips,intent_profile=a.intent_profile,audio_preview=a.audio_preview,suite=a.suite,collections=a.collections,delay=a.delay,detect_fps=a.detect_fps,imgsz=a.imgsz,height=a.height,fps=a.fps,asr=str(a.asr),asr_device=a.asr_device,asr_beam_size=a.asr_beam_size,vision_profile=a.vision_profile,audio_window=a.audio_window,source_sha256={str(f):digest(f) for f in sources},clips=[dict(collection=n,clip_id=r['clip_id'],source=r['source_path'],sha256=digest(Path(r['source_path']))) for n,d,r in tasks])
    a.out.mkdir(parents=True,exist_ok=True);planfile=a.out/'plan.json'
    if planfile.exists():
        if json.loads(planfile.read_text())!=plan:raise ValueError('Plan/source changed; choose a new output directory')
    else:planfile.write_text(json.dumps(plan,ensure_ascii=False,indent=2)+'\n')
    reports={};failures={}
    for name,dataset,row in tasks:
        clip=row['clip_id'];key=name+'/'+clip;run=a.out/'runs'/name/clip
        proxy=a.out/'proxies'/name/(clip+'.mp4');proxy.parent.mkdir(parents=True,exist_ok=True)
        logfile=a.out/'logs'/(name+'_'+clip+'.log');logfile.parent.mkdir(exist_ok=True)
        try:
            if run.exists():
                ready=(run/'metrics.json').exists() and json.loads((run/'metrics.json').read_text()).get('status')=='finished'
                if not ready:
                    if not a.retry_failed:raise ValueError('Incomplete previous run; --resume --retry-failed preserves it and retries')
                    failure=a.out/'failed_runs'/name/(clip+'_'+str(time.time_ns()));failure.parent.mkdir(parents=True,exist_ok=True);run.rename(failure)
            if proxy.exists() and not proxy.with_suffix('.source.json').exists() and a.retry_failed:
                failed_proxy=a.out/'failed_runs'/'proxies'/(name+'_'+clip+'_'+str(time.time_ns())+'.mp4')
                failed_proxy.parent.mkdir(parents=True,exist_ok=True);proxy.rename(failed_proxy)
            if not run.exists():
                with logfile.open('a') as log:
                    if not proxy.exists():subprocess.run([sys.executable,'-m','src.streaming.prepare_stream_proxy','--video',row['source_path'],'--out',str(proxy),'--height',str(a.height),'--fps',str(a.fps)],stdout=log,stderr=subprocess.STDOUT,check=True)
                    if not proxy.with_suffix('.source.json').exists():raise ValueError('Incomplete proxy; remove only failed proxy and retry')
                    print('Running',key,flush=True)
                    run.parent.mkdir(parents=True,exist_ok=True)
                    subprocess.run([sys.executable,'-m','scripts.run_delayed_stream','--video',str(proxy),'--out',str(run),'--device','cuda','--asr',str(a.asr),'--asr-device',a.asr_device,'--asr-beam-size',str(a.asr_beam_size),'--delay',str(a.delay),'--detect-fps',str(a.detect_fps),'--imgsz',str(a.imgsz),'--audio-window',str(a.audio_window),'--vision-profile',a.vision_profile,'--intent-profile',a.intent_profile],stdout=log,stderr=subprocess.STDOUT,check=True)
            report=evaluate(run,dataset,clip,collection=name,reference=a.reference)
            if a.audio_preview:audio_preview(run,proxy)
            (run/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');reports[key]=report
            print('Evaluated',key,json.dumps(report['summary']['stream_av'],ensure_ascii=False),flush=True)
        except Exception as exc:
            failures[key]=repr(exc);print('FAILED',key,repr(exc),flush=True)
        summary={n:{m:summarize([r['scores'][m] for r in reports.values() if r['collection']==n]) for m in ['stream_av','stream_visual','offline_showintent','offline_visual-only']} for n in a.collections if any(r['collection']==n for r in reports.values())}
        (a.out/'summary.json').write_text(json.dumps(dict(planned=len(tasks),evaluated=len(reports),failures=failures,passed=len(reports)==len(tasks) and not failures,summary=summary,clips={k:dict(collection=r['collection'],horizon_ms=r['horizon_ms'],partial=r['partial'],runtime=r['runtime']) for k,r in reports.items()}),ensure_ascii=False,indent=2)+'\n')
    if failures:raise SystemExit('Suite incomplete; inspect summary.json and logs. Failed clips were not counted as successes.')


if __name__=='__main__':main()

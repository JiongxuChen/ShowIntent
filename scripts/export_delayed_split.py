"""Export a time-offset split screen from completed, paced stream records."""
import argparse,bisect,json,math,subprocess,sys,hashlib
from pathlib import Path
from dataclasses import replace


def read(path):
    return [json.loads(s) for s in Path(path).read_text().splitlines() if s.strip()]


def source_time(ms):
    if ms is None:return '--:--.--'
    ticks=round(ms/10)
    minutes,rest=divmod(ticks,6000)
    seconds,hundredths=divmod(rest,100)
    return f'{minutes:02d}:{seconds:02d}.{hundredths:02d}'


def draw_time(canvas,ms,x=0):
    import cv2
    scale=max(.45,canvas.shape[0]/720*.7)
    point=(x+12,round(30*scale/.7))
    text=source_time(ms)
    cv2.putText(canvas,text,point,cv2.FONT_HERSHEY_SIMPLEX,scale,(0,0,0),4,cv2.LINE_AA)
    cv2.putText(canvas,text,point,cv2.FONT_HERSHEY_SIMPLEX,scale,(255,255,255),1,cv2.LINE_AA)


def schedule(timing,playback,delay_ms):
    if not timing or [r['timestamp_ms'] for r in timing]!=[r['timestamp_ms'] for r in playback]:
        raise ValueError('Missing or unaligned playback/timing records')
    source=[r['timestamp_ms'] for r in timing];due=[r['written_at_ms'] for r in timing]
    if any(b<=a for a,b in zip(source,source[1:])) or any(b<a for a,b in zip(due,due[1:])):
        raise ValueError('Nonmonotonic stream clock')
    if any(w<s+delay_ms-1 for s,w in zip(source,due)):
        raise ValueError('Output precedes the configured buffer')
    return due


def replay_b(per,run):
    from src.intent.delayed_adaptation import DelayedIntentStateMachine
    from src.intent.fusion import FusionConfig
    from src.speech.bridge import SpeechEvidenceBridge
    from src.speech.events import SpeechEvent
    from src.speech.presentation_language import ContextualPresentationLanguage
    root=Path(__file__).resolve().parents[1]
    cfg=json.loads((root/'configs/presentation_intent.json').read_text())
    fusion=FusionConfig(**json.loads((root/'configs/semantic_priority_fusion.json').read_text()))
    engine=DelayedIntentStateMachine(**cfg,fusion_config=fusion);language=ContextualPresentationLanguage()
    metrics=json.loads((run/'metrics.json').read_text());events=[]
    for row in read(run/'speech.jsonl'):
        e=SpeechEvent.from_dict({k:v for k,v in row.items() if k in SpeechEvent.__dataclass_fields__})
        available=max(e.end_ms,math.ceil(row['wall_ready_ms']-metrics['delay_seconds']*1000))
        events.append(replace(e,available_at_ms=available,semantics=language.analyze_event(e)))
    bridge=SpeechEvidenceBridge(sorted(events,key=lambda e:e.available_at_ms),enabled=True)
    return [engine.update(p,speech_evidence=bridge.evidence_for(p)) for p in per]


def file_hash(path):
    digest=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(1048576),b''):digest.update(chunk)
    return digest.hexdigest()


def resolve_video(run,metrics,args):
    manifest=run.parent.parent/'render_inputs.json'
    if args.video or not manifest.is_file():
        video=args.video if args.video else Path(metrics['configuration']['video'])
        return video if video.is_absolute() else args.data_root/video
    key=run.parent.name+'/'+run.name
    item=json.loads(manifest.read_text())['clips'][key]
    video=args.data_root/item['proxy'];side=video.with_suffix('.source.json')
    if video.is_file():
        actual=file_hash(video)
        if actual==item['proxy_sha256']:return video
        if side.is_file():
            proof=json.loads(side.read_text())
            if (proof.get('source_sha256')==item['source_sha256'] and proof.get('proxy_sha256')==actual
                    and proof.get('height')==item['height'] and proof.get('fps')==item['fps']):return video
        raise ValueError('Proxy changed without matching source provenance: '+str(video))
    source=args.data_root/item['source']
    if not source.is_file():raise FileNotFoundError('Supply the original video: '+str(source))
    if file_hash(source)!=item['source_sha256']:raise ValueError('Source video differs from the stream recording: '+str(source))
    print('Preparing local stream proxy:',video,flush=True)
    subprocess.run([sys.executable,'-m','src.streaming.prepare_stream_proxy','--video',str(source),
                    '--out',str(video),'--height',str(item['height']),'--fps',str(item['fps'])],check=True)
    return video


class FrameReader:
    def __init__(self,path,fps):
        import cv2
        self.cap=cv2.VideoCapture(str(path));self.fps=fps;self.index=-1;self.last=None
        if not self.cap.isOpened():raise RuntimeError('Cannot read '+str(path))
    def at(self,ts):
        wanted=max(0,int(round(ts*self.fps/1000)))
        while self.index<wanted:
            ok,frame=self.cap.read()
            if not ok:break
            self.last=frame;self.index+=1
        if self.last is None:raise ValueError('Empty proxy')
        return self.last
    def close(self):self.cap.release()


def export(run,out,args):
    import av,cv2,numpy as np
    from src.camera.controller import StreamingCameraController
    from src.camera.render import render_frame
    from src.camera.run import shot_at
    cv2.setNumThreads(2)
    metrics=json.loads((run/'metrics.json').read_text())
    if metrics['status']!='finished' or metrics.get('forced_worker_shutdown') or metrics['worker_exitcodes']!=[0,0] or metrics['input_frames']!=metrics['output_frames']:
        raise ValueError('Only complete, clean stream runs can be exported')
    video=resolve_video(run,metrics,args)
    if not video.is_file():raise FileNotFoundError(video)
    timing=read(run/'timing.jsonl');playback=read(run/'playback-intent.jsonl');delay=metrics['delay_seconds']*1000
    due=schedule(timing,playback,delay)
    if len(timing)!=metrics['output_frames']:raise ValueError('Incomplete playback records')
    per=read(run/'perception.jsonl')
    if not per:raise ValueError('No adopted perception packets')
    readiness={r['timestamp_ms']:r['wall_ready_ms'] for r in read(run/'vision-compute.jsonl')}
    if any(p['timestamp_ms'] not in readiness for p in per):raise ValueError('Missing perception availability evidence')
    intents=replay_b(per,run) if args.intent_source=='current-b' else None
    fps=metrics['input_fps'];left=FrameReader(video,fps);right=FrameReader(video,fps)
    first=left.at(0);h,w=first.shape[:2]
    if abs(left.cap.get(cv2.CAP_PROP_FPS)-fps)>.01:raise ValueError('Use the original upright CFR stream proxy')
    height=args.height;panel_w=max(2,round(height*w/h)//2*2)
    size=(panel_w*2,height)
    end=due[-1]+1000/fps
    if args.max_seconds:end=min(end,args.max_seconds*1000)
    count=math.ceil(end*fps/1000);out.mkdir(parents=True,exist_ok=False)
    silent=out/'split_silent.mp4';controller=StreamingCameraController()
    pi=-1;ri=-1;cached_right=None;cached_source=None;cached_shot=None
    try:
        with av.open(str(silent),'w') as dest,(out/'split.jsonl').open('w') as log:
            from fractions import Fraction
            stream=dest.add_stream('libx264',rate=Fraction(str(fps)).limit_denominator(100000))
            stream.width,stream.height=size;stream.pix_fmt='yuv420p';stream.options={'preset':'veryfast','crf':'22'};stream.codec_context.thread_count=2
            for index in range(count):
                wall=index*1000/fps;latest=bisect.bisect_right(due,wall)-1
                while ri<latest:
                    ri+=1;ts=timing[ri]['timestamp_ms']
                    while pi+1<len(per) and per[pi+1]['timestamp_ms']<=ts and readiness[per[pi+1]['timestamp_ms']]<=due[ri]:pi+=1
                    packet=per[pi] if pi>=0 else dict(timestamp_ms=ts,objects=[],hands=[],relations=[])
                    if args.intent_source=='recorded':active=playback[ri]
                    else:
                        state=intents[pi] if pi>=0 else dict(state='NONE',target_id=None)
                        if pi<0 or ts-packet['timestamp_ms']>1000:state=dict(state,state='NONE',target_id=None)
                        active=dict(state,timestamp_ms=ts)
                    cached_shot=shot_at(controller,ts,packet,active)
                    cached_source=ts;cached_right=render_frame(right.at(ts),cached_shot,(panel_w,height))
                left_source=min(wall,(metrics['input_frames']-1)*1000/fps)
                canvas=np.zeros((height,panel_w*2,3),dtype=np.uint8)
                canvas[:,:panel_w]=cv2.resize(left.at(left_source),(panel_w,height))
                if cached_right is not None:canvas[:,panel_w:]=cached_right
                draw_time(canvas,left_source)
                draw_time(canvas,cached_source,panel_w)
                frame=av.VideoFrame.from_ndarray(canvas,format='bgr24');frame.pts=index
                for packet in stream.encode(frame):dest.mux(packet)
                log.write(json.dumps(dict(wall_ms=wall,left_source_ms=left_source,right_source_ms=cached_source,buffering=cached_source is None,shot=cached_shot),ensure_ascii=False)+'\n')
            for packet in stream.encode():dest.mux(packet)
    finally:left.close();right.close()
    # Only the delayed source audio is audible. Initial audio is silent for the nominal buffer.
    import imageio_ffmpeg
    ffmpeg=imageio_ffmpeg.get_ffmpeg_exe();movie=out/'compare.mp4'
    with av.open(str(video)) as src:has_audio=bool(src.streams.audio)
    if has_audio:
        cmd=[ffmpeg,'-v','error','-n','-i',str(silent),'-itsoffset',str(delay/1000),'-i',str(video),'-map','0:v:0','-map','1:a:0','-c:v','copy','-c:a','aac','-af','apad','-t',str(count/fps),'-movflags','+faststart',str(movie)]
        subprocess.run(cmd,check=True);silent.unlink()
    else:silent.rename(movie)
    meta=dict(camera_profile='streaming',camera_missing_hold_ms=1000,camera_exit_edge_speed=.8,max_recorded_frame_lateness_ms=max(w-t['timestamp_ms']-delay for w,t in zip(due,timing)),audio_timing_note='Audio uses nominal buffer delay; frame lateness and CFR quantization can add audiovisual skew',kind='Recorded stream timing visualization, not fresh live inference',intent_source=args.intent_source,video=str(video),run=str(run),frames=count,fps=fps,delay_seconds=delay/1000,first_right_wall_ms=due[0],duration_seconds=count/fps,width=size[0],height=size[1],audio='Delayed source only' if has_audio else 'No source audio',timing_note='Right presentation uses recorded written_at_ms; C re-render and export costs are not included in measured run latency',complete_recorded_run=not bool(args.max_seconds) or args.max_seconds*1000>=due[-1]+1000/fps)
    (out/'export.json').write_text(json.dumps(meta,indent=2)+'\n');print('Exported',movie,flush=True)


def main():
    p=argparse.ArgumentParser(description=__doc__);g=p.add_mutually_exclusive_group();g.add_argument('--run',type=Path);g.add_argument('--runs',type=Path,nargs='+',help='Collection roots containing scene/clip run directories')
    p.add_argument('--data-root',type=Path,default=Path('.'),help='Root used to resolve the recorded proxy path');p.add_argument('--video',type=Path,help='Explicit upright CFR proxy for a single run');p.add_argument('--out',type=Path,required=True);p.add_argument('--clips',nargs='+');p.add_argument('--intent-source',choices=['current-b','recorded'],default='current-b');p.add_argument('--height',type=int,default=720);p.add_argument('--max-seconds',type=float,default=0)
    a=p.parse_args()
    if a.out.exists():p.error('Choose a new output directory')
    if a.height<128 or a.height%2 or a.max_seconds<0:p.error('Use an even height >=128 and nonnegative duration')
    if a.video and not a.run:p.error('--video requires --run')
    if a.run:export(a.run,a.out,a);return
    if not a.runs:a.runs=[Path('outputs/stream')]
    tasks=[r.parent for root in a.runs for r in sorted(root.glob('*/*/metrics.json')) if not a.clips or r.parent.name in a.clips]
    if not tasks:p.error('No matching runs')
    names=[r.name for r in tasks]
    if len(names)!=len(set(names)):p.error('Duplicate clip IDs')
    if a.clips and set(a.clips)-set(names):p.error('Some requested clips are missing')
    for run in tasks:export(run,a.out/run.name,a)

if __name__=='__main__':main()

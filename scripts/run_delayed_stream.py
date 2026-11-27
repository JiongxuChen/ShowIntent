"""Paced file input, live inference workers, fixed-delay diagnostic video output."""
import argparse
from dataclasses import replace
import importlib.util
import json
import math
import multiprocessing as mp
from pathlib import Path
import queue
import threading
import time


def evidence_time(end_ms, ready_ms, delay_ms):
    """No future transcript content; account for inference within playback buffer."""
    return max(end_ms, math.ceil(ready_ms-delay_ms))


def main():
    launch_started=time.monotonic()
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True,help='Upright constant-FPS proxy made by prepare_stream_proxy')
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--yolo',type=Path,default=Path('models/yolo11m.pt'))
    p.add_argument('--vision-profile',choices=['enhanced'],default='enhanced')
    p.add_argument('--intent-profile',choices=['adapted'],default='adapted',help='B policy; sparse adds measured-hand-motion reacquisition to offline v3')
    p.add_argument('--asr',type=Path,default=Path('models/faster-whisper-base'))
    p.add_argument('--delay',type=float,default=6)
    p.add_argument('--detect-fps',type=float,default=10)
    p.add_argument('--imgsz',type=int,default=1280)
    p.add_argument('--asr-beam-size',type=int,default=1)
    p.add_argument('--audio-window',type=float,default=4)
    p.add_argument('--max-seconds',type=float,default=0)
    p.add_argument('--device',choices=['cpu','cuda'],default='cuda')
    p.add_argument('--asr-device',choices=['cpu','cuda'],default='cpu',help='Default cpu keeps GPU vision while running speech on CPU; cuda is opt-in')
    p.add_argument('--startup-timeout',type=float,default=600,help='Maximum wait for native model warmup before input starts')
    p.add_argument('--shutdown-timeout',type=float,default=120,help='Grace period for native GPU context release after output ends')
    p.add_argument('--check-only',action='store_true')
    p.add_argument('--display',action='store_true',help='Open delayed video window on a machine with a graphical display')
    a=p.parse_args()
    if a.asr_beam_size<1 or a.delay<=0 or a.detect_fps<=0 or a.audio_window<1 or a.imgsz<128 or a.max_seconds<0 or a.shutdown_timeout<=0 or a.startup_timeout<=0:p.error('Invalid timing/size')
    if not a.video.is_file() or not a.yolo.is_file() or not (a.asr/'model.bin').is_file():p.error('Missing local input or weights')
    missing=[m for m in ['cv2','av','numpy','ultralytics','mediapipe','faster_whisper'] if importlib.util.find_spec(m) is None]
    if missing:p.error('Missing packages: '+', '.join(missing))
    if a.check_only:
        print('Local files and package discovery passed; GPU/model loading not checked.');return
    if a.out.exists():p.error('Choose a fresh --out')
    asr_device=a.asr_device or a.device
    if asr_device=='cuda':
        from src.streaming.stream_cuda_runtime import prepare_cuda_runtime
        prepare_cuda_runtime()
    import cv2
    cv2.setNumThreads(2)
    import numpy as np
    from faster_whisper.audio import decode_audio
    from src.speech.bridge import SpeechEvidenceBridge
    from src.speech.presentation_language import ContextualPresentationLanguage
    from src.intent.semantic_priority import SemanticPriorityIntentStateMachine
    from src.intent.persistent import PersistentIntentStateMachine
    from src.intent.rest_aware import RestAwareIntentStateMachine
    from src.intent.delayed_adaptation import DelayedIntentStateMachine
    from src.intent.streaming import StreamIntentStateMachine,SparseVisualIntentStateMachine
    from src.intent.fusion import FusionConfig
    from scripts.render_b_preview import PreviewFraming
    from src.streaming.stream_timeline import next_sample_time,playback_state,visual_feed_stalled
    # Initialization and audio decoding precede the timed input clock; no inference on future data.
    audio=decode_audio(str(a.video),sampling_rate=16000)
    cap=cv2.VideoCapture(str(a.video));fps=cap.get(cv2.CAP_PROP_FPS)
    if not cap.isOpened() or fps<=0:raise RuntimeError('Cannot open CFR proxy')
    width,height=int(cap.get(3)),int(cap.get(4))
    a.out.mkdir(parents=True)
    config=json.loads(Path('configs/presentation_intent.json').read_text())
    engine={'v2':SemanticPriorityIntentStateMachine,'rest':RestAwareIntentStateMachine,'sparse':StreamIntentStateMachine,'shared':RestAwareIntentStateMachine,'adapted':DelayedIntentStateMachine}[a.intent_profile](**config,fusion_config=FusionConfig(**json.loads(Path('configs/semantic_priority_fusion.json').read_text())))
    visual=(SparseVisualIntentStateMachine if a.intent_profile in {'sparse','adapted'} else PersistentIntentStateMachine)(**config)
    if a.intent_profile=='adapted':visual=SparseVisualIntentStateMachine(**vars(engine.config))
    if a.intent_profile=='shared':
        from src.intent.shared_runtime import shared_engines
        engine,visual=shared_engines(config,FusionConfig(**json.loads(Path('configs/semantic_priority_fusion.json').read_text())),recovery=True)
    framing=PreviewFraming();bridge=SpeechEvidenceBridge(enabled=True);language=ContextualPresentationLanguage()
    frames=queue.Queue(maxsize=max(8,math.ceil((a.delay+2)*fps)))
    context=mp.get_context('spawn')
    jobs=context.Queue(maxsize=2);results=context.Queue();speech=context.Queue();errors=context.Queue();stop=context.Event()
    ready_queue=context.Queue();start=context.Event();shared_origin=context.Value('d',0)
    from src.streaming.stream_workers import vision_worker,speech_worker
    from src.streaming.stream_process import isolated_entry
    workers=[context.Process(name='vision',target=isolated_entry,args=(vision_worker,(jobs,results,errors,ready_queue,start,stop,shared_origin,str(a.yolo),a.device,a.imgsz,a.vision_profile,(height,width)),[results,errors,ready_queue])),
             context.Process(name='speech',target=isolated_entry,args=(speech_worker,(audio,speech,errors,ready_queue,start,stop,shared_origin,str(a.asr),asr_device,a.audio_window,a.max_seconds,a.asr_beam_size),[speech,errors,ready_queue]))]
    for worker in workers:worker.start()
    waiting={'vision','speech'}
    startup_deadline=time.monotonic()+a.startup_timeout
    try:
        while waiting:
            if stop.is_set() or any(w.exitcode is not None for w in workers) or time.monotonic()>startup_deadline:
                raise RuntimeError('Inference worker startup failed or timed out; inspect worker messages. No stream started.')
            try:waiting.discard(ready_queue.get(timeout=.2))
            except queue.Empty:pass
    except BaseException:
        stop.set()
        for worker in workers:
            if worker.is_alive():worker.terminate()
            worker.join()
        raise
    stats=dict(input_frames=0,output_frames=0,detection_jobs_dropped=0,late_visual_packets=0,visual_packets=0,speech_events=0,asr_windows=0,max_playback_lateness_ms=0)
    origin=time.monotonic();shared_origin.value=origin
    stats['startup_seconds']=origin-launch_started
    stats.update(input_fps=fps,output_width=width,output_height=height,version='delayed_stream_v2')
    def now():return (time.monotonic()-origin)*1000
    def wait_until(ms):return stop.wait(max(0,(ms-now())/1000))
    def put(q,value):
        while not stop.is_set():
            try:q.put(value,timeout=.1);return
            except queue.Full:pass
    def guarded(fn):
        def run():
            try:fn()
            except BaseException as e:errors.put(repr(e));stop.set()
        return run
    def capture():
        i=0;next_detection=0
        while not stop.is_set():
            ok,frame=cap.read()
            if not ok:break
            ts=round(i*1000/fps)
            if a.max_seconds and ts>=a.max_seconds*1000:break
            if wait_until(ts):break
            if ts>=next_detection:
                next_detection=next_sample_time(ts,next_detection,a.detect_fps)
                try:jobs.put_nowait((ts,i,frame.copy()))
                except queue.Full:stats['detection_jobs_dropped']+=1
            put(frames,(ts,frame));stats['input_frames']+=1;i+=1
        put(frames,None);put(jobs,None)
    threads=[threading.Thread(target=guarded(capture))]
    writer=cv2.VideoWriter(str(a.out/'silent.mp4'),cv2.VideoWriter_fourcc(*'mp4v'),fps,(width,height))
    if not writer.isOpened():raise RuntimeError('Video encoder unavailable')
    logs={n:(a.out/(n+'.jsonl')).open('w') for n in ['perception','showintent','visual-only','speech','timing','playback-intent','playback-visual','speech-windows','evidence','vision-compute']}
    def log(name,row):logs[name].write(json.dumps(row,ensure_ascii=False)+'\n')
    start.set()
    for t in threads:t.start()
    pending=[];latest=None;intent=dict(state='NONE',target_id=None);visual_intent=dict(intent);previous_ts=-1
    completed=False
    try:
        while not stop.is_set():
            try:item=frames.get(timeout=.1)
            except queue.Empty:continue
            if item is None:break
            ts,frame=item;deadline=ts+a.delay*1000
            if any(w.exitcode not in (None,0) for w in workers):raise RuntimeError('Inference worker crashed')
            if wait_until(deadline):break
            while not speech.empty():
                ready,e=speech.get()
                if isinstance(e,dict) and e.get('kind')=='asr_window':
                    log('speech-windows',dict(e,wall_ready_ms=ready));stats['asr_windows']+=1;continue
                log('speech',dict(e.to_dict(),wall_ready_ms=ready))
                bridge.events.append(replace(e,available_at_ms=evidence_time(e.end_ms,ready,a.delay*1000),semantics=language.analyze_event(e)))
                stats['speech_events']+=1
            bridge.events.sort(key=lambda e:e.available_at_ms)
            while not results.empty():
                ready,result=results.get()
                if result.get('kind')=='vision_compute':
                    log('vision-compute',dict(result,wall_ready_ms=ready));continue
                pending.append((ready,result))
            keep=[]
            for ready,packet in pending:
                pts=packet['timestamp_ms']
                if pts>ts:keep.append((ready,packet));continue
                if ready>pts+a.delay*1000 or pts<=previous_ts:
                    stats['late_visual_packets']+=1;continue
                previous_ts=pts;latest=packet
                intent=engine.update(packet,speech_evidence=bridge.evidence_for(packet))
                visual_intent=visual.update(packet)
                log('perception',packet);log('showintent',intent);log('visual-only',visual_intent)
                log('evidence',dict(engine.fusion.diagnostics,intent=intent));stats['visual_packets']+=1
            pending=keep
            if visual_feed_stalled(latest['timestamp_ms'] if latest else None,ts):
                raise RuntimeError(f'Visual inference stalled at media {ts}ms; latest packet: {previous_ts}ms. Inspect vision worker warmup and compute logs.')
            # Never keep a zoom indefinitely when visual inference stalls.
            active=playback_state(intent,latest['timestamp_ms'] if latest else None,ts,stale_ms=1000 if a.intent_profile=='adapted' else 500)
            log('playback-visual',playback_state(visual_intent,latest['timestamp_ms'] if latest else None,ts,stale_ms=1000 if a.intent_profile=='adapted' else 500))
            log('playback-intent',dict(active,timestamp_ms=ts))
            crop=framing.update(ts,active,latest['objects'] if latest else [],width/height)
            x1,y1,x2,y2=[round(v*d) for v,d in zip(crop,[width,height,width,height])]
            image=cv2.resize(frame[y1:max(y1+1,y2),x1:max(x1+1,x2)],(width,height))
            writer.write(image)
            if a.display:
                cv2.imshow('ShowIntent delayed stream',image)
                if cv2.waitKey(1)&255 in (27,ord('q')):break
            late=max(0,now()-deadline)
            stats['max_playback_lateness_ms']=max(stats['max_playback_lateness_ms'],late)
            log('timing',dict(timestamp_ms=ts,deadline_ms=deadline,written_at_ms=now(),state=active['state'],crop=crop))
            stats['output_frames']+=1
            if stats['output_frames']%round(fps*5)==0:print(f"output {ts/1000:.1f}s; visual={stats['visual_packets']}; max lag={stats['max_playback_lateness_ms']:.0f}ms",flush=True)
        if not errors.empty():raise RuntimeError(errors.get())
        if stats['visual_packets']==0:raise RuntimeError('No visual results were adopted; stream validation failed')
        if stats['asr_windows']==0:raise RuntimeError('No ASR window completed before playback ended; stream validation failed')
        completed=True
    finally:
        stats['playback_finished_wall_ms']=now()
        stop.set()
        for t in threads:t.join(timeout=5)
        from src.streaming.stream_shutdown import shutdown_workers
        print('Video output stopped. Waiting for inference processes and GPU context release...',flush=True)
        forced=shutdown_workers(workers,[jobs,results,speech],timeout=a.shutdown_timeout)
        stats['forced_worker_shutdown']=forced
        stats['worker_exitcodes']=[w.exitcode for w in workers]
        if forced or any(w.exitcode!=0 for w in workers):completed=False
        # Consumers are gone: discard any unconsumed frame jobs before closing
        # the parent feeder, otherwise its finalizer can wait on a full pipe.
        from src.streaming.stream_shutdown import close_channels
        stats['queue_cleanup_errors']=close_channels([jobs,results,speech,errors,ready_queue],discard=jobs)
        if stats['queue_cleanup_errors']:completed=False
        for worker in workers:
            if not worker.is_alive():worker.close()
        cap.release();writer.release()
        for f in logs.values():f.close()
        stats.update(stream_output_completed=stats['input_frames']==stats['output_frames'] and stats['visual_packets']>0,wall_seconds=now()/1000,delay_seconds=a.delay,mode='PACED_FILE_LIVE_INFERENCE',status='finished' if completed else 'failed',configuration={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()})
        (a.out/'metrics.json').write_text(json.dumps(stats,indent=2)+'\n')
    if not completed:raise RuntimeError('Stream did not shut down cleanly; inspect metrics.json and worker exit codes')
    print('Finished. silent.mp4 is video-only; see docs/B.md for audio muxing and interpretation.')


if __name__=='__main__':main()

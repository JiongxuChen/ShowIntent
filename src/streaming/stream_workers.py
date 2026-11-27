"""Process-isolated native inference; each process owns its model lifecycle."""
import math
import os
from pathlib import Path
import queue
import time
from dataclasses import replace

# Keep models alive until the dedicated process exits; avoid native destructors
# while returning from the worker function. MediaPipe is explicitly closed.
_PROCESS_MODELS=[]


def warmup_recognizer(recognizer,window_seconds,beam_size):
    """Warm decoder search and lazy VAD before starting the media clock."""
    import numpy as np
    from src.speech.asr import transcribe_samples
    dummy=np.zeros(max(16000,round(window_seconds*16000)),dtype=np.float32)
    segments,_=recognizer.transcribe(dummy,language='zh',beam_size=beam_size,
                                    temperature=0.0,condition_on_previous_text=False,vad_filter=False)
    list(segments)
    transcribe_samples(dummy,recognizer,mode='offline',language='zh',beam_size=beam_size)


def vision_worker(jobs,results,errors,ready,start,stop,origin,model,device,imgsz,profile='basic',frame_shape=None):
    models=None
    try:
        # Hand landmarks use CPU; avoid a redundant NVIDIA EGL context competing
        # with CUDA inference and blocking during graph shutdown.
        mesa=Path('/usr/share/glvnd/egl_vendor.d/50_mesa.json')
        if mesa.is_file():
            os.environ['__EGL_VENDOR_LIBRARY_FILENAMES']=str(mesa)
            os.environ['LIBGL_ALWAYS_SOFTWARE']='1'
            os.environ['EGL_PLATFORM']='surfaceless'
        import numpy as np
        import torch
        import cv2
        torch.set_num_threads(4);cv2.setNumThreads(2)
        from src.perception.run import PerceptionModels,normalize_box
        from src.perception.tracker import SimpleObjectTracker
        from src.perception.types import Detection,HandDetection,PerceptionPacket
        from src.perception.relations import infer_relations
        print('Vision worker: loading and warming up',flush=True)
        models=PerceptionModels(model,.25 if profile.startswith('enhanced') else .35,True);_PROCESS_MODELS.append(models);models.yolo.to(device)
        dummy=np.zeros((*(frame_shape or (imgsz,imgsz)),3),dtype=np.uint8)
        models.detect_objects(dummy,imgsz);models.detect_hands(dummy)
        if profile=='enhanced-batched':
            from src.streaming.stream_perception import warmup_crop_batches
            warmup_crop_batches(models,imgsz)
        tracker=SimpleObjectTracker()
        adapter=None
        if profile.startswith('enhanced'):
            from src.streaming.stream_perception import EnhancedStreamPerception
            adapter=EnhancedStreamPerception(models,imgsz,batch_crops=profile=='enhanced-batched')
        ready.put('vision');print('Vision worker: ready',flush=True)
        start.wait()
        while not stop.is_set():
            try:job=jobs.get(timeout=.2)
            except queue.Empty:continue
            if job is None:break
            ts,i,frame=job;h,w=frame.shape[:2]
            if adapter is not None:
                before=time.monotonic();old_calls=adapter.crop_calls;old_images=adapter.crop_images
                packet=adapter.update(ts,i,frame)
                results.put(((time.monotonic()-origin.value)*1000,dict(kind='vision_compute',timestamp_ms=ts,compute_ms=(time.monotonic()-before)*1000,full_model_calls=1,crop_model_calls=adapter.crop_calls-old_calls,crop_images=adapter.crop_images-old_images)))
                results.put(((time.monotonic()-origin.value)*1000,packet));continue
            raw=models.detect_objects(frame,imgsz)
            tracked=tracker.update([(n,normalize_box(b,w,h),s) for n,b,s in raw])
            objects=[Detection(t,n,list(b),round(s,3)) for t,n,b,s in tracked]
            hands=[HandDetection(t,b,round(s,3)) for t,b,s in models.detect_hands(frame)]
            packet=PerceptionPacket(ts,i,objects,hands,infer_relations(hands,objects)).to_dict()
            results.put(((time.monotonic()-origin.value)*1000,packet))
    except BaseException as e:errors.put('vision: '+repr(e));stop.set();raise
    finally:
        print('Vision worker: closing hand detector',flush=True)
        if models is not None and models.hands is not None:models.hands.close()
        print('Vision worker: model loop stopped; flushing results',flush=True)


def speech_worker(audio,results,errors,ready,start,stop,origin,model,device,window_seconds,max_seconds,beam_size=1):
    try:
        import numpy as np
        from faster_whisper import WhisperModel
        from src.speech.asr import transcribe_samples
        print('Speech worker: loading and warming up',flush=True)
        recognizer=WhisperModel(model,device=device,compute_type='int8' if device=='cpu' else 'float16',cpu_threads=4)
        _PROCESS_MODELS.append(recognizer)
        # Exercise the native backend before declaring startup successful.
        warmup_recognizer(recognizer,window_seconds,beam_size)
        ready.put('speech');print('Speech worker: ready',flush=True);start.wait()
        window=round(window_seconds*16000)
        limit=min(len(audio),round(max_seconds*16000)) if max_seconds else len(audio)
        for begin in range(0,limit,window):
            end=min(begin+window,limit);end_ms=math.ceil(end/16)
            if stop.wait(max(0,(origin.value+end_ms/1000)-time.monotonic())):break
            before=time.monotonic()
            print(f'Speech worker: window {begin//16}-{end_ms}ms started; beam={beam_size}',flush=True)
            result=transcribe_samples(audio[begin:end],recognizer,mode='offline',media_offset_ms=begin//16,language='zh',beam_size=beam_size)
            finished=(time.monotonic()-origin.value)*1000
            print(f'Speech worker: window {begin//16}-{end_ms}ms completed; compute={(time.monotonic()-before)*1000:.0f}ms; ready={finished:.0f}ms',flush=True)
            results.put((finished,dict(kind='asr_window',start_ms=begin//16,end_ms=end_ms,compute_ms=(time.monotonic()-before)*1000,event_count=len(result.events))))
            for e in result.events:results.put((finished,replace(e,available_at_ms=max(e.end_ms,math.ceil(finished)))))
        print('Speech worker: model loop stopped; flushing results',flush=True)
    except BaseException as e:errors.put('speech: '+repr(e));stop.set();raise

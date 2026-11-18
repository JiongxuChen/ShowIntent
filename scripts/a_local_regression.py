"""Fuse new local inference with cached global small detections and hands."""
import json
from pathlib import Path
import cv2
from ultralytics import YOLO
from scripts.a_full_regression import run
from src.perception.local_detection import fuse_detections


def main():
    root=Path(__file__).resolve().parents[1]
    model=YOLO(str(root/'yolo11s.pt'))
    for clip in ['B03','C01','C03']:
        source=root/f'outputs/a_final_candidate/{clip}/raw.jsonl'
        rows=[json.loads(line) for line in source.read_text(encoding='utf-8').splitlines()]
        video=next((root.parent/'received_media').rglob(clip+'.mp4'))
        dest=root/f'outputs/a_local_candidate/{clip}'
        dest.mkdir(parents=True,exist_ok=True)
        cap=cv2.VideoCapture(str(video)); cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
        angle=int(cap.get(cv2.CAP_PROP_ORIENTATION_META))%360
        codes={90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}
        print('START '+clip,flush=True)
        with (dest/'raw.jsonl').open('w',encoding='utf-8') as out:
            for packet in rows:
                ok,frame=cap.read()
                if not ok:
                    raise RuntimeError('Missing source frame')
                if angle in codes:
                    frame=cv2.rotate(frame,codes[angle])
                h,w=frame.shape[:2]; top=int(h*.45)
                prediction=model(frame[top:,:],imgsz=1280,conf=.35,verbose=False)[0]
                local=[]
                for box in prediction.boxes:
                    name=prediction.names[int(box.cls[0])]
                    if name not in ['apple','orange']:
                        continue
                    b=box.xyxyn[0].tolist()
                    b[1]=(b[1]*(h-top)+top)/h; b[3]=(b[3]*(h-top)+top)/h
                    local.append(dict(class_name=name,bbox_norm=b,score=float(box.conf[0])))
                objects=fuse_detections(packet['objects'],local)
                packet['objects']=[dict(o,track_id=f'det_{i}') for i,o in enumerate(objects)]
                packet['relations']=[]
                out.write(json.dumps(packet)+'\n')
        cap.release()
        run('scripts.a_unified_replay','--input',dest/'raw.jsonl','--out-dir',dest/'unified')
        run('src.perception.validate_jsonl',dest/'unified/perception.jsonl')
        run('src.perception.review_overlay','--video',video,'--perception',dest/'unified/perception.jsonl','--out',dest/'review.mp4','--legend')
        (dest/'provenance.json').write_text(json.dumps(dict(global_source=str(source),local_crop_top=.45,imgsz=1280,confidence=.35,hands='reused same-frame cached hands',status='unaccepted_development_candidate'),indent=2),encoding='utf-8')
        print('DONE '+clip,flush=True)


if __name__=='__main__':
    main()

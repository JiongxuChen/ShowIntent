"""Independent enhanced adapter; A's original detector/hands entry is preserved.

Stronger local detector, hand-region second pass, appearance-aware instance IDs,
and measured color attributes feed the same PerceptionPacket interface.
"""

import argparse
import json
import math
from pathlib import Path
import time

import cv2
import numpy as np

from .run import PerceptionModels, normalize_box, resolve_rotation, apply_rotation
from .timing import load_timestamps
from .tracker import iou
from .appearance_tracker import AppearanceTracker


def visual_attributes(frame, box):
    height,width=frame.shape[:2]
    x1,y1,x2,y2=box
    # Exclude some border/background without pretending this is segmentation.
    dx=(x2-x1)*.12;dy=(y2-y1)*.12
    patch=frame[max(0,int(y1+dy)):min(height,int(y2-dy)),max(0,int(x1+dx)):min(width,int(x2-dx))]
    if not patch.size:return [],{'colors':[]}
    hsv=cv2.cvtColor(cv2.resize(patch,(48,48)),cv2.COLOR_BGR2HSV)
    hist=cv2.calcHist([hsv],[0,1],None,[12,4],[0,180,0,256]).flatten()
    # Achromatic colors need value bins: white/black books may share hue.
    value=cv2.calcHist([hsv],[2],None,[8],[0,256]).flatten()
    vector=np.concatenate((hist*.7,value*.3));vector/=max(float(vector.sum()),1e-9)
    h,s,v=hsv[:,:,0],hsv[:,:,1],hsv[:,:,2]
    masks={'white':(s<45)&(v>190),'black':v<60,'gray':(s<45)&(v>=60)&(v<=190),
           'red':((h<7)|(h>172))&(s>65)&(v>60),'orange':(h>=7)&(h<20)&(s>65)&(v>60),
           'yellow':(h>=20)&(h<36)&(s>65)&(v>60),'green':(h>=36)&(h<85)&(s>65)&(v>60),
           'blue':(h>=85)&(h<130)&(s>65)&(v>60),'purple':(h>=130)&(h<=172)&(s>65)&(v>60)}
    colors=[k for k,m in sorted(masks.items(),key=lambda row:-float(row[1].mean())) if m.mean()>.22][:2]
    return [round(float(x),6) for x in vector],{'colors':colors}


def hand_regions(hands,width,height):
    boxes=[h['bbox_norm'] for h in hands]
    if len(boxes)>1:
        boxes.append([min(b[0] for b in boxes),min(b[1] for b in boxes),
                      max(b[2] for b in boxes),max(b[3] for b in boxes)])
    regions=[]
    for b in boxes:
        cx,cy=(b[0]+b[2])/2,(b[1]+b[3])/2
        w=max(.32,(b[2]-b[0])*2.8);h=max(.32,(b[3]-b[1])*2.8)
        region=(max(0,int((cx-w/2)*width)),max(0,int((cy-h/2)*height)),
                min(width,int((cx+w/2)*width)),min(height,int((cy+h/2)*height)))
        if region[2]>region[0] and region[3]>region[1] and not any(iou(region,r)>.7 for r in regions):
            regions.append(region)
    return regions[:3]


def merge_detections(rows):
    kept=[]
    for row in sorted(rows,key=lambda r:-r[2]):
        if not any(row[0]==old[0] and iou(row[1],old[1])>.45 for old in kept):kept.append(row)
    return kept


def interaction_relations(hands,objects):
    relations=[]
    for h in hands:
        hb=h['bbox_norm'];hx,hy=(hb[0]+hb[2])/2,(hb[1]+hb[3])/2
        for o in objects:
            if o['class_name']=='person':continue
            b=o['bbox_norm']
            dx=max(b[0]-hx,0.,hx-b[2]);dy=max(b[1]-hy,0.,hy-b[3])
            distance=math.hypot(dx,dy)
            if distance<.09:
                relations.append(dict(hand_id=h['hand_id'],object_id=o['track_id'],
                                      type='near_or_moving_together',score=round(1-distance/.12,3)))
    return relations


def main():
    parser=argparse.ArgumentParser(description='增强视觉适配器：全类别检测、手部局部复检与短期外观跟踪')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--timestamp-map',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--metrics-out',type=Path,required=True)
    parser.add_argument('--yolo-model',type=Path,default=Path('models/yolo11m.pt'))
    parser.add_argument('--detector-backend',choices=['yolo','rtdetr'],default=None,help='Optional closed-set comparison adapter')
    parser.add_argument('--device',default='0')
    parser.add_argument('--crop-imgsz',type=int,default=960)
    parser.add_argument('--confidence',type=float,default=.25)
    parser.add_argument('--imgsz',type=int,default=1280)
    parser.add_argument('--rotate',choices=['auto','cw','ccw','180','none'],default='auto')
    parser.add_argument('--max-frames',type=int,default=0)
    parser.add_argument('--no-hand-crops',action='store_true')
    args=parser.parse_args()
    if args.out.exists() or args.metrics_out.exists():parser.error('output exists; use new paths')
    if not args.yolo_model.is_file():parser.error('local detector weights missing')
    timestamps=load_timestamps(args.timestamp_map)
    if args.detector_backend:
        from .closed_set import ClosedSetModels
        models=ClosedSetModels(args.yolo_model,args.confidence,True,backend=args.detector_backend,device=args.device)
    else:models=PerceptionModels(str(args.yolo_model),args.confidence,True)
    cap=cv2.VideoCapture(str(args.input))
    if not cap.isOpened():raise RuntimeError(f'cannot open {args.input}')
    if hasattr(cv2,'CAP_PROP_ORIENTATION_AUTO'):cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
    rotation,name=resolve_rotation(cap.get(cv2.CAP_PROP_ORIENTATION_META),args.rotate)
    tracker=AppearanceTracker();count=0;crop_calls=0;started=time.perf_counter()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as out:
        while True:
            ok,frame=cap.read()
            if not ok:break
            if count>=len(timestamps):raise ValueError('more decoded frames than source PTS')
            frame=apply_rotation(frame,rotation);height,width=frame.shape[:2]
            hands=[dict(hand_id=f'{hid}_{i}',bbox_norm=box,score=score)
                   for i,(hid,box,score) in enumerate(models.detect_hands(frame))]
            detections=models.detect_objects(frame,args.imgsz)
            if not args.no_hand_crops:
                for x1,y1,x2,y2 in hand_regions(hands,width,height):
                    crop_calls+=1
                    for cls,b,score in models.detect_objects(frame[y1:y2,x1:x2],args.crop_imgsz):
                        if cls=='person':continue
                        mapped=(b[0]+x1,b[1]+y1,b[2]+x1,b[3]+y1)
                        # Reject boxes truncated by the artificial crop border.
                        if b[0]<3 or b[1]<3 or b[2]>x2-x1-3 or b[3]>y2-y1-3:continue
                        detections.append((cls,mapped,score))
            rows=[]
            for cls,b,score in merge_detections(detections):
                box=normalize_box(b,width,height)
                if box[0]>=box[2] or box[1]>=box[3]:continue
                appearance,attributes=visual_attributes(frame,b)
                rows.append(dict(class_name=cls,bbox_norm=box,score=round(score,4),
                                 appearance=appearance,attributes=attributes))
            objects=tracker.update(timestamps[count],rows)
            packet=dict(timestamp_ms=timestamps[count],frame_id=count,objects=objects,hands=hands,
                        relations=interaction_relations(hands,objects))
            out.write(json.dumps(packet,ensure_ascii=False)+'\n');count+=1
            if count%300==0:print(f'{count}/{len(timestamps)} frames',flush=True)
            if args.max_frames and count>=args.max_frames:break
    cap.release()
    if not args.max_frames and count!=len(timestamps):raise ValueError('incomplete decoded video')
    elapsed=time.perf_counter()-started
    metrics=dict(frames_processed=count,elapsed_seconds=elapsed,processing_fps=count/max(elapsed,1e-9),
                 class_policy='all_detector_classes',adapter='enhanced_hand_crops_appearance',
                 detector_classes=list(models.yolo.names.values()),timestamp_source='decoded_video_pts',
                 crop_inference_calls=crop_calls,rotation_applied=name,width=width,height=height,
                 yolo_model=str(args.yolo_model),imgsz=args.imgsz,crop_imgsz=args.crop_imgsz,detector_backend=args.detector_backend or 'original_yolo',device=args.device if args.detector_backend else 'automatic')
    args.metrics_out.parent.mkdir(parents=True,exist_ok=True)
    args.metrics_out.write_text(json.dumps(metrics,ensure_ascii=False,indent=2)+'\n')
    print(f'Completed {count} frames in {elapsed:.1f}s',flush=True)


if __name__=='__main__':main()

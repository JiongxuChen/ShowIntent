"""Targeted detector-only comparison. Counts are diagnostics, not recall."""
import json
from pathlib import Path
import cv2
from ultralytics import YOLO


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'outputs/a_medium_compare'
    out.mkdir(parents=True,exist_ok=True)
    small=YOLO(str(root/'yolo11s.pt'))
    models=[small,YOLO('yolo11m.pt'),small]
    report=[]
    for clip in ['B03','C01']:
        rows=[json.loads(x) for x in (root/f'outputs/a_final_candidate/{clip}/raw.jsonl').read_text().splitlines()]
        # One low-count frame per temporal third; no semantic truth is assumed.
        selected=[]
        for third in range(3):
            section=rows[third*len(rows)//3:(third+1)*len(rows)//3]
            selected.append(min(section,key=lambda r:sum(o['class_name']=='apple' for o in r['objects'])))
        cap=cv2.VideoCapture(str(next((root.parent/'received_media').rglob(clip+'.mp4'))))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
        angle=int(cap.get(cv2.CAP_PROP_ORIENTATION_META))%360
        codes={90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}
        for row in selected:
            cap.set(cv2.CAP_PROP_POS_FRAMES,row['frame_id'])
            ok,frame=cap.read()
            if not ok:
                raise RuntimeError('Cannot decode')
            if angle in codes:
                frame=cv2.rotate(frame,codes[angle])
            panels=[]
            for name,model in zip(['small','medium','small_crop'],models):
                crop_top=int(frame.shape[0]*.45) if name=='small_crop' else 0
                result=model(frame[crop_top:,:],imgsz=1280,conf=.35,verbose=False)[0]
                h,w=frame.shape[:2]; top=int(h*.45)
                canvas=cv2.resize(frame[top:,:],(960,round((h-top)*960/w)))
                ch=canvas.shape[0]; objects=[]
                for box in result.boxes:
                    label=result.names[int(box.cls[0])]
                    if label not in ['apple','orange']:
                        continue
                    b=box.xyxyn[0].tolist(); score=float(box.conf[0])
                    if crop_top:
                        b[1]=(b[1]*(h-crop_top)+crop_top)/h
                        b[3]=(b[3]*(h-crop_top)+crop_top)/h
                    x1,y1,x2,y2=b
                    x1,x2=int(x1*960),int(x2*960)
                    y1,y2=int((y1*h-top)*ch/(h-top)),int((y2*h-top)*ch/(h-top))
                    cv2.rectangle(canvas,(x1,y1),(x2,y2),(60,255,60),2)
                    cv2.putText(canvas,f'{label} {score:.2f}',(x1,max(25,y1)),cv2.FONT_HERSHEY_SIMPLEX,.6,(60,255,60),2)
                    objects.append(dict(class_name=label,bbox=b,score=score))
                canvas=cv2.copyMakeBorder(canvas,45,0,0,0,cv2.BORDER_CONSTANT)
                cv2.putText(canvas,f'{name} {clip} {row["timestamp_ms"]/1000:.2f}s',(10,30),cv2.FONT_HERSHEY_SIMPLEX,.8,(255,255,255),2)
                panels.append(canvas)
                report.append(dict(clip=clip,frame_id=row['frame_id'],timestamp_ms=row['timestamp_ms'],model=name,objects=objects))
            cv2.imwrite(str(out/f'{clip}_{row["frame_id"]}.jpg'),cv2.hconcat(panels))
        cap.release()
    (out/'detections.json').write_text(json.dumps(report,indent=2),encoding='utf-8')
    print(out,flush=True)


if __name__=='__main__':
    main()

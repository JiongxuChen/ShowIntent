"""Five-FPS diagnostic replay, not a realtime benchmark or replacement output."""
import json
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO
from src.perception.tracker import SimpleObjectTracker
from src.perception.relink_tracker import RelinkObjectTracker


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--tracker-compare', action='store_true')
    args = parser.parse_args()
    cache = {}
    if args.tracker_compare:
        for row in json.loads((root/'outputs/a_sequence_compare/detections.json').read_text(encoding='utf-8')):
            if row['model'] == 'small':
                cache[(row['clip'], round(row['time_s'], 1))] = row['objects']
    out = root / ('outputs/a_tracker_compare' if args.tracker_compare else 'outputs/a_sequence_compare')
    out.mkdir(parents=True, exist_ok=True)
    models = [None, None] if args.tracker_compare else [YOLO(str(root.parent/'yolo11n.pt')), YOLO(str(root/'yolo11s.pt'))]
    report = []
    for clip, start, end in [('B03',0,12),('C01',2,14),('C03',3,17)]:
        source = next((root.parent/'received_media').rglob(clip+'.mp4'))
        cap = cv2.VideoCapture(str(source))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO,0)
        angle = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
        codes = {90:cv2.ROTATE_90_CLOCKWISE,180:cv2.ROTATE_180,270:cv2.ROTATE_90_COUNTERCLOCKWISE}
        trackers = [SimpleObjectTracker(max_missed=1),SimpleObjectTracker(max_missed=1)]
        if args.tracker_compare:
            trackers[1] = RelinkObjectTracker(max_missed=8)
        writer = cv2.VideoWriter(str(out/(clip+'_compare.mp4')),cv2.VideoWriter_fourcc(*'mp4v'),5,(1600,1000))
        if not writer.isOpened():
            raise RuntimeError('Cannot create output')
        try:
            for tick in range(start*5,end*5):
                cap.set(cv2.CAP_PROP_POS_MSEC,tick*200)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError('Cannot read frame')
                if angle in codes:
                    frame = cv2.rotate(frame,codes[angle])
                panels=[]
                for index, model in enumerate(models):
                    detections=[]
                    if args.tracker_compare:
                        detections = [(o['class_name'],tuple(o['bbox']),o['score']) for o in cache[(clip, round(tick/5,1))]]
                    else:
                        result=model(frame,imgsz=1280,conf=.35,verbose=False)[0]
                        for box in result.boxes:
                            name=result.names[int(box.cls[0])]
                            if name in ('apple','orange'):
                                detections.append((name,tuple(box.xyxyn[0].tolist()),float(box.conf[0])))
                    tracks=trackers[index].update(detections)
                    # Fixed lower-frame view shows hands and table with readable labels.
                    fh,fw=frame.shape[:2]
                    top=int(fh*.45)
                    canvas=cv2.resize(frame[top:,:],(800,650))
                    legend = []
                    for number, (tid,name,box,score) in enumerate(tracks, 1):
                        x1,y1,x2,y2=box
                        x1,x2=int(x1*800),int(x2*800)
                        y1,y2=int((y1*fh-top)*650/(fh-top)),int((y2*fh-top)*650/(fh-top))
                        palette = [(80,255,80),(0,190,255),(255,200,60),(220,100,255),(255,255,80),(120,150,255)]
                        color=palette[(int(tid.rsplit('_',1)[-1])-1)%len(palette)]
                        cv2.rectangle(canvas,(x1,y1),(x2,y2),color,2)
                        # Small frame-local marker; full identity stays outside the image.
                        tx=max(18,min((x1+x2)//2,780)); ty=max(22,min((y1+y2)//2,625))
                        cv2.rectangle(canvas,(tx-8,ty-24),(min(799,tx+112),ty+10),(0,0,0),-1)
                        cv2.putText(canvas,tid,(tx-5,ty+3),cv2.FONT_HERSHEY_SIMPLEX,.65,color,2,cv2.LINE_AA)
                        legend.append((f'{tid} {name.upper()} {score:.2f}',color))
                    header=cv2.copyMakeBorder(canvas,50,300,0,0,cv2.BORDER_CONSTANT)
                    for row,(label,color) in enumerate(legend[:16]):
                        column=row//8; line=row%8
                        cv2.putText(header,label,(10+column*400,730+line*32),cv2.FONT_HERSHEY_SIMPLEX,.62,color,2,cv2.LINE_AA)
                    title = ('OLD tracker' if index==0 else 'RELINK candidate') if args.tracker_compare else ('BASELINE nano' if index==0 else 'CANDIDATE small')
                    cv2.putText(header,f'{clip} {tick/5:.1f}s | '+title,(10,32),cv2.FONT_HERSHEY_SIMPLEX,.75,(255,255,255),2)
                    panels.append(header)
                    report.append(dict(clip=clip,time_s=tick/5,model=(title if args.tracker_compare else ('nano' if index==0 else 'small')),objects=[dict(id=t,class_name=n,bbox=list(b),score=s) for t,n,b,s in tracks]))
                writer.write(cv2.hconcat(panels))
        finally:
            writer.release(); cap.release()
        print('Completed '+clip,flush=True)
    (out/'detections.json').write_text(json.dumps(report,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

"""Small offline diagnostic; preserves original inference outputs."""
import json
import argparse
from pathlib import Path
import cv2
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default=str(root.parent / 'yolo11n.pt'))
    parser.add_argument('--tag', default='nano')
    args = parser.parse_args()
    media = root.parent / 'received_media'
    out = root / 'outputs/a_detection_compare' / args.tag
    out.mkdir(parents=True, exist_ok=True)
    model = YOLO(args.model)
    configs = [(1280, .35), (1280, .20), (1536, .35)]
    summary = []
    for clip, seconds in [('B03', 1), ('B03', 6), ('C01', 5), ('C01', 10), ('C02', 10), ('C02', 20)]:
        path = next(media.rglob(clip + '.mp4'))
        cap = cv2.VideoCapture(str(path))
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
        cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
        ok, frame = cap.read()
        cap.release()
        if not ok:
            raise RuntimeError(str(path))
        codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
        if rotation in codes:
            frame = cv2.rotate(frame, codes[rotation])
        panels = []
        for size, conf in configs:
            result = model(frame, imgsz=size, conf=conf, verbose=False)[0]
            height = round(frame.shape[0] * 960 / frame.shape[1])
            canvas = cv2.resize(frame, (960, height))
            detections = []
            for box in result.boxes:
                name = result.names[int(box.cls[0])]
                if name not in ('apple', 'orange'):
                    continue
                norm = box.xyxyn[0].tolist()
                score = float(box.conf[0])
                detections.append(dict(class_name=name, bbox_norm=norm, score=score))
                x1,y1,x2,y2 = [int(v*d) for v,d in zip(norm,(960,height,960,height))]
                cv2.rectangle(canvas,(x1,y1),(x2,y2),(0,255,0),2)
                cv2.putText(canvas,f'{name} {score:.2f}',(x1,max(50,y1)),cv2.FONT_HERSHEY_SIMPLEX,.65,(0,255,0),2)
            cv2.rectangle(canvas,(0,0),(960,35),(0,0,0),-1)
            cv2.putText(canvas,f'{clip} {seconds}s | imgsz={size} conf={conf}',(10,25),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
            panels.append(canvas)
            summary.append(dict(clip=clip,seconds=seconds,imgsz=size,conf=conf,detections=detections))
        cv2.imwrite(str(out / f'{clip}_{seconds}s.jpg'),cv2.vconcat(panels))
    (out/'detections.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
    print(out, flush=True)


if __name__ == '__main__':
    main()

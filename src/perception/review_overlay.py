"""Render large labels from cached perception; no model inference needed."""
import argparse
import json
from pathlib import Path

import cv2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--video', required=True)
    parser.add_argument('--perception', required=True)
    parser.add_argument('--out', required=True)
    parser.add_argument('--legend', action='store_true')
    args = parser.parse_args()
    rows = [json.loads(line) for line in Path(args.perception).read_text(encoding='utf-8').splitlines() if line.strip()]
    packets = {row['frame_id']: row for row in rows}
    cap = cv2.VideoCapture(args.video)
    if not cap.isOpened():
        raise RuntimeError('Cannot open video')
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    rotation = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
    codes = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    writer = None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if rotation in codes:
                frame = cv2.rotate(frame, codes[rotation])
            h, w = frame.shape[:2]
            frame = cv2.resize(frame, (1920, round(h * 1920 / w)), interpolation=cv2.INTER_AREA)
            h, w = frame.shape[:2]
            top = int(h * .45) if args.legend else 0
            if args.legend:
                frame = cv2.copyMakeBorder(frame[top:,:],0,600,0,0,cv2.BORDER_CONSTANT)
            for number, obj in enumerate(packets.get(index, {}).get('objects', [])):
                x1, y1, x2, y2 = [int(v*d) for v, d in zip(obj['bbox_norm'], (w,h,w,h))]
                color = (50,220,80) if obj['class_name'] == 'apple' else (0,180,255)
                label = f"{obj['class_name'].upper()} {obj['track_id']} {obj['score']:.2f}"
                if args.legend:
                    y1 -= top
                    y2 -= top
                    identity = int(obj['track_id'].rsplit('_',1)[-1])
                    color = [(70,255,70),(0,190,255),(255,190,80),(220,100,255),(255,255,80),(160,160,255)][(identity-1)%6]
                    cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
                    cx = max(30,min(w-50,(x1+x2)//2))
                    cy = max(30,min(h-top-30,(y1+y2)//2))
                    marker = str(identity)
                    (mw,mh),_ = cv2.getTextSize(marker,cv2.FONT_HERSHEY_SIMPLEX,1.1,3)
                    cv2.rectangle(frame,(cx-6,cy-mh-6),(cx+mw+6,cy+8),(0,0,0),-1)
                    cv2.putText(frame,marker,(cx,cy),cv2.FONT_HERSHEY_SIMPLEX,1.1,color,3,cv2.LINE_AA)
                    tx = 20 + (number % 2) * 960
                    ty = h-top + 60 + (number // 2) * 60
                    if ty < frame.shape[0]-10:
                        cv2.putText(frame,label,(tx,ty),cv2.FONT_HERSHEY_SIMPLEX,1.1,color,3,cv2.LINE_AA)
                    continue
                (tw, th), base = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
                tx = max(0, min(x1, w-tw-12))
                ty = max(th+14, y1-8)
                cv2.rectangle(frame,(x1,y1),(x2,y2),color,3)
                cv2.rectangle(frame,(tx,ty-th-8),(tx+tw+10,ty+base+4),(0,0,0),-1)
                cv2.putText(frame,label,(tx+4,ty),cv2.FONT_HERSHEY_SIMPLEX,0.9,color,2,cv2.LINE_AA)
            cv2.putText(frame, f'{index / (cap.get(cv2.CAP_PROP_FPS) or 30):.2f}s | APPLE=apple ORANGE=tangerine', (20,40), cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2,cv2.LINE_AA)
            if writer is None:
                writer = cv2.VideoWriter(args.out,cv2.VideoWriter_fourcc(*'mp4v'),cap.get(cv2.CAP_PROP_FPS) or 30,(frame.shape[1],frame.shape[0]))
                if not writer.isOpened():
                    raise RuntimeError('Cannot create output')
            writer.write(frame)
            index += 1
    finally:
        cap.release()
        if writer is not None:
            writer.release()
    print(f'Rendered {index} frames: {args.out}')


if __name__ == '__main__':
    main()

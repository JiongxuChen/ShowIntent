"""Probe overlapping crops on the first second of C01; counts are diagnostic only."""
import json
from pathlib import Path

import cv2
from ultralytics import YOLO


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "outputs/a_local_candidate/C01/raw.jsonl"
    rows = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines()]
    video = next((root.parent / "received_media").rglob("C01.mp4"))
    model = YOLO(str(root / "yolo11s.pt"))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    angle = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
    rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    dest = root / "outputs/a_c01_stack_probe"
    dest.mkdir(parents=True, exist_ok=True)
    report = []
    # Probe every half-second through the initial tabletop period.
    selected = [min(rows, key=lambda r: abs(r["timestamp_ms"] - t)) for t in range(0, 6500, 500)]
    for row in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame_id"])
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode C01 frame {row['frame_id']}")
        if angle in rotations:
            frame = cv2.rotate(frame, rotations[angle])
        h, w = frame.shape[:2]
        # Crops deliberately overlap and cover the fruit table band only.
        specs = [("full", 0, 0, w, h), ("table", 0, int(.62*h), w, h),
                 ("left", 0, int(.60*h), int(.68*w), h),
                 ("center", int(.22*w), int(.60*h), int(.78*w), h),
                 ("right", int(.32*w), int(.60*h), w, h)]
        panels = []
        frame_row = {"frame_id": row["frame_id"], "timestamp_ms": row["timestamp_ms"], "variants": []}
        for label, x0, y0, x1, y1 in specs:
            crop = frame[y0:y1, x0:x1]
            pred = model(crop, imgsz=1280, conf=.20, verbose=False)[0]
            objects = []
            canvas = cv2.resize(frame.copy(), (960, round(h*960/w)))
            sh, sw = canvas.shape[:2]
            for box in pred.boxes:
                name = pred.names[int(box.cls[0])]
                if name != "apple":
                    continue
                cx1, cy1, cx2, cy2 = [float(v) for v in box.xyxy[0].tolist()]
                b = [(cx1+x0)/w, (cy1+y0)/h, (cx2+x0)/w, (cy2+y0)/h]
                objects.append({"bbox_norm": b, "score": float(box.conf[0])})
                X1,Y1,X2,Y2 = [int(v*d) for v,d in zip(b,(sw,sh,sw,sh))]
                cv2.rectangle(canvas,(X1,Y1),(X2,Y2),(40,230,80),3)
                cv2.putText(canvas,f"{float(box.conf[0]):.2f}",(X1,max(24,Y1)),cv2.FONT_HERSHEY_SIMPLEX,.65,(40,230,80),2)
            frame_row["variants"].append({"name":label,"apple_count":len(objects),"objects":objects})
            cv2.rectangle(canvas,(0,0),(sw,38),(0,0,0),-1)
            cv2.putText(canvas,f"C01 frame {row['frame_id']} {row['timestamp_ms']/1000:.2f}s | {label} | apples={len(objects)}",(10,27),cv2.FONT_HERSHEY_SIMPLEX,.62,(255,255,255),2)
            panels.append(canvas)
        cv2.imwrite(str(dest/f"C01_{row['frame_id']:03d}.jpg"),cv2.vconcat(panels))
        report.append(frame_row)
    cap.release()
    (dest/"detections.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(dest)


if __name__ == "__main__":
    main()

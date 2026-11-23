"""Test whether a focused right-side crop reveals books missed in clip 3."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.perception.run import apply_rotation, open_capture_without_auto_rotation, resolve_rotation


ROOT = Path(__file__).resolve().parents[1]
VIDEO = ROOT.parent / "new_video" / "video_20261118_184102.mp4"
OUTPUT = ROOT / "outputs" / "new_video_right_crop_probe"
MODEL = ROOT / "yolo11m.pt"


def sample_frame(seconds=30):
    cap = open_capture_without_auto_rotation(str(VIDEO))
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open {VIDEO}")
    try:
        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        cap.set(cv2.CAP_PROP_POS_FRAMES, round(seconds * fps))
        rotation, _ = resolve_rotation(cap.get(cv2.CAP_PROP_ORIENTATION_META), "auto")
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError("Cannot decode target frame")
        return apply_rotation(frame, rotation)
    finally:
        cap.release()


def main():
    OUTPUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    records = []
    for seconds in (30, 35, 38):
        frame = sample_frame(seconds)
        height, width = frame.shape[:2]
        x0, y0, x1, y1 = round(.42 * width), round(.65 * height), width, height
        crop = frame[y0:y1, x0:x1]
        for conf in (.10, .15, .20, .35):
            result = model(crop, imgsz=1280, conf=conf, verbose=False)[0]
            boxes = []
            canvas = frame.copy()
            for i, box in enumerate(result.boxes, 1):
                if result.names[int(box.cls[0])] != "book":
                    continue
                bx1, by1, bx2, by2 = [float(v) for v in box.xyxy[0].tolist()]
                full_box = [bx1 + x0, by1 + y0, bx2 + x0, by2 + y0]
                score = float(box.conf[0])
                boxes.append({"bbox_xyxy": [round(v, 1) for v in full_box],
                              "score": round(score, 4)})
                ax1, ay1, ax2, ay2 = [int(v) for v in full_box]
                cv2.rectangle(canvas, (ax1, ay1), (ax2, ay2), (40, 230, 70), 8)
                cv2.rectangle(canvas, (ax1, max(0, ay1 - 76)), (ax1 + 300, ay1),
                              (40, 230, 70), -1)
                cv2.putText(canvas, f"{i} {score:.2f}", (ax1 + 8, max(55, ay1 - 14)),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.9, (0, 0, 0), 5, cv2.LINE_AA)
            title = f"t={seconds}s right_mid conf={conf:.2f} | books={len(boxes)}"
            cv2.rectangle(canvas, (0, 0), (width, 96), (0, 0, 0), -1)
            cv2.putText(canvas, title, (20, 68), cv2.FONT_HERSHEY_SIMPLEX,
                        2.0, (255, 255, 255), 5, cv2.LINE_AA)
            out_name = f"t{seconds:02d}_conf{int(conf * 100):02d}.jpg"
            cv2.imwrite(str(OUTPUT / out_name), canvas)
            zoom_name = None
            if conf in (0.10, 0.20):
                zx0, zy0, zx1, zy1 = 1000, 2650, width, 3700
                zoom = canvas[zy0:zy1, zx0:zx1]
                zoom = cv2.resize(zoom, None, fx=1.5, fy=1.5,
                                  interpolation=cv2.INTER_CUBIC)
                zoom_name = out_name.replace(".jpg", "_books_zoom.jpg")
                cv2.imwrite(str(OUTPUT / zoom_name), zoom)
            records.append({"seconds": seconds, "region": "right_mid",
                            "crop_xyxy": [x0, y0, x1, y1],
                            "confidence": conf, "book_count": len(boxes),
                            "detections": boxes, "image": out_name,
                            "books_zoom": zoom_name})
            print(f"t={seconds}s conf={conf:.2f}: {len(boxes)} books")
    (OUTPUT / "detections.json").write_text(
        json.dumps(records, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(OUTPUT)


if __name__ == "__main__":
    main()

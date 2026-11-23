"""Diagnose whether tiled, higher-resolution inference helps separate books.

This experiment preserves raw tiled detections and compares optional NMS levels;
it does not modify the production A runner or claim accuracy without labels.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from ultralytics import YOLO

from src.perception.run import apply_rotation, open_capture_without_auto_rotation, resolve_rotation


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT.parent / "new_video"
OUTPUT_DIR = ROOT / "outputs" / "new_video_tiling_probe"
MODEL_PATH = ROOT / "yolo11m.pt"
SAMPLES = [
    ("video_20261118_183631.mp4", 20),
    ("video_20261118_183847.mp4", 40),
    ("video_20261118_184102.mp4", 30),
]


def read_upright_frame(path: Path, seconds: float):
    capture = open_capture_without_auto_rotation(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_id = round(seconds * fps)
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id):
            raise RuntimeError(f"Cannot seek to frame {frame_id}: {path}")
        rotation, rotation_name = resolve_rotation(
            capture.get(cv2.CAP_PROP_ORIENTATION_META), "auto"
        )
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode frame {frame_id}: {path}")
        return apply_rotation(frame, rotation), frame_id, rotation_name
    finally:
        capture.release()


def infer_books(model, image, x_offset=0, y_offset=0, imgsz=1280, confidence=0.20):
    result = model(image, imgsz=imgsz, conf=confidence, verbose=False)[0]
    detections = []
    for box in result.boxes:
        if result.names[int(box.cls[0])] != "book":
            continue
        x1, y1, x2, y2 = (float(v) for v in box.xyxy[0].tolist())
        detections.append({"bbox": [x1 + x_offset, y1 + y_offset,
                                     x2 + x_offset, y2 + y_offset],
                           "score": float(box.conf[0])})
    return detections


def iou(a, b):
    x1, y1 = max(a[0], b[0]), max(a[1], b[1])
    x2, y2 = min(a[2], b[2]), min(a[3], b[3])
    intersection = max(0, x2 - x1) * max(0, y2 - y1)
    area_a = max(0, a[2] - a[0]) * max(0, a[3] - a[1])
    area_b = max(0, b[2] - b[0]) * max(0, b[3] - b[1])
    return intersection / max(area_a + area_b - intersection, 1e-9)


def nms(detections, threshold):
    kept = []
    for detection in sorted(detections, key=lambda item: item["score"], reverse=True):
        if all(iou(detection["bbox"], other["bbox"]) <= threshold for other in kept):
            kept.append(detection)
    return kept


def tiled_detections(model, frame):
    height, width = frame.shape[:2]
    tile_width, tile_height = round(width * 0.62), round(height * 0.62)
    x_starts = [0, width - tile_width]
    y_starts = [0, height - tile_height]
    detections = []
    for y0 in y_starts:
        for x0 in x_starts:
            tile = frame[y0:y0 + tile_height, x0:x0 + tile_width]
            detections.extend(infer_books(model, tile, x0, y0))
    return detections


def lower_region_detections(model, frame, top_fraction):
    height = frame.shape[0]
    y0 = round(height * top_fraction)
    return infer_books(model, frame[y0:, :], y_offset=y0)


def annotate(frame, detections, title):
    image = frame.copy()
    cv2.rectangle(image, (0, 0), (image.shape[1], 100), (0, 0, 0), -1)
    cv2.putText(image, f"{title} | books={len(detections)}", (24, 72),
                cv2.FONT_HERSHEY_SIMPLEX, 2.2, (255, 255, 255), 6, cv2.LINE_AA)
    for index, detection in enumerate(detections, 1):
        x1, y1, x2, y2 = [int(v) for v in detection["bbox"]]
        cv2.rectangle(image, (x1, y1), (x2, y2), (40, 230, 70), 8)
        label = f"{index} {detection['score']:.2f}"
        cv2.rectangle(image, (x1, max(0, y1 - 78)), (x1 + 330, y1), (40, 230, 70), -1)
        cv2.putText(image, label, (x1 + 8, max(60, y1 - 16)),
                    cv2.FONT_HERSHEY_SIMPLEX, 2.0, (0, 0, 0), 6, cv2.LINE_AA)
    return image


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL_PATH))
    report = []
    for filename, seconds in SAMPLES:
        frame, frame_id, rotation = read_upright_frame(VIDEO_DIR / filename, seconds)
        name = f"{Path(filename).stem}_{seconds:02d}s"
        full_1280 = infer_books(model, frame, imgsz=1280)
        full_1920 = infer_books(model, frame, imgsz=1920)
        lower_30 = lower_region_detections(model, frame, 0.30)
        lower_40 = lower_region_detections(model, frame, 0.40)
        tiled = tiled_detections(model, frame)
        variants = {
            "full_1280": full_1280,
            "full_1920": full_1920,
            "lower_30pct": lower_30,
            "lower_40pct": lower_40,
            "tiles_raw": tiled,
            "tiles_nms70": nms(tiled, 0.70),
            "tiles_nms85": nms(tiled, 0.85),
        }
        panels = []
        sample_result = {"video": filename, "seconds": seconds,
                         "frame_id": frame_id, "rotation": rotation,
                         "size": [frame.shape[1], frame.shape[0]],
                         "variants": {}}
        for variant_name, detections in variants.items():
            sample_result["variants"][variant_name] = [
                {"bbox_xyxy": [round(v, 1) for v in d["bbox"]],
                 "score": round(d["score"], 4)} for d in detections
            ]
            image = annotate(frame, detections, variant_name)
            cv2.imwrite(str(OUTPUT_DIR / f"{name}_{variant_name}.jpg"), image)
            if variant_name in {"full_1280", "lower_30pct", "lower_40pct", "tiles_raw"}:
                scaled_h = round(image.shape[0] * 720 / image.shape[1])
                panels.append(cv2.resize(image, (720, scaled_h), interpolation=cv2.INTER_AREA))
        cv2.imwrite(str(OUTPUT_DIR / f"{name}_compare.jpg"),
                    np.vstack([np.hstack(panels[:2]), np.hstack(panels[2:])]))
        report.append(sample_result)
        print(name + ": " + ", ".join(
            f"{variant}={len(detections)}" for variant, detections in variants.items()
        ))
    (OUTPUT_DIR / "detections.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved diagnostic comparisons: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

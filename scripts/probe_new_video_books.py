"""Compare local YOLO11 checkpoints on representative new-video book frames.

This is a detector-only diagnostic, not a recall/precision benchmark and not
the full A packet pipeline. It writes annotated frame comparisons under outputs/.
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
OUTPUT_DIR = ROOT / "outputs" / "new_video_model_compare"
MODEL_PATHS = {
    "nano": ROOT.parent / "yolo11n.pt",
    "small": ROOT / "yolo11s.pt",
    "medium": ROOT / "yolo11m.pt",
}
SAMPLES = {
    "video_20261118_183631.mp4": [0, 20, 40],
    "video_20261118_183847.mp4": [0, 40, 80],
    "video_20261118_184102.mp4": [0, 30, 55],
}
FOCUS_CLASSES = {"book", "person"}


def read_upright_frame(path: Path, seconds: float):
    capture = open_capture_without_auto_rotation(str(path))
    if not capture.isOpened():
        raise RuntimeError(f"Cannot open video: {path}")
    try:
        fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        frame_id = min(max(0, round(seconds * fps)), max(0, frame_count - 1))
        if not capture.set(cv2.CAP_PROP_POS_FRAMES, frame_id):
            raise RuntimeError(f"Cannot seek to frame {frame_id}: {path}")
        rotation_code, rotation_name = resolve_rotation(
            capture.get(cv2.CAP_PROP_ORIENTATION_META), "auto"
        )
        ok, frame = capture.read()
        if not ok:
            raise RuntimeError(f"Cannot decode frame {frame_id}: {path}")
        return apply_rotation(frame, rotation_code), frame_id, rotation_name
    finally:
        capture.release()


def annotate(frame, result, model_name: str):
    image = frame.copy()
    found = []
    for box in result.boxes:
        label = result.names[int(box.cls[0])]
        if label not in FOCUS_CLASSES:
            continue
        score = float(box.conf[0])
        x1, y1, x2, y2 = (int(value) for value in box.xyxy[0].tolist())
        color = (30, 230, 60) if label == "book" else (0, 150, 255)
        thickness = 8 if label == "book" else 5
        cv2.rectangle(image, (x1, y1), (x2, y2), color, thickness)
        text = f"{label} {score:.2f}"
        font_scale, text_thickness = 2.3, 6
        (text_w, text_h), baseline = cv2.getTextSize(
            text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, text_thickness
        )
        top = max(text_h + baseline + 4, y1)
        cv2.rectangle(image, (x1, top - text_h - baseline - 8),
                      (x1 + text_w + 8, top), color, -1)
        cv2.putText(image, text, (x1 + 4, top - baseline - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), text_thickness,
                    cv2.LINE_AA)
        found.append({"class": label, "score": round(score, 4),
                      "bbox_xyxy": [x1, y1, x2, y2]})

    title = f"{model_name} | book={sum(x['class'] == 'book' for x in found)}"
    cv2.rectangle(image, (0, 0), (image.shape[1], 90), (0, 0, 0), -1)
    cv2.putText(image, title, (20, 66), cv2.FONT_HERSHEY_SIMPLEX,
                2.0, (255, 255, 255), 5, cv2.LINE_AA)
    return image, found


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    report = []
    for filename, sample_seconds in SAMPLES.items():
        video = VIDEO_DIR / filename
        if not video.is_file():
            raise FileNotFoundError(video)
        model_outputs = {}
        models = {name: YOLO(str(path)) for name, path in MODEL_PATHS.items()}
        for seconds in sample_seconds:
            frame, frame_id, rotation = read_upright_frame(video, seconds)
            scene_key = f"{Path(filename).stem}_{seconds:02d}s"
            scene_results = {"video": filename, "seconds": seconds,
                             "frame_id": frame_id, "rotation": rotation,
                             "upright_size": [frame.shape[1], frame.shape[0]],
                             "models": {}}
            panels = []
            for model_name, model in models.items():
                result = model(frame, imgsz=1280, conf=0.35, verbose=False)[0]
                image, detections = annotate(frame, result, model_name)
                panel_path = OUTPUT_DIR / f"{scene_key}_{model_name}.jpg"
                cv2.imwrite(str(panel_path), image)
                scene_results["models"][model_name] = {
                    "book_count": sum(x["class"] == "book" for x in detections),
                    "person_count": sum(x["class"] == "person" for x in detections),
                    "detections": detections,
                }
                scaled_h = round(image.shape[0] * 720 / image.shape[1])
                panel = cv2.resize(image, (720, scaled_h), interpolation=cv2.INTER_AREA)
                panels.append(panel)
            comparison_path = OUTPUT_DIR / f"{scene_key}_compare.jpg"
            cv2.imwrite(str(comparison_path), np.hstack(panels))
            scene_results["comparison_image"] = comparison_path.name
            report.append(scene_results)
            print(f"{scene_key}: " + ", ".join(
                f"{name}={scene_results['models'][name]['book_count']} books"
                for name in MODEL_PATHS
            ))
        del models

    (OUTPUT_DIR / "detections.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved comparison set: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

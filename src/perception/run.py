"""Run A/P0 visual perception on a video file or camera.

The runner emits the agreed PerceptionPacket JSONL contract and an optional
debug overlay. It has no input for a predeclared target object by design.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np

from .relations import infer_relations
from .tracker import SimpleObjectTracker
from .types import Detection, HandDetection, PerceptionPacket


COCO_INTERESTING = {"apple", "banana", "bottle", "cup", "orange", "cell phone", "remote", "book", "scissors"}


def normalize_box(box: Iterable[float], width: int, height: int) -> list[float]:
    x1, y1, x2, y2 = box
    return [round(max(0.0, min(1.0, x1 / width)), 6), round(max(0.0, min(1.0, y1 / height)), 6), round(max(0.0, min(1.0, x2 / width)), 6), round(max(0.0, min(1.0, y2 / height)), 6)]


def clamp_box_unit(box: Iterable[float]) -> list[float]:
    """Clamp an already-normalized box into [0, 1] without rescaling."""
    x1, y1, x2, y2 = box
    return [round(max(0.0, min(1.0, float(v))), 6) for v in (x1, y1, x2, y2)]


def resolve_rotation(meta_degrees: float, override: str) -> tuple[int, str]:
    """Pick the OpenCV rotation that yields the upright display orientation.

    Phone clips carry an EXIF-style rotation flag (CAP_PROP_ORIENTATION_META).
    OpenCV does not apply it by default, so frames decode lying sideways.
    """
    if override != "auto":
        codes = {"cw": (cv2.ROTATE_90_CLOCKWISE, "cw"), "ccw": (cv2.ROTATE_90_COUNTERCLOCKWISE, "ccw"), "180": (cv2.ROTATE_180, "180"), "none": (None, "none")}
        return codes[override]
    degrees = int(meta_degrees) % 360
    if degrees == 90:
        return cv2.ROTATE_90_CLOCKWISE, "cw"
    if degrees == 180:
        return cv2.ROTATE_180, "180"
    if degrees == 270:
        return cv2.ROTATE_90_COUNTERCLOCKWISE, "ccw"
    return None, "none"


def apply_rotation(frame: np.ndarray, rotation_code: int | None) -> np.ndarray:
    # NB: cv2.ROTATE_90_CLOCKWISE == 0, so None (not 0) must mean "no rotation".
    return cv2.rotate(frame, rotation_code) if rotation_code is not None else frame


class PerceptionModels:
    def __init__(self, yolo_model: str, confidence: float, use_hands: bool) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise RuntimeError("Missing YOLO dependency. Run: pip install -r requirements-a.txt") from exc
        self.yolo = YOLO(yolo_model)
        self.confidence = confidence
        self.hands = None
        if use_hands:
            try:
                import mediapipe as mp
                self.hands = mp.solutions.hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5, min_tracking_confidence=0.5)
            except ImportError as exc:
                raise RuntimeError("Missing MediaPipe dependency. Run: pip install -r requirements-a.txt") from exc

    def detect_objects(self, frame: np.ndarray, imgsz: int) -> list[tuple[str, tuple[float, float, float, float], float]]:
        result = self.yolo(frame, verbose=False, conf=self.confidence, imgsz=imgsz)[0]
        output = []
        for box in result.boxes:
            class_name = result.names[int(box.cls[0])]
            if class_name not in COCO_INTERESTING:
                continue
            x1, y1, x2, y2 = (float(value) for value in box.xyxy[0].tolist())
            output.append((class_name, (x1, y1, x2, y2), float(box.conf[0])))
        return output

    def detect_hands(self, frame: np.ndarray) -> list[tuple[str, tuple[float, float, float, float], float]]:
        if self.hands is None:
            return []
        result = self.hands.process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
        if not result.multi_hand_landmarks:
            return []
        output = []
        for index, landmarks in enumerate(result.multi_hand_landmarks):
            xs = [point.x for point in landmarks.landmark]
            ys = [point.y for point in landmarks.landmark]
            label = result.multi_handedness[index].classification[0].label.lower() if result.multi_handedness else str(index)
            # Landmarks are already normalized to the frame; only clamp, never rescale.
            output.append((f"hand_{label}", clamp_box_unit((min(xs), min(ys), max(xs), max(ys))), 1.0))
        return output


def draw_overlay(frame: np.ndarray, objects: list[Detection], hands: list[HandDetection]) -> np.ndarray:
    height, width = frame.shape[:2]
    canvas = frame.copy()
    for obj in objects:
        x1, y1, x2, y2 = [int(value * dimension) for value, dimension in zip(obj.bbox_norm, (width, height, width, height))]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 220, 0), 2)
        cv2.putText(canvas, f"{obj.class_name} {obj.track_id} {obj.score:.2f}", (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 220, 0), 2)
    for hand in hands:
        x1, y1, x2, y2 = [int(value * dimension) for value, dimension in zip(hand.bbox_norm, (width, height, width, height))]
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (255, 170, 0), 2)
        cv2.putText(canvas, hand.hand_id, (x1, max(20, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 170, 0), 2)
    return canvas


def parse_source(value: str) -> str | int:
    return int(value) if value.isdigit() else value


def main() -> None:
    parser = argparse.ArgumentParser(description="ShowIntent A/P0 visual perception")
    parser.add_argument("--input", required=True, help="Video path or camera index, e.g. 0")
    parser.add_argument("--out", required=True, type=Path, help="Output PerceptionPacket JSONL")
    parser.add_argument("--overlay-out", type=Path, help="Optional annotated MP4")
    parser.add_argument("--yolo-model", default="yolo11n.pt")
    parser.add_argument("--confidence", type=float, default=0.35)
    parser.add_argument("--imgsz", type=int, default=1280, help="YOLO inference size; 1280 keeps small fruit visible in 4K frames")
    parser.add_argument("--rotate", default="auto", choices=["auto", "cw", "ccw", "180", "none"], help="Upright-orientation handling; auto follows the container rotation flag")
    parser.add_argument("--no-hands", action="store_true", help="Run object detection only")
    parser.add_argument("--display", action="store_true")
    parser.add_argument("--max-frames", type=int, default=0, help="Useful for a short smoke test")
    parser.add_argument("--metrics-out", type=Path, help="Optional JSON file with throughput and input metadata")
    args = parser.parse_args()

    models = PerceptionModels(args.yolo_model, args.confidence, not args.no_hands)
    capture = cv2.VideoCapture(parse_source(args.input))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open input: {args.input}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    rotation_code, rotation_name = resolve_rotation(capture.get(cv2.CAP_PROP_ORIENTATION_META), args.rotate)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    tracker = SimpleObjectTracker()
    frame_id = 0
    started = time.perf_counter()

    with args.out.open("w", encoding="utf-8") as log:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = apply_rotation(frame, rotation_code)
            height, width = frame.shape[:2]
            timestamp_ms = int(capture.get(cv2.CAP_PROP_POS_MSEC)) if not isinstance(parse_source(args.input), int) else int(frame_id * 1000 / fps)
            raw_objects = models.detect_objects(frame, args.imgsz)
            tracked = tracker.update([(name, normalize_box(box, width, height), score) for name, box, score in raw_objects])
            objects = [Detection(track_id, name, list(box), round(score, 3)) for track_id, name, box, score in tracked]
            hands = [HandDetection(hand_id, box, round(score, 3)) for hand_id, box, score in models.detect_hands(frame)]
            packet = PerceptionPacket(timestamp_ms, frame_id, objects, hands, infer_relations(hands, objects))
            log.write(json.dumps(packet.to_dict(), ensure_ascii=False) + "\n")

            overlay = draw_overlay(frame, objects, hands)
            if args.overlay_out:
                if writer is None:
                    args.overlay_out.parent.mkdir(parents=True, exist_ok=True)
                    writer = cv2.VideoWriter(str(args.overlay_out), cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, height))
                writer.write(overlay)
            if args.display:
                cv2.imshow("ShowIntent A - Perception", overlay)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            frame_id += 1
            if args.max_frames and frame_id >= args.max_frames:
                break

    capture.release()
    if writer is not None:
        writer.release()
    cv2.destroyAllWindows()
    elapsed = time.perf_counter() - started
    metrics = {
        "frames_processed": frame_id,
        "elapsed_seconds": round(elapsed, 3),
        "processing_fps": round(frame_id / max(elapsed, 1e-6), 2),
        "input_fps": round(float(fps), 2),
        "yolo_model": args.yolo_model,
        "imgsz": args.imgsz,
        "hands_enabled": not args.no_hands,
        "rotation_applied": rotation_name,
    }
    if args.metrics_out:
        args.metrics_out.parent.mkdir(parents=True, exist_ok=True)
        args.metrics_out.write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Processed {frame_id} frames in {elapsed:.2f}s ({metrics['processing_fps']:.1f} FPS). JSONL: {args.out}")


if __name__ == "__main__":
    main()


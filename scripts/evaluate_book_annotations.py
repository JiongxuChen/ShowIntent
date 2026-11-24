"""Compare YOLO book detections with manually drawn YOLO-format labels."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
from ultralytics import YOLO


ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "outputs" / "book_annotation_seed"
OUT = ROOT / "outputs" / "book_annotation_eval"
MODEL = ROOT / "yolo11m.pt"
CONFIDENCES = (0.35, 0.20, 0.10)


def read_labels(path: Path, width: int, height: int) -> list[tuple[float, float, float, float]]:
    boxes = []
    for row in path.read_text(encoding="utf-8").splitlines():
        if not row.strip():
            continue
        class_id, cx, cy, bw, bh = map(float, row.split())
        if int(class_id) != 0:
            continue
        boxes.append(((cx - bw / 2) * width, (cy - bh / 2) * height,
                      (cx + bw / 2) * width, (cy + bh / 2) * height))
    return boxes


def box_iou(a, b) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def match_boxes(ground_truth, predictions):
    # Maximum-cardinality bipartite matching over qualifying IoU edges avoids
    # a greedy high-IoU pair hiding two valid lower-IoU matches.
    overlaps = [[box_iou(gt, pred[0]) for pred in predictions]
                for gt in ground_truth]
    pred_to_gt: dict[int, int] = {}

    def assign(gi: int, visited: set[int]) -> bool:
        candidates = sorted(range(len(predictions)),
                            key=lambda pi: overlaps[gi][pi], reverse=True)
        for pi in candidates:
            if overlaps[gi][pi] < 0.5 or pi in visited:
                continue
            visited.add(pi)
            if pi not in pred_to_gt or assign(pred_to_gt[pi], visited):
                pred_to_gt[pi] = gi
                return True
        return False

    for gi in range(len(ground_truth)):
        assign(gi, set())
    return [
        {"gt_index": gi, "prediction_index": pi,
         "iou": round(overlaps[gi][pi], 4),
         "score": round(predictions[pi][1], 4)}
        for pi, gi in sorted(pred_to_gt.items())
    ]


def draw_box(image, box, label, color):
    x1, y1, x2, y2 = [int(round(v)) for v in box]
    cv2.rectangle(image, (x1, y1), (x2, y2), color, 5)
    (tw, th), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    top = max(th + baseline + 4, y1)
    cv2.rectangle(image, (x1, max(0, top - th - baseline - 6)),
                  (x1 + tw + 8, top), color, -1)
    cv2.putText(image, label, (x1 + 4, top - baseline - 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 0), 2, cv2.LINE_AA)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    model = YOLO(str(MODEL))
    report = []
    labelled_stems = {
        path.stem for path in (DATA / "labels").glob("*.txt")
        if path.read_text(encoding="utf-8").strip()
    }
    for image_path in sorted((DATA / "images").glob("*.jpg")):
        if image_path.stem not in labelled_stems:
            continue
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read image: {image_path}")
        height, width = image.shape[:2]
        gt = read_labels(DATA / "labels" / f"{image_path.stem}.txt", width, height)
        for confidence in CONFIDENCES:
            result = model(image, imgsz=1280, conf=confidence, verbose=False)[0]
            predictions = []
            for box in result.boxes:
                if result.names[int(box.cls[0])] != "book":
                    continue
                coords = tuple(float(v) for v in box.xyxy[0].tolist())
                predictions.append((coords, float(box.conf[0])))
            matches = match_boxes(gt, predictions)
            true_positives = sum(item["iou"] >= 0.5 for item in matches)
            report.append({
                "image": image_path.name,
                "confidence": confidence,
                "ground_truth_count": len(gt),
                "prediction_count": len(predictions),
                "matches_iou_0_5": true_positives,
                "matches": matches,
                "ground_truth_xyxy": [[round(v, 1) for v in b] for b in gt],
                "predictions": [{"bbox_xyxy": [round(v, 1) for v in box],
                                 "score": round(score, 4)}
                                for box, score in predictions],
            })
            if confidence == 0.35:
                overlay = image.copy()
                for index, box in enumerate(gt, 1):
                    draw_box(overlay, box, f"GT {index}", (255, 180, 0))
                for index, (box, score) in enumerate(predictions, 1):
                    draw_box(overlay, box, f"P{index} {score:.2f}", (0, 80, 255))
                title = f"{image_path.stem} | GT blue | YOLO red | {len(gt)} vs {len(predictions)}"
                cv2.rectangle(overlay, (0, 0), (width, 52), (0, 0, 0), -1)
                cv2.putText(overlay, title, (12, 37), cv2.FONT_HERSHEY_SIMPLEX,
                            0.8, (255, 255, 255), 2, cv2.LINE_AA)
                cv2.imwrite(str(OUT / f"{image_path.stem}_comparison.jpg"), overlay)
            print(f"{image_path.stem} conf={confidence:.2f}: "
                  f"pred={len(predictions)} match@0.5={true_positives}/{len(gt)}")
    (OUT / "metrics.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved comparisons and metrics to {OUT}")


if __name__ == "__main__":
    main()

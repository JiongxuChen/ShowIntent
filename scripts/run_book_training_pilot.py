"""Fine-tune a local YOLO11s book detector and compare on held-out video frames.

All inputs and outputs live under ignored ``outputs/``. The script never uploads
images, labels, checkpoints, or predictions to the repository.
"""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import torch
from ultralytics import YOLO

from scripts.evaluate_book_annotations import match_boxes, read_labels


ROOT = Path(__file__).resolve().parents[1]
DATASET = ROOT / "outputs" / "book_detector_dataset"
ANNOTATIONS = ROOT / "outputs" / "book_annotation_seed"
BASE_MODEL = ROOT / "yolo11m.pt"
RUN_DIR = ROOT / "outputs" / "book_training_pilot" / "yolo11m"
REPORT = ROOT / "outputs" / "book_training_pilot" / "comparison.json"
CONFIDENCES = (0.10, 0.20, 0.35)
IMGSZ = 960


def evaluate(model_path: Path, tag: str) -> list[dict]:
    model = YOLO(str(model_path))
    rows: list[dict] = []
    for image_path in sorted((DATASET / "val" / "images").glob("*.jpg")):
        image = cv2.imread(str(image_path))
        if image is None:
            raise RuntimeError(f"Cannot read validation image: {image_path}")
        height, width = image.shape[:2]
        label_path = DATASET / "val" / "labels" / f"{image_path.stem}.txt"
        gt = read_labels(label_path, width, height)
        for confidence in CONFIDENCES:
            result = model(image, imgsz=IMGSZ, conf=confidence, verbose=False)[0]
            predictions = [
                (tuple(float(v) for v in box.xyxy[0].tolist()), float(box.conf[0]))
                for box in result.boxes
                if result.names[int(box.cls[0])] == "book"
            ]
            matches = match_boxes(gt, predictions)
            tp = sum(match["iou"] >= 0.5 for match in matches)
            rows.append({
                "model": tag,
                "image": image_path.name,
                "confidence": confidence,
                "ground_truth": len(gt),
                "predictions": len(predictions),
                "tp_iou50": tp,
                "precision": tp / len(predictions) if predictions else 0.0,
                "recall": tp / len(gt) if gt else 0.0,
            })
    return rows


def main() -> None:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable; activate the repository .venv with CUDA PyTorch.")
    if not BASE_MODEL.is_file():
        raise FileNotFoundError(BASE_MODEL)
    if not (DATASET / "data.yaml").is_file():
        raise FileNotFoundError("Run scripts/prepare_book_training_dataset.py first")

    RUN_DIR.mkdir(parents=True, exist_ok=True)
    gpu = torch.cuda.get_device_name(0)
    print(f"GPU: {gpu}; imgsz={IMGSZ}; validation split is video-disjoint")
    baseline = evaluate(BASE_MODEL, "yolo11m_pretrained")

    model = YOLO(str(BASE_MODEL))
    model.train(
        data=str(DATASET / "data.yaml"),
        epochs=30,
        imgsz=IMGSZ,
        batch=1,
        device=0,
        workers=0,
        project=str(RUN_DIR),
        name="finetune",
        exist_ok=True,
        patience=10,
        plots=False,
        cache=False,
        seed=42,
    )
    best = RUN_DIR / "finetune" / "weights" / "best.pt"
    if not best.is_file():
        raise FileNotFoundError(f"Training did not produce {best}")
    finetuned = evaluate(best, "yolo11m_finetuned")

    payload = {
        "scope": "small pilot only; 17 manually annotated frames from three clips",
        "warning": (
            "Six validation frames are from one held-out clip. Do not treat this as a "
            "generalization claim or production model selection."
        ),
        "gpu": gpu,
        "imgsz": IMGSZ,
        "epochs_requested": 30,
        "train_images": 11,
        "validation_images": 6,
        "validation_clip": "20261118_183847",
        "baseline_model": str(BASE_MODEL),
        "finetuned_model": str(best),
        "results": baseline + finetuned,
    }
    REPORT.parent.mkdir(parents=True, exist_ok=True)
    REPORT.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Saved baseline/fine-tuned comparison: {REPORT}")
    for tag in ("yolo11m_pretrained", "yolo11m_finetuned"):
        for confidence in CONFIDENCES:
            selected = [r for r in payload["results"]
                        if r["model"] == tag and r["confidence"] == confidence]
            tp = sum(r["tp_iou50"] for r in selected)
            pred = sum(r["predictions"] for r in selected)
            gt = sum(r["ground_truth"] for r in selected)
            print(f"{tag} conf={confidence:.2f}: TP={tp}/{gt}, predictions={pred}, "
                  f"precision={tp / pred if pred else 0:.3f}, recall={tp / gt if gt else 0:.3f}")


if __name__ == "__main__":
    main()

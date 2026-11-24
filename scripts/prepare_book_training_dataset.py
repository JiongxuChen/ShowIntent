"""Create a local, video-disjoint YOLO dataset from manual book annotations."""

from __future__ import annotations

import csv
import json
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SEED = ROOT / "outputs" / "book_annotation_seed"
OUT = ROOT / "outputs" / "book_detector_dataset"
# Hold out the entire middle clip so no neighboring frames leak across splits.
VAL_CLIP = "20261118_183847"


def main() -> None:
    rows = list(csv.DictReader((SEED / "frames.csv").open(encoding="utf-8-sig")))
    manifest: list[dict[str, str | int]] = []
    split_clips: dict[str, set[str]] = {"train": set(), "val": set()}
    expected_names: dict[str, set[str]] = {"train": set(), "val": set()}

    for row in rows:
        image_path = SEED / row["image"]
        label_path = SEED / row["label"]
        if not image_path.is_file() or not label_path.is_file():
            raise FileNotFoundError(image_path if not image_path.is_file() else label_path)
        label_rows = [line for line in label_path.read_text(encoding="utf-8").splitlines()
                      if line.strip()]
        if not label_rows:
            raise ValueError(f"Missing annotations: {label_path}")
        for line_number, label in enumerate(label_rows, start=1):
            values = label.split()
            if len(values) != 5 or values[0] != "0":
                raise ValueError(f"Invalid class/YOLO row in {label_path}:{line_number}")
            class_id, cx, cy, width, height = map(float, values)
            if class_id != 0 or not (0 <= cx <= 1 and 0 <= cy <= 1
                                     and 0 < width <= 1 and 0 < height <= 1):
                raise ValueError(f"Out-of-range YOLO box in {label_path}:{line_number}")

        clip = Path(row["video"]).stem.removeprefix("video_")
        split = "val" if clip == VAL_CLIP else "train"
        split_clips[split].add(clip)
        expected_names[split].add(image_path.name)
        for folder, source in (("images", image_path), ("labels", label_path)):
            destination = OUT / split / folder / source.name
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source, destination)
        manifest.append({"image": image_path.name, "clip": clip, "split": split,
                         "seconds": int(row["seconds"]), "book_count": len(label_rows)})

    if split_clips["train"] & split_clips["val"]:
        raise AssertionError("Video leakage across train/validation split")
    for split, names in expected_names.items():
        image_dir = OUT / split / "images"
        actual_names = {path.name for path in image_dir.glob("*.jpg")}
        if actual_names != names:
            stale = sorted(actual_names - names)
            raise RuntimeError(
                f"Stale files in {image_dir}: {stale}. Remove only this generated "
                "split directory before rebuilding the dataset."
            )
    yaml_path = OUT / "data.yaml"
    yaml_path.write_text(
        f"path: {json.dumps(OUT.as_posix())}\ntrain: train/images\n"
        "val: val/images\nnames:\n  0: book\n",
        encoding="utf-8",
    )
    (OUT / "split_manifest.json").write_text(
        json.dumps({"validation_clip": VAL_CLIP, "splits": manifest},
                   ensure_ascii=False, indent=2), encoding="utf-8",
    )
    for split, clips in split_clips.items():
        count = sum(row["split"] == split for row in manifest)
        boxes = sum(int(row["book_count"]) for row in manifest if row["split"] == split)
        print(f"{split}: {count} frames, {boxes} boxes, clips={sorted(clips)}")
    print(f"Created local YOLO dataset: {yaml_path}")


if __name__ == "__main__":
    main()

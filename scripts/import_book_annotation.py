"""Import a user-marked cyan-box screenshot as a YOLO label file."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
BATCH = ROOT / "outputs" / "book_annotation_batch3"
SEED = ROOT / "outputs" / "book_annotation_seed"
FRAMES_CSV = SEED / "frames.csv"


def read_candidates() -> dict[str, dict[str, str]]:
    with (BATCH / "frames.csv").open(encoding="utf-8-sig", newline="") as handle:
        return {row["image"]: row for row in csv.DictReader(handle)}


def find_cyan_boxes(image: np.ndarray) -> list[tuple[int, int, int, int]]:
    # Annotation strokes are cyan/blue. Restrict by hue and reject tiny artwork
    # details on book covers by requiring a minimum rectangle area.
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array((82, 120, 150)), np.array((112, 255, 255)))
    mask = cv2.morphologyEx(
        mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    )
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = []
    for contour in contours:
        x, y, width, height = cv2.boundingRect(contour)
        if width < 35 or height < 50 or width * height < 8_000:
            continue
        boxes.append((x, y, x + width, y + height))
    return sorted(boxes, key=lambda box: (box[1], box[0]))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotation", type=Path, required=True,
                        help="User-marked screenshot with cyan book boxes")
    parser.add_argument("--image", required=True,
                        help="Candidate filename from book_annotation_batch3/frames.csv")
    args = parser.parse_args()

    candidates = read_candidates()
    if args.image not in candidates:
        raise SystemExit(f"Unknown candidate image: {args.image}")
    row = candidates[args.image]
    source = BATCH / "images" / args.image
    marked = cv2.imread(str(args.annotation))
    original = cv2.imread(str(source))
    if marked is None or original is None:
        raise SystemExit("Cannot read annotated screenshot or source image")
    if marked.shape[:2] != original.shape[:2]:
        raise SystemExit(
            f"Size mismatch: marked={marked.shape[1]}x{marked.shape[0]}, "
            f"source={original.shape[1]}x{original.shape[0]}"
        )

    boxes = find_cyan_boxes(marked)
    if not boxes:
        raise SystemExit("No cyan book boxes found; nothing imported")
    width, height = original.shape[1], original.shape[0]
    if any(x1 < 0 or y1 < 0 or x2 > width or y2 > height
           for x1, y1, x2, y2 in boxes):
        raise SystemExit("A detected box falls outside the source image")

    destination_image = SEED / "images" / args.image
    destination_label = SEED / "labels" / f"{Path(args.image).stem}.txt"
    if destination_image.exists() or destination_label.exists():
        raise SystemExit(f"Annotation already imported: {args.image}")
    shutil.copy2(source, destination_image)
    label_lines = []
    for x1, y1, x2, y2 in boxes:
        center_x = (x1 + x2) / (2 * width)
        center_y = (y1 + y2) / (2 * height)
        box_width = (x2 - x1) / width
        box_height = (y2 - y1) / height
        label_lines.append(
            f"0 {center_x:.6f} {center_y:.6f} {box_width:.6f} {box_height:.6f}"
        )
    destination_label.write_text("\n".join(label_lines) + "\n", encoding="utf-8")

    with FRAMES_CSV.open(encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames
        existing = list(reader)
    video = f"video_{row['clip']}.mp4"
    existing.append({
        "image": f"images/{args.image}",
        "label": f"labels/{Path(args.image).stem}.txt",
        "video": video,
        "seconds": row["seconds"],
        "upright_width": str(width),
        "upright_height": str(height),
    })
    with FRAMES_CSV.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(existing)

    preview = original.copy()
    for index, box in enumerate(boxes, start=1):
        cv2.rectangle(preview, box[:2], box[2:], (255, 180, 0), 6)
        cv2.putText(preview, str(index), (box[0] + 8, box[1] + 45),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0, 0, 0), 5, cv2.LINE_AA)
    preview_path = SEED / "previews" / f"{Path(args.image).stem}_boxes.jpg"
    preview_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(preview_path), preview, [cv2.IMWRITE_JPEG_QUALITY, 95])
    print(f"Imported {args.image}: {len(boxes)} book boxes, {width}x{height}")
    for index, box in enumerate(boxes, start=1):
        print(f"  {index}: xyxy={box}")
    print(f"Label: {destination_label}")
    print(f"Preview: {preview_path}")


if __name__ == "__main__":
    main()

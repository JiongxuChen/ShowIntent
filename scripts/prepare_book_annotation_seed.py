"""Extract a small, upright frame set for manual per-book box annotations."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np

from src.perception.run import apply_rotation, open_capture_without_auto_rotation, resolve_rotation


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT.parent / "new_video"
OUT = ROOT / "outputs" / "book_annotation_seed"
SAMPLES = {
    "video_20261118_183631.mp4": (0, 10, 20, 30, 38),
    "video_20261118_183847.mp4": (0, 15, 30, 45, 60, 78),
    "video_20261118_184102.mp4": (0, 20, 30, 35, 45, 55),
}
TARGET_WIDTH = 1080


def main() -> None:
    images_dir = OUT / "images"
    labels_dir = OUT / "labels"
    images_dir.mkdir(parents=True, exist_ok=True)
    labels_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, str | int]] = []

    for video_name, times in SAMPLES.items():
        video_path = VIDEO_DIR / video_name
        if not video_path.exists():
            raise FileNotFoundError(video_path)
        cap = open_capture_without_auto_rotation(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        try:
            fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
            rotation, _ = resolve_rotation(cap.get(cv2.CAP_PROP_ORIENTATION_META), "auto")
            clip_key = video_name.removeprefix("video_").removesuffix(".mp4")
            for seconds in times:
                cap.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
                ok, frame = cap.read()
                if not ok:
                    raise RuntimeError(f"Could not decode {video_name} at {seconds}s")
                frame = apply_rotation(frame, rotation)
                height, width = frame.shape[:2]
                scaled_height = round(height * TARGET_WIDTH / width)
                frame = cv2.resize(frame, (TARGET_WIDTH, scaled_height), interpolation=cv2.INTER_AREA)
                stem = f"{clip_key}_t{seconds:02d}s"
                image_path = images_dir / f"{stem}.jpg"
                if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 94]):
                    raise RuntimeError(f"Could not write {image_path}")
                (labels_dir / f"{stem}.txt").write_text("", encoding="utf-8")
                rows.append({
                    "image": f"images/{image_path.name}",
                    "label": f"labels/{stem}.txt",
                    "video": video_name,
                    "seconds": seconds,
                    "upright_width": TARGET_WIDTH,
                    "upright_height": scaled_height,
                })
        finally:
            cap.release()

    with (OUT / "frames.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    thumbs = []
    for row in rows:
        image = cv2.imread(str(OUT / row["image"]))
        thumb = cv2.resize(image, (270, 480), interpolation=cv2.INTER_AREA)
        clip_short = Path(str(row["video"])).stem.split("_")[-1]
        title = f"{clip_short} | {row['seconds']}s"
        cv2.rectangle(thumb, (0, 0), (270, 36), (0, 0, 0), -1)
        cv2.putText(thumb, title, (8, 25), cv2.FONT_HERSHEY_SIMPLEX,
                    0.45, (255, 255, 255), 1, cv2.LINE_AA)
        thumbs.append(thumb)
    cols = 4
    rows_count = (len(thumbs) + cols - 1) // cols
    blank = np.zeros((480, 270, 3), dtype="uint8")
    sheet = cv2.copyMakeBorder(
        cv2.vconcat([
            cv2.hconcat(thumbs[i * cols:(i + 1) * cols] +
                        [blank] * (cols - len(thumbs[i * cols:(i + 1) * cols])))
            for i in range(rows_count)
        ]), 16, 16, 16, 16, cv2.BORDER_CONSTANT, value=(35, 35, 35)
    )
    cv2.imwrite(str(OUT / "contact_sheet.jpg"), sheet, [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(f"Prepared {len(rows)} frames in {OUT}")


if __name__ == "__main__":
    main()

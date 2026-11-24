"""Extract one-at-a-time, full-resolution book annotation candidates locally."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2


ROOT = Path(__file__).resolve().parents[1]
VIDEO_DIR = ROOT.parent / "new_video"
OUT = ROOT / "outputs" / "book_annotation_batch3"

# New times are deliberately between already annotated timestamps. The middle
# clip remains validation-only; the other two clips add training examples.
CANDIDATES = {
    "20261118_183631": (25, 15),
    "20261118_183847": (8, 38, 68),
    "20261118_184102": (10, 40, 50),
}


def main() -> None:
    image_dir = OUT / "images"
    image_dir.mkdir(parents=True, exist_ok=True)
    manifest = []
    index = 1
    for clip, timestamps in CANDIDATES.items():
        video = VIDEO_DIR / f"video_{clip}.mp4"
        if not video.is_file():
            raise FileNotFoundError(video)
        capture = cv2.VideoCapture(str(video))
        if not capture.isOpened():
            raise RuntimeError(f"Cannot open video: {video}")
        fps = capture.get(cv2.CAP_PROP_FPS)
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = frame_count / fps if fps else 0
        for seconds in timestamps:
            if seconds >= duration:
                raise ValueError(f"{clip} has no frame at {seconds}s ({duration:.2f}s total)")
            capture.set(cv2.CAP_PROP_POS_MSEC, seconds * 1000)
            ok, frame = capture.read()
            if not ok:
                raise RuntimeError(f"Cannot read {clip} at {seconds}s")
            # Keep original 4K pixels for accurate manual box placement.
            image_name = f"book_batch3_{index:02d}_{clip}_t{seconds:02d}s.jpg"
            image_path = image_dir / image_name
            if not cv2.imwrite(str(image_path), frame, [cv2.IMWRITE_JPEG_QUALITY, 97]):
                raise RuntimeError(f"Cannot save {image_path}")
            manifest.append({"image": image_name, "clip": clip, "seconds": seconds,
                             "width": frame.shape[1], "height": frame.shape[0]})
            print(f"{index}. {image_name} ({frame.shape[1]}x{frame.shape[0]})")
            index += 1
        capture.release()
    with (OUT / "frames.csv").open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=manifest[0].keys())
        writer.writeheader()
        writer.writerows(manifest)
    print(f"Saved {len(manifest)} candidates to {image_dir}")


if __name__ == "__main__":
    main()

"""Create portrait 1080x1920 derivatives without touching original videos."""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2

from .run import apply_rotation, resolve_rotation


def convert(source: Path, destination: Path, width: int = 1080, height: int = 1920, rotate: str = "auto") -> None:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {source}")
    fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    rotation_code, _ = resolve_rotation(capture.get(cv2.CAP_PROP_ORIENTATION_META), rotate)
    destination.parent.mkdir(parents=True, exist_ok=True)
    writer = None
    frames = 0
    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            frame = apply_rotation(frame, rotation_code)
            if writer is None:
                out_height, out_width = frame.shape[:2]
                writer = cv2.VideoWriter(str(destination), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_width, out_height))
                if not writer.isOpened():
                    raise RuntimeError(f"Could not create {destination}")
            writer.write(cv2.resize(frame, (out_width, out_height), interpolation=cv2.INTER_AREA))
            frames += 1
    finally:
        capture.release()
        if writer is not None:
            writer.release()
    print(f"Wrote {frames} frames: {source.name} -> {destination}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Downsample portrait ShowIntent videos")
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--width", type=int, default=1080)
    parser.add_argument("--height", type=int, default=1920)
    parser.add_argument("--rotate", default="auto", choices=["auto", "cw", "ccw", "180", "none"])
    args = parser.parse_args()
    for source in sorted(args.input_dir.rglob("*.mp4")):
        relative = source.relative_to(args.input_dir)
        convert(source, args.output_dir / relative, args.width, args.height, args.rotate)


if __name__ == "__main__":
    main()


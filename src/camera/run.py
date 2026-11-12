"""CLI entry for the C (camera controller) module.

Reads A's PerceptionPacket JSONL and B's IntentPacket JSONL, writes one
ShotPacket per frame, and optionally renders a cropped preview video.

Examples:
    python -m src.camera.run --perception p.jsonl --intent i.jsonl --out outputs\\shot.jsonl
    python -m src.camera.run --video clip.mp4 --perception p.jsonl --intent i.jsonl --out outputs\\shot.jsonl --overlay-out outputs\\preview.mp4
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .controller import MODES, CameraController


def load_jsonl(path: Path) -> list[dict]:
    with path.open("r", encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]


def run(perception_path: Path, intent_path: Path, out_path: Path, force_wide: bool = False) -> list[dict]:
    perception = load_jsonl(perception_path)
    intent = load_jsonl(intent_path)
    if len(perception) != len(intent):
        raise SystemExit(
            f"Perception ({len(perception)}) and intent ({len(intent)}) streams must have "
            "one packet per frame from the same source; lengths differ."
        )
    controller = CameraController()
    controller.set_manual_override(force_wide)
    shots = []
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as dst:
        for per, it in zip(perception, intent):
            shot = controller.update(it, per)
            if shot["mode"] not in MODES:
                raise ValueError(f"invalid shot mode: {shot['mode']}")
            dst.write(json.dumps(shot, ensure_ascii=False) + "\n")
            shots.append({**shot, "frame_id": per.get("frame_id", 0)})
    print(f"Wrote {len(shots)} ShotPackets -> {out_path}")
    return shots


def main() -> None:
    parser = argparse.ArgumentParser(description="ShowIntent C: perception + intent -> shot")
    parser.add_argument("--perception", required=True, type=Path)
    parser.add_argument("--intent", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="Output ShotPacket JSONL")
    parser.add_argument("--video", type=Path, help="Optional source video for preview rendering")
    parser.add_argument("--overlay-out", type=Path, help="Preview MP4 (cropped camera feed)")
    parser.add_argument("--out-width", type=int, default=1280)
    parser.add_argument("--out-height", type=int, default=720)
    parser.add_argument("--rotate", default="auto", choices=["auto", "cw", "ccw", "180", "none"])
    parser.add_argument("--force-wide", action="store_true", help="Manual override: always keep the wide shot")
    args = parser.parse_args()

    shots = run(args.perception, args.intent, args.out, force_wide=args.force_wide)

    if args.overlay_out:
        if args.video is None or not args.video.exists():
            print(f"Skipped preview rendering: video not provided or missing ({args.video})")
        else:
            from .render import render_preview

            perception = load_jsonl(args.perception)
            objects_by_frame = {p.get("frame_id", i): p.get("objects", []) for i, p in enumerate(perception)}
            count = render_preview(
                args.video,
                shots,
                args.overlay_out,
                objects_by_frame=objects_by_frame,
                out_width=args.out_width,
                out_height=args.out_height,
                rotate=args.rotate,
            )
            print(f"Rendered {count} preview frames -> {args.overlay_out}")


if __name__ == "__main__":
    main()

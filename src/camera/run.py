"""Replay aligned A/B packets, optionally render the original video with audio."""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
from pathlib import Path
from tempfile import TemporaryDirectory

from src.perception.timing import validate_timestamps
from .controller import DEFAULT_MARGIN_RATIO, DEFAULT_MAX_ZOOM, CameraController, visible_box


def load_jsonl(path):
    with Path(path).open(encoding="utf-8") as src:
        return [json.loads(line) for line in src if line.strip()]


def load_pair(perception_path, intent_path):
    perception, intents = load_jsonl(perception_path), load_jsonl(intent_path)
    times = validate_timestamps([p["timestamp_ms"] for p in perception])
    if times != [i.get("timestamp_ms") for i in intents]:
        raise ValueError("A/B row counts or media timestamps differ")
    for i in intents:
        if i.get("state") not in {"NONE", "CANDIDATE", "SHOWING", "ENDING"}:
            raise ValueError(f"Unknown B state at {i['timestamp_ms']} ms")
    return list(zip(perception, intents))


@contextmanager
def new_outputs(*paths):
    """Stage siblings and publish exclusively; clean up partial exports on error."""
    from contextlib import ExitStack
    from os import link
    paths = [Path(p) for p in paths]
    if len({p.resolve() for p in paths}) != len(paths) or any(p.exists() for p in paths):
        raise FileExistsError("Output names must be distinct and new")
    published = []
    with ExitStack() as stack:
        staged = []
        for p in paths:
            p.parent.mkdir(parents=True, exist_ok=True)
            tmp = stack.enter_context(TemporaryDirectory(dir=p.parent))
            staged.append(Path(tmp)/p.name)
        yield staged
        try:
            for tmp, path in zip(staged, paths):
                link(tmp, path)
                published.append(path)
        except BaseException:
            for path in published:
                path.unlink()
            raise


def shot_at(controller, ts, per, intent):
    shot = controller.update(intent, per, timestamp_ms=ts)
    bbox = None
    if intent.get("target_id") is not None:
        obj = next((o for o in per["objects"] if o["track_id"] == intent["target_id"]), {})
        bbox = visible_box(obj.get("bbox_norm"))
    return dict(shot, reason=controller.reason, intent_state=intent["state"],
                intent_target_id=intent.get("target_id"), target_bbox_norm=bbox,
                packet_timestamp_ms=per.get("timestamp_ms"), frame_id=per.get("frame_id"))


def run(perception_path, intent_path, out_path, force_wide=False, *, aspect=1., **options):
    pairs = load_pair(perception_path, intent_path)
    controller = CameraController(aspect=aspect, **options)
    controller.set_manual_override(force_wide)
    shots = [shot_at(controller, i["timestamp_ms"], p, i) for p, i in pairs]
    with new_outputs(out_path) as (log,):
        with log.open("w", encoding="utf-8") as out:
            for shot in shots:
                out.write(json.dumps(shot) + "\n")
    return shots


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--perception", required=True, type=Path)
    parser.add_argument("--intent", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path, help="ShotPacket JSONL")
    parser.add_argument("--video", type=Path)
    parser.add_argument("--overlay-out", "--video-out", type=Path, help="Rendered MP4 with source audio")
    parser.add_argument("--hud", action="store_true", help="Draw target and shot state on output")
    parser.add_argument("--compare", action="store_true", help="Export fixed wide and rendered shot side by side")
    parser.add_argument("--out-width", type=int)
    parser.add_argument("--out-height", type=int)
    parser.add_argument("--rotate", default="auto", choices=["auto", "cw", "ccw", "180", "none"])
    parser.add_argument("--max-zoom", type=float, default=DEFAULT_MAX_ZOOM)
    parser.add_argument("--margin", type=float, default=DEFAULT_MARGIN_RATIO)
    parser.add_argument("--miss-hold-ms", type=int, default=500)
    parser.add_argument("--exit-hold-ms", type=int, default=200)
    parser.add_argument("--force-wide", action="store_true")
    args = parser.parse_args()
    options = dict(max_zoom=args.max_zoom, margin_ratio=args.margin,
                   miss_hold_ms=args.miss_hold_ms, exit_hold_ms=args.exit_hold_ms)
    if bool(args.overlay_out) != bool(args.video):
        parser.error("use --video and --video-out together")
    if args.video:
        from .render import export
        pairs = load_pair(args.perception, args.intent)
        stats = export(args.video, pairs, args.out, args.overlay_out, width=args.out_width,
                       height=args.out_height, rotate=args.rotate, force_wide=args.force_wide,
                       hud=args.hud, compare=args.compare, **options)
        print(json.dumps(stats))
    else:
        if args.out_width or args.out_height:
            parser.error("a source video is required to resolve output dimensions")
        shots = run(args.perception, args.intent, args.out, args.force_wide, **options)
        print(f"Wrote {len(shots)} ShotPackets -> {args.out}")


if __name__ == "__main__":
    main()

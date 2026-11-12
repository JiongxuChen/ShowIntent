"""CLI entry for the B (intent) module.

Reads A's PerceptionPacket JSONL and writes one IntentPacket per frame.

Examples:
    python -m src.intent.run --perception fixtures\\demo_object_show\\perception.jsonl --out outputs\\intent.jsonl
    python -m src.intent.run --perception p.jsonl --out i.jsonl --mode pickup-baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import PickupBaseline
from .state_machine import STATES, ShowIntentStateMachine


def build_engine(mode: str, ms_kwargs: dict | None = None):
    if mode == "showintent":
        return ShowIntentStateMachine(**(ms_kwargs or {}))
    if mode == "pickup-baseline":
        return PickupBaseline()
    raise ValueError(f"unknown mode: {mode} (expected showintent or pickup-baseline)")


def run(perception_path: Path, out_path: Path, mode: str, ms_kwargs: dict | None = None) -> int:
    engine = build_engine(mode, ms_kwargs)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with perception_path.open("r", encoding="utf-8") as src, out_path.open("w", encoding="utf-8") as dst:
        for line in src:
            line = line.strip()
            if not line:
                continue
            packet = json.loads(line)
            intent = engine.update(packet)
            if intent["state"] not in STATES:
                raise ValueError(f"invalid intent state: {intent['state']}")
            dst.write(json.dumps(intent, ensure_ascii=False) + "\n")
            count += 1
    print(f"Wrote {count} IntentPackets ({mode}) -> {out_path}")
    return count


def main() -> None:
    parser = argparse.ArgumentParser(description="ShowIntent B: perception -> intent")
    parser.add_argument("--perception", required=True, type=Path, help="PerceptionPacket JSONL from module A")
    parser.add_argument("--out", required=True, type=Path, help="Output IntentPacket JSONL")
    parser.add_argument("--mode", default="showintent", choices=["showintent", "pickup-baseline"])
    parser.add_argument("--candidate-min-ms", type=float, default=None,
                        help="Millisecond equivalent of candidate_frames (time-based thresholds; default: frame-based)")
    parser.add_argument("--showing-min-ms", type=float, default=None,
                        help="Millisecond equivalent of showing_frames")
    parser.add_argument("--missing-tolerance-ms", type=float, default=None,
                        help="Millisecond equivalent of miss_tolerance")
    parser.add_argument("--ending-min-ms", type=float, default=None,
                        help="Millisecond equivalent of end_frames")
    parser.add_argument("--ending-max-ms", type=float, default=None,
                        help="Millisecond equivalent of ending_max_frames (hard cap on ENDING duration)")
    args = parser.parse_args()
    ms_kwargs = {}
    for name in ("candidate_min_ms", "showing_min_ms", "missing_tolerance_ms", "ending_min_ms", "ending_max_ms"):
        value = getattr(args, name)
        if value is not None:
            ms_kwargs[name] = value
    run(args.perception, args.out, args.mode, ms_kwargs or None)


if __name__ == "__main__":
    main()

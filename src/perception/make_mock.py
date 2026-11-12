"""Generate the shared A-side mock fixture for parallel A/B/C development."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .relations import infer_relations
from .types import Detection, HandDetection, PerceptionPacket


def packet(timestamp_ms: int, frame_id: int, cup_box: list[float], apple_box: list[float], hand_box: list[float] | None) -> PerceptionPacket:
    objects = [
        Detection("obj_cup_1", "cup", cup_box, 0.94),
        Detection("obj_apple_1", "apple", apple_box, 0.92),
        Detection("obj_bottle_1", "bottle", [0.10, 0.48, 0.20, 0.82], 0.88),
    ]
    hands = [] if hand_box is None else [HandDetection("hand_right", hand_box, 0.96)]
    return PerceptionPacket(timestamp_ms, frame_id, objects, hands, infer_relations(hands, objects))


def main() -> None:
    parser = argparse.ArgumentParser(description="Create A/P0 perception mock JSONL")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--interval-ms", type=int, default=200)
    args = parser.parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)
    packets = []
    # 0–2s idle; 2–6s cup rises and remains in the presentation area.
    # 6–7s cup is put down; 7–9s apple is moved sideways only; 9–12s apple is shown.
    for timestamp_ms in range(0, 12_001, args.interval_ms):
        cup = [0.40, 0.62, 0.54, 0.87]
        apple = [0.67, 0.66, 0.79, 0.80]
        hand = None
        if 2_000 <= timestamp_ms < 6_000:
            progress = min(1.0, (timestamp_ms - 2_000) / 1_000)
            cup = [0.40, 0.62 - 0.30 * progress, 0.54, 0.87 - 0.30 * progress]
            hand = [0.37, cup[1] + 0.05, 0.52, cup[3]]
        elif 6_000 <= timestamp_ms < 7_000:
            hand = [0.37, 0.62, 0.52, 0.87]
        elif 7_000 <= timestamp_ms < 9_000:
            progress = (timestamp_ms - 7_000) / 2_000
            apple = [0.67 - 0.36 * progress, 0.66, 0.79 - 0.36 * progress, 0.80]
            hand = [apple[0] - 0.08, 0.63, apple[0] + 0.05, 0.82]
        elif timestamp_ms >= 9_000:
            progress = min(1.0, (timestamp_ms - 9_000) / 1_000)
            apple = [0.49, 0.66 - 0.35 * progress, 0.63, 0.80 - 0.35 * progress]
            hand = [0.44, apple[1] + 0.04, 0.59, apple[3]]
        packets.append(packet(timestamp_ms, len(packets), cup, apple, hand))
    args.out.write_text("".join(json.dumps(value.to_dict(), ensure_ascii=False) + "\n" for value in packets), encoding="utf-8")
    print(f"Wrote {len(packets)} mock PerceptionPackets to {args.out}")


if __name__ == "__main__":
    main()


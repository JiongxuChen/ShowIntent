"""CLI entry for the B (intent) module.

Reads A's PerceptionPacket JSONL and writes one IntentPacket per frame.

Examples:
    python -m src.intent.run --perception fixtures/demo_object_show/perception.jsonl --out outputs/intent.jsonl
    python -m src.intent.run --perception p.jsonl --out i.jsonl --mode pickup-baseline
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .baseline import PickupBaseline
from .types import STATES
from .temporal import IntentConfig, TemporalIntentStateMachine


def build_engine(mode: str, ms_kwargs: dict | None = None):
    if mode == "showintent":
        return TemporalIntentStateMachine(**(ms_kwargs or {}))
    if mode == "pickup-baseline":
        if ms_kwargs:
            raise ValueError("pickup-baseline does not accept intent configuration")
        return PickupBaseline()
    raise ValueError(f"unknown mode: {mode}")


def run(perception_path: Path, out_path: Path, mode: str, ms_kwargs: dict | None = None) -> int:
    if perception_path.resolve() == out_path.resolve():
        raise ValueError("output must not overwrite the input perception file")
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
    parser.add_argument("--config", type=Path, help="JSON IntentConfig; only for showintent")
    parser.add_argument("--candidate-min-ms", type=float, default=None,
                        help="Minimum hand-object association duration in milliseconds")
    parser.add_argument("--showing-min-ms", type=float, default=None,
                        help="Minimum accumulated presentation evidence in milliseconds")
    parser.add_argument("--missing-tolerance-ms", type=float, default=None,
                        help="Target dropout tolerance in milliseconds")
    parser.add_argument("--ending-min-ms", type=float, default=None,
                        help="Minimum ENDING duration in milliseconds")
    parser.add_argument("--ending-max-ms", type=float, default=None,
                        help="Maximum ENDING duration in milliseconds")
    args = parser.parse_args()
    ms_kwargs = {}
    if args.config:
        if args.mode != "showintent":
            parser.error("--config is supported only for showintent")
        try:
            ms_kwargs = json.loads(args.config.read_text(encoding="utf-8"))
            if not isinstance(ms_kwargs, dict):
                raise ValueError("config must contain a JSON object")
            IntentConfig(**ms_kwargs)
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
    for name in ("candidate_min_ms", "showing_min_ms", "missing_tolerance_ms", "ending_min_ms", "ending_max_ms"):
        value = getattr(args, name)
        if value is not None:
            ms_kwargs[name] = value
    try:
        run(args.perception, args.out, args.mode, ms_kwargs or None)
    except (ValueError, TypeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

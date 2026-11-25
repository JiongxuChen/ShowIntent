"""CLI entry for the B (intent) module.

Reads A's PerceptionPacket JSONL and writes one IntentPacket per frame.

Examples:
    python -m src.intent.run --perception tests/fixtures/demo_object_show/perception.jsonl --out outputs/intent.jsonl
    python -m src.intent.run --perception p.jsonl --out i.jsonl --mode pickup-baseline
"""

from __future__ import annotations

import argparse
import json
from contextlib import ExitStack
from pathlib import Path

from .baseline import PickupBaseline
from .types import STATES
from .temporal import IntentConfig, TemporalIntentStateMachine
from .fusion import FusionConfig


def build_engine(mode: str, ms_kwargs: dict | None = None, *, fusion_config: FusionConfig | None = None):
    if mode == "showintent":
        return TemporalIntentStateMachine(fusion_config=fusion_config, **(ms_kwargs or {}))
    if mode == "pickup-baseline":
        if ms_kwargs or fusion_config is not None:
            raise ValueError("pickup-baseline does not accept intent configuration")
        return PickupBaseline()
    raise ValueError(f"unknown mode: {mode}")


def run(perception_path: Path, out_path: Path, mode: str, ms_kwargs: dict | None = None, *,
        speech_path: Path | None = None, fusion_config: FusionConfig | None = None,
        evidence_out: Path | None = None) -> int:
    if perception_path.resolve() == out_path.resolve():
        raise ValueError("output must not overwrite the input perception file")
    bridge = None
    if speech_path is not None:
        if mode != 'showintent':
            raise ValueError('speech events are supported only for showintent')
        from src.speech.bridge import SpeechEvidenceBridge
        bridge = SpeechEvidenceBridge.from_jsonl(speech_path, enabled=True)
        fusion_config = fusion_config or FusionConfig()
    elif fusion_config is not None or evidence_out is not None:
        raise ValueError('fusion config and evidence output require --speech-events')
    outputs = [p for p in (out_path, evidence_out) if p is not None]
    inputs = [p for p in (perception_path, speech_path) if p is not None]
    if len({p.resolve() for p in outputs}) != len(outputs) or any(
            p.resolve() == q.resolve() for p in outputs for q in inputs):
        raise ValueError('outputs must be distinct and must not overwrite inputs')
    if bridge is not None and any(p.exists() for p in outputs):
        raise ValueError('fusion output already exists; choose a new output path')
    engine = build_engine(mode, ms_kwargs, fusion_config=fusion_config)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with ExitStack() as stack:
        src = stack.enter_context(perception_path.open('r', encoding='utf-8'))
        dst = stack.enter_context(out_path.open('x' if bridge else 'w', encoding='utf-8'))
        trace = None
        if evidence_out is not None:
            evidence_out.parent.mkdir(parents=True, exist_ok=True)
            trace = stack.enter_context(evidence_out.open('x', encoding='utf-8'))
        for line in src:
            line = line.strip()
            if not line:
                continue
            packet = json.loads(line)
            observation = bridge.evidence_for(packet) if bridge is not None else None
            intent = engine.update(packet, speech_evidence=observation) if bridge is not None else engine.update(packet)
            if trace is not None:
                trace.write(json.dumps({**engine.fusion.diagnostics, 'intent': intent}, ensure_ascii=False) + '\n')
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
    parser.add_argument('--speech-events', type=Path, help='Final D SpeechEvent JSONL; enables peer evidence fusion')
    parser.add_argument('--fusion-config', type=Path, help='JSON FusionConfig; requires speech events')
    parser.add_argument('--evidence-out', type=Path, help='New per-frame fusion diagnostic JSONL')
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
        fusion = None
        if args.fusion_config:
            fusion = FusionConfig(**json.loads(args.fusion_config.read_text(encoding='utf-8')))
        run(args.perception, args.out, args.mode, ms_kwargs or None,
            speech_path=args.speech_events, fusion_config=fusion, evidence_out=args.evidence_out)
    except (ValueError, TypeError, OSError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()

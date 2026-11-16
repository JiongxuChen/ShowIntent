"""Replay B's synthetic cases and expose failures without hiding them.

No video, GPU, ASR, or third-party dependency is required. Expectations are
examined only AFTER inference and are never supplied to an engine.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

from .baseline import PickupBaseline
from .mock_scenarios import Scenario, make_scenarios, write_scenario
from .types import STATES
from .temporal import IntentConfig, TemporalIntentStateMachine

MODES = ("showintent", "pickup-baseline")


def make_engine(mode: str):
    if mode == "showintent":
        return TemporalIntentStateMachine()
    if mode == "pickup-baseline":
        return PickupBaseline()
    raise ValueError(f"unknown mode: {mode}")


def intent_contract_errors(packets: list[dict], intents: list[dict]) -> list[str]:
    errors = []
    if len(packets) != len(intents):
        return ["input/output packet count differs"]
    known_ids: set[str] = set()
    keys = {"timestamp_ms", "state", "target_id", "score", "reason_codes"}
    for index, (packet, intent) in enumerate(zip(packets, intents)):
        known_ids.update(o["track_id"] for o in packet["objects"])
        if set(intent) != keys:
            errors.append(f"packet {index}: incorrect IntentPacket fields")
        if intent.get("timestamp_ms") != packet["timestamp_ms"]:
            errors.append(f"packet {index}: timestamp mismatch")
        state, target = intent.get("state"), intent.get("target_id")
        if state not in STATES:
            errors.append(f"packet {index}: unknown state")
        if state == "NONE" and target is not None:
            errors.append(f"packet {index}: NONE must release target")
        if state in ("CANDIDATE", "SHOWING", "ENDING") and target not in known_ids:
            errors.append(f"packet {index}: target not observed in current/past input")
        score = intent.get("score")
        if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
            errors.append(f"packet {index}: invalid evidence score")
        reasons = intent.get("reason_codes")
        if not isinstance(reasons, list) or not all(isinstance(r, str) for r in reasons):
            errors.append(f"packet {index}: invalid reason codes")
    return errors


def showing_segments(intents: list[dict]) -> list[dict]:
    """Use half-open timestamp intervals; the last packet has no inferred duration."""
    segments = []
    for index, row in enumerate(intents):
        if row["state"] != "SHOWING":
            continue
        end = intents[index + 1]["timestamp_ms"] if index + 1 < len(intents) else row["timestamp_ms"]
        if segments and segments[-1]["target_id"] == row["target_id"] and segments[-1]["end_ms"] == row["timestamp_ms"]:
            segments[-1]["end_ms"] = end
        else:
            segments.append({"target_id": row["target_id"], "start_ms": row["timestamp_ms"], "end_ms": end})
    return segments


def diagnose_scenario(scenario: Scenario, mode: str) -> tuple[dict, list[dict]]:
    engine = make_engine(mode)
    # Deliberately pass only perception packets, not the scenario or labels.
    intents = [engine.update(packet) for packet in scenario.packets]
    checks = []

    def add(code: str, passed: bool, **detail) -> None:
        checks.append({"code": code, "passed": passed, **detail})

    errors = intent_contract_errors(scenario.packets, intents)
    add("INTENT_CONTRACT", not errors, errors=errors)
    for start, end in scenario.forbidden:
        unexpected = [r for r in intents if start <= r["timestamp_ms"] < end and r["state"] == "SHOWING"]
        add("NO_FALSE_SHOWING", not unexpected, start_ms=start, end_ms=end,
            showing_packet_count=len(unexpected), first_showing_ms=unexpected[0]["timestamp_ms"] if unexpected else None)
    for event in scenario.events:
        hits = [r for r in intents if event["start_ms"] - 1000 <= r["timestamp_ms"] < event["end_ms"]
                and r["state"] == "SHOWING" and r["target_id"] == event["target_id"]]
        first = hits[0]["timestamp_ms"] if hits else None
        add("EXPECTED_TARGET_DETECTED", bool(hits), target_id=event["target_id"],
            event_start_ms=event["start_ms"], first_showing_ms=first)
        add("START_DEADLINE", first is not None and first <= event["latest_start_ms"],
            target_id=event["target_id"], latest_start_ms=event["latest_start_ms"], first_showing_ms=first)
        wrong = [r for r in intents if event["start_ms"] <= r["timestamp_ms"] < event["end_ms"]
                 and r["state"] == "SHOWING" and r["target_id"] != event["target_id"]]
        add("NO_WRONG_TARGET", not wrong, target_id=event["target_id"], wrong_packet_count=len(wrong))
    for hold in scenario.holds:
        before = [r for r in intents if r["timestamp_ms"] < hold["start_ms"]]
        during = [r for r in intents if hold["start_ms"] <= r["timestamp_ms"] < hold["end_ms"]]
        ready = bool(before) and before[-1]["state"] == "SHOWING" and before[-1]["target_id"] == hold["target_id"]
        held = bool(during) and all(r["state"] == "SHOWING" and r["target_id"] == hold["target_id"] for r in during)
        add("BRIEF_LOSS_HOLD", ready and held, **hold, showing_before_loss=ready)
    for release in scenario.releases:
        late = [r for r in intents if release["release_by_ms"] <= r["timestamp_ms"] < release["until_ms"]
                and r["state"] in ("SHOWING", "ENDING") and r["target_id"] == release["target_id"]]
        add("LONG_LOSS_RELEASE", not late, **release, late_packet_count=len(late))
    if scenario.final_none:
        add("FINAL_NONE", intents[-1]["state"] == "NONE", final_state=intents[-1]["state"])
    return {"scenario": scenario.name, "description": scenario.description, "mode": mode,
            "passed": all(c["passed"] for c in checks), "checks": checks,
            "showing_segments": showing_segments(intents), "packet_count": len(intents)}, intents


def run_suite(out_dir: Path, fps_values: list[int], modes: list[str]) -> dict:
    # Validate all arguments before creating outputs.
    if not fps_values or not modes or any(m not in MODES for m in modes):
        raise ValueError("provide at least one FPS and one supported mode")
    groups = [(fps, make_scenarios(fps)) for fps in dict.fromkeys(fps_values)]
    results = []
    for fps, scenarios in groups:
        for scenario in scenarios:
            case_dir = out_dir / f"fps_{fps}" / scenario.name
            write_scenario(scenario, case_dir)
            for mode in dict.fromkeys(modes):
                result, intents = diagnose_scenario(scenario, mode)
                result["fps"] = fps
                results.append(result)
                (case_dir / f"intent_{mode}.jsonl").write_text(
                    "".join(json.dumps(i, ensure_ascii=False) + "\n" for i in intents), encoding="utf-8")
    source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in
                     (Path(__file__), Path(__file__).with_name("mock_scenarios.py"),
                      Path(__file__).with_name("types.py"), Path(__file__).with_name("baseline.py"),
                      Path(__file__).with_name("temporal.py"))}
    report = {"scope": "synthetic_behavior_diagnostics_not_real_video_evaluation",
              "interval_convention": "half-open [start_ms, end_ms)",
              "source_sha256": source_hashes,
              "showintent_parameters": asdict(IntentConfig()),
              "total_cases": len(results), "passed_cases": sum(r["passed"] for r in results),
              "results": results}
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="回放 B 的合成场景并如实报告行为缺口")
    parser.add_argument("--out", type=Path, default=Path("outputs/intent_diagnostics"))
    parser.add_argument("--fps", nargs="+", type=int, default=[5, 30])
    parser.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    parser.add_argument("--strict", action="store_true", help="任一行为检查不通过则退出码为 1")
    args = parser.parse_args()
    try:
        report = run_suite(args.out, args.fps, args.modes)
    except ValueError as exc:
        parser.error(str(exc))
    for result in report["results"]:
        failed = [c["code"] for c in result["checks"] if not c["passed"]]
        status = "PASS" if result["passed"] else "GAP"
        print(f"{status} {result['fps']:2d} FPS {result['mode']:15s} {result['scenario']}: {', '.join(failed) or '-'}")
    print(f"{report['passed_cases']}/{report['total_cases']} cases passed; report: {args.out / 'report.json'}")
    if args.strict and report["passed_cases"] != report["total_cases"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

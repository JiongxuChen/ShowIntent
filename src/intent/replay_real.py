"""Validate provided A packets and replay all B methods without video inference.

No labels, filenames, or scenario descriptions are passed to the engines.
Without reviewed annotations this produces diagnostics, never accuracy scores.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict
from pathlib import Path
from statistics import median

from src.perception.validate_jsonl import validate_packet
from .diagnose import intent_contract_errors, showing_segments
from .run import build_engine
from .temporal import IntentConfig

CLIPS = ("A01", "A02", "A03", "A04", "B01", "B02", "B03", "C01", "C02", "C03", "C04", "D01")


def load_and_check(path: Path) -> tuple[list[dict], dict]:
    rows = []
    last_ts = last_frame = -1
    tracks = defaultdict(set)
    warnings = set()
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        packet = json.loads(line)
        errors = validate_packet(packet, number)
        if errors:
            raise ValueError(f"{path}: " + "; ".join(errors))
        ts, frame = packet["timestamp_ms"], packet["frame_id"]
        if isinstance(ts, bool) or not isinstance(ts, int) or ts <= last_ts:
            raise ValueError(f"{path}:{number}: timestamp_ms must strictly increase")
        if isinstance(frame, bool) or not isinstance(frame, int) or frame < 0 or frame <= last_frame:
            raise ValueError(f"{path}:{number}: frame_id must strictly increase")
        last_ts, last_frame = ts, frame
        ids = {o["track_id"] for o in packet["objects"]}
        if len(ids) != len(packet["objects"]):
            raise ValueError(f"{path}:{number}: duplicate object track IDs")
        hand_ids = {h["hand_id"] for h in packet["hands"]}
        if len(hand_ids) != len(packet["hands"]):
            warnings.add("duplicate hand IDs observed; A uses handedness as hand identity")
        for item in packet["objects"] + packet["hands"]:
            score = item.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{path}:{number}: invalid detection score")
        for obj in packet["objects"]:
            if not isinstance(obj.get("class_name"), str):
                raise ValueError(f"{path}:{number}: class_name must be a string")
            tracks[obj["class_name"]].add(obj["track_id"])
        if not isinstance(packet["relations"], list):
            raise ValueError(f"{path}:{number}: relations must be a list")
        for rel in packet["relations"]:
            if not isinstance(rel, dict) or rel.get("object_id") not in ids or rel.get("hand_id") not in hand_ids:
                raise ValueError(f"{path}:{number}: relation must reference visible hand/object IDs")
            score = rel.get("score")
            if isinstance(score, bool) or not isinstance(score, (int, float)) or not math.isfinite(score) or not 0 <= score <= 1:
                raise ValueError(f"{path}:{number}: invalid relation score")
        rows.append(packet)
    if not rows:
        raise ValueError(f"{path}: no perception packets")
    gaps = [b["timestamp_ms"]-a["timestamp_ms"] for a, b in zip(rows, rows[1:])]
    quality = {
        "packet_count": len(rows), "first_timestamp_ms": rows[0]["timestamp_ms"],
        "last_timestamp_ms": rows[-1]["timestamp_ms"],
        "median_packet_gap_ms": median(gaps) if gaps else None,
        "max_packet_gap_ms": max(gaps) if gaps else None,
        "frames_with_hands": sum(bool(p["hands"]) for p in rows),
        "frames_with_relations": sum(bool(p["relations"]) for p in rows),
        "frames_without_objects": sum(not p["objects"] for p in rows),
        "track_count_by_class": {k: len(v) for k, v in tracks.items()},
        "warnings": sorted(warnings),
    }
    return rows, quality


def replay_clip(source: Path, destination: Path, config: dict) -> dict:
    rows, quality = load_and_check(source)
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    marker = source.with_name(source.stem.replace("perception_", "completion_", 1)+".json")
    provenance = None
    if marker.exists():
        provenance = json.loads(marker.read_text(encoding="utf-8"))
        if provenance.get("perception_sha256") != digest:
            raise ValueError(f"input differs from completion record: {source}")
    names = [f"intent_{mode}.jsonl" for mode in ("showintent", "pickup-baseline")]+["report.json"]
    if any((destination / name).exists() for name in names):
        raise FileExistsError(f"refusing to overwrite replay artifacts: {destination}")
    results, outputs = {}, {}
    for mode in ("showintent", "pickup-baseline"):
        engine = build_engine(mode, config if mode == "showintent" else None)
        started = time.perf_counter()
        intents = [engine.update(p) for p in rows]
        elapsed = (time.perf_counter()-started)*1000
        errors = intent_contract_errors(rows, intents)
        if errors:
            raise ValueError("; ".join(errors))
        results[mode] = {"states": dict(Counter(i["state"] for i in intents)),
                         "showing_segments": showing_segments(intents),
                         "final_state": intents[-1]["state"], "engine_wall_ms": round(elapsed, 3),
                         "timing_scope": "B_update_only_excludes_A_IO_render_and_confirmation_wait"}
        outputs[mode] = intents
    report = {
        "source": str(source.resolve()), "perception_sha256": digest,
        "input_provenance": provenance,
        "source_kind": provenance.get("source_kind", "unverified") if provenance else "provided_perception_video_origin_unverified",
        "quality": quality, "showintent_config": asdict(IntentConfig(**config)), "methods": results,
        "b_source_sha256": {name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                            for name in ("temporal.py", "types.py", "baseline.py", "replay_real.py")},
        "evaluation_status": "diagnostics_only_no_reviewed_event_annotations_no_accuracy_or_event_latency",
    }
    destination.mkdir(parents=True, exist_ok=True)
    for mode, intents in outputs.items():
        (destination / f"intent_{mode}.jsonl").write_text(
            "".join(json.dumps(i, ensure_ascii=False)+"\n" for i in intents), encoding="utf-8")
    (destination / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="校验真实 A 输出并回放 B；无标注时只生成诊断")
    parser.add_argument("--perception-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", choices=CLIPS, default=list(CLIPS))
    parser.add_argument("--config", type=Path, default=Path("configs/presentation_intent.json"))
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    IntentConfig(**config)
    sources = {clip: args.perception_dir/f"perception_{clip}.jsonl" for clip in dict.fromkeys(args.clips)}
    if any(not p.is_file() for p in sources.values()):
        parser.error("missing perception files: " + ", ".join(str(p) for p in sources.values() if not p.is_file()))
    for clip, path in sources.items():
        result = replay_clip(path, args.out/clip, config)
        print(f"{clip}: {result['quality']['packet_count']} packets; "
              f"SHOWING segments={len(result['methods']['showintent']['showing_segments'])}; "
              f"source_kind={result['source_kind']}")


if __name__ == "__main__":
    main()

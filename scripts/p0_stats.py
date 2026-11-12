"""P0 evaluation statistics for the ShowIntent A/B/C pipeline.

Usage:
    python -m scripts.p0_stats --outputs-dir outputs --perception-dir ../outputs

Reads intent_<clip>.jsonl / intent_<clip>_baseline.jsonl / shot_<clip>.jsonl
plus perception_<clip>.jsonl and prints a per-clip summary. Also writes
outputs/p0_stats.json with the machine-readable version.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

CLIPS = ["A01", "A02", "A03", "A04", "B01", "B02", "B03", "C01", "C02", "C03", "C04", "D01"]
GROUND_TRUTH = {  # from the received-media brief; True = a real presentation occurs
    "A01": True, "A02": True, "A03": True, "A04": True,
    "B01": False, "B02": False, "B03": False,
    "C01": True, "C02": True, "C03": True, "C04": True,
    "D01": True,
}


def load_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def intent_segments(rows: list[dict]) -> list[dict]:
    """Contiguous (state, target_id) segments from an IntentPacket stream."""
    segments: list[dict] = []
    current: dict | None = None
    for r in rows:
        key = (r["state"], r["target_id"])
        if current is None or (current["state"], current["target_id"]) != key:
            if current:
                segments.append(current)
            current = {"state": r["state"], "target_id": r["target_id"],
                       "t_start_ms": r["timestamp_ms"], "t_end_ms": r["timestamp_ms"]}
        else:
            current["t_end_ms"] = r["timestamp_ms"]
    if current:
        segments.append(current)
    return segments


def showing_runs(segments: list[dict]) -> list[dict]:
    return [s for s in segments if s["state"] == "SHOWING"]


def passed_candidate(rows: list[dict], run_start_ms: int) -> bool:
    """Was there a CANDIDATE packet shortly before this SHOWING run started?"""
    for r in rows:
        if r["state"] == "CANDIDATE" and 0 <= run_start_ms - r["timestamp_ms"] <= 3000:
            return True
    return False


def clip_stats(clip: str, out_dir: Path, perception_dir: Path) -> dict:
    intent = load_jsonl(out_dir / f"intent_{clip}.jsonl")
    baseline = load_jsonl(out_dir / f"intent_{clip}_baseline.jsonl")
    shots = load_jsonl(out_dir / f"shot_{clip}.jsonl")
    perception = load_jsonl(perception_dir / f"perception_{clip}.jsonl")
    if not intent:
        return {"clip": clip, "status": "missing"}

    segs = intent_segments(intent)
    runs = showing_runs(segs)
    fps_est = 1000.0 / max(1, intent[1]["timestamp_ms"] - intent[0]["timestamp_ms"]) if len(intent) > 1 else 30.0

    # candidate visibility before each SHOWING run
    runs_detail = []
    for run in runs:
        runs_detail.append({
            "start_s": round(run["t_start_ms"] / 1000, 1),
            "end_s": round(run["t_end_ms"] / 1000, 1),
            "duration_s": round((run["t_end_ms"] - run["t_start_ms"]) / 1000, 1),
            "target_id": run["target_id"],
            "candidate_before": passed_candidate(intent, run["t_start_ms"]),
        })

    # camera stats
    cam = {"TRANSITION_IN": 0, "TRANSITION_OUT": 0, "closeup_frames": 0, "ends_wide": None, "modes": {}}
    if shots:
        modes = Counter(s["mode"] for s in shots)
        cam["modes"] = dict(modes)
        for i, s in enumerate(shots):
            prev = shots[i - 1]["mode"] if i else None
            if s["mode"] == "TRANSITION_IN" and prev != "TRANSITION_IN":
                cam["TRANSITION_IN"] += 1
            if s["mode"] == "TRANSITION_OUT" and prev != "TRANSITION_OUT":
                cam["TRANSITION_OUT"] += 1
        cam["closeup_frames"] = modes.get("CLOSEUP", 0) + modes.get("TRANSITION_IN", 0) + modes.get("TRANSITION_OUT", 0)
        cam["ends_wide"] = shots[-1]["mode"] == "WIDE"
        cam["closeup_seconds"] = round(cam["closeup_frames"] / fps_est, 1)

    # baseline comparison
    b_runs = showing_runs(intent_segments(baseline)) if baseline else []

    # track fragmentation in perception
    tracks_by_class: dict[str, set] = defaultdict(set)
    for r in perception:
        for o in r["objects"]:
            tracks_by_class[o["class_name"]].add(o["track_id"])

    # target changes during SHOWING (state stays SHOWING, target flips)
    target_flips = 0
    last = None
    for r in intent:
        if r["state"] == "SHOWING":
            if last is not None and r["target_id"] != last:
                target_flips += 1
            last = r["target_id"]
        else:
            last = None if r["state"] in ("NONE",) else last

    return {
        "clip": clip,
        "gt_presentation": GROUND_TRUTH[clip],
        "frames": len(intent),
        "fps_est": round(fps_est, 1),
        "states_seen": sorted({r["state"] for r in intent}),
        "showing_run_count": len(runs),
        "showing_runs": runs_detail,
        "total_showing_s": round(sum(x["duration_s"] for x in runs_detail), 1),
        "max_showing_s": round(max((x["duration_s"] for x in runs_detail), default=0.0), 1),
        "ending_seen": "ENDING" in {r["state"] for r in intent},
        "returns_to_none": intent[-1]["state"] == "NONE",
        "camera": cam,
        "baseline_showing_runs": len(b_runs),
        "baseline_total_showing_s": round(sum((s["t_end_ms"] - s["t_start_ms"]) / 1000 for s in b_runs), 1),
        "tracks_by_class": {k: sorted(v) for k, v in sorted(tracks_by_class.items())},
        "track_count_total": sum(len(v) for v in tracks_by_class.values()),
        "showing_target_flips": target_flips,
        "false_trigger": len(runs) > 0 and not GROUND_TRUTH[clip],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--outputs-dir", type=Path, default=Path("outputs"))
    parser.add_argument("--perception-dir", type=Path, default=Path("../outputs"))
    args = parser.parse_args()

    results = [clip_stats(c, args.outputs_dir, args.perception_dir) for c in CLIPS]
    (args.outputs_dir / "p0_stats.json").write_text(json.dumps(results, indent=1), encoding="utf-8")

    for r in results:
        if r.get("status") == "missing":
            print(f"{r['clip']}: MISSING intent output")
            continue
        cam = r["camera"]
        runs = "; ".join(f"{x['start_s']}-{x['end_s']}s({x['target_id']},cand={x['candidate_before']})" for x in r["showing_runs"]) or "-"
        print(f"{r['clip']}: SHOWING runs={r['showing_run_count']} [{runs}] "
              f"END={r['ending_seen']} backNONE={r['returns_to_none']} "
              f"| cam TIN={cam.get('TRANSITION_IN')} TOUT={cam.get('TRANSITION_OUT')} closeup={cam.get('closeup_seconds')}s endsWide={cam.get('ends_wide')} "
              f"| baseline runs={r['baseline_showing_runs']} ({r['baseline_total_showing_s']}s) "
              f"| tracks={r['track_count_total']} flips={r['showing_target_flips']}")
    print("\nwrote outputs/p0_stats.json")


if __name__ == "__main__":
    main()

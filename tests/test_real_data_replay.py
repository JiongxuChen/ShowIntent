"""Validate the real-data boundary using explicitly synthetic test packets."""

import copy
import hashlib
import json
import subprocess
import sys

import pytest

from src.intent.mock_scenarios import make_scenarios
from src.intent.replay_real import load_and_check, replay_clip


def write_packets(path, packets):
    path.write_text("".join(json.dumps(p)+"\n" for p in packets), encoding="utf-8")
    return path


def test_methods_replay_without_mutating_input_or_inventing_accuracy(tmp_path):
    scenario = next(s for s in make_scenarios(5) if s.name == "lift_hold_putdown")
    source = write_packets(tmp_path/"perception_A01.jsonl", scenario.packets)
    before = source.read_bytes()
    out = tmp_path/"replay"
    report = replay_clip(source, out, {})
    assert source.read_bytes() == before
    assert report["source_kind"] == "provided_perception_video_origin_unverified"
    assert "no_accuracy" in report["evaluation_status"]
    assert report["quality"]["packet_count"] == len(scenario.packets)
    for mode in ("showintent", "pickup-baseline"):
        rows = [json.loads(line) for line in (out/f"intent_{mode}.jsonl").read_text().splitlines()]
        assert len(rows) == len(scenario.packets)
        assert [r["timestamp_ms"] for r in rows] == [p["timestamp_ms"] for p in scenario.packets]
        assert report["methods"][mode]["showing_segments"]
    with pytest.raises(FileExistsError):
        replay_clip(source, out, {})


@pytest.mark.parametrize("issue", ["duplicate_time", "backward_time", "duplicate_track", "bad_relation", "bad_score", "bad_frame"])
def test_invalid_perception_is_rejected_before_outputs_are_written(tmp_path, issue):
    packets = copy.deepcopy(make_scenarios(5)[5].packets[:4])
    if issue == "duplicate_time":
        packets[1]["timestamp_ms"] = packets[0]["timestamp_ms"]
    elif issue == "backward_time":
        packets[2]["timestamp_ms"] = 1
    elif issue == "duplicate_track":
        packets[1]["objects"].append(packets[1]["objects"][0])
    elif issue == "bad_frame":
        packets[1]["frame_id"] = True
    else:
        packets[1]["hands"] = [{"hand_id": "h", "bbox_norm": [.1,.1,.2,.2], "score": 1.}]
        packets[1]["relations"] = [{"hand_id": "h", "object_id": "track_cup" if issue == "bad_score" else "never_seen",
                                    "type": "near_or_moving_together", "score": float("nan") if issue == "bad_score" else .8}]
    source = write_packets(tmp_path/"perception_A01.jsonl", packets)
    out = tmp_path/"replay"
    with pytest.raises(ValueError):
        replay_clip(source, out, {})
    assert not out.exists()


def test_empty_input_is_not_a_successful_video(tmp_path):
    path = tmp_path/"empty.jsonl"
    path.write_text("\n")
    with pytest.raises(ValueError, match="no perception"):
        load_and_check(path)


def test_gaps_in_frame_ids_are_allowed_and_duplicate_hands_are_diagnosed(tmp_path):
    packets = copy.deepcopy(make_scenarios(5)[1].packets[:4])
    for p in packets:
        p["frame_id"] *= 6
        p["hands"].append(copy.deepcopy(p["hands"][0]))
    _, quality = load_and_check(write_packets(tmp_path/"p.jsonl", packets))
    assert quality["median_packet_gap_ms"] == 200
    assert quality["warnings"]


def test_partial_smoke_provenance_is_retained_and_hash_checked(tmp_path):
    source = write_packets(tmp_path/"perception_A01.jsonl", make_scenarios(5)[0].packets[:3])
    marker = tmp_path/"completion_A01.json"
    marker.write_text(json.dumps({"source_kind": "synthetic_partial_unit_test", "perception_sha256": "wrong"}))
    with pytest.raises(ValueError, match="completion"):
        replay_clip(source, tmp_path/"out", {})
    marker.write_text(json.dumps({"source_kind": "synthetic_partial_unit_test",
                                  "perception_sha256": hashlib.sha256(source.read_bytes()).hexdigest()}))
    result = replay_clip(source, tmp_path/"out", {})
    assert result["source_kind"] == "synthetic_partial_unit_test"



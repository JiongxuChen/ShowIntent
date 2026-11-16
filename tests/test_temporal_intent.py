"""Acceptance tests for the presentation intent engine."""

import copy
import json
import math
import subprocess
import sys
from dataclasses import asdict
from pathlib import Path

import pytest

from src.intent.diagnose import diagnose_scenario
from src.intent.mock_scenarios import make_scenarios
from src.intent.run import build_engine
from src.intent.temporal import IntentConfig, TemporalIntentStateMachine


def packet(t, y=.80, x=.4, related=False, tid="a", cls="cup"):
    return {"timestamp_ms": t, "frame_id": t * 3 if isinstance(t, int) else 0,
            "objects": [{"track_id": tid, "class_name": cls,
                         "bbox_norm": [x-.05, y-.05, x+.05, y+.05], "score": .95}],
            "hands": [{"hand_id": "h", "bbox_norm": [.3, .3, .5, .5], "score": .9}] if related else [],
            "relations": [{"hand_id": "h", "object_id": tid, "type": "near_or_moving_together", "score": .85}] if related else []}


def replay(packets, **config):
    engine = TemporalIntentStateMachine(**config)
    return [engine.update(p) for p in packets]


def established_show():
    return [packet(t, y=.8 if t < 1000 else .45, related=t >= 1000) for t in range(0, 2501, 100)]


def test_public_entrypoint_uses_temporal_engine():
    assert isinstance(build_engine("showintent"), TemporalIntentStateMachine)



@pytest.mark.parametrize("fps", [5, 15, 30])
def test_synthetic_presentation_scenarios_meet_acceptance_criteria(fps):
    for scenario in make_scenarios(fps):
        result, _ = diagnose_scenario(scenario, "showintent")
        assert result["passed"], (scenario.name, result["checks"])


def test_confirmation_times_agree_within_sampling_resolution():
    starts = []
    for fps in (5, 15, 30):
        scenario = next(s for s in make_scenarios(fps) if s.name == "lift_hold_putdown")
        _, rows = diagnose_scenario(scenario, "showintent")
        starts.append(next(r["timestamp_ms"] for r in rows if r["state"] == "SHOWING"))
    assert max(starts) - min(starts) <= 250


def test_decimated_frame_numbers_do_not_change_timing():
    scenario = next(s for s in make_scenarios(5) if s.name == "two_targets")
    sparse = copy.deepcopy(scenario.packets)
    for row in sparse:
        row["frame_id"] *= 17
    assert replay(sparse) == replay(scenario.packets)


def test_irregular_packet_intervals_use_elapsed_time():
    times = [0, 100, 300, 600, 900, 1000, 1100, 1275, 1400, 1530, 1700, 1900, 2200, 2600, 3000, 3200, 3500]
    rows = replay([packet(t, y=.45 if 1000 <= t < 3000 else .8, related=1000 <= t < 3000) for t in times])
    first = next(r["timestamp_ms"] for r in rows if r["state"] == "SHOWING")
    assert 1600 <= first <= 1900
    assert rows[-1]["state"] == "NONE"


def test_long_prior_contact_does_not_count_as_showing_dwell():
    rows = replay([packet(t, y=.45 if 2000 <= t < 2400 else .8, related=True) for t in range(0, 4001, 100)])
    assert not any(r["state"] == "SHOWING" for r in rows)


@pytest.mark.parametrize("interruption", ["relation", "object"])
def test_interrupted_evidence_does_not_accumulate_across_gaps(interruption):
    packets = [packet(t, y=.45 if 1000 <= t < 2000 else .8, related=1000 <= t < 2000) for t in range(0, 3001, 100)]
    interrupted = next(p for p in packets if p["timestamp_ms"] == 1400)
    interrupted["relations"] = []
    if interruption == "object":
        interrupted["objects"] = []
    assert not any(r["state"] == "SHOWING" for r in replay(packets))


def test_initial_hold_can_trigger_after_a_later_observed_lift():
    packets = []
    for t in range(0, 6001, 100):
        y = .6 if t < 2000 else .6 - .3 * min(1, (t-2000)/1000) if t < 5000 else .8
        packets.append(packet(t, y=y, related=t < 5000))
    rows = replay(packets)
    assert not any(r["state"] == "SHOWING" for r in rows if r["timestamp_ms"] < 3000)
    assert any(r["state"] == "SHOWING" for r in rows if 3000 <= r["timestamp_ms"] < 5000)
    assert rows[-1]["state"] == "NONE"


def test_small_bounded_motion_is_allowed_during_hold():
    packets = [packet(t, y=.8 if t < 1000 else .45,
                      x=.4 if t < 1000 else .4 + .004 * math.sin(t/1000 * math.pi),
                      related=t >= 1000) for t in range(0, 3501, 100)]
    assert replay(packets)[-1]["state"] == "SHOWING"


def test_window_span_rejects_drift_below_the_instantaneous_speed_limit():
    packets = [packet(t, y=.8 if t < 1000 else .45,
                      x=.3 if t < 1000 else .3 + .06 * (t-1000)/1000,
                      related=t >= 1000) for t in range(0, 6001, 100)]
    assert not any(r["state"] == "SHOWING" for r in replay(packets))


def test_relation_loss_with_visible_raised_object_eventually_exits():
    packets = established_show() + [packet(t, y=.45) for t in range(2600, 4501, 100)]
    rows = replay(packets)
    assert any(r["state"] == "SHOWING" for r in rows)
    assert rows[-1]["state"] == "NONE"


def test_another_qualified_object_does_not_replace_an_active_show():
    packets = []
    for t in range(0, 5001, 100):
        p = packet(t, y=.8 if t < 1000 else .45, related=t >= 1000)
        other = packet(t, y=.8 if t < 2500 else .4, x=.7, related=t >= 2500, tid="b", cls="apple")
        p["objects"] += other["objects"]
        p["relations"] += other["relations"]
        packets.append(p)
    rows = replay(packets)
    showing = [r for r in rows if r["state"] == "SHOWING"]
    assert showing and all(r["target_id"] == "a" for r in showing)


@pytest.mark.parametrize("cls,x,allowed", [("cup", .42, True), ("apple", .42, False), ("cup", .85, False)])
def test_relink_requires_same_class_nearby_continuation(cls, x, allowed):
    rows = replay(established_show() + [packet(t, y=.46, x=x, related=True, tid="b", cls=cls) for t in range(2600, 4501, 100)])
    relinks = [r for r in rows if "TARGET_RELINKED" in r["reason_codes"]]
    assert bool(relinks) == allowed
    if allowed:
        assert all(r["target_id"] == "b" for r in relinks)
    else:
        assert not any(r["target_id"] == "b" and r["state"] == "SHOWING" for r in rows)


def test_large_packet_gap_cannot_reuse_an_old_confirmation():
    packets = established_show() + [packet(t, y=.45, related=True) for t in range(5000, 6501, 100)]
    rows = replay(packets)
    after_gap = [r for r in rows if r["timestamp_ms"] >= 5000]
    assert "EVIDENCE_GAP" in after_gap[0]["reason_codes"]
    assert not any(r["state"] == "SHOWING" for r in after_gap)


@pytest.mark.parametrize("bad_ts", [100, 99, -1, True, 1.5, None])
def test_bad_timestamps_are_rejected(bad_ts):
    engine = TemporalIntentStateMachine()
    engine.update(packet(100))
    with pytest.raises(ValueError):
        engine.update(packet(bad_ts))


@pytest.mark.parametrize("parameters", [
    {"showing_min_ms": 0}, {"missing_tolerance_ms": -1}, {"raise_margin": float("nan")},
    {"stable_speed_tolerance": float("inf")}, {"putdown_margin": .2},
    {"ending_max_ms": 10}, {"history_window_ms": 100}, {"track_ttl_ms": 1},
])
def test_invalid_configurations_fail_before_inference(parameters):
    with pytest.raises(ValueError):
        TemporalIntentStateMachine(**parameters)


def test_checked_in_configuration_matches_active_defaults():
    config = json.loads(Path("configs/presentation_intent.json").read_text())
    assert config == asdict(IntentConfig())


def test_configured_cli_preserves_input_and_runs_current_engine(tmp_path):
    source = tmp_path / "perception.jsonl"
    source.write_text("".join(json.dumps(p)+"\n" for p in established_show()))
    before = source.read_bytes()
    output = tmp_path / "intent.jsonl"
    result = subprocess.run([sys.executable, "-B", "-m", "src.intent.run", "--perception", str(source),
                             "--out", str(output), "--config", "configs/presentation_intent.json"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert source.read_bytes() == before
    rows = [json.loads(line) for line in output.read_text().splitlines()]
    assert rows[-1]["state"] == "SHOWING"
    assert "LIFT_MOTION_OBSERVED" in next(r for r in rows if r["state"] == "SHOWING")["reason_codes"]


def test_cli_rejects_unknown_config_and_input_overwrite(tmp_path):
    source = tmp_path / "p.jsonl"
    source.write_text(json.dumps(packet(0))+"\n")
    before = source.read_bytes()
    config = tmp_path / "bad.json"
    config.write_text('{"not_a_setting": 2}')
    for extra in ([], ["--config", str(config)]):
        result = subprocess.run([sys.executable, "-B", "-m", "src.intent.run", "--perception", str(source),
                                 "--out", str(source), *extra], capture_output=True, text=True)
        assert result.returncode == 2
        assert source.read_bytes() == before

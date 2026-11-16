"""Public packet contract, causality, fixtures and CLI behavior."""

from __future__ import annotations

import copy
import json
import subprocess
import sys
from dataclasses import replace

import pytest

from src.intent.diagnose import (
    MODES, diagnose_scenario, intent_contract_errors, make_engine, run_suite,
    showing_segments,
)
from src.intent.mock_scenarios import make_scenarios
from src.perception.validate_jsonl import validate_packet


@pytest.mark.parametrize("fps", [5, 15, 30])
def test_mock_inputs_follow_a_contract_and_reference_visible_objects(fps):
    for scenario in make_scenarios(fps):
        previous = -1
        for index, packet in enumerate(scenario.packets):
            assert not validate_packet(packet, index + 1)
            assert packet["timestamp_ms"] > previous
            assert packet["frame_id"] == index
            assert set(packet) == {"timestamp_ms", "frame_id", "objects", "hands", "relations"}
            previous = packet["timestamp_ms"]
            objects = {o["track_id"] for o in packet["objects"]}
            hands = {h["hand_id"] for h in packet["hands"]}
            assert len(objects) == len(packet["objects"])
            assert len(hands) == len(packet["hands"])
            for relation in packet["relations"]:
                assert relation["object_id"] in objects
                assert relation["hand_id"] in hands


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("fps", [5, 30])
def test_every_scenario_emits_one_aligned_intent_per_input(mode, fps):
    for scenario in make_scenarios(fps):
        _, intents = diagnose_scenario(scenario, mode)
        assert not intent_contract_errors(scenario.packets, intents)


@pytest.mark.parametrize("mode", MODES)
def test_processing_does_not_mutate_a_packets(mode):
    packets = make_scenarios(5)[6].packets
    original = copy.deepcopy(packets)
    engine = make_engine(mode)
    for packet in packets:
        engine.update(packet)
    assert packets == original


@pytest.mark.parametrize("mode", MODES)
def test_future_packets_do_not_change_prefix_decisions(mode):
    packets = make_scenarios(5)[6].packets
    full_engine = make_engine(mode)
    full = [full_engine.update(p) for p in packets]
    for count in (10, 22, 50):
        prefix_engine = make_engine(mode)
        assert [prefix_engine.update(p) for p in packets[:count]] == full[:count]


@pytest.mark.parametrize("mode", MODES)
def test_changing_oracles_does_not_change_inference(mode):
    scenario = make_scenarios(5)[6]
    _, original = diagnose_scenario(scenario, mode)
    oracle_changed = replace(scenario, events=[], forbidden=[], holds=[], releases=[], final_none=False)
    _, changed = diagnose_scenario(oracle_changed, mode)
    assert changed == original


@pytest.mark.parametrize("mode", MODES)
def test_track_names_are_not_predeclared_targets(mode):
    packets = make_scenarios(5)[6].packets
    renamed = copy.deepcopy(packets)
    mapping = {"track_cup": "id_72", "track_apple": "id_9", "track_bottle": "id_300"}
    for packet in renamed:
        for obj in packet["objects"]:
            obj["track_id"] = mapping[obj["track_id"]]
        for rel in packet["relations"]:
            rel["object_id"] = mapping[rel["object_id"]]
    first, second = make_engine(mode), make_engine(mode)
    original = [first.update(p) for p in packets]
    changed = [second.update(p) for p in renamed]
    for before, after in zip(original, changed):
        assert after["state"] == before["state"]
        assert after["target_id"] == mapping.get(before["target_id"])


@pytest.mark.parametrize("mode", ["showintent"])
def test_basic_show_has_four_states_and_releases_target(mode):
    scenario = next(s for s in make_scenarios(30) if s.name == "lift_hold_putdown")
    result, intents = diagnose_scenario(scenario, mode)
    assert result["passed"], result["checks"]
    states = [i["state"] for i in intents]
    assert states.index("CANDIDATE") < states.index("SHOWING") < states.index("ENDING")
    assert intents[-1]["state"] == "NONE"
    assert intents[-1]["target_id"] is None


@pytest.mark.parametrize("fps", [5, 30])
def test_ms_mode_holds_short_loss_and_releases_long_loss(fps):
    scenarios = make_scenarios(fps)
    for name in ("brief_occlusion", "long_loss"):
        scenario = next(s for s in scenarios if s.name == name)
        result, _ = diagnose_scenario(scenario, "showintent")
        assert result["passed"], result["checks"]


def test_segments_use_actual_timestamps_and_split_on_target_change():
    rows = [
        {"timestamp_ms": 0, "state": "NONE", "target_id": None},
        {"timestamp_ms": 100, "state": "SHOWING", "target_id": "a"},
        {"timestamp_ms": 350, "state": "SHOWING", "target_id": "a"},
        {"timestamp_ms": 420, "state": "SHOWING", "target_id": "b"},
        {"timestamp_ms": 800, "state": "NONE", "target_id": None},
    ]
    assert showing_segments(rows) == [
        {"target_id": "a", "start_ms": 100, "end_ms": 420},
        {"target_id": "b", "start_ms": 420, "end_ms": 800},
    ]


@pytest.mark.parametrize("bad_fps", [0, -1, 121, 2.5, True])
def test_invalid_sampling_rate_is_rejected(bad_fps):
    with pytest.raises(ValueError):
        make_scenarios(bad_fps)


def test_diagnostics_files_keep_expectations_separate_and_allow_replay(tmp_path):
    report = run_suite(tmp_path, [5], ["showintent"])
    assert report["total_cases"] == 9
    assert json.loads((tmp_path / "report.json").read_text())["results"] == report["results"]
    case = tmp_path / "fps_5/lift_hold_putdown"
    assert (case / "expectations.json").exists()
    destination = tmp_path / "replayed.jsonl"
    run = subprocess.run([sys.executable, "-B", "-m", "src.intent.run", "--perception",
                          str(case / "perception.jsonl"), "--out", str(destination)], capture_output=True, text=True)
    assert run.returncode == 0, run.stderr
    rows = [json.loads(line) for line in destination.read_text().splitlines()]
    inputs = [json.loads(line) for line in (case / "perception.jsonl").read_text().splitlines()]
    assert not intent_contract_errors(inputs, rows)
    assert "SHOWING" in {r["state"] for r in rows}
    assert rows[-1]["state"] == "NONE"


def test_strict_diagnostics_reports_unresolved_behavior_with_nonzero_exit(tmp_path):
    result = subprocess.run([sys.executable, "-B", "-m", "src.intent.diagnose", "--out", str(tmp_path),
                             "--fps", "5", "--modes", "pickup-baseline", "--strict"], capture_output=True, text=True)
    assert result.returncode == 1, result.stderr
    report = json.loads((tmp_path / "report.json").read_text())
    assert report["passed_cases"] < report["total_cases"]
    assert "GAP" in result.stdout


# Presentation behavior acceptance checks.
def test_acceptance_raised_transport_does_not_show():
    scenario = next(s for s in make_scenarios(30) if s.name == "raised_transport")
    result, _ = diagnose_scenario(scenario, "showintent")
    assert result["passed"], result["checks"]


def test_acceptance_initial_hold_stays_uncertain():
    scenario = next(s for s in make_scenarios(30) if s.name == "initial_hold")
    result, _ = diagnose_scenario(scenario, "showintent")
    assert result["passed"], result["checks"]


def test_acceptance_second_target_is_confirmed_before_deadline():
    scenario = next(s for s in make_scenarios(5) if s.name == "two_targets")
    result, _ = diagnose_scenario(scenario, "showintent")
    assert result["passed"], result["checks"]

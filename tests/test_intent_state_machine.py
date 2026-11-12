"""Tests for the B module ShowIntent state machine (rule-based, pure vision)."""

from src.intent.run import build_engine
from src.intent.state_machine import ShowIntentStateMachine


def packet(ts, frame, objects=(), relations=(), hands=()):
    return {
        "timestamp_ms": ts,
        "frame_id": frame,
        "objects": list(objects),
        "hands": list(hands),
        "relations": list(relations),
    }


def obj(track_id, cls="cup", center=(0.47, 0.74), size=(0.14, 0.25), score=0.9):
    cx, cy = center
    w, h = size
    return {
        "track_id": track_id,
        "class_name": cls,
        "bbox_norm": [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2],
        "score": score,
    }


def rel(hand_id, object_id, score=0.85):
    return {"hand_id": hand_id, "object_id": object_id, "type": "near_or_moving_together", "score": score}


def hand(hand_id="hand_right"):
    return {"hand_id": hand_id, "bbox_norm": [0.4, 0.5, 0.5, 0.6], "score": 1.0}


def feed(state_machine, packets):
    return [state_machine.update(p) for p in packets]


def test_no_relation_stays_none():
    sm = ShowIntentStateMachine()
    out = feed(sm, [packet(i * 200, i, objects=[obj("obj_1")]) for i in range(20)])
    assert all(o["state"] == "NONE" for o in out)


def test_sustained_relation_enters_candidate_then_showing():
    sm = ShowIntentStateMachine()
    # table frames first so the baseline is learned, then raised + held
    packets = [packet(i * 200, i, objects=[obj("obj_1")]) for i in range(6)]
    packets += [
        packet(6 * 200 + j * 200, 6 + j, objects=[obj("obj_1", center=(0.47, 0.45))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(15)
    ]
    out = feed(sm, packets)
    states = [o["state"] for o in out]
    assert states[0] == "NONE"
    assert "CANDIDATE" in states
    assert "SHOWING" in states
    showing = out[states.index("SHOWING")]
    assert showing["target_id"] == "obj_1"
    assert set(showing["reason_codes"]) == {"HAND_OBJECT_ASSOCIATED", "RAISED_IN_FRAME", "TEMPORAL_HOLD"}
    # CANDIDATE must come before SHOWING
    assert states.index("CANDIDATE") < states.index("SHOWING")


def test_carry_at_table_level_never_shows():
    sm = ShowIntentStateMachine()
    # object slides across the table (x moves, y stays at table height), hand relation present
    packets = [packet(i * 200, i, objects=[obj("obj_1")]) for i in range(6)]
    packets += [
        packet(6 * 200 + j * 200, 6 + j, objects=[obj("obj_1", center=(0.2 + 0.02 * j, 0.74))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(25)
    ]
    out = feed(sm, packets)
    assert all(o["state"] in ("NONE", "CANDIDATE") for o in out)


def test_putdown_returns_to_none():
    sm = ShowIntentStateMachine()
    packets = [packet(i * 200, i, objects=[obj("obj_1")]) for i in range(6)]
    packets += [
        packet(6 * 200 + j * 200, 6 + j, objects=[obj("obj_1", center=(0.47, 0.45))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(15)
    ]
    packets += [
        packet((21 + j) * 200, 21 + j, objects=[obj("obj_1")])
        for j in range(12)
    ]
    out = feed(sm, packets)
    states = [o["state"] for o in out]
    assert "SHOWING" in states
    assert "ENDING" in states
    # hand gone and object back on the table: must settle in NONE and stay there
    assert states[-1] == "NONE"
    assert states[-3] == "NONE"


def test_brief_dropout_keeps_showing():
    sm = ShowIntentStateMachine()
    packets = [packet(i * 200, i, objects=[obj("obj_1")]) for i in range(6)]
    packets += [
        packet(6 * 200 + j * 200, 6 + j, objects=[obj("obj_1", center=(0.47, 0.45))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(15)
    ]
    last_frame = len(packets)
    # 5 frames where the target vanishes entirely (occlusion)
    packets += [packet((last_frame + j) * 200, last_frame + j) for j in range(5)]
    packets += [
        packet((last_frame + 5 + j) * 200, last_frame + 5 + j, objects=[obj("obj_1", center=(0.47, 0.45))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(3)
    ]
    out = feed(sm, packets)
    states = [o["state"] for o in out]
    assert "SHOWING" in states
    # never left SHOWING during the dropout
    dropout_states = states[len(states) - 8 : len(states) - 3]
    assert all(s == "SHOWING" for s in dropout_states)


def test_showing_target_id_is_locked():
    sm = ShowIntentStateMachine()
    packets = [packet(i * 200, i, objects=[obj("obj_1"), obj("obj_2", cls="apple", center=(0.7, 0.74), size=(0.12, 0.14))]) for i in range(6)]
    packets += [
        packet(6 * 200 + j * 200, 6 + j, objects=[obj("obj_1", center=(0.47, 0.45)), obj("obj_2", cls="apple", center=(0.7, 0.74), size=(0.12, 0.14))], relations=[rel("hand_right", "obj_1"), rel("hand_right", "obj_2")], hands=[hand()])
        for j in range(20)
    ]
    out = feed(sm, packets)
    showing = [o for o in out if o["state"] == "SHOWING"]
    assert showing, "expected SHOWING"
    assert all(o["target_id"] == "obj_1" for o in showing)


def test_schema_of_intent_packet():
    sm = ShowIntentStateMachine()
    out = sm.update(packet(0, 0, objects=[obj("obj_1")]))
    assert set(out.keys()) == {"timestamp_ms", "state", "target_id", "score", "reason_codes"}
    assert out["state"] in ("NONE", "CANDIDATE", "SHOWING", "ENDING")
    assert isinstance(out["score"], float)
    assert isinstance(out["reason_codes"], list)


def test_baseline_pickup_triggers_but_schema_matches():
    engine = build_engine("pickup-baseline")
    packets = [packet(i * 200, i, objects=[obj("obj_1")]) for i in range(6)]
    packets += [
        packet(6 * 200 + j * 200, 6 + j, objects=[obj("obj_1", center=(0.47, 0.5))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(6)
    ]
    out = feed(engine, packets)
    assert "SHOWING" in [o["state"] for o in out]
    assert all(set(o.keys()) == {"timestamp_ms", "state", "target_id", "score", "reason_codes"} for o in out)
    # and the transport-only negative does NOT lift -> stays NONE for the baseline too
    engine2 = build_engine("pickup-baseline")
    carry = [packet(i * 200, i, objects=[obj("obj_1")]) for i in range(6)]
    carry += [
        packet(6 * 200 + j * 200, 6 + j, objects=[obj("obj_1", center=(0.2 + 0.02 * j, 0.74))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(10)
    ]
    out2 = feed(engine2, carry)
    assert all(o["state"] == "NONE" for o in out2)


def test_ms_thresholds_equivalent_to_frame_thresholds_at_30fps():
    # 30 fps: 33 ms per frame. ms thresholds chosen to round up to the same
    # frame counts as the legacy defaults (3 / 8), so behaviour must match.
    ms_kwargs = {"candidate_min_ms": 99, "showing_min_ms": 264}
    sm_frame = ShowIntentStateMachine()
    sm_ms = ShowIntentStateMachine(**ms_kwargs)
    packets = [packet(i * 33, i, objects=[obj("obj_1")]) for i in range(6)]
    packets += [
        packet(6 * 33 + j * 33, 6 + j, objects=[obj("obj_1", center=(0.47, 0.45))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(15)
    ]
    states_frame = [o["state"] for o in feed(sm_frame, packets)]
    states_ms = [o["state"] for o in feed(sm_ms, packets)]
    assert states_frame == states_ms
    assert "SHOWING" in states_ms


def test_ms_thresholds_scale_with_packet_rate():
    # Same wall-clock ms thresholds at 5 fps (200 ms/frame): SHOWING must be
    # reached after ~showing_min_ms of evidence instead of 8 frames = 1.6 s.
    ms_kwargs = {"candidate_min_ms": 100, "showing_min_ms": 267}
    sm_ms = ShowIntentStateMachine(**ms_kwargs)
    packets = [packet(i * 200, i, objects=[obj("obj_1")]) for i in range(6)]
    packets += [
        packet(6 * 200 + j * 200, 6 + j, objects=[obj("obj_1", center=(0.47, 0.45))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(8)
    ]
    out = feed(sm_ms, packets)
    states = [o["state"] for o in out]
    assert "SHOWING" in states
    # legacy frame-based machine needs 8 candidate frames, ms-based needs 2
    assert states.index("SHOWING") < 12


def _carry_candidate(target="obj_table", frames=110):
    """A table fruit whose resting hand keeps a permanent relation: it can
    enter CANDIDATE but never qualifies (cy stays at table level)."""
    return [
        packet(i * 33, i, objects=[obj(target, cls="apple", center=(0.3, 0.78))], relations=[rel("hand_right", target)], hands=[hand()])
        for i in range(frames)
    ]


def test_table_object_cannot_lock_candidate_forever():
    sm = ShowIntentStateMachine()
    out = feed(sm, _carry_candidate(frames=110))
    assert all(o["state"] == "CANDIDATE" for o in out[3:])  # locked as candidate
    # now a real presented object appears: raised + stable
    packets = []
    for j in range(14):
        ts = (110 + j) * 33
        packets.append(packet(ts, 110 + j, objects=[obj("obj_table", cls="apple", center=(0.3, 0.78)), obj("obj_real", cls="orange", center=(0.55, 0.55))], relations=[rel("hand_right", "obj_table"), rel("hand_right", "obj_real")], hands=[hand()]))
    out2 = feed(sm, packets)
    states = [o["state"] for o in out2]
    assert "SHOWING" in states
    showing = out2[states.index("SHOWING")]
    assert showing["target_id"] == "obj_real"
    # the real target must have passed through CANDIDATE first (replacement)
    cand_targets = [o["target_id"] for o in out2 if o["state"] == "CANDIDATE"]
    assert "obj_real" in cand_targets


def test_candidate_replacement_emits_candidate_not_showing():
    sm = ShowIntentStateMachine()
    feed(sm, _carry_candidate(frames=110))
    # exactly the switch frame: replacement must be CANDIDATE, not SHOWING
    p = packet(110 * 33, 110, objects=[obj("obj_table", cls="apple", center=(0.3, 0.78)), obj("obj_real", cls="orange", center=(0.55, 0.55))], relations=[rel("hand_right", "obj_table"), rel("hand_right", "obj_real")], hands=[hand()])
    it = sm.update(p)
    if it["target_id"] == "obj_real":
        assert it["state"] == "CANDIDATE"
        assert "CANDIDATE_REPLACED" in it["reason_codes"]


def test_showing_target_not_replaced_by_new_raised_object():
    sm = ShowIntentStateMachine()
    packets = [packet(i * 33, i, objects=[obj("obj_1")]) for i in range(6)]
    packets += [
        packet(6 * 33 + j * 33, 6 + j, objects=[obj("obj_1", center=(0.47, 0.45))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(12)
    ]
    out = feed(sm, packets)
    assert out[-1]["state"] == "SHOWING" and out[-1]["target_id"] == "obj_1"
    # a second raised object appears during SHOWING: target must stay locked
    for j in range(8):
        ts = (18 + j) * 33
        it = sm.update(packet(ts, 18 + j, objects=[obj("obj_1", center=(0.47, 0.45)), obj("obj_2", cls="apple", center=(0.6, 0.5))], relations=[rel("hand_right", "obj_1"), rel("hand_right", "obj_2")], hands=[hand()]))
        assert it["target_id"] == "obj_1"


def test_ending_does_not_deadlock_on_new_track():
    sm = ShowIntentStateMachine()
    packets = [packet(i * 33, i, objects=[obj("obj_1", cls="orange")]) for i in range(6)]
    packets += [
        packet(6 * 33 + j * 33, 6 + j, objects=[obj("obj_1", cls="orange", center=(0.47, 0.45))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(12)
    ]
    out = feed(sm, packets)
    assert out[-1]["state"] == "SHOWING"
    # track dies; a brand-new unrelated track appears far away
    packets2 = []
    for j in range(70):
        ts = (18 + j) * 33
        if j < 3:
            packets2.append(packet(ts, 18 + j, objects=[obj("obj_1", cls="orange", center=(0.47, 0.45))], relations=[rel("hand_right", "obj_1")], hands=[hand()]))
        else:
            packets2.append(packet(ts, 18 + j, objects=[obj("obj_9", cls="apple", center=(0.9, 0.2))], relations=[rel("hand_right", "obj_9")], hands=[hand()]))
    out2 = feed(sm, packets2)
    states = [o["state"] for o in out2]
    assert "ENDING" in states
    # the deadlock is gone: ENDING terminates into NONE (a later re-candidate
    # on the new track is legal behaviour, so the final state is free)
    assert "NONE" in states


def test_ending_relink_nearby_same_class_only():
    sm = ShowIntentStateMachine()
    packets = [packet(i * 33, i, objects=[obj("obj_1", cls="orange")]) for i in range(6)]
    packets += [
        packet(6 * 33 + j * 33, 6 + j, objects=[obj("obj_1", cls="orange", center=(0.47, 0.45))], relations=[rel("hand_right", "obj_1")], hands=[hand()])
        for j in range(12)
    ]
    feed(sm, packets)
    # track obj_1 disappears (16 frames of miss tolerance -> ENDING), then a
    # nearby same-class raised track appears while still inside the relink window
    for j in range(16):
        ts = (18 + j) * 33
        it = sm.update(packet(ts, 18 + j, objects=[], relations=[], hands=[]))
    assert it["state"] == "ENDING"
    p = packet(34 * 33, 34, objects=[obj("obj_3", cls="orange", center=(0.49, 0.46))], relations=[rel("hand_right", "obj_3")], hands=[hand()])
    it = sm.update(p)
    assert it["state"] == "SHOWING" and it["target_id"] == "obj_3"
    assert "TARGET_RELINKED" in it["reason_codes"]

    # negative: a wrong-class nearby track must NOT resume; a far same-class
    # track must NOT resume either
    for cls, cx, tag in [("apple", 0.49, "wrong-class"), ("orange", 0.95, "too-far")]:
        sm2 = ShowIntentStateMachine()
        feed(sm2, [packet(i * 33, i, objects=[obj("obj_1", cls="orange")]) for i in range(6)])
        feed(sm2, [packet(6 * 33 + j * 33, 6 + j, objects=[obj("obj_1", cls="orange", center=(0.47, 0.45))], relations=[rel("hand_right", "obj_1")], hands=[hand()]) for j in range(12)])
        for j in range(16):
            it2 = sm2.update(packet((18 + j) * 33, 18 + j, objects=[], relations=[], hands=[]))
        assert it2["state"] == "ENDING"
        p2 = packet(34 * 33, 34, objects=[obj("obj_5", cls=cls, center=(cx, 0.46))], relations=[rel("hand_right", "obj_5")], hands=[hand()])
        it2 = sm2.update(p2)
        assert not (it2["state"] == "SHOWING" and it2["target_id"] == "obj_5"), tag


def _run_real_clip(name):
    """Run the state machine over a real perception JSONL (skipped when the
    local evaluation outputs are absent, e.g. in CI)."""
    import json
    from pathlib import Path
    path = Path(__file__).resolve().parent.parent.parent / "outputs" / f"perception_{name}.jsonl"
    if not path.exists():
        import pytest
        pytest.skip(f"real perception output not available: {path}")
    sm = ShowIntentStateMachine()
    with path.open(encoding="utf-8") as f:
        return [sm.update(json.loads(line)) for line in f if line.strip()]


def test_regression_A02_ends_and_returns_to_none():
    out = _run_real_clip("A02")
    assert any(o["state"] == "SHOWING" for o in out)
    assert out[-1]["state"] in ("NONE", "CANDIDATE")  # was stuck in ENDING before D2 fix


def test_regression_C01_C03_detect_showing():
    for name in ("C01", "C03"):
        out = _run_real_clip(name)
        assert any(o["state"] == "SHOWING" for o in out), f"{name}: SHOWING missed"


def test_regression_B02_B03_no_false_trigger():
    for name in ("B02", "B03"):
        out = _run_real_clip(name)
        assert not any(o["state"] == "SHOWING" for o in out), f"{name}: false trigger"

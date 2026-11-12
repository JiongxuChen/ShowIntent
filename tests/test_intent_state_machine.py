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

from src.intent.run import build_engine

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



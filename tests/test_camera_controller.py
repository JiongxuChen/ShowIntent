"""Tests for the C module camera controller."""

from src.camera.controller import FULL_CROP, CameraController, compute_crop


def intent(ts, state, target_id=None):
    return {"timestamp_ms": ts, "state": state, "target_id": target_id, "score": 0.9, "reason_codes": []}


def perception(objs=(), frame=0, ts=0):
    return {"timestamp_ms": ts, "frame_id": frame, "objects": list(objs), "hands": [], "relations": []}


def obj(track_id, bbox):
    return {"track_id": track_id, "class_name": "cup", "bbox_norm": list(bbox), "score": 0.9}


def bbox_at(center, size=(0.14, 0.25)):
    cx, cy = center
    w, h = size
    return [cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2]


def test_wide_when_no_intent():
    c = CameraController()
    shot = c.update(intent(0, "NONE"), perception())
    assert shot["mode"] == "WIDE"
    assert shot["crop_norm"] == [0.0, 0.0, 1.0, 1.0]
    assert shot["zoom"] == 1.0


def test_candidate_stays_wide():
    c = CameraController()
    for _ in range(10):
        shot = c.update(intent(0, "CANDIDATE", "obj_1"), perception([obj("obj_1", bbox_at((0.47, 0.5)))]))
    assert shot["mode"] == "WIDE"


def test_showing_zooms_in_and_crop_contains_target():
    c = CameraController()
    bbox = bbox_at((0.47, 0.45))
    for t in range(0, 2000, 33):
        shot = c.update(intent(t, "SHOWING", "obj_1"), perception([obj("obj_1", bbox)], ts=t))
    assert shot["mode"] == "CLOSEUP"
    x1, y1, x2, y2 = shot["crop_norm"]
    assert x1 < bbox[0] and bbox[2] < x2  # target horizontally inside crop
    assert y1 < bbox[1] and bbox[3] < y2  # target vertically inside crop
    assert shot["zoom"] > 1.3
    assert shot["target_id"] == "obj_1"


def test_crop_stays_in_bounds_and_aspect():
    c = CameraController()
    bbox = [0.9, 0.85, 1.2, 1.1]  # deliberately out of frame
    for t in range(0, 2000, 33):
        shot = c.update(intent(t, "SHOWING", "obj_1"), perception([obj("obj_1", bbox)], ts=t))
    x1, y1, x2, y2 = shot["crop_norm"]
    assert 0.0 <= x1 < x2 <= 1.0
    assert 0.0 <= y1 < y2 <= 1.0
    aspect = (x2 - x1) / (y2 - y1)
    # With matching source/output shape the normalized aspect is one.
    assert abs(aspect - 1.0) < 0.001


def test_target_id_stable_and_brief_dropout_holds_crop():
    c = CameraController()
    bbox = bbox_at((0.47, 0.45))
    for t in range(0, 2000, 33):
        shot = c.update(intent(t, "SHOWING", "obj_1"), perception([obj("obj_1", bbox)], ts=t))
    stable_crop = list(shot["crop_norm"])
    # target vanishes for a few frames: crop must hold, mode stays CLOSEUP
    for t in range(2013, 2160, 33):
        shot = c.update(intent(t, "SHOWING", "obj_1"), perception([], ts=t))
    assert shot["mode"] == "CLOSEUP"
    assert shot["crop_norm"] == stable_crop
    assert shot["target_id"] == "obj_1"


def test_timeout_after_long_dropout_returns_to_wide():
    c = CameraController()
    bbox = bbox_at((0.47, 0.45))
    for t in range(0, 2000, 33):
        c.update(intent(t, "SHOWING", "obj_1"), perception([obj("obj_1", bbox)], ts=t))
    for t in range(2013, 3600, 33):
        shot = c.update(intent(t, "SHOWING", "obj_1"), perception([], ts=t))
    assert shot["mode"] == "WIDE"
    assert all(abs(v - f) < 0.01 for v, f in zip(shot["crop_norm"], [0.0, 0.0, 1.0, 1.0]))


def test_ending_transitions_out():
    c = CameraController()
    bbox = bbox_at((0.47, 0.45))
    for t in range(0, 2000, 33):
        c.update(intent(t, "SHOWING", "obj_1"), perception([obj("obj_1", bbox)], ts=t))
    for t in range(2013, 3600, 33):
        shot = c.update(intent(t, "ENDING", "obj_1"), perception([obj("obj_1", bbox)], ts=t))
    assert shot["mode"] == "WIDE"


def test_manual_override_forces_wide():
    c = CameraController()
    bbox = bbox_at((0.47, 0.45))
    for t in range(0, 2000, 33):
        c.update(intent(t, "SHOWING", "obj_1"), perception([obj("obj_1", bbox)], ts=t))
    c.set_manual_override(True)
    shot = c.update(intent(2013, "SHOWING", "obj_1"), perception([obj("obj_1", bbox)], ts=2013))
    assert shot["mode"] == "WIDE"
    assert shot["manual_override"] is True


def test_smoothing_no_jump_cut():
    c = CameraController()
    bbox = bbox_at((0.47, 0.45))
    shots = [c.update(intent(t, "SHOWING", "obj_1"), perception([obj("obj_1", bbox)], ts=t))
             for t in range(0, 330, 33)]
    moves = [max(abs(a - b) for a, b in zip(s0["crop_norm"], s1["crop_norm"])) for s0, s1 in zip(shots, shots[1:])]
    assert all(m < 0.25 for m in moves)  # gradual zoom, never a hard cut


def test_shotpacket_schema():
    c = CameraController()
    shot = c.update(intent(0, "SHOWING", "obj_1"), perception([obj("obj_1", bbox_at((0.47, 0.45)))]))
    assert set(shot.keys()) == {"timestamp_ms", "mode", "target_id", "crop_norm", "zoom", "manual_override"}
    assert shot["mode"] in ("WIDE", "TRANSITION_IN", "CLOSEUP", "TRANSITION_OUT")
    assert len(shot["crop_norm"]) == 4


def test_compute_crop_unit_edges():
    crop = compute_crop([0.0, 0.0, 0.2, 0.2], 16 / 9)
    x1, y1, x2, y2 = crop
    assert 0.0 <= x1 < x2 <= 1.0
    assert 0.0 <= y1 < y2 <= 1.0
    assert abs((x2 - x1) / (y2 - y1) - 16 / 9) < 0.05

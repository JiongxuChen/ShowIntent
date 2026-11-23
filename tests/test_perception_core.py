from src.perception.relations import infer_relations
from src.perception.tracker import SimpleObjectTracker
from src.perception.types import Detection, HandDetection
from src.perception.validate_jsonl import validate_packet
from src.perception.run import open_capture_without_auto_rotation


def test_tracker_keeps_id_for_nearby_same_class_detection():
    tracker = SimpleObjectTracker()
    first = tracker.update([("cup", (0.4, 0.4, 0.6, 0.7), 0.9)])
    second = tracker.update([("cup", (0.42, 0.4, 0.62, 0.7), 0.9)])
    assert first[0][0] == second[0][0]


def test_relation_is_candidate_only_when_hand_is_near_object():
    hand = HandDetection("hand_right", [0.4, 0.4, 0.5, 0.6], 1.0)
    cup = Detection("obj_1", "cup", [0.48, 0.45, 0.65, 0.7], 0.9)
    relations = infer_relations([hand], [cup])
    assert relations[0].type == "near_or_moving_together"
    assert relations[0].object_id == "obj_1"


def test_validator_accepts_a_minimum_packet():
    value = {"timestamp_ms": 0, "frame_id": 0, "objects": [], "hands": [], "relations": []}
    assert validate_packet(value, 1) == []


def test_validator_rejects_dangling_relation():
    value = {"timestamp_ms": 0, "frame_id": 0, "objects": [], "hands": [],
             "relations": [{"hand_id": "h", "object_id": "o", "type": "near", "score": 0.8}]}
    assert any("missing object or hand" in error for error in validate_packet(value, 1))


def test_validator_rejects_invalid_object_confidence():
    value = {"timestamp_ms": 0, "frame_id": 0, "objects": [
        {"track_id": "o", "class_name": "apple", "bbox_norm": [0.1, 0.1, 0.2, 0.2], "score": float("nan")}],
        "hands": [], "relations": []}
    assert any("score" in error for error in validate_packet(value, 1))


def test_capture_disables_opencv_automatic_video_rotation(monkeypatch):
    import cv2

    class FakeCapture:
        def __init__(self):
            self.auto_rotation = 1.0
            self.released = False

        def isOpened(self):
            return True

        def get(self, prop):
            assert prop == cv2.CAP_PROP_ORIENTATION_AUTO
            return self.auto_rotation

        def set(self, prop, value):
            assert prop == cv2.CAP_PROP_ORIENTATION_AUTO
            self.auto_rotation = value
            return True

        def release(self):
            self.released = True

    fake = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: fake)

    capture = open_capture_without_auto_rotation("phone.mp4")

    assert capture is fake
    assert fake.auto_rotation == 0


def test_capture_fails_safely_if_auto_rotation_cannot_be_disabled(monkeypatch):
    import cv2
    import pytest

    class FakeCapture:
        released = False

        def isOpened(self):
            return True

        def get(self, prop):
            return 1.0

        def set(self, prop, value):
            return False

        def release(self):
            self.released = True

    fake = FakeCapture()
    monkeypatch.setattr(cv2, "VideoCapture", lambda source: fake)

    with pytest.raises(RuntimeError, match="auto-rotation"):
        open_capture_without_auto_rotation("phone.mp4")
    assert fake.released

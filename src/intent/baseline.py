"""Pick-up Trigger baseline: fires whenever an object clearly lifts off the table.

Deliberately naive, for false-trigger comparison against ShowIntent:
- no hand-object temporal hold required,
- no showing dwell required,
- single displacement threshold against the per-track table baseline.

It emits the exact same IntentPacket schema as the main method, so C and the
evaluation harness can consume either one interchangeably.
"""

from __future__ import annotations

from statistics import median

from .state_machine import (
    REASON_BACK_ON_TABLE,
    REASON_HAND_ASSOCIATED,
    REASON_INSUFFICIENT,
    REASON_RAISED,
    REASON_TARGET_LOST,
    IntentPacket,
    TableBaseline,
    TrackHistory,
    bbox_center,
)

STATE_NONE = "NONE"
STATE_SHOWING = "SHOWING"
STATE_ENDING = "ENDING"


class PickupBaseline:
    def __init__(
        self,
        pickup_margin: float = 0.10,
        putdown_margin: float = 0.03,
        end_frames: int = 2,
        baseline_min_samples: int = 3,
        miss_tolerance: int = 15,
    ) -> None:
        self.pickup_margin = pickup_margin
        self.putdown_margin = putdown_margin
        self.end_frames = end_frames
        self.miss_tolerance = miss_tolerance
        self._baseline = TableBaseline(min_samples=baseline_min_samples)
        self._histories: dict[str, TrackHistory] = {}
        self.state = STATE_NONE
        self.target_id: str | None = None
        self._miss = 0
        self._end_streak = 0

    def _baseline_y(self, track_id: str) -> float | None:
        return self._baseline.baseline(track_id)

    def _packet(self, ts: int, state: str, score: float, reasons: list[str]) -> dict:
        return IntentPacket(ts, state, self.target_id, score, reasons).to_dict()

    def update(self, packet: dict) -> dict:
        frame_id = int(packet.get("frame_id", 0))
        ts = int(packet.get("timestamp_ms", frame_id * 1000 // 30))
        objects = {o["track_id"]: o for o in packet.get("objects", [])}
        related_ids = {r.get("object_id") for r in packet.get("relations", [])}

        for track_id, obj in objects.items():
            cx, cy = bbox_center(obj["bbox_norm"])
            self._histories.setdefault(track_id, TrackHistory()).add(frame_id, cx, cy)
            self._baseline.observe(track_id, cy, track_id in related_ids)

        if self.state == STATE_NONE:
            for track_id, obj in objects.items():
                cx, cy = bbox_center(obj["bbox_norm"])
                base = self._baseline_y(track_id)
                lifted = cy < (base - self.pickup_margin if base is not None else 0.55)
                if lifted:
                    self.state = STATE_SHOWING
                    self.target_id = track_id
                    self._miss = 0
                    return self._packet(ts, STATE_SHOWING, float(obj.get("score", 0.0)), [REASON_RAISED])
            return self._packet(ts, STATE_NONE, 0.0, [REASON_INSUFFICIENT])

        if self.state == STATE_SHOWING:
            obj = objects.get(self.target_id)
            if obj is None:
                self._miss += 1
                if self._miss > self.miss_tolerance:
                    self.state = STATE_ENDING
                    self._end_streak = 0
                    return self._packet(ts, STATE_ENDING, 0.0, [REASON_TARGET_LOST])
                return self._packet(ts, STATE_SHOWING, 0.0, [REASON_RAISED])
            self._miss = 0
            _, cy = bbox_center(obj["bbox_norm"])
            base = self._baseline_y(self.target_id)
            on_table = cy >= (base - self.putdown_margin if base is not None else 0.55)
            if on_table:
                self.state = STATE_ENDING
                self._end_streak = 0
                return self._packet(ts, STATE_ENDING, 0.0, [REASON_BACK_ON_TABLE])
            score = float(obj.get("score", 0.0))
            return self._packet(ts, STATE_SHOWING, score, [REASON_RAISED])

        # ENDING
        obj = objects.get(self.target_id)
        if obj is not None:
            _, cy = bbox_center(obj["bbox_norm"])
            base = self._baseline_y(self.target_id)
            lifted = cy < (base - self.putdown_margin if base is not None else 0.55)
            if lifted:
                self.state = STATE_SHOWING
                self._miss = 0
                return self._packet(ts, STATE_SHOWING, float(obj.get("score", 0.0)), [REASON_HAND_ASSOCIATED, REASON_RAISED])
            reliable = False
        else:
            reliable = False
        self._end_streak = 0 if reliable else self._end_streak + 1
        if self._end_streak >= self.end_frames:
            self.state = STATE_NONE
            self.target_id = None
            return self._packet(ts, STATE_NONE, 0.0, [REASON_INSUFFICIENT])
        return self._packet(ts, STATE_ENDING, 0.0, [REASON_TARGET_LOST if obj is None else REASON_BACK_ON_TABLE])

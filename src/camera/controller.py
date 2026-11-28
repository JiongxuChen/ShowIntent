"""Causal camera control in normalized coordinates of the upright source."""
from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

MODES = ("WIDE", "TRANSITION_IN", "CLOSEUP", "TRANSITION_OUT")
FULL_CROP = (0.0, 0.0, 1.0, 1.0)
DEFAULT_MARGIN_RATIO = .13
DEFAULT_MAX_ZOOM = 3.0


@dataclass(frozen=True)
class ShotPacket:
    timestamp_ms: int
    mode: str
    target_id: str | None
    crop_norm: list[float]
    zoom: float
    manual_override: bool

    def to_dict(self):
        return dict(timestamp_ms=self.timestamp_ms, mode=self.mode, target_id=self.target_id,
                    crop_norm=list(self.crop_norm), zoom=self.zoom, manual_override=self.manual_override)


def visible_box(box):
    if not isinstance(box, (list, tuple)) or len(box) != 4:
        return None
    if not all(isinstance(v, (float, int)) and isfinite(v) for v in box):
        return None
    x1, y1, x2, y2 = [min(1., max(0., v)) for v in box]
    return [x1, y1, x2, y2] if x2 > x1 and y2 > y1 else None


def centered_crop(cx, cy, width, height):
    left = min(max(0., cx - width / 2), 1. - width)
    top = min(max(0., cy - height / 2), 1. - height)
    return [left, top, min(1., left + width), min(1., top + height)]


def compute_crop(bbox, aspect, margin_ratio=DEFAULT_MARGIN_RATIO, min_size=0.12):
    """aspect = output pixel aspect / upright source pixel aspect."""
    x1, y1, x2, y2 = bbox
    wide_w = min(1., aspect)
    w = min(wide_w, max(min_size, (1 + 2 * margin_ratio) * max(x2-x1, (y2-y1)*aspect)))
    return centered_crop((x1+x2)/2, (y1+y2)/2, w, min(1., w/aspect))


def crop_distance(a, b):
    return max(abs(x-y) for x, y in zip(a, b))


def keep_visible(crop, old, box, budget):
    """Move the smoothed window just enough to include the box, within its speed limit."""
    for axis in (0, 1):
        end = axis+2
        span = crop[end]-crop[axis]
        left_limit = max(0., box[end]-span)
        right_limit = min(1.-span, box[axis])
        if left_limit > right_limit:
            continue
        wanted = min(max(crop[axis], left_limit), right_limit)-crop[axis]
        min_shift = max(old[j]-budget-crop[j] for j in (axis, end))
        max_shift = min(old[j]+budget-crop[j] for j in (axis, end))
        shift = min(max(wanted, min_shift), max_shift)
        crop[axis] = max(0., crop[axis]+shift)
        crop[end] = min(1., crop[end]+shift)


def related_context(box, target_id, packet):
    """Include the nearest part of an associated hand, capped by object size."""
    hands = {h.get("hand_id"): h.get("bbox_norm") for h in packet.get("hands", [])}
    context = list(box)
    w, h = box[2]-box[0], box[3]-box[1]
    for r in packet.get("relations", []):
        if r.get("object_id") != target_id:
            continue
        hand = visible_box(hands.get(r.get("hand_id")))
        if hand is None:
            continue
        # A distant hand must not pull the frame away from the confirmed object.
        if hand[0] > box[2]+w/2 or hand[2] < box[0]-w/2 or hand[1] > box[3]+h/2 or hand[3] < box[1]-h/2:
            continue
        for i, extent in enumerate((w, h)):
            context[i] = min(context[i], max(hand[i], box[i] - .35*extent))
            context[i+2] = max(context[i+2], min(hand[i+2], box[i+2] + .35*extent))
    return context


class CameraController:
    def __init__(self, aspect=1., margin_ratio=DEFAULT_MARGIN_RATIO, smoothing=.20, converge_eps=.008,
                 miss_hold_frames=15, min_crop_size=.12, *, max_zoom=DEFAULT_MAX_ZOOM,
                 miss_hold_ms=None, exit_hold_ms=200, max_edge_speed=1.6, exit_edge_speed=None):
        if not all(isfinite(x) and x > 0 for x in (aspect, max_edge_speed, min_crop_size, converge_eps)):
            raise ValueError("aspect, speed, crop size and tolerance must be positive")
        if not (isfinite(max_zoom) and max_zoom >= 1 and 0 < smoothing <= 1
                and isfinite(margin_ratio) and margin_ratio >= 0):
            raise ValueError("invalid zoom, smoothing or margin")
        self.hold_ms = miss_hold_ms if miss_hold_ms is not None else miss_hold_frames * 1000/30
        if not all(isfinite(x) and x >= 0 for x in (self.hold_ms, exit_hold_ms)):
            raise ValueError("invalid hold duration")
        self.aspect = aspect
        self.wide_w = min(1., aspect)
        self.wide = centered_crop(.5, .5, self.wide_w, min(1., 1/aspect))
        self.min_width = min(self.wide_w, max(self.wide_w/max_zoom, min_crop_size*self.wide_w))
        self.margin_ratio = margin_ratio
        self.smoothing = smoothing
        self.converge_eps = converge_eps
        self.speed = max_edge_speed
        self.exit_speed = max_edge_speed if exit_edge_speed is None else exit_edge_speed
        if not isfinite(self.exit_speed) or self.exit_speed <= 0:
            raise ValueError('exit speed must be positive')
        self.exit_hold_ms = exit_hold_ms
        self.current = list(self.wide)
        self.target_id = None
        self.last_seen = self.last_ts = self.exit_since = None
        self.lock_width = None
        self.mode = "WIDE"
        self.reason = "IDLE"
        self.visible = None
        self.manual_override = self.paused = False

    def set_manual_override(self, force_wide):
        self.manual_override = bool(force_wide)

    def set_paused(self, paused):
        self.paused = bool(paused)

    def _release(self):
        self.target_id = self.last_seen = self.lock_width = None

    def _goal(self, state, tid, packet, ts):
        self.visible = None
        if self.manual_override:
            self._release()
            self.exit_since = None
            return self.wide, "FORCED_WIDE"
        if self.paused:
            return self.current, "PAUSED"
        if state == "ENDING" and self.target_id and tid in (None, self.target_id):
            if self.exit_since is None:
                self.exit_since = ts
            if ts - self.exit_since < self.exit_hold_ms:
                return self.current, "ENDING_HOLD"
        else:
            self.exit_since = None
        if state != "SHOWING" or tid is None:
            self._release()
            return self.wide, state
        if self.target_id != tid:
            self._release()
        observed = packet.get("timestamp_ms", ts)
        obj = next((o for o in packet.get("objects", []) if o.get("track_id") == tid), {})
        box = visible_box(obj.get("bbox_norm"))
        if box is None or ts - observed > self.hold_ms:
            if self.last_seen is not None and ts - self.last_seen <= self.hold_ms:
                return self.current, "TARGET_MISSING_HOLD"
            self._release()
            return self.wide, "TARGET_UNAVAILABLE"
        new_target = self.target_id != tid
        self.target_id, self.last_seen = tid, observed
        self.visible = box
        context = related_context(box, tid, packet)
        desired = compute_crop(context, self.aspect, self.margin_ratio, self.min_width)
        width = desired[2] - desired[0]
        # Use the unpadded subject span so the wide-shot threshold is independent of padding.
        if self.aspect == 1. and max(context[2]-context[0], context[3]-context[1]) >= .60:
            width = self.wide_w
        # Keep scale stable within an episode; widen when a growing object needs room.
        self.lock_width = width if new_target else max(self.lock_width, width)
        # A nearly full-sized crop adds imperceptible magnification. With matching
        # aspect ratios the full source also gives the most room around the target.
        if self.aspect == 1. and self.lock_width >= .96:
            return self.wide, "FULL_FRAME_FIT"
        cx, cy = (context[0]+context[2])/2, (context[1]+context[3])/2
        w = self.lock_width
        goal = centered_crop(cx, cy, w, min(1., w/self.aspect))
        if self.mode == "CLOSEUP" and not new_target and crop_distance(goal, self.current) < .009:
            goal = self.current
        return goal, "ACQUIRE" if new_target else "TRACK"

    def update(self, intent_packet, perception_packet, *, timestamp_ms=None):
        ts = intent_packet["timestamp_ms"] if timestamp_ms is None else timestamp_ms
        if self.last_ts is not None and ts < self.last_ts:
            raise ValueError("camera clock moved backwards")
        per = perception_packet or {}
        if max(intent_packet["timestamp_ms"], per.get("timestamp_ms", 0)) > ts:
            raise ValueError("camera received a future packet")
        dt = 0 if self.last_ts is None else ts-self.last_ts
        self.last_ts = ts
        goal, reason = self._goal(intent_packet["state"], intent_packet.get("target_id"), per, ts)
        old = list(self.current)
        distance = crop_distance(goal, self.current)
        if reason == "FORCED_WIDE":
            self.current = list(self.wide)
        elif distance and dt:
            # Bound movement even after a long gap in delivery.
            step_ms = min(dt, 100)
            alpha = 1 - (1-self.smoothing)**(step_ms*30/1000)
            speed = self.exit_speed if self.target_id is None else self.speed
            step = min(alpha*distance, speed*step_ms/1000)
            if distance <= min(self.converge_eps, speed*step_ms/1000):
                step = distance
            self.current = list(goal) if step == distance else [
                a + (b-a)*step/distance for a, b in zip(self.current, goal)]
        if self.visible and dt:
            # Project each axis toward a containing window. Respect the same edge
            # speed budget if the ID jumps too far to reach in a single frame.
            keep_visible(self.current, old, self.visible, self.speed * min(dt, 100) / 1000)
        if reason not in ("PAUSED", "TARGET_MISSING_HOLD", "ENDING_HOLD"):
            if self.target_id is None or reason == "FULL_FRAME_FIT":
                self.mode = "WIDE" if self.current == self.wide else "TRANSITION_OUT"
            elif reason == "ACQUIRE" or self.mode != "CLOSEUP":
                self.mode = "CLOSEUP" if crop_distance(self.current, goal) <= self.converge_eps else "TRANSITION_IN"
        self.reason = reason
        zoom = self.wide_w / (self.current[2]-self.current[0])
        return ShotPacket(ts, self.mode, None if self.mode == "WIDE" else self.target_id, list(self.current), zoom,
                          self.manual_override or self.paused).to_dict()


class StreamingCameraController(CameraController):
    """Keep framing through the same bounded observation gaps as stream B."""
    def __init__(self, **kwargs):
        kwargs.setdefault('miss_hold_ms', 1000)
        kwargs.setdefault('exit_hold_ms', 200)
        kwargs.setdefault('exit_edge_speed', .8)
        super().__init__(**kwargs)

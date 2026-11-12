"""IntentPacket contract and the ShowIntent rule-based state machine.

B consumes A's PerceptionPacket stream and emits one IntentPacket per frame.
It is pure vision: no ASR, no predeclared target, no learned model.

States: NONE -> CANDIDATE -> SHOWING -> ENDING -> NONE.

Core idea: a hand-object relation is only *candidate* evidence. Showing
requires the object to be raised above its table baseline and held stably
for several consecutive frames. Ordinary carrying/tidying stays at table
height and therefore never reaches SHOWING.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from statistics import median

STATES = ("NONE", "CANDIDATE", "SHOWING", "ENDING")

REASON_HAND_ASSOCIATED = "HAND_OBJECT_ASSOCIATED"
REASON_RAISED = "RAISED_IN_FRAME"
REASON_HOLD = "TEMPORAL_HOLD"
REASON_TARGET_LOST = "TARGET_LOST"
REASON_BACK_ON_TABLE = "BACK_ON_TABLE"
REASON_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"
REASON_CANDIDATE_REPLACED = "CANDIDATE_REPLACED"
REASON_RELINKED = "TARGET_RELINKED"


@dataclass(frozen=True)
class IntentPacket:
    timestamp_ms: int
    state: str
    target_id: str | None
    score: float
    reason_codes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "timestamp_ms": self.timestamp_ms,
            "state": self.state,
            "target_id": self.target_id,
            "score": round(float(self.score), 3),
            "reason_codes": list(self.reason_codes),
        }


def bbox_center(bbox: list[float]) -> tuple[float, float]:
    return (bbox[0] + bbox[2]) / 2.0, (bbox[1] + bbox[3]) / 2.0


class TableBaseline:
    """Estimate each track's resting (table) center-y from unrelated frames.

    Frames where the object has no hand relation are assumed to be on the
    table. The median is robust against a few unrelated in-air frames.
    """

    def __init__(self, min_samples: int = 3, clamp: tuple[float, float] = (0.45, 1.0)) -> None:
        self.min_samples = min_samples
        self.clamp = clamp
        self._samples: dict[str, list[float]] = {}

    def observe(self, track_id: str, center_y: float, related: bool) -> None:
        if related:
            return
        bucket = self._samples.setdefault(track_id, [])
        bucket.append(center_y)
        if len(bucket) > 120:  # bounded memory; the median barely changes later
            del bucket[:60]

    def baseline(self, track_id: str) -> float | None:
        bucket = self._samples.get(track_id, [])
        if len(bucket) < self.min_samples:
            return None
        return max(self.clamp[0], min(self.clamp[1], median(bucket)))


class TrackHistory:
    """Bounded per-track center history used for stability checks."""

    def __init__(self, maxlen: int = 30) -> None:
        self._points: list[tuple[int, float, float]] = []  # (frame_id, cx, cy)
        self.maxlen = maxlen

    def add(self, frame_id: int, cx: float, cy: float) -> None:
        self._points.append((frame_id, cx, cy))
        if len(self._points) > self.maxlen:
            del self._points[: len(self._points) - self.maxlen]

    def max_step(self, count: int) -> float | None:
        """Largest consecutive-frame centre displacement in the last `count` points."""
        pts = self._points[-count:]
        if len(pts) < 2:
            return None
        worst = 0.0
        for (f0, x0, y0), (f1, x1, y1) in zip(pts, pts[1:]):
            if f1 - f0 > 2:  # gap: skip across a detection dropout
                continue
            worst = max(worst, ((x1 - x0) ** 2 + (y1 - y0) ** 2) ** 0.5)
        return worst

    def span(self, count: int) -> float | None:
        pts = self._points[-count:]
        if len(pts) < 2:
            return None
        xs = [p[1] for p in pts]
        ys = [p[2] for p in pts]
        return max(max(xs) - min(xs), max(ys) - min(ys))

    def last_point(self) -> tuple[int, float, float] | None:
        return self._points[-1] if self._points else None


class _TrackState:
    __slots__ = ("related_streak", "history", "unrelated_streak")

    def __init__(self) -> None:
        self.related_streak = 0
        self.unrelated_streak = 0
        self.history = TrackHistory()


class ShowIntentStateMachine:
    """Frame-based hysteresis state machine over PerceptionPacket dicts."""

    def __init__(
        self,
        candidate_frames: int = 3,
        showing_frames: int = 8,
        miss_tolerance: int = 15,
        putdown_frames: int = 3,
        end_frames: int = 3,
        candidate_decay_frames: int = 8,
        relation_min_score: float = 0.3,
        raise_margin: float = 0.12,
        putdown_margin: float = 0.05,
        step_tolerance: float = 0.05,
        raised_fallback_y: float = 0.72,
        table_fallback_y: float = 0.75,
        candidate_min_ms: float | None = None,
        showing_min_ms: float | None = None,
        missing_tolerance_ms: float | None = None,
        ending_min_ms: float | None = None,
        candidate_switch_frames: int = 90,
        ending_max_frames: int = 60,
        ending_max_ms: float | None = None,
        relink_max_frames: int = 45,
        relink_max_dist: float = 0.15,
    ) -> None:
        self.candidate_frames = candidate_frames
        self.showing_frames = showing_frames
        self.miss_tolerance = miss_tolerance
        self.putdown_frames = putdown_frames
        self.end_frames = end_frames
        self.candidate_decay_frames = candidate_decay_frames
        self.relation_min_score = relation_min_score
        self.raise_margin = raise_margin
        self.putdown_margin = putdown_margin
        self.step_tolerance = step_tolerance
        self.raised_fallback_y = raised_fallback_y
        self.table_fallback_y = table_fallback_y
        # Optional millisecond thresholds. When provided, the corresponding
        # frame-count thresholds are recomputed from the observed packet rate
        # (median inter-packet timestamp gap), so "3 frames" means the same
        # wall-clock time at 5 FPS mock data and 30 FPS real video. When
        # None (default), the legacy frame-count semantics are kept.
        self.candidate_min_ms = candidate_min_ms
        self.showing_min_ms = showing_min_ms
        self.missing_tolerance_ms = missing_tolerance_ms
        self.ending_min_ms = ending_min_ms
        # D1: a candidate that fails to promote for this many frames may be
        # replaced by another object that currently satisfies the full
        # qualification (related + raised + stable).
        self.candidate_switch_frames = candidate_switch_frames
        # D2: hard cap on how long ENDING may last, and a bounded window in
        # which a same-class nearby raised track may resume the presentation.
        self.ending_max_frames = ending_max_frames
        self.ending_max_ms = ending_max_ms
        self.relink_max_frames = relink_max_frames
        self.relink_max_dist = relink_max_dist
        self._ts_gaps: list[int] = []
        self._ms_per_frame: float | None = None

        self.state = "NONE"
        self.target_id: str | None = None
        self._tracks: dict[str, _TrackState] = {}
        self._baseline = TableBaseline()
        self._candidate_streak = 0
        self._candidate_age = 0
        self._candidate_decay = 0
        self._miss = 0
        self._putdown_streak = 0
        self._end_streak = 0
        self._ending_frames = 0
        self._last_relation_score = 0.0
        self._last_ts: int | None = None
        self._track_classes: dict[str, str] = {}

    # -- helpers ---------------------------------------------------------

    def _observe_rate(self, ts: int) -> None:
        """Track the median inter-packet gap to convert ms thresholds to frames."""
        if not self._ts_gaps and self._last_ts is None:
            self._last_ts = ts
            return
        gap = ts - int(self._last_ts)
        self._last_ts = ts
        if 0 < gap <= 2000:  # ignore playback jumps / out-of-order ts
            self._ts_gaps.append(gap)
            if len(self._ts_gaps) > 60:
                del self._ts_gaps[:30]
            self._ms_per_frame = median(self._ts_gaps)

    def _thr(self, frame_value: int, ms_value: float | None) -> int:
        """Effective frame threshold: ms-based when available, else legacy frames."""
        if ms_value is None or self._ms_per_frame is None or self._ms_per_frame <= 0:
            return frame_value
        return max(1, math.ceil(ms_value / self._ms_per_frame))

    def _related_map(self, packet: dict) -> dict[str, float]:
        out: dict[str, float] = {}
        for rel in packet.get("relations", []):
            if rel.get("type") == "near_or_moving_together" and rel.get("score", 0.0) >= self.relation_min_score:
                object_id = rel.get("object_id")
                if object_id:
                    out[object_id] = max(out.get(object_id, 0.0), float(rel["score"]))
        return out

    def _is_raised(self, track_id: str, cy: float) -> bool:
        base = self._baseline.baseline(track_id)
        if base is not None:
            return cy < base - self.raise_margin
        # No resting samples (e.g. the hand rests next to the object on the
        # table the whole clip): fall back to an absolute display region.
        return cy < self.raised_fallback_y

    def _is_on_table(self, track_id: str, cy: float) -> bool:
        base = self._baseline.baseline(track_id)
        if base is None:
            return cy >= self.table_fallback_y
        return cy >= base - self.putdown_margin

    def _is_stable(self, track_id: str) -> bool:
        track = self._tracks.get(track_id)
        if track is None:
            return False
        window = max(self._thr(self.showing_frames, self.showing_min_ms), 4)
        step = track.history.max_step(window)
        return step is not None and step <= self.step_tolerance

    def _packet(self, ts: int, state: str, score: float, reasons: list[str], target: str | None = None) -> dict:
        return IntentPacket(ts, state, target if target else self.target_id, score, reasons).to_dict()

    # -- main entry ------------------------------------------------------

    def update(self, packet: dict) -> dict:
        frame_id = int(packet.get("frame_id", 0))
        ts = int(packet.get("timestamp_ms", frame_id * 1000 // 30))
        self._observe_rate(ts)
        objects = {o["track_id"]: o for o in packet.get("objects", [])}
        related = self._related_map(packet)

        for track_id, obj in objects.items():
            cx, cy = bbox_center(obj["bbox_norm"])
            track = self._tracks.setdefault(track_id, _TrackState())
            if obj.get("class_name"):
                self._track_classes[track_id] = obj["class_name"]
            is_related = track_id in related
            track.related_streak = track.related_streak + 1 if is_related else 0
            track.unrelated_streak = 0 if is_related else track.unrelated_streak + 1
            track.history.add(frame_id, cx, cy)
            self._baseline.observe(track_id, cy, is_related)

        if self.state == "NONE":
            return self._update_none(ts, objects, related)
        if self.state == "CANDIDATE":
            return self._update_candidate(ts, objects, related)
        if self.state == "SHOWING":
            return self._update_showing(ts, objects, related)
        return self._update_ending(ts, objects, related)

    # -- states ----------------------------------------------------------

    def _update_none(self, ts, objects, related) -> dict:
        self.target_id = None
        best_id, best_streak = None, 0
        for track_id in objects:
            streak = self._tracks[track_id].related_streak
            if streak > best_streak:
                best_id, best_streak = track_id, streak
        if best_id is not None and best_streak >= self._thr(self.candidate_frames, self.candidate_min_ms):
            self.state = "CANDIDATE"
            self.target_id = best_id
            self._candidate_streak = best_streak
            self._candidate_age = 0
            self._candidate_decay = 0
            # emit CANDIDATE for the transition frame; promotion to SHOWING
            # can only happen on a later frame so the stage is always visible
            return self._packet(ts, "CANDIDATE", related.get(best_id, 0.0), [REASON_HAND_ASSOCIATED])
        return self._packet(ts, "NONE", 0.0, [REASON_INSUFFICIENT])

    def _update_candidate(self, ts, objects, related) -> dict:
        track_id = self.target_id
        obj = objects.get(track_id)
        if obj is None or track_id not in related:
            self._candidate_decay += 1
            if self._candidate_decay > self.candidate_decay_frames:
                self._reset_to_none()
                return self._packet(ts, "NONE", 0.0, [REASON_INSUFFICIENT])
            return self._packet(ts, "CANDIDATE", 0.0, [REASON_INSUFFICIENT])
        # D1: re-evaluate every frame. If the current candidate has been
        # stuck below promotion for candidate_switch_frames while another
        # object fully qualifies (related + raised + stable), replace it.
        # The replacement emits CANDIDATE (never SHOWING) and must earn its
        # own promotion streak afterwards.
        if self._candidate_age >= self.candidate_switch_frames:
            for tid, other in objects.items():
                if tid == track_id or tid not in related:
                    continue
                ocx, ocy = bbox_center(other["bbox_norm"])
                if (
                    self._tracks[tid].related_streak >= self.candidate_frames
                    and self._is_raised(tid, ocy)
                    and self._is_stable(tid)
                ):
                    self.target_id = tid
                    self._candidate_streak = self._tracks[tid].related_streak
                    self._candidate_age = 0
                    self._candidate_decay = 0
                    return self._packet(ts, "CANDIDATE", related[tid], [REASON_HAND_ASSOCIATED, REASON_CANDIDATE_REPLACED])
        self._candidate_decay = 0
        self._candidate_streak += 1
        self._candidate_age += 1
        cx, cy = bbox_center(obj["bbox_norm"])
        raised = self._is_raised(track_id, cy)
        stable = self._is_stable(track_id)
        score = related[track_id]
        reasons = [REASON_HAND_ASSOCIATED]
        if raised:
            reasons.append(REASON_RAISED)
        if stable:
            reasons.append(REASON_HOLD)
        # _candidate_age >= 1 guarantees every target visibly passes through
        # CANDIDATE for at least one frame before SHOWING can be announced.
        if self._candidate_age >= 1 and self._candidate_streak >= self._thr(self.showing_frames, self.showing_min_ms) and raised and stable:
            self.state = "SHOWING"
            self._miss = 0
            self._putdown_streak = 0
            return self._packet(ts, "SHOWING", score, [REASON_HAND_ASSOCIATED, REASON_RAISED, REASON_HOLD])
        return self._packet(ts, "CANDIDATE", score, reasons)

    def _update_showing(self, ts, objects, related) -> dict:
        track_id = self.target_id
        obj = objects.get(track_id)
        if obj is None:
            self._miss += 1
            if self._miss > self._thr(self.miss_tolerance, self.missing_tolerance_ms):
                self.state = "ENDING"
                self._end_streak = 0
                self._ending_frames = 0
                return self._packet(ts, "ENDING", 0.0, [REASON_TARGET_LOST])
            return self._packet(ts, "SHOWING", self._last_relation_score, [REASON_HAND_ASSOCIATED])
        self._miss = 0
        cx, cy = bbox_center(obj["bbox_norm"])
        score = related.get(track_id, self._last_relation_score)
        self._last_relation_score = score
        if self._is_on_table(track_id, cy):
            self._putdown_streak += 1
        else:
            self._putdown_streak = 0
        if self._putdown_streak >= self.putdown_frames:
            self.state = "ENDING"
            self._end_streak = 0
            self._ending_frames = 0
            return self._packet(ts, "ENDING", score, [REASON_BACK_ON_TABLE])
        return self._packet(ts, "SHOWING", score, [REASON_HAND_ASSOCIATED, REASON_RAISED, REASON_HOLD])

    def _update_ending(self, ts, objects, related) -> dict:
        # D2: hard cap — ENDING must never outlive ending_max_frames.
        self._ending_frames += 1
        if self._ending_frames > self._thr(self.ending_max_frames, self.ending_max_ms):
            self._reset_to_none()
            return self._packet(ts, "NONE", 0.0, [REASON_TARGET_LOST, REASON_INSUFFICIENT])
        track_id = self.target_id  # the original SHOWING target
        obj = objects.get(track_id)
        if obj is not None:
            cx, cy = bbox_center(obj["bbox_norm"])
            if track_id in related and self._is_raised(track_id, cy):
                # brief table touch then lifted again: resume the same target
                self.state = "SHOWING"
                self._miss = 0
                self._putdown_streak = 0
                self._ending_frames = 0
                self._last_relation_score = related.get(track_id, 0.0)
                return self._packet(ts, "SHOWING", self._last_relation_score, [REASON_HAND_ASSOCIATED, REASON_RAISED, REASON_HOLD])
            # still visible but either back on the table or floating without a
            # relation: in both cases the presentation evidence is fading.
            self._end_streak += 1
        else:
            # D2: the original track is gone (perception ID fragmentation).
            # Within a short window, allow an explicit re-link to a track that
            # is the same class, spatially close to the last known position,
            # currently raised and hand-related. Any other new track must not
            # reset the ending timer.
            if self._ending_frames <= self.relink_max_frames:
                relinked = self._try_relink(objects, related)
                if relinked is not None:
                    self.state = "SHOWING"
                    self._miss = 0
                    self._putdown_streak = 0
                    self._ending_frames = 0
                    self._last_relation_score = related.get(relinked, 0.0)
                    return self._packet(ts, "SHOWING", self._last_relation_score,
                                        [REASON_HAND_ASSOCIATED, REASON_RAISED, REASON_HOLD, REASON_RELINKED])
            self._end_streak += 1
        if self._end_streak >= self._thr(self.end_frames, self.ending_min_ms):
            self._reset_to_none()
            return self._packet(ts, "NONE", 0.0, [REASON_INSUFFICIENT])
        reason = REASON_TARGET_LOST if obj is None else REASON_BACK_ON_TABLE
        return self._packet(ts, "ENDING", 0.0, [reason])

    def _try_relink(self, objects, related) -> str | None:
        orig_class = self._track_classes.get(self.target_id)
        track = self._tracks.get(self.target_id)
        if orig_class is None or track is None:
            return None
        last = track.history.last_point()
        if last is None:
            return None
        best_id, best_dist = None, self.relink_max_dist
        for tid, o in objects.items():
            if tid == self.target_id or tid not in related:
                continue
            if self._track_classes.get(tid) != orig_class:
                continue
            cx, cy = bbox_center(o["bbox_norm"])
            dist = ((cx - last[1]) ** 2 + (cy - last[2]) ** 2) ** 0.5
            if dist <= best_dist and self._is_raised(tid, cy):
                best_id, best_dist = tid, dist
        if best_id is not None:
            self.target_id = best_id
        return best_id

    def _reset_to_none(self) -> None:
        self.state = "NONE"
        self.target_id = None
        self._candidate_streak = 0
        self._candidate_age = 0
        self._candidate_decay = 0
        self._miss = 0
        self._putdown_streak = 0
        self._end_streak = 0
        self._ending_frames = 0

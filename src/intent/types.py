"""Shared intent packets and small geometry/history helpers."""

from __future__ import annotations

from dataclasses import dataclass, field
from statistics import median

STATES = ("NONE", "CANDIDATE", "SHOWING", "ENDING")

REASON_HAND_ASSOCIATED = "HAND_OBJECT_ASSOCIATED"
REASON_RAISED = "RAISED_IN_FRAME"
REASON_TARGET_LOST = "TARGET_LOST"
REASON_BACK_ON_TABLE = "BACK_ON_TABLE"
REASON_INSUFFICIENT = "INSUFFICIENT_EVIDENCE"


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


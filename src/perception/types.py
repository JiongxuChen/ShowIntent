"""Data structures for the A -> B/C PerceptionPacket contract."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any


@dataclass(frozen=True)
class Detection:
    track_id: str
    class_name: str
    bbox_norm: list[float]
    score: float


@dataclass(frozen=True)
class HandDetection:
    hand_id: str
    bbox_norm: list[float]
    score: float


@dataclass(frozen=True)
class Relation:
    hand_id: str
    object_id: str
    type: str
    score: float


@dataclass(frozen=True)
class PerceptionPacket:
    timestamp_ms: int
    frame_id: int
    objects: list[Detection]
    hands: list[HandDetection]
    relations: list[Relation]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


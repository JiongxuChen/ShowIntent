"""Explainable hand-object candidate association features."""

from __future__ import annotations

from .tracker import BBox, center_distance
from .types import HandDetection, Relation, Detection


def _expanded_box(box: BBox, margin: float) -> BBox:
    return (max(0.0, box[0] - margin), max(0.0, box[1] - margin), min(1.0, box[2] + margin), min(1.0, box[3] + margin))


def _contains(box: BBox, x: float, y: float) -> bool:
    return box[0] <= x <= box[2] and box[1] <= y <= box[3]


def infer_relations(hands: list[HandDetection], objects: list[Detection], near_distance: float = 0.16) -> list[Relation]:
    """Return proximity evidence. This is not a grasp or intent classifier."""
    relations: list[Relation] = []
    for hand in hands:
        hand_box: BBox = tuple(hand.bbox_norm)  # type: ignore[assignment]
        for obj in objects:
            object_box: BBox = tuple(obj.bbox_norm)  # type: ignore[assignment]
            expanded = _expanded_box(object_box, near_distance)
            hx, hy = (hand_box[0] + hand_box[2]) / 2, (hand_box[1] + hand_box[3]) / 2
            distance = center_distance(hand_box, object_box)
            if _contains(expanded, hx, hy) or distance <= near_distance:
                score = max(0.0, min(1.0, 1.0 - distance / max(near_distance * 2, 1e-6)))
                relations.append(Relation(hand.hand_id, obj.track_id, "near_or_moving_together", round(score, 3)))
    return relations


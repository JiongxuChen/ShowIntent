"""Small, dependency-free tracker suitable for P0 short clips.

It deliberately trades long-term re-identification for deterministic, explainable
IoU/centre-distance association. A track survives brief detector dropouts only.
"""

from __future__ import annotations

from dataclasses import dataclass


BBox = tuple[float, float, float, float]


def iou(a: BBox, b: BBox) -> float:
    left, top = max(a[0], b[0]), max(a[1], b[1])
    right, bottom = min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, right - left) * max(0.0, bottom - top)
    if inter == 0:
        return 0.0
    area_a = max(0.0, a[2] - a[0]) * max(0.0, a[3] - a[1])
    area_b = max(0.0, b[2] - b[0]) * max(0.0, b[3] - b[1])
    return inter / max(area_a + area_b - inter, 1e-9)


def center_distance(a: BBox, b: BBox) -> float:
    ax, ay = (a[0] + a[2]) / 2, (a[1] + a[3]) / 2
    bx, by = (b[0] + b[2]) / 2, (b[1] + b[3]) / 2
    return ((ax - bx) ** 2 + (ay - by) ** 2) ** 0.5


@dataclass
class _Track:
    track_id: str
    class_name: str
    bbox: BBox
    missed: int = 0


class SimpleObjectTracker:
    def __init__(self, iou_threshold: float = 0.15, max_center_distance: float = 0.18, max_missed: int = 8) -> None:
        self.iou_threshold = iou_threshold
        self.max_center_distance = max_center_distance
        self.max_missed = max_missed
        self._tracks: dict[str, _Track] = {}
        self._next_id = 1

    def update(self, detections: list[tuple[str, BBox, float]]) -> list[tuple[str, str, BBox, float]]:
        """Associate `(class_name, bbox, score)` detections and return IDs."""
        candidates: list[tuple[float, str, int]] = []
        for index, (class_name, bbox, _) in enumerate(detections):
            for track_id, track in self._tracks.items():
                if track.class_name != class_name:
                    continue
                overlap = iou(track.bbox, bbox)
                distance = center_distance(track.bbox, bbox)
                if overlap >= self.iou_threshold or distance <= self.max_center_distance:
                    candidates.append((overlap - distance * 0.1, track_id, index))

        assigned_tracks: set[str] = set()
        assigned_detections: set[int] = set()
        matches: dict[int, str] = {}
        for _, track_id, index in sorted(candidates, reverse=True):
            if track_id not in assigned_tracks and index not in assigned_detections:
                assigned_tracks.add(track_id)
                assigned_detections.add(index)
                matches[index] = track_id

        for track in self._tracks.values():
            track.missed += 1
        output: list[tuple[str, str, BBox, float]] = []
        for index, (class_name, bbox, score) in enumerate(detections):
            track_id = matches.get(index)
            if track_id is None:
                track_id = f"obj_{self._next_id}"
                self._next_id += 1
                self._tracks[track_id] = _Track(track_id, class_name, bbox)
            else:
                track = self._tracks[track_id]
                track.bbox, track.missed = bbox, 0
            output.append((track_id, class_name, bbox, score))

        self._tracks = {key: value for key, value in self._tracks.items() if value.missed <= self.max_missed}
        return output


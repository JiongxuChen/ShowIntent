"""ShotPacket contract and the P0 camera controller.

C consumes A's PerceptionPacket + B's IntentPacket and emits one ShotPacket
per frame. The crop is computed with plain geometry (bbox + margin, aspect
fit, clamped to [0, 1]) and smoothed with an exponential moving average.
No future frames are used; no LLM anywhere near the lens logic.
"""

from __future__ import annotations

from dataclasses import dataclass

MODES = ("WIDE", "TRANSITION_IN", "CLOSEUP", "TRANSITION_OUT")
FULL_CROP = (0.0, 0.0, 1.0, 1.0)


@dataclass(frozen=True)
class ShotPacket:
    timestamp_ms: int
    mode: str
    target_id: str | None
    crop_norm: list[float]
    zoom: float
    manual_override: bool

    def to_dict(self) -> dict:
        return {
            "timestamp_ms": self.timestamp_ms,
            "mode": self.mode,
            "target_id": self.target_id,
            "crop_norm": [round(v, 4) for v in self.crop_norm],
            "zoom": round(float(self.zoom), 3),
            "manual_override": bool(self.manual_override),
        }


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def compute_crop(bbox: list[float], aspect: float, margin_ratio: float = 0.30, min_size: float = 0.12) -> list[float]:
    """Expand a normalized bbox by a margin and fit the output aspect ratio.

    Returns [x1, y1, x2, y2] fully inside [0, 1].
    """
    x1, y1, x2, y2 = bbox
    w = max(x2 - x1, min_size)
    h = max(y2 - y1, min_size)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
    w = w * (1 + 2 * margin_ratio)
    h = h * (1 + 2 * margin_ratio)
    if w / h < aspect:  # widen to match aspect
        w = h * aspect
    else:
        h = w / aspect
    # keep the crop inside the frame, shifting only as much as needed
    left = clamp01(min(max(cx - w / 2, 0.0), 1.0 - w))
    top = clamp01(min(max(cy - h / 2, 0.0), 1.0 - h))
    return [left, top, left + w, top + h]


def crop_distance(a: list[float], b: list[float]) -> float:
    return max(abs(a[i] - b[i]) for i in range(4))


class CameraController:
    def __init__(
        self,
        aspect: float = 16 / 9,
        margin_ratio: float = 0.30,
        smoothing: float = 0.2,
        converge_eps: float = 0.015,
        miss_hold_frames: int = 15,
        min_crop_size: float = 0.12,
    ) -> None:
        self.aspect = aspect
        self.margin_ratio = margin_ratio
        self.smoothing = smoothing
        self.converge_eps = converge_eps
        self.miss_hold_frames = miss_hold_frames
        self.min_crop_size = min_crop_size
        self.mode = "WIDE"
        self.current = list(FULL_CROP)
        self.target_id: str | None = None
        self._miss = 0
        self.manual_override = False

    def set_manual_override(self, force_wide: bool) -> None:
        """Human-in-the-loop hook: force the camera back to the wide shot."""
        self.manual_override = bool(force_wide)

    def _lerp(self, target: list[float]) -> list[float]:
        a = self.smoothing
        return [cur + (tgt - cur) * a for cur, tgt in zip(self.current, target)]

    def update(self, intent_packet: dict, perception_packet: dict | None) -> dict:
        ts = int(intent_packet.get("timestamp_ms", 0))
        state = intent_packet.get("state", "NONE")
        self.target_id = intent_packet.get("target_id")

        if self.manual_override:
            self.current = self._lerp(list(FULL_CROP))
            mode = "WIDE" if crop_distance(self.current, FULL_CROP) <= self.converge_eps else "TRANSITION_OUT"
            self.mode = mode
            zoom = 1.0 / max(self.current[2] - self.current[0], 1e-6)
            return ShotPacket(ts, mode, None, self.current, zoom, True).to_dict()

        if state == "SHOWING" and self.target_id and perception_packet is not None:
            obj = next((o for o in perception_packet.get("objects", []) if o.get("track_id") == self.target_id), None)
            if obj is not None:
                self._miss = 0
                target = compute_crop(obj["bbox_norm"], self.aspect, self.margin_ratio, self.min_crop_size)
            else:
                self._miss += 1
                if self._miss <= self.miss_hold_frames:
                    target = list(self.current)  # brief dropout: hold the last stable crop
                else:
                    target = list(FULL_CROP)  # timed out: fall back to wide
            self.current = self._lerp(target)
            if target == list(FULL_CROP):
                mode = "WIDE" if crop_distance(self.current, FULL_CROP) <= self.converge_eps else "TRANSITION_OUT"
            else:
                converged = crop_distance(self.current, target) <= self.converge_eps
                mode = "CLOSEUP" if converged else "TRANSITION_IN"
        else:
            self._miss = 0
            self.current = self._lerp(list(FULL_CROP))
            mode = "WIDE" if crop_distance(self.current, FULL_CROP) <= self.converge_eps else "TRANSITION_OUT"

        self.mode = mode
        zoom = 1.0 / max(self.current[2] - self.current[0], 1e-6)
        return ShotPacket(ts, mode, self.target_id, self.current, zoom, False).to_dict()

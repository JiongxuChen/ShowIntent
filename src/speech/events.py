"""D1's minimal SpeechEvent contract, using the team's exact wire fields."""

from dataclasses import asdict, dataclass, field
from typing import Any


def require_ms(value: Any, name: str) -> int:
    if type(value) is not int or value < 0:
        raise ValueError(f"{name} must be a non-negative integer in milliseconds")
    return value


def _labels(value: Any, name: str) -> None:
    if not isinstance(value, list) or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"{name} must be a list of non-empty strings")
    if len(value) != len(set(value)):
        raise ValueError(f"{name} must not contain duplicates")


@dataclass(frozen=True)
class SpeechEvent:
    start_ms: int
    end_ms: int
    available_at_ms: int
    text: str
    mentioned_classes: list[str] = field(default_factory=list)
    cue_types: list[str] = field(default_factory=list)
    is_final: bool = True

    def __post_init__(self) -> None:
        for name in ("start_ms", "end_ms", "available_at_ms"):
            require_ms(getattr(self, name), name)
        if self.end_ms <= self.start_ms:
            raise ValueError("end_ms must be greater than start_ms")
        if self.available_at_ms < self.end_ms:
            raise ValueError("available_at_ms must be >= end_ms (no future words)")
        if not isinstance(self.text, str) or not self.text.strip():
            raise ValueError("text must be a non-empty string")
        _labels(self.mentioned_classes, "mentioned_classes")
        _labels(self.cue_types, "cue_types")
        if type(self.is_final) is not bool:
            raise ValueError("is_final must be boolean")

    def to_dict(self) -> dict:
        # Lists may have been modified by a caller after construction.
        self.__post_init__()
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SpeechEvent":
        if not isinstance(data, dict):
            raise ValueError("SpeechEvent must be a JSON object")
        expected = set(cls.__dataclass_fields__)
        if set(data) != expected:
            raise ValueError(f"SpeechEvent fields must be exactly {sorted(expected)}")
        return cls(**data)


def available_events(events: list[SpeechEvent], timestamp_ms: int) -> list[SpeechEvent]:
    """Offline fixture query only; does not measure or simulate ASR execution."""
    require_ms(timestamp_ms, "timestamp_ms")
    return [event for event in events if event.available_at_ms <= timestamp_ms]

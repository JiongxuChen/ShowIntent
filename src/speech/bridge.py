"""Stable D-to-B adapter with JSONL event loading; B's source stays unchanged."""

import json
from pathlib import Path

from .cue_adapter import SpeechCueAdapter
from .events import SpeechEvent


def read_events_jsonl(path: str | Path) -> list[SpeechEvent]:
    events = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                events.append(SpeechEvent.from_dict(json.loads(line)))
            except (ValueError, TypeError, json.JSONDecodeError) as exc:
                raise ValueError(f"{path}: event line {line_no}: {exc}") from exc
    if any(left.available_at_ms > right.available_at_ms for left, right in zip(events, events[1:])):
        raise ValueError("events must be ordered by available_at_ms")
    return events


class SpeechEvidenceBridge:
    """B may call ``evidence_for(packet)`` before its visual update.

    The returned dict intentionally has no ``state``, ``target_id`` or camera
    field. With ``enabled=False`` (default) it is a no-op diagnostic.
    """
    def __init__(self, events: list[SpeechEvent] | None = None, *, enabled: bool = False, ttl_ms: int = 3000, lexicon_path=None):
        self.events = list(events or [])
        self.adapter = SpeechCueAdapter(enabled=enabled, ttl_ms=ttl_ms, lexicon_path=lexicon_path)

    @classmethod
    def from_jsonl(cls, path: str | Path, **options) -> "SpeechEvidenceBridge":
        return cls(read_events_jsonl(path), **options)

    def evidence_for(self, perception_packet: dict) -> dict:
        return self.adapter.evidence(self.events, perception_packet).to_dict()

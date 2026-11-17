"""Optional causal speech evidence for B. This module cannot create an intent packet."""

from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterable

from .events import SpeechEvent, require_ms
from .language import LanguageCueExtractor


@dataclass(frozen=True)
class CueEvidence:
    timestamp_ms: int
    enabled: bool
    has_evidence: bool = False
    suggested_classes: list[str] = field(default_factory=list)
    cue_types: list[str] = field(default_factory=list)
    presentation_language: bool = False
    reason_codes: list[str] = field(default_factory=list)
    source_start_ms: int | None = None
    source_end_ms: int | None = None
    source_available_at_ms: int | None = None

    def to_dict(self) -> dict:
        return asdict(self)


class SpeechCueAdapter:
    def __init__(self, *, enabled: bool = False, ttl_ms: int = 3000, lexicon_path: str | Path | None = None):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        require_ms(ttl_ms, "ttl_ms")
        self.enabled, self.ttl_ms, self.lexicon_path, self.extractor = enabled, ttl_ms, lexicon_path, None

    def evidence(self, events: Iterable[SpeechEvent] | None, perception_packet: dict, *, timestamp_ms: int | None = None) -> CueEvidence:
        now = require_ms(perception_packet.get("timestamp_ms") if timestamp_ms is None and isinstance(perception_packet, dict) else timestamp_ms, "timestamp_ms")
        def result(reason: str, source: SpeechEvent | None = None, **updates) -> CueEvidence:
            values = {"timestamp_ms": now, "enabled": self.enabled, "reason_codes": [reason]}
            if source:
                values.update(source_start_ms=source.start_ms, source_end_ms=source.end_ms, source_available_at_ms=source.available_at_ms)
            values.update(updates)
            return CueEvidence(**values)
        if not self.enabled:
            return result("DISABLED")
        if not isinstance(perception_packet, dict) or type(perception_packet.get("timestamp_ms")) is not int or perception_packet["timestamp_ms"] > now:
            return result("INVALID_PERCEPTION")
        objects = perception_packet.get("objects")
        if not isinstance(objects, list) or any(not isinstance(item, dict) or not isinstance(item.get("class_name"), str) or not item["class_name"].strip() for item in objects):
            return result("INVALID_PERCEPTION")
        ready = []
        for event in events or ():
            if not isinstance(event, SpeechEvent):
                raise ValueError("events must contain SpeechEvent instances")
            if event.available_at_ms <= now:
                if self.extractor is None:
                    self.extractor = LanguageCueExtractor(self.lexicon_path)
                lexical = self.extractor.extract(event.text)
                ready.append((event, list(dict.fromkeys(event.mentioned_classes + lexical.mentioned_classes)), set(event.cue_types) | set(lexical.cue_types)))
        if not ready:
            return result("NO_AVAILABLE_RELEVANT_SPEECH")
        event, classes, cues = max(ready, key=lambda item: (item[0].end_ms, item[0].start_ms, item[0].available_at_ms, item[0].is_final))
        if now - event.end_ms > self.ttl_ms:
            return result("EXPIRED_SPEECH", event)
        if not event.is_final:
            return result("PARTIAL_TRANSCRIPT", event)
        for cue in ("NEGATED_LANGUAGE", "ENDING_LANGUAGE", "TOPIC_CHANGE"):
            if cue in cues:
                return result(cue, event, cue_types=[cue])
        if len(classes) != 1:
            return result("CONFLICTING_CLASSES" if classes else "NO_CLASS_MENTION", event)
        visible = Counter(item["class_name"] for item in objects)
        if not visible[classes[0]]:
            return result("NO_VISIBLE_MATCH", event)
        if visible[classes[0]] > 1:
            return result("AMBIGUOUS_INSTANCES", event)
        presentation = "PRESENTATION_LANGUAGE" in cues
        return result("VISIBLE_CATEGORY_MATCH", event, has_evidence=True, suggested_classes=classes, cue_types=["PRESENTATION_LANGUAGE"] if presentation else [], presentation_language=presentation)

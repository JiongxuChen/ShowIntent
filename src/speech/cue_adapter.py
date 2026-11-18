"""Optional D5/D6 category evidence for B, with causal and ambiguity gates.

This module never emits an IntentPacket, ShotPacket, target ID or track ID.
Freshness is measured from spoken-content end_ms, so a delayed ASR return does
not make old words fresh again. Choosing the latest spoken relevant cue also
prevents an out-of-order old ASR result from replacing a more recent topic.
"""

from __future__ import annotations

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
    def __init__(
        self, *, enabled: bool = False, ttl_ms: int = 3000,
        lexicon_path: str | Path | None = None,
    ):
        if type(enabled) is not bool:
            raise ValueError("enabled must be boolean")
        require_ms(ttl_ms, "ttl_ms")
        self.enabled = enabled
        self.ttl_ms = ttl_ms
        self.lexicon_path = lexicon_path
        # Disabled/empty optional speech must not need a dictionary on disk.
        self.extractor = None

    def evidence(
        self, events: Iterable[SpeechEvent] | None, perception_packet: dict,
        *, timestamp_ms: int | None = None,
    ) -> CueEvidence:
        if timestamp_ms is None:
            if not isinstance(perception_packet, dict):
                raise ValueError("perception_packet must be an object")
            timestamp_ms = perception_packet.get("timestamp_ms")
        now = require_ms(timestamp_ms, "timestamp_ms")

        def result(reason: str, source: SpeechEvent | None = None, **changes) -> CueEvidence:
            values = dict(timestamp_ms=now, enabled=self.enabled, reason_codes=[reason])
            if source is not None:
                values.update(source_start_ms=source.start_ms, source_end_ms=source.end_ms,
                              source_available_at_ms=source.available_at_ms)
            values.update(changes)
            return CueEvidence(**values)

        if not self.enabled:
            return result("DISABLED")
        if not isinstance(perception_packet, dict):
            return result("INVALID_PERCEPTION")
        packet_time = perception_packet.get("timestamp_ms")
        if type(packet_time) is not int or packet_time < 0:
            return result("INVALID_PERCEPTION")
        if packet_time > now:
            return result("FUTURE_PERCEPTION")
        objects = perception_packet.get("objects")
        if not isinstance(objects, list) or any(
            not isinstance(obj, dict) or not isinstance(obj.get("class_name"), str)
            or not obj["class_name"].strip() for obj in objects
        ):
            return result("INVALID_PERCEPTION")
        visible_counts = Counter(obj["class_name"] for obj in objects)
        available = []
        for event in events or ():
            if not isinstance(event, SpeechEvent):
                raise ValueError("events must contain SpeechEvent instances")
            event.to_dict()
            if event.available_at_ms <= now:
                # Recover negative/ending/topic cues even if the caller passes old
                # manually-labelled fixtures instead of enrich_event output.
                if self.extractor is None:
                    self.extractor = LanguageCueExtractor(self.lexicon_path)
                cues = self.extractor.extract(event.text)
                semantic = set(event.cue_types) | set(cues.cue_types)
                classes = list(dict.fromkeys(event.mentioned_classes + cues.mentioned_classes))
                if semantic or classes:
                    available.append((event, classes, semantic))
        if not available:
            return result("NO_AVAILABLE_RELEVANT_SPEECH")
        event, classes, semantic = max(
            available, key=lambda row: (row[0].end_ms, row[0].start_ms,
                                       row[0].available_at_ms, row[0].is_final)
        )
        if now - event.end_ms > self.ttl_ms:
            return result("EXPIRED_SPEECH", event)
        if not event.is_final:
            return result("PARTIAL_TRANSCRIPT", event)
        if "NEGATED_LANGUAGE" in semantic:
            return result("NEGATED_LANGUAGE", event, cue_types=["NEGATED_LANGUAGE"])
        if "ENDING_LANGUAGE" in semantic:
            return result("ENDING_LANGUAGE", event, cue_types=["ENDING_LANGUAGE"])
        if "TOPIC_CHANGE" in semantic:
            return result("TOPIC_CHANGE", event, cue_types=["TOPIC_CHANGE"])
        if not classes:
            return result("NO_CLASS_MENTION", event)
        if len(classes) > 1:
            return result("CONFLICTING_CLASSES", event)
        class_name = classes[0]
        if not visible_counts[class_name]:
            return result("NO_VISIBLE_MATCH", event)
        if visible_counts[class_name] > 1:
            return result("AMBIGUOUS_INSTANCES", event)
        cue_types = [cue for cue in ("PRESENTATION_LANGUAGE",) if cue in semantic]
        return result(
            "VISIBLE_CATEGORY_MATCH", event, has_evidence=True,
            suggested_classes=[class_name], cue_types=cue_types,
            presentation_language="PRESENTATION_LANGUAGE" in semantic,
        )


def speech_cue_adapter(
    events: Iterable[SpeechEvent] | None, perception_packet: dict, *,
    timestamp_ms: int | None = None, enabled: bool = False, ttl_ms: int = 3000,
    lexicon_path: str | Path | None = None,
) -> dict:
    """Stateless convenience interface; B retains every final decision."""
    return SpeechCueAdapter(enabled=enabled, ttl_ms=ttl_ms, lexicon_path=lexicon_path).evidence(
        events, perception_packet, timestamp_ms=timestamp_ms,
    ).to_dict()

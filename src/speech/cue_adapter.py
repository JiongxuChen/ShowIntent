"""Causal lexical observations. Instance grounding and intent belong to B."""

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
    attributes: dict = field(default_factory=dict)
    referent_scope: str = 'mentioned'
    source_text: str = ''
    semantic_backend: str = 'rules'

    def to_dict(self) -> dict:
        return asdict(self)


class SpeechCueAdapter:
    def __init__(self, *, enabled: bool = False, ttl_ms: int = 3000,
                 max_source_age_ms: int = 8000, lexicon_path: str | Path | None = None):
        if type(enabled) is not bool:
            raise ValueError('enabled must be boolean')
        require_ms(ttl_ms, 'ttl_ms')
        require_ms(max_source_age_ms, 'max_source_age_ms')
        self.enabled, self.ttl_ms = enabled, ttl_ms
        self.max_source_age_ms = max_source_age_ms
        self.lexicon_path = lexicon_path
        self.extractor = None
        self._lexical_cache = {}

    def evidence(self, events: Iterable[SpeechEvent] | None, perception_packet: dict,
                 *, timestamp_ms: int | None = None) -> CueEvidence:
        now = require_ms(perception_packet.get('timestamp_ms') if timestamp_ms is None
                         and isinstance(perception_packet, dict) else timestamp_ms, 'timestamp_ms')

        def result(reason, source=None, **updates):
            values = dict(timestamp_ms=now, enabled=self.enabled, reason_codes=[reason])
            if source is not None:
                values.update(source_start_ms=source.start_ms, source_end_ms=source.end_ms,
                              source_available_at_ms=source.available_at_ms)
            values.update(updates)
            return CueEvidence(**values)

        if not self.enabled:
            return result('DISABLED')
        if (not isinstance(perception_packet, dict)
                or type(perception_packet.get('timestamp_ms')) is not int
                or perception_packet['timestamp_ms'] != now):
            return result('INVALID_PERCEPTION')
        objects = perception_packet.get('objects')
        if not isinstance(objects, list) or any(not isinstance(o, dict)
                or not isinstance(o.get('class_name'), str) or not o['class_name'].strip() for o in objects):
            return result('INVALID_PERCEPTION')
        ready = []
        for event in events or ():
            if not isinstance(event, SpeechEvent):
                raise ValueError('events must contain SpeechEvent instances')
            # Partial and future transcripts cannot supersede a usable final event.
            if event.available_at_ms > now or not event.is_final:
                continue
            if self.extractor is None:
                self.extractor = LanguageCueExtractor(self.lexicon_path)
            key = (event.text, tuple(event.mentioned_classes), tuple(event.cue_types))
            if key not in self._lexical_cache:
                lexical = self.extractor.extract(event.text)
                # Saved rule tags are not immutable truth. Recompute them from
                # text so replaying an old ASR file also fixes obsolete polarity.
                self._lexical_cache[key] = lexical
            lexical = self._lexical_cache[key]
            classes, cues = lexical.mentioned_classes, set(lexical.cue_types)
            attrs, scope, backend = lexical.attributes or {}, lexical.referent_scope, 'rules'
            if event.semantics is not None:
                from .semantics import semantic_cues
                classes, cues, attrs, scope = semantic_cues(event.semantics)
                backend = 'validated_semantics'
            if 'CONTINUATION_LANGUAGE' in cues:
                cues = cues - {'NEGATED_LANGUAGE', 'ENDING_LANGUAGE'}
            # Unrelated chat does not erase a still-relevant previous observation.
            if classes or cues:
                ready.append((event, classes, cues, attrs, scope, backend))
        if not ready:
            return result('NO_AVAILABLE_RELEVANT_SPEECH')
        event, classes, cues, attrs, scope, backend = max(ready, key=lambda item: (item[0].end_ms, item[0].start_ms,
                                                          item[0].available_at_ms))
        # Bound both availability age and semantic age. Delayed stale results never revive.
        if now-event.available_at_ms > self.ttl_ms or now-event.end_ms > self.max_source_age_ms:
            return result('EXPIRED_SPEECH', event)
        fields = dict(suggested_classes=classes, cue_types=sorted(cues),
                      presentation_language='PRESENTATION_LANGUAGE' in cues, attributes=attrs,
                      referent_scope=scope, source_text=event.text, semantic_backend=backend)
        for cue in ('NON_CURRENT_LANGUAGE', 'NEGATED_LANGUAGE', 'ENDING_LANGUAGE', 'TOPIC_CHANGE'):
            if cue in cues:
                return result(cue, event, **fields)
        if len(classes) > 1:
            return result('CONFLICTING_CLASSES', event, **fields)
        if not classes:
            # Keep deictic semantics; B can resolve against its active/interacted instance.
            return result('DEICTIC_REFERENCE' if cues else 'NO_CLASS_MENTION', event,
                          has_evidence=bool(cues), **fields)
        visible = Counter(o['class_name'] for o in objects)
        if not visible[classes[0]]:
            return result('NO_VISIBLE_MATCH', event, **fields)
        if visible[classes[0]] > 1:
            return result('AMBIGUOUS_INSTANCES', event, **fields)
        return result('VISIBLE_CATEGORY_MATCH', event, has_evidence=True, **fields)

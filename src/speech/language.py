"""Small, inspectable bilingual lexical cues for D3.

The ASR engine and the visual intent module remain separate. These rules find
words and presentation-like expressions, not a person's intent or object ID.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
import re

from .events import SpeechEvent


DEFAULT_LEXICON = Path(__file__).resolve().parents[2] / "config" / "speech_lexicon.json"


@dataclass(frozen=True)
class LanguageCues:
    mentioned_classes: list[str]
    cue_types: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _phrase_pattern(phrase: str) -> re.Pattern:
    """English phrases use word boundaries; Chinese aliases use literal spans."""
    pattern = re.escape(phrase)
    pattern = pattern.replace(r"\ ", r"\s+")
    if re.match(r"[a-zA-Z0-9]", phrase):
        pattern = r"\b" + pattern
    if re.search(r"[a-zA-Z0-9]$", phrase):
        pattern += r"\b"
    return re.compile(pattern, re.IGNORECASE)


def _string_list(value, field: str) -> list[str]:
    if not isinstance(value, list) or not value or any(
        not isinstance(item, str) or not item.strip() for item in value
    ):
        raise ValueError(f"lexicon {field} must be a non-empty list of strings")
    return value


class LanguageCueExtractor:
    def __init__(self, lexicon_path: str | Path | None = None):
        self.lexicon_path = Path(lexicon_path) if lexicon_path else DEFAULT_LEXICON
        config = json.loads(self.lexicon_path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict) or type(config.get("version")) is not int or config["version"] != 1:
            raise ValueError("lexicon must be an object with version=1")
        classes = config.get("classes")
        if not isinstance(classes, dict) or not classes:
            raise ValueError("lexicon classes must be a non-empty object")
        self._aliases = []
        for class_name, aliases in classes.items():
            if not isinstance(class_name, str) or not class_name.strip():
                raise ValueError("lexicon class names must be non-empty strings")
            self._aliases.append((class_name, [
                _phrase_pattern(alias) for alias in _string_list(aliases, f"classes.{class_name}")
            ]))
        self._excluded = [
            _phrase_pattern(phrase)
            for phrase in _string_list(config.get("excluded_phrases"), "excluded_phrases")
        ]
        self._cue_patterns = {}
        for field, cue_type in (
            ("presentation_patterns", "PRESENTATION_LANGUAGE"),
            ("negative_patterns", "NEGATED_LANGUAGE"),
            ("topic_change_patterns", "TOPIC_CHANGE"),
            ("ending_patterns", "ENDING_LANGUAGE"),
        ):
            try:
                # Version-1 custom dictionaries written before ending cues stay
                # valid. Missing/empty ending_patterns explicitly disables them.
                patterns = config.get(field, []) if field == "ending_patterns" else config.get(field)
                if field != "ending_patterns" or patterns != []:
                    patterns = _string_list(patterns, field)
                self._cue_patterns[cue_type] = [
                    re.compile(pattern, re.IGNORECASE)
                    for pattern in patterns
                ]
            except re.error as exc:
                raise ValueError(f"invalid regex in lexicon {field}: {exc}") from exc

    def extract(self, text: str) -> LanguageCues:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        # Spaces in excluded spans preserve word positions and prevent substring
        # aliases such as 苹果 in 苹果汁; a later independent 苹果 still matches.
        masked = text
        for pattern in self._excluded:
            masked = pattern.sub(lambda match: " " * len(match.group()), masked)
        found = []
        for class_name, patterns in self._aliases:
            positions = [match.start() for pattern in patterns for match in pattern.finditer(masked)]
            if positions:
                found.append((min(positions), class_name))
        mentioned_classes = [name for _, name in sorted(found)]
        negated = any(p.search(text) for p in self._cue_patterns["NEGATED_LANGUAGE"])
        topic_change = any(p.search(text) for p in self._cue_patterns["TOPIC_CHANGE"])
        presentation = any(p.search(text) for p in self._cue_patterns["PRESENTATION_LANGUAGE"])
        ending = any(p.search(text) for p in self._cue_patterns["ENDING_LANGUAGE"])
        cue_types = []
        # Sentence-level rejection is deliberate: mixed closing/opening or
        # positive/negative clauses need richer grounding. Closing is a lexical
        # cue for B, never an instruction to put B's state into ENDING.
        if presentation and not negated and not ending:
            cue_types.append("PRESENTATION_LANGUAGE")
        if negated:
            cue_types.append("NEGATED_LANGUAGE")
        elif ending:
            cue_types.append("ENDING_LANGUAGE")
        if topic_change:
            cue_types.append("TOPIC_CHANGE")
        return LanguageCues(mentioned_classes, cue_types)

    def enrich(self, event: SpeechEvent) -> SpeechEvent:
        event.to_dict()  # validate mutable nested fields before accepting input
        cues = self.extract(event.text)
        # Extraction replaces fixture annotations; it never trusts a positive
        # hand-written label more than the current text's negation checks.
        return replace(event, mentioned_classes=cues.mentioned_classes, cue_types=cues.cue_types)


@lru_cache(maxsize=8)
def _extractor(lexicon_path: str | None) -> LanguageCueExtractor:
    return LanguageCueExtractor(lexicon_path)


def extract_cues(text: str, lexicon_path: str | Path | None = None) -> LanguageCues:
    """Extract lexical categories and cues using the bundled or supplied rules."""
    path = str(Path(lexicon_path).resolve()) if lexicon_path else None
    return _extractor(path).extract(text)


def enrich_event(event: SpeechEvent, *, lexicon_path: str | Path | None = None) -> SpeechEvent:
    """Keep timing/text unchanged while replacing semantic fields with D3 rules."""
    path = str(Path(lexicon_path).resolve()) if lexicon_path else None
    return _extractor(path).enrich(event)

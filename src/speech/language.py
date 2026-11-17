"""Small inspectable bilingual lexical rules for member D.

Lexical results describe words that were recognized; they do not decide visual
presentation intent, an object track, B's state, or any camera action.
"""

from dataclasses import asdict, dataclass, replace
from functools import lru_cache
import json
from pathlib import Path
import re

from .events import SpeechEvent

DEFAULT_LEXICON = Path(__file__).resolve().parents[2] / "configs" / "speech_lexicon.json"


@dataclass(frozen=True)
class LanguageCues:
    mentioned_classes: list[str]
    cue_types: list[str]

    def to_dict(self) -> dict:
        return asdict(self)


def _phrased(pattern: str) -> re.Pattern:
    expression = re.escape(pattern).replace(r"\ ", r"\s+")
    if re.match(r"[A-Za-z0-9]", pattern):
        expression = r"\b" + expression
    if re.search(r"[A-Za-z0-9]$", pattern):
        expression += r"\b"
    return re.compile(expression, re.IGNORECASE)


def _list(value, name: str, *, allow_empty: bool = False) -> list[str]:
    if not isinstance(value, list) or (not allow_empty and not value) or any(not isinstance(x, str) or not x.strip() for x in value):
        raise ValueError(f"lexicon {name} must be a {'possibly empty ' if allow_empty else 'non-empty '}list of strings")
    return value


class LanguageCueExtractor:
    def __init__(self, lexicon_path: str | Path | None = None):
        path = Path(lexicon_path) if lexicon_path else DEFAULT_LEXICON
        config = json.loads(path.read_text(encoding="utf-8-sig"))
        if not isinstance(config, dict) or config.get("version") != 1 or not isinstance(config.get("classes"), dict):
            raise ValueError("lexicon must be version 1 with classes")
        self.aliases = [(name, [_phrased(x) for x in _list(words, f"classes.{name}")]) for name, words in config["classes"].items()]
        self.excluded = [_phrased(x) for x in _list(config.get("excluded_phrases"), "excluded_phrases", allow_empty=True)]
        self.patterns = {}
        for field, cue in (("presentation_patterns", "PRESENTATION_LANGUAGE"), ("negative_patterns", "NEGATED_LANGUAGE"), ("ending_patterns", "ENDING_LANGUAGE"), ("topic_change_patterns", "TOPIC_CHANGE")):
            try:
                self.patterns[cue] = [re.compile(x, re.IGNORECASE) for x in _list(config.get(field, []), field, allow_empty=field == "ending_patterns")]
            except re.error as exc:
                raise ValueError(f"invalid regex in {field}: {exc}") from exc

    def extract(self, text: str) -> LanguageCues:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        masked = text
        for pattern in self.excluded:
            masked = pattern.sub(lambda item: " " * len(item.group()), masked)
        mentioned = []
        for name, patterns in self.aliases:
            positions = [match.start() for pattern in patterns for match in pattern.finditer(masked)]
            if positions:
                mentioned.append((min(positions), name))
        hit = {cue: any(pattern.search(text) for pattern in patterns) for cue, patterns in self.patterns.items()}
        cues = []
        if hit["PRESENTATION_LANGUAGE"] and not hit["NEGATED_LANGUAGE"] and not hit["ENDING_LANGUAGE"]:
            cues.append("PRESENTATION_LANGUAGE")
        if hit["NEGATED_LANGUAGE"]:
            cues.append("NEGATED_LANGUAGE")
        elif hit["ENDING_LANGUAGE"]:
            cues.append("ENDING_LANGUAGE")
        if hit["TOPIC_CHANGE"]:
            cues.append("TOPIC_CHANGE")
        return LanguageCues([name for _, name in sorted(mentioned)], cues)

    def enrich(self, event: SpeechEvent) -> SpeechEvent:
        cues = self.extract(event.text)
        return replace(event, mentioned_classes=cues.mentioned_classes, cue_types=cues.cue_types)


@lru_cache(maxsize=8)
def _cached(path: str | None) -> LanguageCueExtractor:
    return LanguageCueExtractor(path)


def extract_cues(text: str, lexicon_path: str | Path | None = None) -> LanguageCues:
    return _cached(str(Path(lexicon_path).resolve()) if lexicon_path else None).extract(text)


def enrich_event(event: SpeechEvent, *, lexicon_path: str | Path | None = None) -> SpeechEvent:
    return _cached(str(Path(lexicon_path).resolve()) if lexicon_path else None).enrich(event)

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
    attributes: dict = None
    referent_scope: str = 'mentioned'
    source_clause: str = ''

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
        for field, cue in (("presentation_patterns", "PRESENTATION_LANGUAGE"), ("negative_patterns", "NEGATED_LANGUAGE"), ("ending_patterns", "ENDING_LANGUAGE"), ("topic_change_patterns", "TOPIC_CHANGE"), ("reference_patterns", "REFERENCE_LANGUAGE"), ("continuation_patterns", "CONTINUATION_LANGUAGE"), ("non_current_patterns", "NON_CURRENT_LANGUAGE")):
            try:
                self.patterns[cue] = [re.compile(x, re.IGNORECASE) for x in _list(config.get(field, []), field, allow_empty=True)]
            except re.error as exc:
                raise ValueError(f"invalid regex in {field}: {exc}") from exc

    def extract(self, text: str) -> LanguageCues:
        if not isinstance(text, str):
            raise ValueError("text must be a string")
        # Scope polarity and time to the relevant clause. An introductory
        # conditional ("如果你刚学微积分，看看这本书") is not a future display.
        clauses = [c.strip() for c in re.split(r'[，。！？；,;!?\n]+', text) if c.strip()]
        relevant = []
        for clause in clauses or [text]:
            probe = re.sub(r'不妨|不得不|不能不|not\s+only', '', clause, flags=re.I)
            hits = {cue: any(p.search(probe) for p in patterns) for cue, patterns in self.patterns.items()}
            introduction = bool(re.search(r'(?:先看(?:这|這|一[下眼])|先给.{0,4}介绍|先給.{0,4}介紹|接下来.{0,4}(?:介绍|介紹|看(?:这|這|一下))|首先是|第一[个本件款]|推荐.{0,5}这|推薦.{0,5}這)|\b(?:here\s+is|first\s+up|take\s+a\s+look)\b', probe, re.I))
            attribute = bool(re.search(r'(?:这个|这本|这款|它|這個|這本|這款).{0,12}(?:颜色|顏色|质地|質地|大小|封面|材质|材質|重量|尺寸|特点|特點)|\b(?:its|this).{0,20}(?:cover|color|texture|size|material)\b', probe, re.I))
            if introduction: hits['INTRODUCTION_LANGUAGE'] = True
            if attribute: hits['ATTRIBUTE_LANGUAGE'] = True
            action = any(hits.get(k) for k in ('PRESENTATION_LANGUAGE','NEGATED_LANGUAGE','ENDING_LANGUAGE','TOPIC_CHANGE','CONTINUATION_LANGUAGE','INTRODUCTION_LANGUAGE'))
            relevant.append((int(action), clause, hits))
        _, clause, hit = max(enumerate(relevant), key=lambda pair:(pair[1][0], pair[0]))[1]
        masked = clause
        for pattern in self.excluded:
            masked = pattern.sub(lambda item: " " * len(item.group()), masked)
        mentioned = []
        for name, patterns in self.aliases:
            positions = [match.start() for pattern in patterns for match in pattern.finditer(masked)]
            if positions:
                mentioned.append((min(positions), name))
        if hit["CONTINUATION_LANGUAGE"]:
            hit["NEGATED_LANGUAGE"] = hit["ENDING_LANGUAGE"] = False
        cues = []
        if hit["PRESENTATION_LANGUAGE"] and not hit["NEGATED_LANGUAGE"] and not hit["ENDING_LANGUAGE"]:
            cues.append("PRESENTATION_LANGUAGE")
        if hit["NEGATED_LANGUAGE"]:
            cues.append("NEGATED_LANGUAGE")
        elif hit["ENDING_LANGUAGE"]:
            cues.append("ENDING_LANGUAGE")
        if hit["TOPIC_CHANGE"]:
            cues.append("TOPIC_CHANGE")
        for cue in ("REFERENCE_LANGUAGE", "CONTINUATION_LANGUAGE", "NON_CURRENT_LANGUAGE", "INTRODUCTION_LANGUAGE", "ATTRIBUTE_LANGUAGE"):
            if hit.get(cue):
                cues.append(cue)
        colors = {'red':r'红色|紅色|红的|紅的|\bred\b','orange':r'橙色|\borange-colored\b',
                  'yellow':r'黄色|黃色|黄的|黃的|\byellow\b','green':r'绿色|綠色|绿的|綠的|\bgreen\b',
                  'blue':r'蓝色|藍色|蓝的|藍的|\bblue\b','purple':r'紫色|\bpurple\b',
                  'white':r'白色|白的|\bwhite\b','black':r'黑色|黑的|\bblack\b'}
        attrs = {'colors':[name for name, pattern in colors.items() if re.search(pattern, clause, re.I)]}
        scope = 'previous' if re.search(r'刚才那个|剛才那個|上一本|上一个|上一個|previous\s+(?:one|item)', clause, re.I) else 'mentioned'
        return LanguageCues([name for _, name in sorted(mentioned)], cues, attrs, scope, clause)

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

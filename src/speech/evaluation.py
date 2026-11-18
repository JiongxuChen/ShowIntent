"""Reproducible speech-text diagnostics for D's independent public-media tests.

jiwer and OpenCC are optional and imported only when scores are requested.
These helpers evaluate lexical ASR output, not presentation intent or fusion.
"""

from __future__ import annotations

from collections import Counter
import html
import math
from pathlib import Path
import re
import statistics
from typing import Iterable
import unicodedata

from third_party.srt import srt

from .events import SpeechEvent


_APOSTROPHES = str.maketrans({"’": "'", "‘": "'", "ʼ": "'", "＇": "'"})
_EMPTY_REFERENCE_POLICY = (
    "jiwer-4: both empty => 0; empty reference with nonempty hypothesis => "
    "number of inserted units, not a percentage of reference length"
)


def read_reference(path: str | Path) -> str:
    """Read plain text or caption text from SRT/simple WebVTT.

    WebVTT support covers the downloaded numeric/no-ID cues and timestamp cue
    settings. Advanced VTT NOTE/STYLE/REGION blocks and arbitrary cue IDs are
    deliberately rejected by the strict upstream SRT parser rather than silently
    omitted. Caption markup is removed, entities decoded and lines joined.
    """
    path = Path(path)
    text = path.read_text(encoding="utf-8-sig")
    suffix = path.suffix.lower()
    if suffix == ".txt":
        return text.strip()
    if suffix not in (".srt", ".vtt"):
        raise ValueError("reference must be .txt, .srt or simple .vtt")
    if suffix == ".vtt":
        lines = text.splitlines()
        if not lines or not lines[0].strip().startswith("WEBVTT"):
            raise ValueError(f"{path.name}: missing WEBVTT header")
        text = "\n".join(lines[1:])
        # SRT's repair-friendly content parser can absorb VTT metadata into a
        # previous cue. Validate only the outer simple-cue shape first, leaving
        # all timestamp/content parsing to the existing upstream implementation.
        for block in re.split(r"\n\s*\n", text.strip()):
            if not block.strip():
                continue
            block_lines = block.splitlines()
            first = block_lines[0].strip()
            if first.isdecimal() and len(block_lines) > 1:
                first = block_lines[1].strip()
            if "-->" not in first:
                raise ValueError(f"{path.name}: unsupported WebVTT metadata or nonnumeric cue ID")
    try:
        cues = list(srt.parse(text, ignore_errors=False))
    except srt.SRTParseError as exc:
        raise ValueError(f"{path.name}: malformed caption reference: {exc}") from exc
    parts = [html.unescape(re.sub(r"<[^>]*>", "", cue.content)) for cue in cues]
    return " ".join(" ".join(parts).split())


def _text(value: str, name: str) -> str:
    if not isinstance(value, str):
        raise ValueError(f"{name} must be a string")
    return unicodedata.normalize("NFKC", value).translate(_APOSTROPHES).lower()


def _normalise_words(value: str, name: str) -> str:
    text = _text(value, name)
    normalised = []
    for index, char in enumerate(text):
        category = unicodedata.category(char)
        if category[0] in ("L", "N", "M"):
            normalised.append(char)
        elif char == "'" and 0 < index < len(text) - 1 and text[index - 1].isalnum() and text[index + 1].isalnum():
            normalised.append(char)
        else:
            normalised.append(" ")
    return " ".join("".join(normalised).split())


def _normalise_characters(value: str, name: str) -> str:
    return "".join(char for char in _text(value, name)
                   if unicodedata.category(char)[0] in ("L", "N", "M"))


def _counts(output, reference: str, hypothesis: str, metric: str) -> dict:
    word_units = metric == "wer"
    return {
        metric: float(getattr(output, metric)),
        "reference_units": len(reference.split()) if word_units else len(reference),
        "hypothesis_units": len(hypothesis.split()) if word_units else len(hypothesis),
        "hits": output.hits,
        "substitutions": output.substitutions,
        "deletions": output.deletions,
        "insertions": output.insertions,
    }


def score_transcript(reference: str, hypothesis: str, language: str) -> dict:
    """English WER, or Chinese raw-script CER plus separately reported t2s CER.

    Scores are fractions and may exceed 1 when insertions dominate. For an empty
    reference, jiwer 4's insertion-count convention is explicit in the result.
    No reference/hypothesis text is returned, avoiding accidental republication
    of complete downloaded transcripts in compact experiment reports.
    """
    if not isinstance(language, str):
        raise ValueError("language must be 'en' or 'zh'")
    language = language.lower().replace("_", "-")
    if language == "english" or language == "en" or language.startswith("en-"):
        language = "en"
    elif language in ("zh", "cmn", "chinese") or language.startswith("zh-"):
        language = "zh"
    else:
        raise ValueError("only English WER and Chinese CER are supported")
    # Imports remain lazy so optional ASR/evaluation packages cannot break D1.
    import jiwer

    if language == "en":
        ref = _normalise_words(reference, "reference")
        hyp = _normalise_words(hypothesis, "hypothesis")
        output = jiwer.process_words(ref, hyp)
        return {
            "language": "en", "metric": "wer", **_counts(output, ref, hyp, "wer"),
            "normalization": "NFKC, lowercase, normalized inner-word apostrophes; other non-letter/number/mark characters become spaces",
            "empty_reference_policy": _EMPTY_REFERENCE_POLICY,
        }
    from opencc import OpenCC

    ref = _normalise_characters(reference, "reference")
    hyp = _normalise_characters(hypothesis, "hypothesis")
    output = jiwer.process_characters(ref, hyp)
    converter = OpenCC("t2s")
    simple_ref, simple_hyp = converter.convert(ref), converter.convert(hyp)
    simple_output = jiwer.process_characters(simple_ref, simple_hyp)
    return {
        "language": "zh", "metric": "cer", **_counts(output, ref, hyp, "cer"),
        "t2s": _counts(simple_output, simple_ref, simple_hyp, "cer"),
        "normalization": "NFKC, lowercase; whitespace, punctuation, symbols and controls removed; raw-script score retains traditional/simplified distinctions",
        "script_normalization": "OpenCC t2s, reported separately without replacing raw CER",
        "empty_reference_policy": _EMPTY_REFERENCE_POLICY,
    }


def event_summary(events: Iterable[SpeechEvent | dict]) -> dict:
    """Count extracted cues and availability delay; no correctness claim.

    Latency is available_at_ms - end_ms. p95 is nearest-rank ceil(0.95*n).
    Empty streams return null latency values instead of implying measured zero.
    """
    classes = Counter()
    latency = []
    event_count = presentation_count = negative_count = 0
    for value in events:
        event = SpeechEvent.from_dict(value) if isinstance(value, dict) else value
        if not isinstance(event, SpeechEvent):
            raise ValueError("events must contain SpeechEvent instances or wire dictionaries")
        event.to_dict()
        event_count += 1
        classes.update(event.mentioned_classes)
        presentation_count += "PRESENTATION_LANGUAGE" in event.cue_types
        negative_count += "NEGATED_LANGUAGE" in event.cue_types
        latency.append(event.available_at_ms - event.end_ms)
    latency.sort()
    return {
        "event_count": event_count,
        "class_counts": dict(sorted(classes.items())),
        "presentation_count": presentation_count,
        "negative_count": negative_count,
        "latency_ms": {
            "min": latency[0] if latency else None,
            "median": statistics.median(latency) if latency else None,
            "p95": latency[math.ceil(0.95 * len(latency)) - 1] if latency else None,
            "max": latency[-1] if latency else None,
        },
    }

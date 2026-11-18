"""Reuse cdown/srt for parsing; add ShowIntent's D1 fixture conversion."""

import json
from pathlib import Path

from third_party.srt import srt

from .events import SpeechEvent, require_ms


def _timedelta_ms(value) -> int:
    return (value.days * 86400 + value.seconds) * 1000 + value.microseconds // 1000


def load_mock_events(
    path: str | Path, *, simulated_delay_ms: int = 200, media_offset_ms: int = 0
) -> list[SpeechEvent]:
    """SRT has no language annotations. JSONL may carry manual annotations.

    SRT/JSONL files are prewritten fixtures, never actual online ASR observations.
    JSONL records: start_ms, end_ms, text; optionally mentioned_classes/cue_types.
    All timestamps in input are relative to the beginning of the audio clip.
    """
    require_ms(simulated_delay_ms, "simulated_delay_ms")
    require_ms(media_offset_ms, "media_offset_ms")
    path = Path(path)
    content = path.read_text(encoding="utf-8-sig")
    records = []
    if path.suffix.lower() == ".srt":
        # Strict upstream parser: malformed/unconsumed text is an error.
        try:
            records = [
                {"start_ms": _timedelta_ms(sub.start), "end_ms": _timedelta_ms(sub.end),
                 "text": sub.content.strip()}
                for sub in srt.parse(content, ignore_errors=False)
            ]
        except srt.SRTParseError as exc:
            raise ValueError(f"{path.name}: malformed SRT: {exc}") from exc
    elif path.suffix.lower() == ".jsonl":
        allowed = {"start_ms", "end_ms", "text", "mentioned_classes", "cue_types"}
        for line_no, line in enumerate(content.splitlines(), 1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
                if not isinstance(record, dict) or set(record) - allowed:
                    raise ValueError(f"allowed subtitle fields: {sorted(allowed)}")
                if not {"start_ms", "end_ms", "text"} <= set(record):
                    raise ValueError("subtitle requires start_ms, end_ms and text")
                records.append(record)
            except (ValueError, TypeError) as exc:
                raise ValueError(f"{path.name}:{line_no}: {exc}") from exc
    else:
        raise ValueError("subtitle input must be .srt or .jsonl")
    if not records:
        raise ValueError("subtitle input contains no cues")
    events = []
    for index, record in enumerate(records, 1):
        try:
            start = require_ms(record["start_ms"], "start_ms") + media_offset_ms
            end = require_ms(record["end_ms"], "end_ms") + media_offset_ms
            events.append(SpeechEvent(
                start_ms=start, end_ms=end, available_at_ms=end + simulated_delay_ms,
                text=record["text"],
                mentioned_classes=record.get("mentioned_classes", []),
                cue_types=record.get("cue_types", []), is_final=True,
            ))
        except (ValueError, TypeError) as exc:
            raise ValueError(f"{path.name}:cue {index}: {exc}") from exc
    # Overlapping cues may end in another order; output by availability for B.
    return sorted(events, key=lambda e: (e.available_at_ms, e.start_ms))


def read_events(path: str | Path) -> list[SpeechEvent]:
    events = []
    for line_no, line in enumerate(Path(path).read_text(encoding="utf-8-sig").splitlines(), 1):
        if line.strip():
            try:
                events.append(SpeechEvent.from_dict(json.loads(line)))
            except (ValueError, TypeError) as exc:
                raise ValueError(f"event line {line_no}: {exc}") from exc
    if any(a.available_at_ms > b.available_at_ms for a, b in zip(events, events[1:])):
        raise ValueError("events must be ordered by available_at_ms")
    return events

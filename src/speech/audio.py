"""Bounded-memory recording input built on python-soundfile.

Sample positions are the source of truth. Milliseconds are floored only when
exporting timestamps, so a block's rounding never accumulates into later blocks.
The reader preserves the recording's sample rate and channel count; preparing
ASR-specific mono/resampled input belongs to D2.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterator

import numpy as np
import soundfile as sf


@dataclass(frozen=True)
class AudioInfo:
    """Recording metadata; ``frame_count`` counts multichannel sample frames."""

    path: str
    sample_rate: int
    channels: int
    frame_count: int
    duration_ms: int
    duration_seconds: float
    format: str
    subtype: str

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass(frozen=True)
class AudioChunk:
    """A half-open sample interval and its corresponding media timestamps.

    ``samples`` has shape ``(end_sample - start_sample, channels)`` and dtype
    float32, including for mono recordings. Sample positions are relative to the
    recording; timestamps additionally include ``media_offset_ms``.
    """

    start_ms: int
    end_ms: int
    start_sample: int
    end_sample: int
    sample_rate: int
    samples: np.ndarray


def _require_integer(value: int, name: str, minimum: int) -> None:
    if type(value) is not int or value < minimum:
        raise ValueError(f"{name} must be an integer >= {minimum}")


def _recording_path(path: str | Path) -> Path:
    recording = Path(path)
    if not recording.is_file():
        raise FileNotFoundError(f"Audio file does not exist: {recording}")
    return recording


def _validate_recording(recording: sf.SoundFile, path: Path) -> None:
    if recording.frames <= 0:
        raise ValueError(f"Audio file contains no sample frames: {path}")
    if recording.samplerate <= 0 or recording.channels <= 0:
        raise ValueError(f"Audio file has invalid sample rate or channels: {path}")


def inspect_audio(path: str | Path) -> AudioInfo:
    """Read metadata without decoding the entire recording into memory.

    WAV/FLAC and other formats supported by the installed libsndfile are accepted.
    An invalid or empty recording raises ValueError; an absent path raises
    FileNotFoundError. ``duration_ms`` is floored, while ``frame_count`` and
    ``sample_rate`` preserve the exact duration as a rational number.
    """
    recording_path = _recording_path(path)
    try:
        with sf.SoundFile(recording_path) as recording:
            _validate_recording(recording, recording_path)
            return AudioInfo(
                path=str(recording_path.resolve()),
                sample_rate=recording.samplerate,
                channels=recording.channels,
                frame_count=recording.frames,
                duration_ms=recording.frames * 1000 // recording.samplerate,
                duration_seconds=recording.frames / recording.samplerate,
                format=recording.format,
                subtype=recording.subtype,
            )
    except (sf.LibsndfileError, sf.SoundFileError) as exc:
        raise ValueError(f"Cannot read audio file {recording_path}: {exc}") from exc


def iter_audio_chunks(
    path: str | Path,
    block_ms: int = 100,
    media_offset_ms: int = 0,
) -> Iterator[AudioChunk]:
    """Decode a recording incrementally, preserving its final partial block.

    Every boundary is derived from an absolute sample position. For rates such
    as 44.1 kHz and 1 ms blocks, block lengths alternate as needed rather than
    repeatedly rounding a fixed size. Memory is bounded by one requested block;
    this function does not sleep or claim to capture live microphone audio.
    """
    _require_integer(block_ms, "block_ms", 1)
    _require_integer(media_offset_ms, "media_offset_ms", 0)
    recording_path = _recording_path(path)
    try:
        with sf.SoundFile(recording_path) as recording:
            _validate_recording(recording, recording_path)
            sample_rate = recording.samplerate
            start_sample = 0
            block_number = 1
            while start_sample < recording.frames:
                boundary = block_number * sample_rate * block_ms // 1000
                end_sample = min(recording.frames, max(start_sample + 1, boundary))
                samples = recording.read(
                    end_sample - start_sample, dtype="float32", always_2d=True
                )
                if len(samples) != end_sample - start_sample:
                    raise ValueError(
                        f"Audio ended before its declared sample count: {recording_path}"
                    )
                yield AudioChunk(
                    start_ms=media_offset_ms + start_sample * 1000 // sample_rate,
                    end_ms=media_offset_ms + end_sample * 1000 // sample_rate,
                    start_sample=start_sample,
                    end_sample=end_sample,
                    sample_rate=sample_rate,
                    samples=samples,
                )
                start_sample = end_sample
                block_number += 1
    except (sf.LibsndfileError, sf.SoundFileError) as exc:
        raise ValueError(f"Cannot decode audio file {recording_path}: {exc}") from exc

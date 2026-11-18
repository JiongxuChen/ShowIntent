"""Incremental PyAV decode/resample, without predecoding an entire recording."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class MediaWindow:
    samples: object
    start_sample: int
    end_sample: int


def inspect_media_clock(path) -> dict:
    """Inspect stream starts; use container start as the common clip origin."""
    import av
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"media file does not exist: {path}")
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("media has no audio stream")
        audio = container.streams.audio[0]
        video = container.streams.video[0] if container.streams.video else None
        def start_ms(stream):
            return (float(stream.start_time * stream.time_base) * 1000
                    if stream is not None and stream.start_time is not None else None)
        audio_start = start_ms(audio)
        video_start = start_ms(video)
        origin = (container.start_time / 1000 if container.start_time is not None
                  else min([value for value in (audio_start, video_start) if value is not None], default=0))
        offset = max(0, round((audio_start if audio_start is not None else origin) - origin))
        return {"container_origin_ms": origin, "audio_stream_start_ms": audio_start,
                "video_stream_start_ms": video_start, "audio_offset_ms": offset,
                "timeline_policy": "decoded_audio_start_relative_to_container_start",
                "audio_codec": audio.codec_context.name}


def iter_media_windows(path, *, step_ms=2000, sample_rate=16000):
    """Yield exact sample-count windows at the requested native output timeline.

    libav decodes and resamples. The small pending tail plus one codec frame are
    retained; no full-file waveform is constructed. Timestamp discontinuities are
    rejected, because concatenating them would silently compress media time.
    """
    import av
    import numpy as np
    if type(step_ms) is not int or not 100 <= step_ms <= 10000:
        raise ValueError("step_ms must be an integer between 100 and 10000")
    if type(sample_rate) is not int or sample_rate <= 0:
        raise ValueError("sample_rate must be positive")
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"media file does not exist: {path}")
    block_size = step_ms * sample_rate // 1000
    if not block_size:
        raise ValueError("window contains no samples")
    pending = np.empty(0, dtype=np.float32)
    emitted = 0
    expected_pts = None
    with av.open(str(path)) as container:
        if not container.streams.audio:
            raise ValueError("media has no audio stream")
        resampler = av.AudioResampler(format="fltp", layout="mono", rate=sample_rate)
        def frames():
            for original in container.decode(audio=0):
                yield from resampler.resample(original)
            yield from resampler.resample(None)
        for frame in frames():
            if frame.pts is not None:
                position = round(float(frame.pts * frame.time_base) * sample_rate)
                if expected_pts is not None and abs(position - expected_pts) > 2:
                    raise ValueError("audio timestamp discontinuity; normalize gaps before replay")
                expected_pts = position + frame.samples
            values = np.asarray(frame.to_ndarray()[0], dtype=np.float32)
            if not np.isfinite(values).all():
                raise ValueError("audio contains nonfinite samples")
            pending = np.concatenate((pending, values))
            while len(pending) >= block_size:
                end = emitted + block_size
                yield MediaWindow(pending[:block_size].copy(), emitted, end)
                pending = pending[block_size:]
                emitted = end
        if len(pending):
            yield MediaWindow(pending.copy(), emitted, emitted + len(pending))
        elif emitted == 0:
            raise ValueError("audio contains no decoded samples")

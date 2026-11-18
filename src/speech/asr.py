"""D2/D4 adapter around pretrained Faster-Whisper, with auditable replay timing.

In causal mode each recognizer call sees only its completed nonoverlapping window.
Media arrival is simulated; processing is measured, including generator iteration
and language extraction. This is file replay, not a live microphone service.
"""

from dataclasses import dataclass
import math
from pathlib import Path
import time

from .events import SpeechEvent, require_ms
from .quality import POLICY, segment_quality

SAMPLE_RATE = 16000


@dataclass
class ASRRun:
    events: list[SpeechEvent]
    windows: list[dict]
    metadata: dict
    transcripts: list[dict]


def transcribe_samples(
    samples, recognizer, *, mode="offline", window_ms=4000, media_offset_ms=0,
    language=None, beam_size=3, extractor=None, clock=time.perf_counter,
) -> ASRRun:
    """Recognize a mono 16 kHz array; accept fake backends/clocks for causal tests.

    Offline mode releases every result only after EOF plus actual computation.
    Causal mode models a single worker: max(window EOF, previous completion) +
    measured processing time. It never assigns an offline transcript to the time
    a word was spoken. Disjoint windows can split words; this limitation is logged.
    """
    import numpy as np
    require_ms(media_offset_ms, "media_offset_ms")
    if mode not in {"offline", "causal"}:
        raise ValueError("mode must be offline or causal")
    if type(window_ms) is not int or not 1000 <= window_ms <= 30000:
        raise ValueError("window_ms must be an integer between 1000 and 30000")
    if type(beam_size) is not int or not 1 <= beam_size <= 10:
        raise ValueError("beam_size must be an integer between 1 and 10")
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("audio must be a nonempty finite mono 16 kHz sample array")
    duration_ms = math.ceil(len(audio) * 1000 / SAMPLE_RATE)
    step = window_ms * SAMPLE_RATE // 1000 if mode == "causal" else len(audio)
    events, windows, transcripts = [], [], []
    ready_ms = media_offset_ms
    total_compute_ms = 0
    discarded = 0
    rejected_quality = 0
    detected_languages = []
    for window_index, begin in enumerate(range(0, len(audio), step)):
        end = min(begin + step, len(audio))
        window_start = begin * 1000 // SAMPLE_RATE
        window_end = math.ceil(end * 1000 / SAMPLE_RATE)
        started_at_ms = max(media_offset_ms + window_end, ready_ms)
        before = clock()
        segments, info = recognizer.transcribe(
            audio[begin:end].copy(), language=language, beam_size=beam_size,
            temperature=0.0, condition_on_previous_text=False, vad_filter=True,
            word_timestamps=False,
        )
        # Faster-Whisper returns a lazy generator; measure the entire iteration.
        provisional = []
        for segment in segments:
            text = segment.text.strip()
            if not text:
                continue
            if not math.isfinite(segment.start) or not math.isfinite(segment.end):
                discarded += 1
                continue
            start_ms = max(window_start, min(window_end, window_start + round(segment.start * 1000)))
            end_ms = max(window_start, min(window_end, window_start + round(segment.end * 1000)))
            if end_ms <= start_ms:
                discarded += 1
                continue
            quality = segment_quality(segment)
            if not quality["accepted"]:
                rejected_quality += 1
                transcripts.append({
                    "window_index": window_index, "start_ms": media_offset_ms + start_ms,
                    "end_ms": media_offset_ms + end_ms, "text": text,
                    "avg_logprob": getattr(segment, "avg_logprob", None), **quality,
                })
                continue
            event = SpeechEvent(
                start_ms=media_offset_ms + start_ms, end_ms=media_offset_ms + end_ms,
                available_at_ms=media_offset_ms + window_end, text=text,
            )
            if extractor is not None:
                event = extractor(event)
            provisional.append(event)
            transcripts.append({
                "window_index": window_index, "start_ms": event.start_ms,
                "end_ms": event.end_ms, "text": event.text,
                "avg_logprob": getattr(segment, "avg_logprob", None),
                "no_speech_prob": getattr(segment, "no_speech_prob", None),
                "compression_ratio": quality["compression_ratio"], "accepted": True,
                "quality_reason": None, "quality_policy": POLICY,
            })
        elapsed = clock() - before
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("processing clock must be finite and monotonic")
        compute_ms = math.ceil(elapsed * 1000)
        total_compute_ms += compute_ms
        ready_ms = started_at_ms + compute_ms
        from dataclasses import replace
        events.extend(replace(event, available_at_ms=ready_ms) for event in provisional)
        detected = getattr(info, "language", language)
        if detected and detected not in detected_languages:
            detected_languages.append(detected)
        windows.append({
            "window_index": window_index, "start_sample": begin, "end_sample": end,
            "audio_start_ms": media_offset_ms + window_start,
            "received_until_ms": media_offset_ms + window_end,
            "processing_start_ms": started_at_ms, "available_at_ms": ready_ms,
            "compute_ms": compute_ms, "event_count": len(provisional),
        })
    return ASRRun(
        events=events, windows=windows, transcripts=transcripts,
        metadata={
            "stage": "D2-D4", "mode": mode, "sample_rate": SAMPLE_RATE,
            "duration_ms": duration_ms, "media_offset_ms": media_offset_ms,
            "window_ms": window_ms if mode == "causal" else None,
            "timing_kind": "simulated_media_arrival_plus_measured_compute",
            "timing_policy": "single_worker_causal_windows" if mode == "causal" else "offline_eof_barrier",
            "clock_scope": "transcribe_call_generator_iteration_and_language_extraction",
            "live_microphone": False, "total_compute_ms": total_compute_ms,
            "compute_rtf": total_compute_ms / duration_ms,
            "detected_languages": detected_languages, "discarded_segments": discarded,
            "event_count": len(events),
            "quality_policy": POLICY, "rejected_quality_segments": rejected_quality,
            "semantic_annotations": "rule_extractor" if extractor else "empty_no_extractor",
        },
    )


def run_asr(
    path, model_path, *, mode="offline", window_ms=4000, media_offset_ms=0,
    language=None, beam_size=3, device="cpu", compute_type="int8", cpu_threads=4,
    extract_language=True, lexicon_path=None,
) -> ASRRun:
    """Decode audio/video with upstream PyAV support and use local pinned weights."""
    source = Path(path)
    if not source.is_file():
        raise FileNotFoundError(f"media file does not exist: {source}")
    model_path = Path(model_path)
    if not model_path.is_dir() or not (model_path / "model.bin").is_file():
        raise ValueError("local model missing; run scripts/fetch_asr_model.py first")
    if type(cpu_threads) is not int or cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    import faster_whisper
    import ctranslate2
    import av
    from .media_stream import inspect_media_clock
    timing = inspect_media_clock(source)
    before = time.perf_counter()
    model = WhisperModel(
        str(model_path), device=device, compute_type=compute_type,
        cpu_threads=cpu_threads, local_files_only=True,
    )
    model_load_ms = math.ceil((time.perf_counter() - before) * 1000)
    before = time.perf_counter()
    samples = decode_audio(str(source), sampling_rate=SAMPLE_RATE)
    decode_ms = math.ceil((time.perf_counter() - before) * 1000)
    extractor = None
    if extract_language:
        from .language import enrich_event
        extractor = lambda event: enrich_event(event, lexicon_path=lexicon_path)
    result = transcribe_samples(
        samples, model, mode=mode, window_ms=window_ms,
        media_offset_ms=media_offset_ms + timing["audio_offset_ms"],
        language=language, beam_size=beam_size, extractor=extractor,
    )
    result.metadata.update({
        "source_kind": "pretrained_asr_from_media", "source": source.as_posix(),
        "model_path": model_path.as_posix(), "model_load_ms": model_load_ms,
        "decode_ms": decode_ms, "device": device, "compute_type": compute_type,
        "cpu_threads": cpu_threads, "beam_size": beam_size,
        "requested_language": language, "faster_whisper_version": faster_whisper.__version__,
        "media_offset_ms": media_offset_ms, "audio_offset_ms": timing["audio_offset_ms"],
        "media_clock": timing,
        "ctranslate2_version": ctranslate2.__version__, "pyav_version": av.__version__,
        "excluded_from_compute_rtf": ["model_load", "full_file_decode", "file_output"],
        "replay_limitations": "predecoded_file; ASR sees current window only; no live capture; disjoint boundaries may split speech",
    })
    return result

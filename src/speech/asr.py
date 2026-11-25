"""Pretrained Faster-Whisper replay for D's final-event contract.

This is file replay, not a microphone service. Causal mode calls the recognizer
on disjoint completed windows and measures when every result became available.
"""

from dataclasses import dataclass, replace
import math
import re
from pathlib import Path
import time

from .events import SpeechEvent, require_ms
from .language import enrich_event

SAMPLE_RATE = 16000


@dataclass
class ASRRun:
    events: list[SpeechEvent]
    windows: list[dict]
    transcripts: list[dict]
    metadata: dict


def transcribe_samples(samples, recognizer, *, mode: str = "offline", window_ms: int = 4000,
                       media_offset_ms: int = 0, language: str | None = None, beam_size: int = 3,
                       extract_language: bool = True, lexicon_path=None, context_ms: int = 0,
                       clock=time.perf_counter) -> ASRRun:
    """Recognize a mono 16 kHz array through Faster-Whisper or a test backend."""
    import numpy as np
    require_ms(media_offset_ms, "media_offset_ms")
    if mode not in {"offline", "causal"}:
        raise ValueError("mode must be offline or causal")
    if type(window_ms) is not int or not 1000 <= window_ms <= 30000:
        raise ValueError("window_ms must be an integer from 1000 to 30000")
    if type(beam_size) is not int or not 1 <= beam_size <= 10:
        raise ValueError("beam_size must be an integer from 1 to 10")
    if type(context_ms) is not int or not 0<=context_ms<window_ms:
        raise ValueError('context_ms must be nonnegative and smaller than the ASR window')
    audio = np.asarray(samples, dtype=np.float32)
    if audio.ndim != 1 or not len(audio) or not np.isfinite(audio).all():
        raise ValueError("samples must be a nonempty finite mono array")
    step = len(audio) if mode == "offline" else window_ms * SAMPLE_RATE // 1000
    events, windows, transcripts, ready_at = [], [], [], media_offset_ms
    total_compute_ms, rejected = 0, 0
    for index, begin in enumerate(range(0, len(audio), step)):
        end = min(begin + step, len(audio))
        context_begin=max(0,begin-context_ms*SAMPLE_RATE//1000) if mode=='causal' else begin
        context_start_ms=context_begin*1000//SAMPLE_RATE
        start_ms, end_ms = begin * 1000 // SAMPLE_RATE, math.ceil(end * 1000 / SAMPLE_RATE)
        started_at = max(media_offset_ms + end_ms, ready_at)
        before = clock()
        segments, info = recognizer.transcribe(audio[context_begin:end].copy(), language=language, beam_size=beam_size,
            temperature=0.0, condition_on_previous_text=False, vad_filter=True, word_timestamps=context_ms>0)
        provisional = []
        for segment in segments:  # fully iterate the lazy backend inside the measured interval
            text = str(getattr(segment, "text", "")).strip()
            raw_start, raw_end = getattr(segment, "start", None), getattr(segment, "end", None)
            if context_ms>0:
                # Own each word once, by midpoint in the newly received window.
                # Past audio supplies context but cannot repeat an old command.
                words=[w for w in (getattr(segment,'words',None) or [])
                       if start_ms <= context_start_ms+(w.start+w.end)*500 < end_ms]
                if not words:continue
                text=''.join(w.word for w in words).strip()
                raw_start,raw_end=words[0].start,words[-1].end
            if not text or not isinstance(raw_start, (int, float)) or not isinstance(raw_end, (int, float)) or not math.isfinite(raw_start) or not math.isfinite(raw_end):
                rejected += 1
                continue
            if re.search(r'(.{2,8})(?:\s*\1){5,}',text):
                rejected+=1;continue
            segment_start = max(context_start_ms, min(end_ms, context_start_ms + round(raw_start * 1000)))
            segment_end = max(context_start_ms, min(end_ms, context_start_ms + round(raw_end * 1000)))
            if segment_end <= segment_start:
                rejected += 1
                continue
            event = SpeechEvent(media_offset_ms + segment_start, media_offset_ms + segment_end,
                                media_offset_ms + end_ms, text)
            if extract_language:
                event = enrich_event(event, lexicon_path=lexicon_path)
            provisional.append(event)
            transcripts.append({"window_index": index, "start_ms": event.start_ms, "end_ms": event.end_ms,
                                "text": text, "avg_logprob": getattr(segment, "avg_logprob", None),
                                "no_speech_prob": getattr(segment, "no_speech_prob", None)})
        elapsed = clock() - before
        if not math.isfinite(elapsed) or elapsed < 0:
            raise ValueError("processing clock must be finite and monotonic")
        compute_ms = math.ceil(elapsed * 1000)
        total_compute_ms += compute_ms
        ready_at = started_at + compute_ms
        events.extend(replace(item, available_at_ms=ready_at) for item in provisional)
        windows.append({"window_index": index, "start_sample": begin, "end_sample": end,
                        "audio_start_ms": media_offset_ms + start_ms, "received_until_ms": media_offset_ms + end_ms,
                        "context_start_ms":media_offset_ms+context_start_ms,
                        "processing_start_ms": started_at, "available_at_ms": ready_at, "compute_ms": compute_ms,
                        "event_count": len(provisional), "detected_language": getattr(info, "language", language)})
    duration_ms = math.ceil(len(audio) * 1000 / SAMPLE_RATE)
    return ASRRun(events, windows, transcripts, {"stage": "D2-D4", "mode": mode, "sample_rate": SAMPLE_RATE,
        "duration_ms": duration_ms, "media_offset_ms": media_offset_ms, "window_ms": window_ms if mode == "causal" else None,
        "context_ms":context_ms,
        "timing_kind": "simulated_media_arrival_plus_measured_compute", "live_microphone": False,
        "total_compute_ms": total_compute_ms, "compute_rtf": total_compute_ms / duration_ms,
        "event_count": len(events), "discarded_segments": rejected,
        "semantic_annotations": "rule_extractor" if extract_language else "empty_no_extractor"})


def run_asr(path: str | Path, model_path: str | Path, **options) -> ASRRun:
    """Decode a local audio/video file and require an already-downloaded local model."""
    source, model = Path(path), Path(model_path)
    if not source.is_file():
        raise FileNotFoundError(f"media file does not exist: {source}")
    if not model.is_dir() or not (model / "model.bin").is_file():
        raise ValueError("local model missing; provide a Faster-Whisper CTranslate2 model directory")
    try:
        from faster_whisper import WhisperModel
        from faster_whisper.audio import decode_audio
        import av
    except ImportError as exc:
        raise ImportError("install requirements-d.txt for ASR") from exc
    device = options.pop("device", "cpu")
    compute_type = options.pop("compute_type", "int8")
    cpu_threads = options.pop("cpu_threads", 4)
    if type(cpu_threads) is not int or cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    with av.open(str(source)) as container:
        if not container.streams.audio:
            duration_ms=max(0,round((container.duration or 0)/1000))
            return ASRRun([],[],[],dict(stage='D2-D4',mode=options.get('mode','offline'),
                duration_ms=duration_ms,media_offset_ms=options.get('media_offset_ms',0),
                window_ms=options.get('window_ms',4000),event_count=0,discarded_segments=0,
                total_compute_ms=0,compute_rtf=0.,source_kind='media_without_audio_stream',
                source=str(source),model_path=str(model),device=device,compute_type=compute_type,
                timing_kind='no_audio_no_language_evidence'))
    model_instance = WhisperModel(str(model), device=device, compute_type=compute_type,
                                 cpu_threads=cpu_threads, local_files_only=True)
    result = transcribe_samples(decode_audio(str(source), sampling_rate=SAMPLE_RATE), model_instance, **options)
    result.metadata.update(source_kind="pretrained_asr_from_media", source=source.as_posix(), model_path=model.as_posix(),
                           device=device, compute_type=compute_type, cpu_threads=cpu_threads)
    return result

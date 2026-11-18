"""D's incremental file source and publication driver for LocalAgreement ASR."""

from dataclasses import replace
import math
from pathlib import Path
import time

from .asr import ASRRun, SAMPLE_RATE
from .events import require_ms
from .media_stream import inspect_media_clock, iter_media_windows
from .quality import POLICY


def recognize_windows(
    source, session, *, media_offset_ms=0, audio_offset_ms=0, pace=False,
    on_events=None, clock=time.perf_counter, sleep=time.sleep,
) -> ASRRun:
    """Run a stateful ASR session over arriving windows and publish each commit.

    Accelerated replay models arrival and measures compute. Paced replay waits
    for each window and measures elapsed wall time at in-memory publication.
    A callback receives committed events immediately, before file EOF.
    """
    require_ms(media_offset_ms, "media_offset_ms")
    require_ms(audio_offset_ms, "audio_offset_ms")
    if type(pace) is not bool:
        raise ValueError("pace must be boolean")
    epoch = clock()
    ready_ms = media_offset_ms
    total_compute = total_decode = 0
    total_samples = 0
    events, updates, transcripts = [], [], []
    iterator = iter(source)
    index = 0
    first_published_at = None

    def publish(batch, receive_ms, start_sample, end_sample, *, eof=False, decode_ms=0):
        nonlocal ready_ms, total_compute, first_published_at
        if pace:
            target = (receive_ms - media_offset_ms) / 1000
            while (remaining := target - (clock() - epoch)) > 0:
                sleep(min(remaining, 0.1))
        before = clock()
        ready_start = (max(receive_ms, media_offset_ms + math.ceil((before - epoch) * 1000))
                       if pace else max(receive_ms, ready_ms) + decode_ms)
        provisional = batch()
        after = clock()
        if not math.isfinite(after - before) or after < before:
            raise ValueError("processing clock must be finite and monotonic")
        compute_ms = math.ceil((after - before) * 1000)
        total_compute += compute_ms
        ready_ms = (max(receive_ms, ready_ms, media_offset_ms + math.ceil((after - epoch) * 1000))
                    if pace else ready_start + compute_ms)
        committed = [replace(event, available_at_ms=ready_ms) for event in provisional]
        for event in committed:
            if event.end_ms > receive_ms:
                raise ValueError("ASR event references audio beyond the received window")
            event.to_dict()
        events.extend(committed)
        update = dict(session.last_update)
        update.update({"window_index": index, "start_sample": start_sample, "end_sample": end_sample,
                       "received_until_ms": receive_ms, "processing_start_ms": ready_start,
                       "available_at_ms": ready_ms, "compute_ms": compute_ms,
                       "decode_ms": decode_ms,
                       "event_count": len(committed), "eof": eof,
                       "provisional_availability": False})
        updates.append(update)
        transcripts.extend({"window_index": index, "start_ms": event.start_ms,
                            "end_ms": event.end_ms, "text": event.text,
                            "commit_policy": "eof_flush" if eof else "local_agreement"}
                           for event in committed)
        if committed:
            if first_published_at is None:
                first_published_at = ready_ms
            if on_events is not None:
                on_events(committed)

    while True:
        before = clock()
        try:
            window = next(iterator)
        except StopIteration:
            break
        decode_ms = math.ceil(max(0, clock() - before) * 1000)
        total_decode += decode_ms
        if window.start_sample != total_samples or window.end_sample <= window.start_sample:
            raise ValueError("source windows must be nonempty and sample-contiguous")
        if len(window.samples) != window.end_sample - window.start_sample:
            raise ValueError("source sample count differs from its declared interval")
        total_samples = window.end_sample
        received = media_offset_ms + audio_offset_ms + math.ceil(total_samples * 1000 / SAMPLE_RATE)
        publish(lambda: session.push(window.samples, received_until_ms=received), received,
                window.start_sample, window.end_sample, decode_ms=decode_ms)
        index += 1
    if not total_samples:
        raise ValueError("source yielded no audio")
    import numpy as np
    received = media_offset_ms + audio_offset_ms + math.ceil(total_samples * 1000 / SAMPLE_RATE)
    publish(lambda: session.push(np.empty(0, dtype=np.float32), received_until_ms=received, finish=True),
            received, total_samples, total_samples, eof=True)
    duration = math.ceil(total_samples * 1000 / SAMPLE_RATE)
    return ASRRun(events, updates, {
        "stage": "D4-D6", "mode": "agreement", "sample_rate": SAMPLE_RATE,
        "duration_ms": duration, "media_offset_ms": media_offset_ms,
        "audio_offset_ms": audio_offset_ms, "event_count": len(events),
        "timing_kind": "wall_clock_file_replay" if pace else "simulated_media_arrival_plus_measured_compute",
        "timing_policy": "local_agreement_incremental", "paced": pace,
        "live_microphone": False, "full_file_predecode": False,
        "total_compute_ms": total_compute, "incremental_decode_ms": total_decode,
        "compute_rtf": total_compute / duration,
        "compute_and_decode_rtf": (total_compute + total_decode) / duration,
        "first_published_at_ms": first_published_at,
        "publication_clock": "in_memory_after_ASR_and_language; journal transport excluded",
        "eof_flushed_event_count": updates[-1]["event_count"],
        "quality_policy": POLICY,
        "rejected_quality_segments": sum(update.get("rejected_quality_segments", 0) for update in updates),
    }, transcripts)


def run_incremental_asr(
    path, model_path, *, step_ms=2000, buffer_ms=15000, media_offset_ms=0,
    language=None, beam_size=3, device="cpu", compute_type="int8", cpu_threads=4,
    extract_language=True, lexicon_path=None, pace=False, on_events=None,
) -> ASRRun:
    """Construct the upstream model and incremental input; no whole-file decode."""
    require_ms(media_offset_ms, "media_offset_ms")
    if type(step_ms) is not int or not 500 <= step_ms <= 5000:
        raise ValueError("step_ms must be an integer between 500 and 5000")
    if type(buffer_ms) is not int or buffer_ms < max(4000, step_ms * 2) or buffer_ms > 30000:
        raise ValueError("buffer_ms must be between max(4000, twice step_ms) and 30000")
    if type(cpu_threads) is not int or cpu_threads <= 0:
        raise ValueError("cpu_threads must be positive")
    model_path = Path(model_path)
    if not (model_path / "model.bin").is_file():
        raise ValueError("local model missing; run scripts/fetch_asr_model.py first")
    timing = inspect_media_clock(path)
    from faster_whisper import WhisperModel
    from .streaming import AgreementRecognizer
    import faster_whisper, ctranslate2, av
    before = time.perf_counter()
    model = WhisperModel(str(model_path), device=device, compute_type=compute_type,
                         cpu_threads=cpu_threads, local_files_only=True)
    load_ms = math.ceil((time.perf_counter() - before) * 1000)
    extractor = None
    if extract_language:
        from .language import enrich_event
        extractor = lambda event: enrich_event(event, lexicon_path=lexicon_path)
    session = AgreementRecognizer(
        model, language=language, beam_size=beam_size,
        media_offset_ms=media_offset_ms + timing["audio_offset_ms"],
        buffer_limit_ms=buffer_ms, extractor=extractor,
    )
    result = recognize_windows(
        iter_media_windows(path, step_ms=step_ms), session, media_offset_ms=media_offset_ms,
        audio_offset_ms=timing["audio_offset_ms"], pace=pace, on_events=on_events,
    )
    result.metadata.update({
        "source_kind": "pretrained_incremental_asr_from_media", "source": Path(path).as_posix(),
        "step_ms": step_ms, "buffer_ms": buffer_ms, "model_path": model_path.as_posix(),
        "model_load_ms": load_ms, "device": device, "compute_type": compute_type,
        "cpu_threads": cpu_threads, "beam_size": beam_size, "requested_language": language,
        "semantic_annotations": "rule_extractor" if extractor else "empty_no_extractor",
        "faster_whisper_version": faster_whisper.__version__,
        "ctranslate2_version": ctranslate2.__version__, "pyav_version": av.__version__,
        "media_clock": timing, "excluded_from_compute_rtf": ["model_load", "incremental_decode", "file_output"],
    })
    return result

"""Independent member-D input, pretrained ASR, and language-cue entry point."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import tempfile

from src.speech.subtitles import load_mock_events, read_events


def write_jsonl(path: Path, records) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w", encoding="utf-8", newline="\n", dir=path.parent, delete=False
        ) as stream:
            temp_path = Path(stream.name)
            for record in records:
                stream.write(json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n")
        os.replace(temp_path, path)
    finally:
        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description="ShowIntent D: 录音读取 / Mock 字幕 / 预训练 ASR")
    parser.add_argument("--backend", choices=["reader", "faster-whisper"], default="reader",
                        help="reader 保留 D1；faster-whisper 从音视频真实识别")
    parser.add_argument("--subtitles", type=Path, help="手写 .srt 或 .jsonl；不是 ASR")
    parser.add_argument("--audio", type=Path, help="录音或带音轨的视频；ASR 可直接读取 MP4")
    parser.add_argument("--output", type=Path, help="事件 JSONL，默认 outputs/speech_events.jsonl")
    parser.add_argument("--audio-log", type=Path, help="录音分块日志，默认 outputs/audio_chunks.jsonl")
    parser.add_argument("--simulated-delay-ms", type=int, default=200, help="Mock 延迟，非实测 ASR 延迟")
    parser.add_argument("--media-offset-ms", type=int, default=0, help="音频起点在共同媒体时间轴的位置")
    parser.add_argument("--block-ms", type=int, default=100, help="录音分块时长，毫秒")
    parser.add_argument("--validate-events", type=Path, help="只校验现有事件文件")
    parser.add_argument("--model", type=Path, default=Path("models/faster-whisper-base"))
    parser.add_argument("--mode", choices=["offline", "causal", "agreement"], default="offline")
    parser.add_argument("--window-ms", type=int, default=4000)
    parser.add_argument("--step-ms", type=int, default=2000, help="agreement 增量输入步长")
    parser.add_argument("--buffer-ms", type=int, default=15000, help="agreement 音频缓存上限")
    parser.add_argument("--pace", action="store_true", help="agreement 按墙钟等待音频到达，仍是文件回放")
    parser.add_argument("--journal", type=Path, help="agreement 逐步发布事件，新建 JSONL 文件，不覆盖旧运行")
    parser.add_argument("--language", help="如 zh/en；缺省自动检测")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--compute-type", default="int8")
    parser.add_argument("--cpu-threads", type=int, default=4)
    cues = parser.add_mutually_exclusive_group()
    cues.add_argument("--extract-cues", action="store_true", help="为 Mock 字幕启用 D3 自动提取；ASR 默认启用")
    cues.add_argument("--no-language-cues", action="store_true", help="ASR 仅转写，不提取语言线索")
    parser.add_argument("--lexicon", type=Path, help="可选语言规则配置")
    args = parser.parse_args(argv)
    if args.validate_events:
        if args.subtitles or args.audio or args.output or args.audio_log or args.backend != "reader" or args.journal or args.pace:
            parser.error("--validate-events cannot be combined with input/output options")
        try:
            events = read_events(args.validate_events)
            print(f"Valid SpeechEvent JSONL: {len(events)} event(s)")
            return 0
        except (OSError, ValueError, TypeError) as exc:
            parser.error(str(exc))
    if not args.subtitles and not args.audio:
        parser.error("provide --subtitles and/or --audio (D1 does not transcribe audio)")
    is_asr = args.backend == "faster-whisper"
    if args.mode == "agreement" and not is_asr:
        parser.error("--mode agreement requires --backend faster-whisper")
    if is_asr and (not args.audio or args.subtitles or args.audio_log):
        parser.error("faster-whisper requires --audio and cannot combine --subtitles/--audio-log")
    if (args.journal or args.pace) and (not is_asr or args.mode != "agreement"):
        parser.error("--journal/--pace require --backend faster-whisper --mode agreement")
    if args.output and not args.subtitles and not is_asr:
        parser.error("--output requires --subtitles; audio alone produces --audio-log")
    if args.audio_log and not args.audio:
        parser.error("--audio-log requires --audio")
    output = args.output or Path("outputs/speech_events.jsonl")
    meta_path = output.with_suffix(".meta.json")
    audio_log = args.audio_log or Path("outputs/audio_chunks.jsonl")
    transcript_path = output.with_suffix(".transcript.jsonl")
    windows_path = output.with_suffix(".windows.jsonl")
    inputs = {p.resolve() for p in (args.subtitles, args.audio, args.lexicon) if p}
    destinations = ([output, meta_path, transcript_path, windows_path] if is_asr else
                    ([output, meta_path] if args.subtitles else []) + ([audio_log] if args.audio else []))
    if args.journal:
        destinations += [args.journal, args.journal.with_suffix(".status.json")]
    if len({p.resolve() for p in destinations}) != len(destinations):
        parser.error("output paths must be distinct")
    if any(p.resolve() in inputs for p in destinations):
        parser.error("output paths must not overwrite input files")
    if is_asr and any(p.resolve().is_relative_to(args.model.resolve()) for p in destinations):
        parser.error("ASR output paths must not be inside the model directory")
    try:
        # Validate options even in audio-only mode.
        from src.speech.events import require_ms
        require_ms(args.simulated_delay_ms, "simulated_delay_ms")
        require_ms(args.media_offset_ms, "media_offset_ms")
        if args.block_ms <= 0:
            raise ValueError("block_ms must be positive")
        if is_asr:
            from src.speech.asr import run_asr
            from contextlib import nullcontext
            model_identity = {}
            model_manifest = Path(__file__).parent / "config/asr_model.json"
            if model_manifest.is_file():
                pinned = json.loads(model_manifest.read_text(encoding="utf-8"))
                expected_path = Path(__file__).parent / pinned["local_directory"]
                if args.model.resolve() == expected_path.resolve():
                    actual = hashlib.sha256((args.model / "model.bin").read_bytes()).hexdigest()
                    if actual != pinned["files"]["model.bin"]["sha256"]:
                        raise ValueError("model bytes differ from pinned model manifest")
                    model_identity = {"model_repository": pinned["repository"],
                                      "model_revision": pinned["revision"], "model_sha256": actual}
            journal_context = nullcontext(None)
            if args.journal:
                from src.speech.journal import EventJournal
                journal_context = EventJournal(args.journal, status_writer=write_jsonl)
            with journal_context as journal:
                if args.mode == "agreement":
                    from src.speech.incremental import run_incremental_asr
                    result = run_incremental_asr(
                        args.audio, args.model, step_ms=args.step_ms, buffer_ms=args.buffer_ms,
                        media_offset_ms=args.media_offset_ms, language=args.language,
                        beam_size=args.beam_size, device=args.device, compute_type=args.compute_type,
                        cpu_threads=args.cpu_threads, extract_language=not args.no_language_cues,
                        lexicon_path=args.lexicon, pace=args.pace,
                        on_events=journal.append if journal else None,
                    )
                else:
                    result = run_asr(
                        args.audio, args.model, mode=args.mode, window_ms=args.window_ms,
                        media_offset_ms=args.media_offset_ms, language=args.language,
                        beam_size=args.beam_size, device=args.device, compute_type=args.compute_type,
                        cpu_threads=args.cpu_threads, extract_language=not args.no_language_cues,
                        lexicon_path=args.lexicon,
                    )
            result.metadata.update(model_identity)
            result.metadata["source_sha256"] = hashlib.sha256(args.audio.read_bytes()).hexdigest()
            profile = args.lexicon or Path(__file__).parent / "config/speech_lexicon.json"
            result.metadata["lexicon_sha256"] = (None if args.no_language_cues else
                                                hashlib.sha256(profile.read_bytes()).hexdigest())
            write_jsonl(output, (event.to_dict() for event in result.events))
            write_jsonl(transcript_path, result.transcripts)
            write_jsonl(windows_path, result.windows)
            write_jsonl(meta_path, [result.metadata])
            print(f"ASR {args.mode}: {len(result.events)} events; compute RTF={result.metadata['compute_rtf']:.3f}; {output}")
            return 0
        events = load_mock_events(
            args.subtitles, simulated_delay_ms=args.simulated_delay_ms,
            media_offset_ms=args.media_offset_ms,
        ) if args.subtitles else []
        if args.extract_cues and args.subtitles:
            from src.speech.language import enrich_event
            events = [enrich_event(event, lexicon_path=args.lexicon) for event in events]
        audio_info = None
        if args.audio:
            # Lazy import: subtitle fixtures work without SoundFile or NumPy.
            from src.speech.audio import inspect_audio, iter_audio_chunks
            audio_info = inspect_audio(args.audio)
            if any(e.end_ms > args.media_offset_ms + audio_info.duration_ms for e in events):
                raise ValueError("subtitle extends beyond audio duration; check the shared time origin")
            def chunk_records():
                for chunk in iter_audio_chunks(
                    args.audio, block_ms=args.block_ms, media_offset_ms=args.media_offset_ms
                ):
                    yield {"start_ms": chunk.start_ms, "end_ms": chunk.end_ms,
                           "start_sample": chunk.start_sample, "end_sample": chunk.end_sample,
                           "sample_rate": chunk.sample_rate, "frames": len(chunk.samples),
                           "channels": chunk.samples.shape[1]}
            write_jsonl(audio_log, chunk_records())
            print(f"Audio decoded: {audio_info.frame_count} frames @ {audio_info.sample_rate} Hz; {audio_log}")
        if args.subtitles:
            write_jsonl(output, (e.to_dict() for e in events))
            metadata = {
                "stage": "D1", "source_kind": "handwritten_subtitle_mock",
                "timing_kind": "simulated_availability_not_measured_asr",
                "source": args.subtitles.as_posix(),
                "source_sha256": hashlib.sha256(args.subtitles.read_bytes()).hexdigest(),
                "simulated_delay_ms": args.simulated_delay_ms,
                "media_offset_ms": args.media_offset_ms, "event_count": len(events),
                "semantic_annotations": "rule_extractor" if args.extract_cues else "manual_passthrough_or_empty_no_extractor",
                "audio": audio_info.to_dict() if audio_info else None,
            }
            if args.extract_cues:
                profile = args.lexicon or Path(__file__).parent / "config/speech_lexicon.json"
                metadata["lexicon_sha256"] = hashlib.sha256(profile.read_bytes()).hexdigest()
            # One JSON object with a newline is valid JSON and keeps writes atomic.
            write_jsonl(meta_path, [metadata])
            print(f"Mock SpeechEvents: {len(events)}; {output}; provenance: {meta_path}")
        return 0
    except ImportError as exc:
        parser.error(f"dependencies unavailable: {exc}; install requirements-asr.txt for ASR or requirements.txt for reader")
    except (OSError, ValueError, TypeError, RuntimeError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    raise SystemExit(main())

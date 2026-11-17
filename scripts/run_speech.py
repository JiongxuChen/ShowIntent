"""Run the optional member-D ASR replay and write portable JSONL evidence."""

import argparse
import json
from pathlib import Path

from src.speech.asr import run_asr


def write_jsonl(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="ShowIntent D: pretrained ASR file replay")
    parser.add_argument("--audio", required=True, type=Path)
    parser.add_argument("--model", required=True, type=Path, help="local Faster-Whisper CTranslate2 model directory")
    parser.add_argument("--out", required=True, type=Path, help="new SpeechEvent JSONL path")
    parser.add_argument("--mode", choices=("offline", "causal"), default="offline")
    parser.add_argument("--window-ms", type=int, default=4000)
    parser.add_argument("--media-offset-ms", type=int, default=0)
    parser.add_argument("--language")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--no-language-cues", action="store_true")
    args = parser.parse_args()
    if args.out.exists():
        parser.error("--out already exists; choose a new path so a prior run is preserved")
    result = run_asr(args.audio, args.model, mode=args.mode, window_ms=args.window_ms,
        media_offset_ms=args.media_offset_ms, language=args.language, beam_size=args.beam_size,
        extract_language=not args.no_language_cues)
    write_jsonl(args.out, (event.to_dict() for event in result.events))
    write_jsonl(args.out.with_suffix(".windows.jsonl"), result.windows)
    args.out.with_suffix(".meta.json").write_text(json.dumps(result.metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(result.events)} SpeechEvents -> {args.out}")


if __name__ == "__main__":
    main()

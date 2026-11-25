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
    parser.add_argument("--mode", choices=("offline", "causal"), default="causal")
    parser.add_argument("--window-ms", type=int, default=4000)
    parser.add_argument("--media-offset-ms", type=int, default=0)
    parser.add_argument("--language")
    parser.add_argument("--beam-size", type=int, default=3)
    parser.add_argument("--no-language-cues", action="store_true")
    parser.add_argument('--device', choices=('cpu', 'cuda'), default='cpu')
    parser.add_argument('--compute-type', default='int8')
    parser.add_argument('--cpu-threads', type=int, default=4)
    parser.add_argument('--context-ms', type=int, default=0,help='Past-audio context; words are committed once by media time')
    args = parser.parse_args()
    artifacts = (args.out, args.out.with_suffix('.windows.jsonl'), args.out.with_suffix('.meta.json'))
    if len({p.resolve() for p in artifacts}) != 3:
        parser.error('output paths must be distinct; use a .jsonl extension for --out')
    if any(p.exists() or p.resolve() == args.audio.resolve() for p in artifacts):
        parser.error('an output exists or aliases the input; choose a new --out path')
    result = run_asr(args.audio, args.model, mode=args.mode, window_ms=args.window_ms,
        media_offset_ms=args.media_offset_ms, language=args.language, beam_size=args.beam_size,
        extract_language=not args.no_language_cues, device=args.device,
        compute_type=args.compute_type, cpu_threads=args.cpu_threads,context_ms=args.context_ms)
    write_jsonl(args.out, (event.to_dict() for event in result.events))
    write_jsonl(args.out.with_suffix(".windows.jsonl"), result.windows)
    with args.out.with_suffix('.meta.json').open('x', encoding='utf-8') as stream:
        stream.write(json.dumps(result.metadata, ensure_ascii=False, indent=2) + '\n')
    print(f"Wrote {len(result.events)} SpeechEvents -> {args.out}")


if __name__ == "__main__":
    main()

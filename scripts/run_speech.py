"""Run the optional member-D ASR replay and write portable JSONL evidence."""

from pathlib import Path
import sys

from run_speech import main as run_main


def main():
    args = list(sys.argv[1:])
    if '--out' in args:
        args[args.index('--out')] = '--output'
    if '--output' in args:
        index = args.index('--output')
        if index + 1 < len(args):
            path = Path(args[index + 1])
            destinations = [path, *(path.with_suffix(suffix) for suffix in ('.meta.json', '.windows.jsonl', '.transcript.jsonl'))]
            if any(item.exists() for item in destinations):
                raise SystemExit('output or sidecar exists; choose a new --out path')
    return run_main(([] if '--backend' in args else ['--backend', 'faster-whisper']) + args)


if __name__ == "__main__":
    raise SystemExit(main())

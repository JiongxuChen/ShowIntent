"""Generate a 12-second synthetic test tone, not human speech or ASR evidence."""

import argparse
import math
from pathlib import Path
import struct
import wave


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("outputs/audio_probe.wav"))
    args = parser.parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    sample_rate = 16000
    with wave.open(str(args.output), "wb") as stream:
        stream.setnchannels(1)
        stream.setsampwidth(2)
        stream.setframerate(sample_rate)
        for second in range(12):
            samples = [int(1000 * math.sin(2 * math.pi * 440 * (second + n / sample_rate)))
                       for n in range(sample_rate)]
            stream.writeframes(struct.pack(f"<{sample_rate}h", *samples))
    print(f"Synthetic tone (not speech): {args.output}")


if __name__ == "__main__":
    main()

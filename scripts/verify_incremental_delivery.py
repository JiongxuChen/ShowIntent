"""Verify events become readable before an actual paced file replay completes."""

import argparse
import json
import os
from pathlib import Path
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audio", default="data/public/media/mandarin_bottle_air_60s_105s.wav")
    parser.add_argument("--language", default="zh")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "outputs/paced_delivery")
    args = parser.parse_args()
    folder = args.output_dir.resolve()
    folder.mkdir(parents=True, exist_ok=True)
    journal = folder / "events.journal.jsonl"
    status = journal.with_suffix(".status.json")
    if journal.exists() or status.exists():
        parser.error("delivery test already exists; choose a fresh --output-dir")
    command = [sys.executable, str(ROOT / "run_speech.py"), "--backend", "faster-whisper",
               "--audio", args.audio, "--language", args.language, "--mode", "agreement",
               "--pace", "--journal", str(journal), "--output", str(folder / "events.jsonl")]
    started = time.perf_counter()
    observation = None
    with (folder / "process.stdout.txt").open("w", encoding="utf-8") as stdout, \
         (folder / "process.stderr.txt").open("w", encoding="utf-8") as stderr:
        child = subprocess.Popen(command, cwd=ROOT, stdout=stdout, stderr=stderr,
                                 creationflags=subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0)
        try:
            while child.poll() is None:
                elapsed = time.perf_counter() - started
                if elapsed > 180:
                    raise TimeoutError("paced delivery check exceeded 180 seconds")
                if observation is None and journal.exists() and status.exists():
                    # A concurrent reader must ignore an unfinished final line.
                    payload = journal.read_bytes()
                    complete = payload[:payload.rfind(b"\n") + 1]
                    lines = [json.loads(line) for line in complete.decode("utf-8").splitlines() if line.strip()]
                    phase = json.loads(status.read_text(encoding="utf-8")).get("status")
                    if lines and phase == "RUNNING":
                        observation = {"readable_before_process_exit": True, "journal_status": phase,
                                       "observed_elapsed_since_spawn_ms": round(elapsed * 1000),
                                       "readable_event_count": len(lines), "paced": True}
                        print(json.dumps(observation), flush=True)
                time.sleep(0.1)
            if child.returncode != 0:
                raise RuntimeError(f"recognition failed; see {folder / 'process.stderr.txt'}")
        finally:
            if child.poll() is None:
                child.terminate()
                child.wait(timeout=10)
    if observation is None:
        raise AssertionError("no event was observed while the recognizer was still running")
    phase = json.loads(status.read_text(encoding="utf-8"))
    if phase["status"] != "COMPLETE":
        raise AssertionError("journal did not complete")
    canonical = (folder / "events.jsonl").read_text(encoding="utf-8")
    if journal.read_text(encoding="utf-8") != canonical:
        raise AssertionError("journal differs from final canonical event stream")
    observation.update({"final_event_count": phase["event_count"], "final_status": phase["status"],
                        "journal_matches_final_export": True,
                        "process_elapsed_ms": round((time.perf_counter() - started) * 1000)})
    (folder / "observation.json").write_text(json.dumps(observation, indent=2) + "\n", encoding="utf-8")
    print(f"Verified paced incremental delivery: {folder / 'observation.json'}", flush=True)


if __name__ == "__main__":
    main()

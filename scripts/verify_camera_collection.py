"""Independently audit exported video timing, copied audio and the applied shot logs."""
import argparse
import csv
from fractions import Fraction
from hashlib import sha256
import json
from pathlib import Path

import av

from src.camera.evaluate import measure
from src.camera.run import load_jsonl, new_outputs
from src.perception.timing import load_timestamps


def inspect_media(path):
    """Scan encoded packets (no transcoding); decode the first image for orientation."""
    with av.open(str(path)) as src:
        frame = next(src.decode(video=0))
        w, h = frame.width, frame.height
        if frame.rotation % 180:
            w, h = h, w
        rotation = frame.rotation
    with av.open(str(path)) as src:
        video = src.streams.video[0]
        audio = next(iter(src.streams.audio), None)
        streams = [video] + ([audio] if audio else [])
        pts = []
        audio_hash = sha256()
        audio_timing = []
        for packet in src.demux(streams):
            if packet.pts is None:
                continue
            if packet.stream == video:
                pts.append(packet.pts * packet.time_base)
            else:
                audio_hash.update(bytes(packet))
                audio_timing.append((packet.pts*packet.time_base, packet.duration*packet.time_base))
        return dict(pts=sorted(pts), size=(w,h), rotation=rotation, audio_timing=audio_timing,
                    audio_hash=audio_hash.hexdigest() if audio else None,
                    duration=video.duration*video.time_base)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True, type=Path)
    args = parser.parse_args()
    reports = {}
    stats = {}
    for manifest in sorted(Path("evaluation").glob("*/dataset_manifest.csv")):
        scene = manifest.parent.name
        with manifest.open(encoding="utf8") as src:
            rows = list(csv.DictReader(src))
        for meta in rows:
            clip = meta["clip_id"]
            folder = args.root/clip
            shots = load_jsonl(folder/"shots.jsonl")
            expected = load_timestamps(Path("outputs/cache")/scene/"timestamps"/f"{clip}.json")
            if [r["timestamp_ms"] for r in shots] != expected or [r["packet_timestamp_ms"] for r in shots] != expected:
                raise ValueError(f"Shot or A/B clock differs from official PTS for {clip}")
            stats[clip] = measure(shots)
            if not Path(meta["source_path"]).exists():
                reports[clip] = {"frames": len(shots), "status": "source_missing"}
                continue
            original = inspect_media(meta["source_path"])
            output = inspect_media(folder/"camera.mp4")
            if original["pts"] != output["pts"] or [int(t*1000) for t in output["pts"]] != expected:
                raise ValueError(f"Exported video timestamps or frame count differ for {clip}")
            if any(original[k] != output[k] for k in ["audio_timing", "audio_hash"]):
                raise ValueError(f"Audio payload or clock changed for {clip}")
            if output["rotation"] or Fraction(*output["size"]) != Fraction(*original["size"]):
                raise ValueError(f"Orientation or aspect changed for {clip}")
            delta = float(abs(output["duration"]-original["duration"])*1000)
            if delta > 1000/24:
                raise ValueError(f"Video stream duration differs by over one 24fps frame for {clip}")
            reports[clip] = dict(status="verified", frames=len(shots), audio_sha256=original["audio_hash"],
                                 duration_difference_ms=round(delta,5), size=output["size"])
    with new_outputs(args.root/"media_audit.json", args.root/"shot_metrics.json") as (report, metrics):
        report.write_text(json.dumps(reports, indent=2)+"\n", encoding="utf8")
        metrics.write_text(json.dumps(stats, indent=2)+"\n", encoding="utf8")
    print(f"Verified {sum(r['status']=='verified' for r in reports.values())} movies and {len(stats)} shot traces")


if __name__ == "__main__":
    main()

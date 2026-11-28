"""Render and score the team's published showintent packets using the dataset manifests."""
import argparse
import csv
import json
from pathlib import Path

from src.camera.evaluate import measure
from src.camera.run import load_jsonl, load_pair, new_outputs, run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--clips", nargs="+")
    parser.add_argument("--render", action="store_true", help="Render each available source")
    args = parser.parse_args()
    requested = set(args.clips or [])
    clips = []
    for manifest in sorted(Path("evaluation").glob("*/dataset_manifest.csv")):
        with manifest.open(encoding="utf8") as src:
            for row in csv.DictReader(src):
                if not requested or row["clip_id"] in requested:
                    clips.append((manifest.parent.name, row))
    if requested - {row["clip_id"] for _, row in clips}:
        parser.error("unknown clip")
    metrics = {}
    for scene, row in clips:
        clip = row["clip_id"]
        video = Path(row["source_path"])
        per = Path(row["perception_path"])
        intent = Path("outputs/results")/scene/clip/"showintent.jsonl"
        target = args.out/clip
        log = target/"shots.jsonl"
        stats = {}
        if args.render and video.exists():
            from src.camera.render import export
            if row.get("video_size_bytes") and video.stat().st_size != int(row["video_size_bytes"]):
                raise ValueError(f"Wrong source size for {clip}")
            stats = export(video, load_pair(per, intent), log, target/"camera.mp4")
        else:
            run(per, intent, log)
        metrics[clip] = dict(scene=scene, rendered=bool(stats), render=stats,
                             **measure(load_jsonl(log)))
        print(f"{clip}: {metrics[clip]['frames']} frames, {stats or 'packets only'}", flush=True)
    with new_outputs(args.out/"metrics.json") as (path,):
        path.write_text(json.dumps(metrics, indent=2)+"\n", encoding="utf8")


if __name__ == "__main__":
    main()

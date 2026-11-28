"""Produce a shareable two-object camera demo from the team's synthetic A trajectory."""
import argparse
from fractions import Fraction
import json
from pathlib import Path

import av
import cv2
import numpy as np

from src.camera.render import export
from src.camera.run import new_outputs
from src.intent.mock_scenarios import make_scenarios
from src.intent.temporal import TemporalIntentStateMachine


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    rate, w, h = 25, 640, 360
    packets = next(s.packets for s in make_scenarios(rate) if s.name == "two_targets")
    policy = TemporalIntentStateMachine()
    pairs = [(p, policy.update(p)) for p in packets]
    src = args.out/"source.mp4"
    with new_outputs(src, args.out/"perception.jsonl", args.out/"intent.jsonl") as (video, per, intent):
        with av.open(str(video), "w") as dest:
            stream = dest.add_stream("libx264", rate=rate)
            stream.width, stream.height, stream.pix_fmt = w, h, "yuv420p"
            for fid, (p, _) in enumerate(pairs):
                image = np.full((h, w, 3), (52, 49, 42), np.uint8)
                for obj in p["objects"]:
                    x, y, x2, y2 = (int(v*s) for v, s in zip(obj["bbox_norm"], (w,h,w,h)))
                    color = {"cup": (205, 160, 70), "apple": (60, 68, 210), "bottle": (102, 167, 87)}[obj["class_name"]]
                    cv2.rectangle(image, (x,y), (x2,y2), color, -1)
                    cv2.putText(image, obj["class_name"], (x,y-4), cv2.FONT_HERSHEY_SIMPLEX, .45, (225,225,225), 1)
                frame = av.VideoFrame.from_ndarray(image, format="bgr24")
                frame.pts, frame.time_base = fid, Fraction(1, rate)
                for packet in stream.encode(frame):
                    dest.mux(packet)
            for packet in stream.encode():
                dest.mux(packet)
        for path, col in [(per, 0), (intent, 1)]:
            path.write_text("".join(json.dumps(pair[col])+"\n" for pair in pairs), encoding="utf8")
    print(export(src, pairs, args.out/"shots.jsonl", args.out/"camera.mp4", hud=True))


if __name__ == "__main__":
    main()

"""Interactive camera preview using A's detector and B's visual streaming policy."""
import argparse
from concurrent.futures import ThreadPoolExecutor
from contextlib import ExitStack
import json
from pathlib import Path
from time import perf_counter, sleep

import cv2

from .controller import CameraController
from .evaluate import describe
from .render import compare_views, output_size, render_frame
from .run import new_outputs, shot_at
from src.perception.run import apply_rotation, resolve_rotation


def visual_pipeline(weights, device, imgsz):
    from src.perception.closed_set import ClosedSetModels
    from src.streaming.stream_perception import EnhancedStreamPerception
    from src.intent.streaming import SparseVisualIntentStateMachine
    import numpy as np
    import torch

    if device == "cpu":
        torch.set_num_threads(min(4, torch.get_num_threads()))
    models = ClosedSetModels(weights, .25, True, backend="yolo", device=device)
    vision = EnhancedStreamPerception(models, imgsz)
    config = Path(__file__).resolve().parents[2]/"configs"/"presentation_intent.json"
    policy = SparseVisualIntentStateMachine(**json.loads(config.read_text(encoding="utf8")))
    # Exercise lazy model initialization before starting the media clock.
    dummy = np.zeros((360, 640, 3), dtype=np.uint8)
    models.detect_objects(dummy, imgsz)
    models.detect_hands(dummy)

    def infer(ts, fid, frame):
        per = vision.update(ts, fid, frame)
        return per, policy.update(per)
    return infer


def preview(source, infer, *, out, headless=False, seconds=None, detect_hz=10, force_wide=False):
    if detect_hz <= 0 or (seconds is not None and seconds <= 0):
        raise ValueError("duration and detection rate must be positive")
    webcam = isinstance(source, int)
    with new_outputs(out, out.with_suffix(".metrics.json")) as (log_path, report_path), ExitStack() as stack:
        cap = cv2.VideoCapture(source)
        stack.callback(cap.release)
        if not cap.isOpened():
            raise RuntimeError(f"Cannot open capture source {source}")
        cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
        rotation, _ = resolve_rotation(cap.get(cv2.CAP_PROP_ORIENTATION_META), "auto")
        if not headless:
            stack.callback(cv2.destroyAllWindows)
        log = stack.enter_context(log_path.open("w", encoding="utf8"))
        pool = stack.enter_context(ThreadPoolExecutor(max_workers=1))
        controller = CameraController()
        controller.set_manual_override(force_wide)
        pending = None
        per = {"objects": []}
        intent = {"timestamp_ms": 0, "state": "NONE"}
        clock = perf_counter()
        frames = detected = 0
        last_submit = -float("inf")
        ages = []
        first_pts = None
        input_fps = cap.get(cv2.CAP_PROP_FPS) or 30.
        compare = False
        window = "ShowIntent | W wide | P pause | R auto | V compare | Q quit"
        if not headless:
            cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        while True:
            ok, frame = cap.read()
            if not ok:
                if webcam:
                    raise RuntimeError("Camera stopped delivering frames")
                break
            if not webcam:
                pts = cap.get(cv2.CAP_PROP_POS_MSEC)
                if first_pts is None:
                    first_pts = pts
                due = (pts-first_pts)/1000 if pts > first_pts else frames/input_fps
                sleep(max(0., clock + due - perf_counter()))
            now = perf_counter()
            ts = int((now-clock)*1000)
            if seconds is not None and ts >= seconds*1000:
                break
            frame = apply_rotation(frame, rotation)
            h, w = frame.shape[:2]
            size = output_size(w, h, None, None)
            if pending is not None and pending.done():
                per, intent = pending.result()
                detected += 1
                ages.append(ts-per["timestamp_ms"])
                pending = None
            if pending is None and ts-last_submit >= 1000/detect_hz:
                # No queue of stale frames. The worker owns a separate downsampled image.
                scale = min(1., 960/max(w, h))
                sample = cv2.resize(frame, (max(1, round(w*scale)), max(1, round(h*scale))))
                pending = pool.submit(infer, ts, frames, sample)
                last_submit = ts
            shot = shot_at(controller, ts, per, intent)
            canvas = render_frame(frame, shot, size, hud=True)
            if not headless:
                cv2.imshow(window, compare_views(frame, canvas) if compare else canvas)
                key = cv2.waitKey(1) & 0xff
                if key == ord("v"):
                    compare = not compare
                elif key == ord("w"):
                    controller.set_manual_override(True)
                elif key == ord("p"):
                    controller.set_paused(not controller.paused)
                elif key == ord("r"):
                    controller.set_manual_override(False)
                    controller.set_paused(False)
                elif key in (27, ord("q")):
                    break
            log.write(json.dumps(shot)+"\n")
            frames += 1
        elapsed = perf_counter()-clock
        if not frames:
            raise RuntimeError("No preview frames")
        if pending is not None:
            pending.result()  # surface late worker errors; unpublished results do not count
        stats = dict(source="webcam" if webcam else "paced_file", headless=headless,
                     output_frames=frames, received_detections=detected, elapsed_s=round(elapsed, 3),
                     output_fps=round(frames/elapsed, 2), received_detection_hz=round(detected/elapsed, 2),
                     observation_to_consumption_ms=describe(ages))
        report_path.write_text(json.dumps(stats, indent=2)+"\n", encoding="utf8")
    return stats


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", default="0", help="camera index or paced local video")
    parser.add_argument("--weights", type=Path, default=Path("models/yolo11n.pt"))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--detect-hz", type=float, default=10.)
    parser.add_argument("--seconds", type=float)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--force-wide", action="store_true")
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    infer = visual_pipeline(args.weights, args.device, args.imgsz)
    source = int(args.source) if args.source.isdigit() else args.source
    print(json.dumps(preview(source, infer, out=args.out, headless=args.headless,
                             seconds=args.seconds, detect_hz=args.detect_hz, force_wide=args.force_wide)))


if __name__ == "__main__":
    main()

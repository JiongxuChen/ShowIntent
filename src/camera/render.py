"""Video rendering helpers for the C module preview output.

The crop is applied to frames decoded from the original video. Container
rotation metadata is applied exactly once (same rule as module A) so crop
coordinates — defined on the upright frame — line up with the pixels.
"""

from __future__ import annotations

import cv2

from src.perception.run import apply_rotation, resolve_rotation


def crop_frame(frame, crop_norm: list[float]):
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = crop_norm
    px1, py1 = int(x1 * width), int(y1 * height)
    px2, py2 = max(px1 + 1, int(x2 * width)), max(py1 + 1, int(y2 * height))
    return frame[py1:py2, px1:px2]


def draw_hud(canvas, shot: dict, target_bbox_norm: list[float] | None, crop_norm: list[float]) -> None:
    height, width = canvas.shape[:2]
    if target_bbox_norm is not None:
        cx1, cy1, cx2, cy2 = crop_norm
        cw, ch = max(cx2 - cx1, 1e-6), max(cy2 - cy1, 1e-6)
        x1 = int((target_bbox_norm[0] - cx1) / cw * width)
        y1 = int((target_bbox_norm[1] - cy1) / ch * height)
        x2 = int((target_bbox_norm[2] - cx1) / cw * width)
        y2 = int((target_bbox_norm[3] - cy1) / ch * height)
        if x2 > 0 and y2 > 0 and x1 < width and y1 < height:
            cv2.rectangle(canvas, (max(0, x1), max(0, y1)), (min(width, x2), min(height, y2)), (0, 220, 0), 2)
    cv2.putText(canvas, f"{shot['mode']} {shot.get('target_id')} zoom {shot['zoom']:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
    cv2.putText(canvas, f"{shot['mode']} {shot.get('target_id')} zoom {shot['zoom']:.1f}", (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)


def render_preview(
    video_path,
    shot_packets: list[dict],
    out_path,
    objects_by_frame: dict[int, list[dict]] | None = None,
    out_width: int = 1280,
    out_height: int = 720,
    rotate: str = "auto",
) -> int:
    """Write a cropped preview video following ShotPacket crops. Returns frame count."""
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open {video_path}")
    rotation_code, _ = resolve_rotation(capture.get(cv2.CAP_PROP_ORIENTATION_META), rotate)
    source_fps = capture.get(cv2.CAP_PROP_FPS) or 30.0
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), source_fps, (out_width, out_height))
    if not writer.isOpened():
        capture.release()
        raise RuntimeError(f"Could not create {out_path}")
    written = 0
    try:
        for shot in shot_packets:
            ok, frame = capture.read()
            if not ok:
                break
            frame = apply_rotation(frame, rotation_code)
            canvas = crop_frame(frame, shot["crop_norm"])
            canvas = cv2.resize(canvas, (out_width, out_height), interpolation=cv2.INTER_AREA)
            target_bbox = None
            if objects_by_frame is not None and shot.get("target_id"):
                obj = next((o for o in objects_by_frame.get(shot.get("frame_id", -1), []) if o.get("track_id") == shot["target_id"]), None)
                if obj is not None:
                    target_bbox = obj["bbox_norm"]
            draw_hud(canvas, shot, target_bbox, shot["crop_norm"])
            writer.write(canvas)
            written += 1
    finally:
        capture.release()
        writer.release()
    return written

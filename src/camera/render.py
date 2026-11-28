"""Render upright frames once, using their original presentation timestamps."""
from __future__ import annotations

import json
from math import ceil
from tempfile import TemporaryDirectory
from pathlib import Path
from time import perf_counter

import cv2
import av
import numpy as np

from src.perception.run import apply_rotation, resolve_rotation
from .controller import CameraController
from .run import new_outputs, shot_at


def crop_frame(frame, crop_norm):
    h, w = frame.shape[:2]
    x1, y1, x2, y2 = crop_norm
    return frame[int(y1*h):ceil(y2*h), int(x1*w):ceil(x2*w)]


def draw_hud(canvas, shot, target_bbox_norm, crop_norm):
    h, w = canvas.shape[:2]
    if target_bbox_norm:
        left, top, right, bottom = crop_norm
        x1, y1, x2, y2 = target_bbox_norm
        points = (int((x1-left)*w/(right-left)), int((y1-top)*h/(bottom-top))), (
            int((x2-left)*w/(right-left)), int((y2-top)*h/(bottom-top)))
        cv2.rectangle(canvas, *points, (20, 220, 20), 2)
    label = f"{shot['mode']}  {shot.get('target_id') or '-'}  {shot['zoom']:.2f}x"
    cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (10, 10, 10), 4)
    cv2.putText(canvas, label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 1)


def sharpen_focus(canvas, zoom):
    """Boost luminance detail gradually as the camera moves into a close-up."""
    strength = min(1., max(0., (zoom - 1.) * 2))
    if strength == 0:
        return canvas
    luma = cv2.cvtColor(canvas, cv2.COLOR_BGR2GRAY)
    smooth = cv2.GaussianBlur(luma, (0, 0), 1.15)
    # Store signed detail around 128 for an 8-bit lookup. Suppress small
    # differences and cap larger ones to limit noise and edge halos.
    detail = cv2.addWeighted(luma, 1., smooth, -1., 128.)
    levels = np.arange(256, dtype=np.float32) - 128
    boost = np.minimum(10., .85 * np.maximum(np.abs(levels) - 1.25, 0.))
    lut = np.rint(128 + strength * np.sign(levels) * boost).astype(np.uint8)
    adjustment = cv2.cvtColor(cv2.LUT(detail, lut), cv2.COLOR_GRAY2BGR)
    return cv2.addWeighted(canvas, 1., adjustment, 1., -128.)


def render_frame(upright, shot, size, hud=False, *, enhance=False):
    part = crop_frame(upright, shot["crop_norm"])
    interpolation = cv2.INTER_AREA
    if part.shape[1] < size[0]:
        interpolation = cv2.INTER_CUBIC if enhance else cv2.INTER_LINEAR
    canvas = cv2.resize(part, size, interpolation=interpolation)
    if enhance:
        canvas = sharpen_focus(canvas, shot.get("zoom", 1.))
    if hud:
        draw_hud(canvas, shot, shot.get("target_bbox_norm"), shot["crop_norm"])
    return canvas


def compare_views(upright, camera):
    """Show the original and directed image at the same height and media time."""
    h, w = camera.shape[:2]
    original = cv2.resize(upright, (round(h*upright.shape[1]/upright.shape[0]), h),
                          interpolation=cv2.INTER_AREA)
    # H.264 yuv420p needs an even total width.
    if (original.shape[1]+w) % 2:
        original = cv2.copyMakeBorder(original, 0, 0, 0, 1, cv2.BORDER_REPLICATE)
    joined = np.hstack((original, camera))
    for label, x in (("Fixed wide", 12), ("ShowIntent", original.shape[1]+12)):
        cv2.putText(joined, label, (x, h-18), cv2.FONT_HERSHEY_SIMPLEX, .8, (12,12,12), 4)
        cv2.putText(joined, label, (x, h-18), cv2.FONT_HERSHEY_SIMPLEX, .8, (248,248,248), 1)
    return joined


def output_size(w, h, width, height, *, max_long_edge=1280):
    if width is None and height is None:
        if w >= h:
            width = min(w, max_long_edge)
        else:
            height = min(h, max_long_edge)
    if width is None:
        width = max(2, round(height*w/h/2)*2)
    if height is None:
        height = max(2, round(width*h/w/2)*2)
    if min(width, height) < 2 or width % 2 or height % 2:
        raise ValueError("H.264 output width and height must be positive even integers")
    return width, height


def render_preview(video_path, shot_packets, out_path, objects_by_frame=None,
                   out_width=1280, out_height=720, rotate="auto"):
    """Render previously computed shots (original preview API, with the HUD enabled)."""
    if not shot_packets:
        raise ValueError("No ShotPackets for preview")
    pairs = []
    for shot in shot_packets:
        objects = (objects_by_frame or {}).get(shot.get("frame_id"), [])
        target = next((o for o in objects if o["track_id"] == shot.get("target_id")), {})
        per = {"timestamp_ms": shot["timestamp_ms"], "objects": objects}
        pairs.append((per, dict(shot, target_bbox_norm=target.get("bbox_norm", shot.get("target_bbox_norm")))))
    with TemporaryDirectory() as tmp:
        info = export(video_path, pairs, Path(tmp)/"shots.jsonl", out_path, width=out_width,
                      height=out_height, rotate=rotate, hud=True, precomputed=True)
    return info["frames"]


def export(video, pairs, shot_path, mp4_path, *, width=None, height=None, rotate="auto",
           force_wide=False, hud=False, compare=False, enhance=False, precomputed=False, **options):
    """Use the most recent A/B observation at or before each decoded frame PTS."""
    started = perf_counter()
    frame_count = 0
    cursor = 0
    per = {"objects": []}
    intent = {"timestamp_ms": 0, "state": "NONE", "target_id": None}
    last_ts = -1
    with new_outputs(shot_path, mp4_path) as (shot_tmp, video_tmp):
        with av.open(str(video)) as src, av.open(str(video_tmp), "w", options={"movie_timescale": "720000"}) as dest, shot_tmp.open("w", encoding="utf8") as log:
            vin = src.streams.video[0]
            vin.thread_type = "AUTO"
            vin.codec_context.thread_count = 4
            audio = src.streams.audio[0] if src.streams.audio else None
            # Reserve video track first. Dimensions and display matrix are known on first decode.
            vout = dest.add_stream("libx264", rate=vin.average_rate or 30)
            vout.time_base = vout.codec_context.time_base = vin.time_base
            vout.pix_fmt = "yuv420p"
            vout.options = {"preset": "veryfast", "crf": "17" if enhance else "21"}
            vout.codec_context.thread_count = 4
            aout = dest.add_stream_from_template(audio) if audio else None
            # Muxing any audio starts the MP4 header. Initialize dimensions before demux.
            with av.open(str(video)) as probe:
                first = next(probe.decode(video=0))
                rotation, name = resolve_rotation(-first.rotation, rotate)
                w, h = first.width, first.height
            if name in ("cw", "ccw"):
                w, h = h, w
            size = output_size(w, h, width, height, max_long_edge=1920 if enhance else 1280)
            vout.width, vout.height = size
            if compare:
                joined_width = size[0] + round(size[1]*w/h)
                vout.width = joined_width + joined_width % 2
            encoded_width = vout.width
            controller = CameraController(aspect=(size[0]/size[1])/(w/h), **options)
            controller.set_manual_override(force_wide)
            for packet in src.demux([vin] + ([audio] if audio else [])):
                if packet.stream == vin:
                    for frame in packet.decode():
                        if frame.pts is None:
                            raise ValueError("missing source video PTS")
                        ts = int(frame.pts * frame.time_base * 1000)
                        if ts <= last_ts:
                            raise ValueError("source video clock is not strictly increasing")
                        if (frame_count == 0 and pairs[0][0]["timestamp_ms"] < ts - 1):
                            raise ValueError("packet stream starts before the source video")
                        while cursor < len(pairs) and pairs[cursor][0]["timestamp_ms"] <= ts:
                            per, intent = pairs[cursor]
                            cursor += 1
                        if precomputed and cursor:
                            shot = dict(intent, timestamp_ms=ts)
                        else:
                            shot = shot_at(controller, ts, per, intent)
                        shot["source_frame_id"] = frame_count
                        pixels = apply_rotation(frame.to_ndarray(format="bgr24"), rotation)
                        canvas = render_frame(pixels, shot, size, hud, enhance=enhance)
                        if compare:
                            canvas = compare_views(pixels, canvas)
                        out = av.VideoFrame.from_ndarray(canvas, format="bgr24")
                        out.pts, out.time_base = frame.pts, frame.time_base
                        out.duration = frame.duration
                        for encoded in vout.encode(out):
                            dest.mux(encoded)
                        log.write(json.dumps(shot) + "\n")
                        frame_count += 1
                        last_ts = ts
                elif packet.dts is not None:
                    packet.stream = aout
                    dest.mux(packet)
            if not frame_count or cursor != len(pairs):
                raise ValueError("incomplete video or A/B observations beyond last decoded PTS")
            for encoded in vout.encode():
                dest.mux(encoded)
    return {"frames": frame_count, "packets": cursor, "width": encoded_width, "height": size[1],
            "rotation": name, "audio_copied": audio is not None, "enhanced": enhance,
            "render_fps": round(frame_count/(perf_counter()-started), 2)}

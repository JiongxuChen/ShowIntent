import json
from fractions import Fraction
from hashlib import sha256

import av
import numpy as np
import pytest

from src.camera.controller import CameraController, FULL_CROP
from src.camera.render import compare_views, export, render_frame, render_preview
from src.camera.run import load_pair, new_outputs


def pair(ts, tid="book", box=(.65, .66, .75, .8)):
    objects = [{"track_id": tid, "bbox_norm": box}] if box else []
    return ({"timestamp_ms": ts, "objects": objects},
            {"timestamp_ms": ts, "target_id": tid, "state": "SHOWING"})


def test_id_switch_and_stale_observations():
    c = CameraController(miss_hold_ms=250)
    for ts in range(0, 2000, 40):
        p, i = pair(ts)
        s = c.update(i, p)
    assert s["mode"] == "CLOSEUP"
    c.set_paused(True)
    p, i = pair(2000, "other", (.01,.01,.1,.1))
    assert c.update(i, p)["crop_norm"] == s["crop_norm"]
    c.set_paused(False)
    current = c.update(i, p, timestamp_ms=2040)
    assert current["target_id"] == "other" and c.reason == "ACQUIRE"
    assert max(abs(a-b) for a,b in zip(current["crop_norm"], s["crop_norm"])) <= .065
    for ts in range(2080, 4300, 40):
        last = c.update(i, p, timestamp_ms=ts)
    assert last["crop_norm"] == list(FULL_CROP) and c.reason == "TARGET_UNAVAILABLE"
    with pytest.raises(ValueError, match="future"):
        CameraController().update(i, p, timestamp_ms=100)


@pytest.mark.parametrize("aspect", [9/16, 1., 16/9])
@pytest.mark.parametrize("bbox", [[0,0,.02,.02], [.95,.95,1,1], [0,0,1,1]])
def test_geometry_at_edges_and_multiple_aspects(aspect, bbox):
    c = CameraController(aspect=aspect, max_zoom=2)
    for ts in range(0, 2000, 40):
        p,i = pair(ts, box=bbox)
        s = c.update(i,p)
        x,y,r,b = s["crop_norm"]
        assert 0 <= x < r <= 1 and 0 <= y < b <= 1
        assert (r-x)/(b-y) == pytest.approx(aspect)
        assert 1-1e-9 <= s["zoom"] <= 2+1e-9


def test_alignment_refused(tmp_path):
    p, i = tmp_path/"a", tmp_path/"b"
    for times, msg in [([4,4], "increasing"), ([3,5], "differ")]:
        p.write_text("".join(json.dumps(pair(t)[0])+"\n" for t in times))
        i.write_text("".join(json.dumps(pair(t)[1])+"\n" for t in (3,6)))
        with pytest.raises(ValueError, match=msg):
            load_pair(p,i)


def test_vfr_sparse_input_keeps_all_frames_and_has_no_lookahead(tmp_path):
    pts = [80,120,151,196,234,277]
    video = tmp_path/"source.mp4"
    with av.open(str(video), "w") as dest:
        stream = dest.add_stream("libx264", rate=25)
        stream.width, stream.height = 128, 72
        stream.time_base = stream.codec_context.time_base = Fraction(1,1000)
        for t in pts:
            frame = av.VideoFrame.from_ndarray(np.zeros((72,128,3), np.uint8), format="bgr24")
            frame.pts, frame.time_base = t, Fraction(1,1000)
            for packet in stream.encode(frame):
                dest.mux(packet)
        for packet in stream.encode():
            dest.mux(packet)
    trace, movie = tmp_path/"shots.jsonl", tmp_path/"camera.mp4"
    export(video, [pair(151), pair(234)], trace, movie)
    rows = [json.loads(line) for line in trace.read_text().splitlines()]
    assert [r["timestamp_ms"] for r in rows] == pts
    assert [r["packet_timestamp_ms"] for r in rows] == [None,None,151,151,234,234]
    assert [r["mode"] for r in rows[:2]] == ["WIDE","WIDE"]
    with av.open(str(movie)) as c:
        assert not c.streams.audio
        assert [int(f.pts*f.time_base*1000) for f in c.decode(video=0)] == pts
    legacy = tmp_path/"preview.mp4"
    assert render_preview(video, rows[2:], legacy, out_width=128, out_height=72) == len(pts)
    with pytest.raises(FileExistsError):
        export(video, [pair(151)], trace, movie)
    log2, movie2 = tmp_path/"new.jsonl", tmp_path/"new.mp4"
    with pytest.raises(ValueError, match="beyond"):
        export(video, [pair(1000)], log2, movie2)
    assert not log2.exists() and not movie2.exists()


def test_render_uses_the_applied_crop():
    image = np.zeros((40, 80, 3), np.uint8)
    image[:,40:,2] = 255
    result = render_frame(image, {"crop_norm": [.5,0,1,1]}, (32, 32))
    assert result.shape == (32,32,3) and np.all(result[:,:,2] == 255)
    joined = compare_views(image, result)
    assert joined.shape == (32,96,3) and np.all(joined[-1,64:,2] == 255)


def test_enhancement_follows_zoom_and_limits_luma_boost():
    image = np.full((48, 80, 3), (55, 85, 115), np.uint8)
    image[:, 40:] += 75
    size = (80, 48)
    shot = {"crop_norm": FULL_CROP, "zoom": 1.}
    assert np.array_equal(render_frame(image, shot, size), image)
    shot["zoom"] = 2.
    enhanced = render_frame(image, shot, size)
    delta = enhanced.astype(np.int16) - image.astype(np.int16)
    assert delta.min() == -10 and delta.max() == 10
    assert np.array_equal(delta[:,:,0], delta[:,:,2])
    assert not delta[:, :30].any()
    shot["zoom"] = 1.25
    mid = render_frame(image, shot, size)
    assert 0 < np.abs(mid.astype(np.int16) - image).max() < delta.max()


def test_full_frame_fit_records_reason_and_force_wide_beats_pause():
    c = CameraController()
    for ts in range(0,1000,40):
        p,i = pair(ts, box=[.2,.3,.82,.6])
        shot = c.update(i,p)
        assert shot["mode"] == "WIDE" and c.reason == "FULL_FRAME_FIT"
    for ts in range(1000,2600,40):
        p,i = pair(ts,"second")
        shot = c.update(i,p)
    assert shot["mode"] == "CLOSEUP"
    c.set_paused(True)
    c.set_manual_override(True)
    assert c.update(i,p)["crop_norm"] == list(FULL_CROP)
    assert c.reason == "FORCED_WIDE"


def test_concurrent_output_name_collision_preserves_winner(tmp_path):
    log, video = tmp_path/"shots", tmp_path/"movie"
    with pytest.raises(FileExistsError):
        with new_outputs(log, video) as (a,b):
            a.write_bytes(b"log")
            b.write_bytes(b"mine")
            video.write_bytes(b"winner")
    assert not log.exists() and video.read_bytes() == b"winner"


def test_collection_accepts_registered_copy_and_rejects_other_same_size_copy(tmp_path):
    from scripts.run_camera_collection import check_render_source

    video = tmp_path/"clip.mp4"
    video.write_bytes(b"approved copy")
    row = {"clip_id": "sample", "video_size_bytes": "100",
           "render_copy_sha256": sha256(video.read_bytes()).hexdigest()}
    check_render_source(video, row)
    video.write_bytes(b"modified copy")
    with pytest.raises(ValueError, match="Unrecognized source video"):
        check_render_source(video, row)
    video.write_bytes(bytes(100))
    check_render_source(video, row)

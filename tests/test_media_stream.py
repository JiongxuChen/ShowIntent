"""Real local libav decoding checks; generated signals are not ASR evidence."""

from fractions import Fraction
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import av
import numpy as np
import soundfile as sf

from src.speech.media_stream import inspect_media_clock, iter_media_windows


class MediaStreamTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def wav(self, *, frames=48123, rate=48000, channels=2, name="stereo.wav"):
        path = self.root / name
        signal = np.sin(np.arange(frames) * (2 * np.pi * 440 / rate)) * 0.15
        data = np.repeat(signal[:, None], channels, axis=1).astype(np.float32)
        sf.write(path, data, rate, subtype="PCM_16")
        return path

    def video(self, *, with_audio=True):
        path = self.root / ("offset_av.mkv" if with_audio else "video_only.mkv")
        with av.open(str(path), "w", format="matroska") as output:
            video = output.add_stream("ffv1", rate=25)
            video.width, video.height, video.pix_fmt = 16, 16, "yuv420p"
            audio = output.add_stream("pcm_s16le", rate=16000) if with_audio else None
            if audio is not None:
                audio.layout = "mono"
            frame = av.VideoFrame.from_ndarray(np.zeros((16, 16, 3), dtype=np.uint8), format="rgb24")
            frame.pts, frame.time_base = 0, Fraction(1, 25)
            for packet in video.encode(frame):
                output.mux(packet)
            for packet in video.encode(None):
                output.mux(packet)
            if audio is not None:
                frame = av.AudioFrame.from_ndarray(np.zeros((1, 1600), dtype=np.int16),
                                                   format="s16", layout="mono")
                frame.sample_rate, frame.pts, frame.time_base = 16000, 8000, Fraction(1, 16000)
                for packet in audio.encode(frame):
                    output.mux(packet)
                for packet in audio.encode(None):
                    output.mux(packet)
        return path

    def test_stereo_48k_resamples_to_mono_16k_and_keeps_exact_tail(self):
        path = self.wav()
        windows = list(iter_media_windows(path, step_ms=500))
        self.assertEqual([len(window.samples) for window in windows], [8000, 8000, 41])
        self.assertEqual([(w.start_sample, w.end_sample) for w in windows],
                         [(0, 8000), (8000, 16000), (16000, 16041)])
        self.assertTrue(all(w.samples.ndim == 1 and w.samples.dtype == np.float32 for w in windows))
        joined = np.concatenate([w.samples for w in windows])
        self.assertEqual(len(joined), 48123 // 3)
        self.assertTrue(np.isfinite(joined).all())
        self.assertGreater(float(np.max(np.abs(joined))), 0.1)

    def test_native_16k_mono_is_sample_complete(self):
        path = self.wav(frames=32001, rate=16000, channels=1)
        windows = list(iter_media_windows(path, step_ms=1000))
        expected, _ = sf.read(path, dtype="float32")
        np.testing.assert_allclose(np.concatenate([w.samples for w in windows]), expected, atol=1e-7)
        self.assertEqual([len(w.samples) for w in windows], [16000, 16000, 1])

    def test_generator_opens_lazily_and_first_window_does_not_decode_whole_file(self):
        path = self.wav(frames=48000 * 5)
        real_open = av.open
        state = {"opened": 0, "frames_decoded": 0, "closed": False}

        class TrackingContainer:
            def __init__(self, container):
                self.container = container
                self.streams = container.streams

            def __enter__(self):
                return self

            def __exit__(self, *args):
                self.container.close()
                state["closed"] = True

            def decode(self, **kwargs):
                for frame in self.container.decode(**kwargs):
                    state["frames_decoded"] += 1
                    yield frame

        def tracked_open(*args, **kwargs):
            state["opened"] += 1
            return TrackingContainer(real_open(*args, **kwargs))

        with real_open(str(path)) as container:
            full_decode_frames = sum(1 for _ in container.decode(audio=0))
        with patch("av.open", side_effect=tracked_open):
            source = iter_media_windows(path, step_ms=100)
            self.assertEqual(state["opened"], 0)
            first = next(source)
            self.assertEqual(len(first.samples), 1600)
            self.assertEqual(state["opened"], 1)
            self.assertLess(state["frames_decoded"], full_decode_frames)
            source.close()
        self.assertTrue(state["closed"])

    def test_audio_only_clock_has_zero_offset_and_reports_codec(self):
        clock = inspect_media_clock(self.wav())
        self.assertEqual(clock["audio_offset_ms"], 0)
        self.assertIsNone(clock["video_stream_start_ms"])
        self.assertEqual(clock["audio_codec"], "pcm_s16le")
        self.assertEqual(clock["timeline_policy"], "decoded_audio_start_relative_to_container_start")

    def test_video_and_audio_stream_starts_preserve_explicit_half_second_offset(self):
        path = self.video()
        clock = inspect_media_clock(path)
        self.assertEqual(clock["container_origin_ms"], 0)
        self.assertEqual(clock["video_stream_start_ms"], 0)
        self.assertEqual(clock["audio_stream_start_ms"], 500)
        self.assertEqual(clock["audio_offset_ms"], 500)
        windows = list(iter_media_windows(path, step_ms=100))
        self.assertEqual(sum(len(w.samples) for w in windows), 1600)
        self.assertEqual(windows[0].start_sample, 0)

    def test_missing_corrupt_empty_and_video_only_sources_fail_clearly(self):
        absent = self.root / "missing.wav"
        for function in (inspect_media_clock, lambda path: list(iter_media_windows(path))):
            with self.subTest(function=function), self.assertRaises(FileNotFoundError):
                function(absent)
        corrupt = self.root / "corrupt.wav"
        corrupt.write_bytes(b"not a media file")
        for function in (inspect_media_clock, lambda path: list(iter_media_windows(path))):
            with self.subTest(function=function), self.assertRaises(ValueError):
                function(corrupt)
        empty = self.wav(frames=0)
        with self.assertRaises(ValueError):
            list(iter_media_windows(empty))
        video_only = self.video(with_audio=False)
        for function in (inspect_media_clock, lambda path: list(iter_media_windows(path))):
            with self.subTest(function=function), self.assertRaisesRegex(ValueError, "no audio"):
                function(video_only)

    def test_invalid_window_options_reject_integers_outside_contract(self):
        path = self.wav()
        for options in ({"step_ms": 99}, {"step_ms": 10001}, {"step_ms": True},
                        {"step_ms": 100.5}, {"sample_rate": 0}, {"sample_rate": True},
                        {"sample_rate": 1.5}, {"step_ms": 100, "sample_rate": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                list(iter_media_windows(path, **options))

    def test_timestamp_gap_is_rejected_instead_of_compressing_the_media_timeline(self):
        path = self.wav(frames=1600, rate=16000, channels=1)
        frames = []
        for pts in (0, 3200):
            frame = av.AudioFrame.from_ndarray(np.zeros((1, 1600), dtype=np.float32),
                                               format="fltp", layout="mono")
            frame.sample_rate, frame.pts, frame.time_base = 16000, pts, Fraction(1, 16000)
            frames.append(frame)

        class GappedContainer:
            streams = SimpleNamespace(audio=[object()])

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

            def decode(self, **kwargs):
                yield from frames

        with patch("av.open", return_value=GappedContainer()):
            with self.assertRaisesRegex(ValueError, "discontinuity"):
                list(iter_media_windows(path, step_ms=100))


if __name__ == "__main__":
    unittest.main()

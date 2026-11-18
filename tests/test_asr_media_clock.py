"""ASR wrapper preserves the container clock without real inference or I/O media.

Fake decoder/model modules supply only sample arrays and timestamped words; the
real transcribe_samples timing logic still runs under a deterministic clock.
"""

from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from src.speech import asr


class QuarterSecondClock:
    def __init__(self):
        self.calls = 0

    def __call__(self):
        # Two reads per recognized window, with exactly250ms processing time.
        value = (self.calls // 2) * 0.25 + (0.25 if self.calls % 2 else 0.0)
        self.calls += 1
        return value


class FakeWhisperModel:
    def __init__(self, *args, **kwargs):
        self.constructor_args = args
        self.constructor_kwargs = kwargs
        self.calls = []

    def transcribe(self, samples, **kwargs):
        self.calls.append((samples.copy(), kwargs))
        segments = [SimpleNamespace(start=0.1, end=0.6, text="测试语音",
                                    avg_logprob=-0.1, no_speech_prob=0.01)]
        return iter(segments), SimpleNamespace(language="zh")


class ASRMediaClockTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.media = self.root / "recording.mp4"
        self.media.write_bytes(b"not real media; decoder and clock inspection are fake")
        self.model_path = self.root / "model"
        self.model_path.mkdir()
        (self.model_path / "model.bin").write_bytes(b"not real model; constructor is fake")

    def invoke(self, timing, *, mode="offline", media_offset_ms=0):
        samples = np.zeros(2 * asr.SAMPLE_RATE, dtype=np.float32)
        faster_whisper = ModuleType("faster_whisper")
        faster_whisper.__version__ = "test-faster-whisper"
        faster_whisper.__path__ = []
        models = []

        def construct(*args, **kwargs):
            model = FakeWhisperModel(*args, **kwargs)
            models.append(model)
            return model

        faster_whisper.WhisperModel = construct
        decoder = Mock(return_value=samples)
        audio_module = ModuleType("faster_whisper.audio")
        audio_module.decode_audio = decoder
        faster_whisper.audio = audio_module
        ctranslate2 = ModuleType("ctranslate2")
        ctranslate2.__version__ = "test-ctranslate2"
        av = ModuleType("av")
        av.__version__ = "test-pyav"
        real_transcribe_samples = asr.transcribe_samples
        clock = QuarterSecondClock()

        def timed_transcribe_samples(*args, **kwargs):
            return real_transcribe_samples(*args, clock=clock, **kwargs)

        with patch.dict(sys.modules, {
            "faster_whisper": faster_whisper, "faster_whisper.audio": audio_module,
            "ctranslate2": ctranslate2, "av": av,
        }), patch("src.speech.media_stream.inspect_media_clock", return_value=timing) as inspect_clock, patch(
            "src.speech.asr.transcribe_samples", side_effect=timed_transcribe_samples,
        ) as transcribe:
            result = asr.run_asr(
                self.media, self.model_path, mode=mode, window_ms=1000,
                media_offset_ms=media_offset_ms, extract_language=False,
            )
        inspect_clock.assert_called_once_with(self.media)
        decoder.assert_called_once_with(str(self.media), sampling_rate=16000)
        self.assertEqual(len(models), 1)
        self.assertTrue(models[0].constructor_kwargs["local_files_only"])
        self.assertEqual(transcribe.call_args.kwargs["media_offset_ms"],
                         media_offset_ms + timing["audio_offset_ms"])
        self.assertEqual(result.metadata["media_offset_ms"], media_offset_ms)
        self.assertEqual(result.metadata["audio_offset_ms"], timing["audio_offset_ms"])
        self.assertEqual(result.metadata["media_clock"], timing)
        self.assertEqual(result.metadata["duration_ms"], 2000)
        return result

    @staticmethod
    def clock(*, origin=0, audio_start=0, video_start=0, offset=0):
        return {
            "container_origin_ms": origin, "audio_stream_start_ms": audio_start,
            "video_stream_start_ms": video_start, "audio_offset_ms": offset,
            "timeline_policy": "decoded_audio_start_relative_to_container_start",
            "audio_codec": "aac",
        }

    def test_positive_audio_start_shifts_offline_event_and_eof_availability(self):
        result = self.invoke(self.clock(audio_start=85, offset=85))
        event = result.events[0]
        self.assertEqual((event.start_ms, event.end_ms, event.available_at_ms), (185, 685, 2335))
        self.assertEqual(result.windows[0]["audio_start_ms"], 85)
        self.assertEqual(result.windows[0]["received_until_ms"], 2085)
        self.assertEqual(result.windows[0]["processing_start_ms"], 2085)
        self.assertEqual(result.transcripts[0]["start_ms"], 185)
        self.assertEqual(result.transcripts[0]["end_ms"], 685)

    def test_user_offset_and_recording_audio_offset_are_added_once_in_causal_mode(self):
        for audio_offset in (6, 41, 50, 85):
            with self.subTest(audio_offset=audio_offset):
                result = self.invoke(
                    self.clock(audio_start=audio_offset, offset=audio_offset),
                    mode="causal", media_offset_ms=1200,
                )
                origin = 1200 + audio_offset
                self.assertEqual([(e.start_ms, e.end_ms) for e in result.events],
                                 [(origin + 100, origin + 600), (origin + 1100, origin + 1600)])
                self.assertEqual([e.available_at_ms for e in result.events],
                                 [origin + 1250, origin + 2250])
                self.assertEqual([w["received_until_ms"] for w in result.windows],
                                 [origin + 1000, origin + 2000])
                self.assertTrue(all(e.end_ms <= w["received_until_ms"]
                                    for e, w in zip(result.events, result.windows)))

    def test_absolute_container_start_is_not_mistaken_for_relative_audio_offset(self):
        result = self.invoke(
            self.clock(origin=5000, audio_start=5085, video_start=5000, offset=85),
            media_offset_ms=300,
        )
        self.assertEqual(result.events[0].start_ms, 485)
        self.assertEqual(result.events[0].end_ms, 985)
        self.assertEqual(result.events[0].available_at_ms, 2635)

    def test_video_start_after_audio_does_not_shift_speech_to_first_video_frame(self):
        # D01-like video PTS starts around80ms, but audio is already on clock0.
        result = self.invoke(self.clock(audio_start=0, video_start=80, offset=0))
        self.assertEqual((result.events[0].start_ms, result.events[0].end_ms), (100, 600))
        self.assertEqual(result.events[0].available_at_ms, 2250)
        self.assertEqual(result.metadata["media_clock"]["video_stream_start_ms"], 80)

    def test_zero_offset_audio_only_regression_preserves_previous_timestamps(self):
        for mode in ("offline", "causal"):
            with self.subTest(mode=mode):
                result = self.invoke(self.clock(video_start=None), mode=mode)
                self.assertEqual(result.events[0].start_ms, 100)
                self.assertEqual(result.events[0].end_ms, 600)
                self.assertEqual(result.events[0].available_at_ms, 2250 if mode == "offline" else 1250)
                self.assertEqual(result.metadata["audio_offset_ms"], 0)
                self.assertEqual(result.metadata["media_offset_ms"], 0)


if __name__ == "__main__":
    unittest.main()

"""Contract checks for D1 recording input using generated, non-speech audio."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import numpy as np
import soundfile as sf

from src.speech.audio import inspect_audio, iter_audio_chunks


class AudioReaderTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.root = Path(self.directory.name)

    def tearDown(self):
        self.directory.cleanup()

    def write_audio(self, name="recording.wav", *, frames=4001, rate=16000, channels=1):
        path = self.root / name
        signal = np.zeros((frames, channels), dtype=np.float32)
        if frames:
            signal[:, 0] = 0.2 * np.sin(np.arange(frames) * (2 * np.pi * 440 / rate))
        sf.write(path, signal, rate, subtype="PCM_16")
        return path

    def test_metadata_uses_sample_count_and_keeps_duration_precision(self):
        path = self.write_audio(channels=2)
        info = inspect_audio(path)
        self.assertEqual(info.sample_rate, 16000)
        self.assertEqual(info.channels, 2)
        self.assertEqual(info.frame_count, 4001)
        self.assertEqual(info.duration_ms, 250)
        self.assertEqual(info.duration_seconds, 4001 / 16000)
        self.assertEqual(info.to_dict()["format"], "WAV")
        self.assertEqual(info.to_dict()["path"], str(path.resolve()))

    def test_stereo_blocks_keep_tail_and_media_offset(self):
        path = self.write_audio(channels=2)
        chunks = list(iter_audio_chunks(path, block_ms=100, media_offset_ms=700))
        self.assertEqual([len(chunk.samples) for chunk in chunks], [1600, 1600, 801])
        self.assertEqual([(c.start_ms, c.end_ms) for c in chunks],
                         [(700, 800), (800, 900), (900, 950)])
        self.assertEqual([(c.start_sample, c.end_sample) for c in chunks],
                         [(0, 1600), (1600, 3200), (3200, 4001)])
        self.assertTrue(all(c.samples.shape[1] == 2 for c in chunks))
        self.assertTrue(all(c.samples.dtype == np.float32 for c in chunks))
        decoded, _ = sf.read(path, dtype="float32", always_2d=True)
        np.testing.assert_array_equal(np.concatenate([c.samples for c in chunks]), decoded)

    def test_fractional_block_sample_count_does_not_accumulate_rounding_error(self):
        path = self.write_audio(frames=4411, rate=44100)
        chunks = list(iter_audio_chunks(path, block_ms=1))
        self.assertEqual(len(chunks), 101)
        self.assertEqual(chunks[9].end_sample, 441)
        self.assertEqual(chunks[99].end_sample, 4410)
        self.assertEqual(chunks[-1].end_sample, 4411)
        self.assertEqual(chunks[99].end_ms, 100)
        self.assertEqual(sum(len(c.samples) for c in chunks), 4411)
        self.assertTrue(all(c.samples.ndim == 2 and c.samples.shape[1] == 1 for c in chunks))

    def test_flac_uses_the_same_reader(self):
        path = self.write_audio("recording.flac", frames=1600)
        self.assertEqual(inspect_audio(path).format, "FLAC")
        self.assertEqual(len(list(iter_audio_chunks(path))), 1)

    def test_invalid_options_fail_before_decoding(self):
        path = self.write_audio()
        for options in ({"block_ms": 0}, {"block_ms": -1}, {"block_ms": True},
                        {"block_ms": 0.5}, {"media_offset_ms": -1},
                        {"media_offset_ms": True}, {"media_offset_ms": 1.2}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                list(iter_audio_chunks(path, **options))

    def test_missing_invalid_and_empty_recordings_fail_clearly(self):
        missing = self.root / "missing.wav"
        with self.assertRaises(FileNotFoundError):
            inspect_audio(missing)
        with self.assertRaises(FileNotFoundError):
            list(iter_audio_chunks(missing))
        invalid = self.root / "invalid.wav"
        invalid.write_text("not audio", encoding="utf-8")
        empty = self.write_audio("empty.wav", frames=0)
        for path in (invalid, empty):
            with self.subTest(path=path), self.assertRaises(ValueError):
                inspect_audio(path)
            with self.subTest(path=path), self.assertRaises(ValueError):
                list(iter_audio_chunks(path))


if __name__ == "__main__":
    unittest.main()

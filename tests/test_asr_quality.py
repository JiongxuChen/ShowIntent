"""A conservative two-signal guard rejects repetition in likely nonspeech.

The constructed segments exercise policy boundaries, not empirical ASR accuracy.
"""

from types import SimpleNamespace
import unittest

import numpy as np

from src.speech.asr import transcribe_samples
from src.speech.quality import POLICY, segment_quality


def segment(text="Recognized speech.", **stats):
    return SimpleNamespace(start=0.1, end=0.8, text=text, **stats)


class SegmentModel:
    def __init__(self, segments):
        self.segments = segments

    def transcribe(self, audio, **kwargs):
        return iter(self.segments), SimpleNamespace(language="en")


class SegmentQualityTests(unittest.TestCase):
    def run_segments(self, segments, *, extractor=None):
        ticks = iter((0.0, 0.25))
        return transcribe_samples(
            np.zeros(16000, dtype=np.float32), SegmentModel(segments),
            extractor=extractor, clock=lambda: next(ticks),
        )

    def test_likely_nonspeech_with_extreme_repetition_is_rejected_despite_high_logprob(self):
        suspicious = segment("repeated phrase " * 50, no_speech_prob=0.7828,
                             compression_ratio=22.74, avg_logprob=-0.096)
        self.assertFalse(segment_quality(suspicious)["accepted"])
        extracted = []
        result = self.run_segments([suspicious], extractor=lambda event: extracted.append(event.text) or event)
        self.assertEqual(result.events, [])
        self.assertEqual(extracted, [])
        self.assertEqual(len(result.transcripts), 1)
        self.assertEqual(result.transcripts[0]["text"], suspicious.text.strip())
        self.assertFalse(result.transcripts[0]["accepted"])
        self.assertEqual(result.metadata["rejected_quality_segments"], 1)
        self.assertEqual(result.windows[0]["event_count"], 0)
        self.assertEqual(result.windows[0]["available_at_ms"], 1250)
        self.assertEqual(result.metadata["total_compute_ms"], 250)

    def test_policy_thresholds_require_both_signals_including_exact_boundary(self):
        for no_speech, compression, accepted in (
            (0.6, 2.4, False), (0.5999, 2.4, True), (0.6, 2.3999, True),
            (0.0, 50.0, True), (0.99, 1.0, True),
        ):
            with self.subTest(no_speech=no_speech, compression=compression):
                self.assertEqual(segment_quality(segment(
                    no_speech_prob=no_speech, compression_ratio=compression,
                ))["accepted"], accepted)
        self.assertEqual(POLICY, "reject_high_no_speech_and_repetition_v1")

    def test_high_compression_voice_and_benign_repeated_words_are_preserved(self):
        repeated = segment("Very very good.", no_speech_prob=0.03, compression_ratio=15.0)
        self.assertTrue(segment_quality(repeated)["accepted"])
        result = self.run_segments([repeated])
        self.assertEqual([event.text for event in result.events], ["Very very good."])
        self.assertTrue(result.transcripts[0]["accepted"])
        self.assertEqual(result.metadata["rejected_quality_segments"], 0)

    def test_missing_or_nonfinite_statistics_do_not_alone_trigger_rejection(self):
        cases = [
            {}, {"no_speech_prob": 0.99}, {"compression_ratio": 25.0},
            {"no_speech_prob": None, "compression_ratio": 25.0},
            {"no_speech_prob": float("nan"), "compression_ratio": 25.0},
            {"no_speech_prob": float("inf"), "compression_ratio": 25.0},
            {"no_speech_prob": 0.99, "compression_ratio": None},
            {"no_speech_prob": 0.99, "compression_ratio": float("nan")},
            {"no_speech_prob": 0.99, "compression_ratio": float("inf")},
        ]
        for stats in cases:
            with self.subTest(stats=stats):
                item = segment(**stats)
                self.assertTrue(segment_quality(item)["accepted"])
        # The previous standalone high-no-speech suppression is not retained.
        result = self.run_segments([segment(no_speech_prob=0.99)])
        self.assertEqual(len(result.events), 1)

    def test_mixed_transcripts_keep_rejected_raw_diagnostics_out_of_events(self):
        first = segment("first", no_speech_prob=0.01, compression_ratio=1.0)
        rejected = segment("hallucinated loop", no_speech_prob=0.9, compression_ratio=10.0)
        last = segment("last", no_speech_prob=0.01, compression_ratio=1.0)
        result = self.run_segments([first, rejected, last])
        self.assertEqual([event.text for event in result.events], ["first", "last"])
        self.assertEqual([record["accepted"] for record in result.transcripts], [True, False, True])
        self.assertEqual(result.metadata["rejected_quality_segments"], 1)
        self.assertEqual(result.windows[0]["event_count"], 2)


if __name__ == "__main__":
    unittest.main()

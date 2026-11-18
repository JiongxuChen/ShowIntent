"""D4 timing contracts use deterministic clocks and an inspecting ASR backend.

These tests do not download weights or turn fake text into accuracy evidence.
"""

from dataclasses import replace
from types import SimpleNamespace
import unittest

import numpy as np

from src.speech.asr import SAMPLE_RATE, transcribe_samples
from src.speech.events import available_events


class ManualClock:
    def __init__(self):
        self.value = 0.0

    def __call__(self):
        return self.value

    def advance(self, seconds):
        self.value += seconds


def segment(start, end, text="Look at this apple."):
    return SimpleNamespace(
        start=start, end=end, text=text, avg_logprob=-0.2, no_speech_prob=0.01,
    )


class InspectingRecognizer:
    def __init__(self, clock, *, batches=None, call_time=0.0, yield_time=0.0,
                 finish_time=0.0, languages=None):
        self.clock = clock
        self.batches = batches
        self.call_time = call_time
        self.yield_time = yield_time
        self.finish_time = finish_time
        self.languages = languages
        self.calls = []
        self.iterations_finished = 0

    def transcribe(self, audio, **kwargs):
        index = len(self.calls)
        self.calls.append((audio.copy(), kwargs))
        self.clock.advance(self.call_time)
        batch = self.batches[index] if self.batches is not None else [
            segment(0.0, min(0.5, len(audio) / SAMPLE_RATE)),
        ]

        def lazy_segments():
            for item in batch:
                self.clock.advance(self.yield_time)
                yield item
            self.clock.advance(self.finish_time)
            self.iterations_finished += 1

        language = self.languages[index] if self.languages else "en"
        return lazy_segments(), SimpleNamespace(language=language, language_probability=1.0)


class ASRTimingTests(unittest.TestCase):
    def test_offline_results_wait_for_entire_file_and_lazy_computation(self):
        clock = ManualClock()
        recognizer = InspectingRecognizer(
            clock, batches=[[segment(0.2, 0.7), segment(2.0, 3.0, "A cup.")]],
            call_time=0.125, yield_time=0.125, finish_time=0.125,
        )
        samples = np.zeros(4 * SAMPLE_RATE, dtype=np.float32)
        result = transcribe_samples(samples, recognizer, media_offset_ms=1000, clock=clock)
        self.assertEqual(len(recognizer.calls), 1)
        self.assertEqual(recognizer.iterations_finished, 1)
        self.assertEqual(result.windows[0]["compute_ms"], 500)
        self.assertEqual([event.available_at_ms for event in result.events], [5500, 5500])
        self.assertEqual([(e.start_ms, e.end_ms) for e in result.events], [(1200, 1700), (3000, 4000)])
        self.assertEqual(available_events(result.events, 5499), [])
        self.assertEqual(available_events(result.events, 5500), result.events)
        self.assertEqual(result.metadata["timing_policy"], "offline_eof_barrier")
        self.assertFalse(result.metadata["live_microphone"])
        self.assertEqual(result.metadata["compute_rtf"], 0.125)

    def test_causal_backend_only_sees_current_completed_window_and_exact_tail(self):
        clock = ManualClock()
        recognizer = InspectingRecognizer(clock, call_time=0.125, yield_time=0.125)
        samples = np.arange(9 * SAMPLE_RATE + 1, dtype=np.float32)
        result = transcribe_samples(
            samples, recognizer, mode="causal", window_ms=4000,
            media_offset_ms=1200, clock=clock,
        )
        expected_slices = [samples[:64000], samples[64000:128000], samples[128000:]]
        self.assertEqual(len(recognizer.calls), 3)
        for (actual, kwargs), expected in zip(recognizer.calls, expected_slices):
            np.testing.assert_array_equal(actual, expected)
            self.assertFalse(np.shares_memory(actual, samples))
            self.assertFalse(kwargs["condition_on_previous_text"])
            self.assertTrue(kwargs["vad_filter"])
        np.testing.assert_array_equal(np.concatenate([call[0] for call in recognizer.calls]), samples)
        self.assertEqual([w["received_until_ms"] for w in result.windows], [5200, 9200, 10201])
        self.assertEqual([w["available_at_ms"] for w in result.windows], [5450, 9450, 10451])
        self.assertEqual([(w["start_sample"], w["end_sample"]) for w in result.windows],
                         [(0, 64000), (64000, 128000), (128000, len(samples))])
        self.assertEqual(result.metadata["duration_ms"], 9001)
        for event, window in zip(result.events, result.windows):
            self.assertGreaterEqual(event.start_ms, window["audio_start_ms"])
            self.assertLessEqual(event.end_ms, window["received_until_ms"])
            self.assertEqual(event.available_at_ms, window["available_at_ms"])

    def test_slow_worker_accumulates_backlog_instead_of_backdating_results(self):
        clock = ManualClock()
        recognizer = InspectingRecognizer(clock, call_time=2.5)
        result = transcribe_samples(
            np.zeros(3 * SAMPLE_RATE, dtype=np.float32), recognizer,
            mode="causal", window_ms=1000, clock=clock,
        )
        self.assertEqual([w["processing_start_ms"] for w in result.windows], [1000, 3500, 6000])
        self.assertEqual([e.available_at_ms for e in result.events], [3500, 6000, 8500])
        self.assertEqual(result.metadata["total_compute_ms"], 7500)
        self.assertEqual(result.metadata["compute_rtf"], 2.5)

    def test_language_extraction_and_generator_completion_are_in_measured_scope(self):
        clock = ManualClock()
        recognizer = InspectingRecognizer(
            clock, call_time=0.125, yield_time=0.25, finish_time=0.5,
        )

        def annotate(event):
            clock.advance(0.125)
            return replace(event, mentioned_classes=["apple"], cue_types=["PRESENTATION_LANGUAGE"])

        result = transcribe_samples(
            np.zeros(SAMPLE_RATE, dtype=np.float32), recognizer,
            mode="causal", window_ms=1000, extractor=annotate, clock=clock,
        )
        self.assertEqual(result.windows[0]["compute_ms"], 1000)
        self.assertEqual(result.events[0].available_at_ms, 2000)
        self.assertEqual(result.events[0].mentioned_classes, ["apple"])
        self.assertEqual(result.events[0].cue_types, ["PRESENTATION_LANGUAGE"])
        self.assertEqual(result.metadata["semantic_annotations"], "rule_extractor")

    def test_without_extractor_recognized_words_do_not_create_semantic_labels(self):
        clock = ManualClock()
        recognizer = InspectingRecognizer(clock)
        result = transcribe_samples(np.zeros(SAMPLE_RATE), recognizer, clock=clock)
        self.assertEqual(result.events[0].text, "Look at this apple.")
        self.assertEqual(result.events[0].mentioned_classes, [])
        self.assertEqual(result.events[0].cue_types, [])
        self.assertEqual(result.metadata["semantic_annotations"], "empty_no_extractor")
        self.assertEqual(result.transcripts[0]["avg_logprob"], -0.2)
        self.assertEqual(result.transcripts[0]["no_speech_prob"], 0.01)

    def test_silence_still_accounts_for_compute_and_advances_window_clock(self):
        clock = ManualClock()
        recognizer = InspectingRecognizer(clock, batches=[[], []], finish_time=0.125)
        result = transcribe_samples(
            np.zeros(2 * SAMPLE_RATE), recognizer,
            mode="causal", window_ms=1000, clock=clock,
        )
        self.assertEqual(result.events, [])
        self.assertEqual(result.transcripts, [])
        self.assertEqual(recognizer.iterations_finished, 2)
        self.assertEqual([w["available_at_ms"] for w in result.windows], [1125, 2125])
        self.assertEqual(result.metadata["total_compute_ms"], 250)

    def test_segment_timestamps_are_bounded_and_unusable_segments_discarded(self):
        clock = ManualClock()
        recognizer = InspectingRecognizer(clock, batches=[[
            segment(-0.1, 0.2, "  Start  "), segment(0.8, 4.0, "Tail"),
            segment(0.4, 0.4), segment(0.7, 0.3), segment(float("nan"), 0.9),
            segment(0.0, float("inf")), segment(0.1, 0.2, "   "),
        ]])
        result = transcribe_samples(np.zeros(SAMPLE_RATE), recognizer, clock=clock)
        self.assertEqual([(e.start_ms, e.end_ms, e.text) for e in result.events],
                         [(0, 200, "Start"), (800, 1000, "Tail")])
        self.assertEqual(result.metadata["discarded_segments"], 4)
        self.assertTrue(all(event.end_ms <= event.available_at_ms for event in result.events))

    def test_long_session_sample_positions_do_not_drift(self):
        clock = ManualClock()
        recognizer = InspectingRecognizer(clock)
        samples = np.zeros(65 * SAMPLE_RATE + 16, dtype=np.float32)
        result = transcribe_samples(samples, recognizer, mode="causal", window_ms=1000, clock=clock)
        self.assertEqual(len(result.windows), 66)
        self.assertEqual(result.windows[-1]["start_sample"], 65 * SAMPLE_RATE)
        self.assertEqual(result.windows[-1]["end_sample"], len(samples))
        self.assertEqual(result.windows[-1]["received_until_ms"], 65001)
        self.assertEqual(result.events[-1].start_ms, 65000)
        self.assertEqual(result.events[-1].end_ms, 65001)
        self.assertEqual(result.events[-1].available_at_ms, 65001)

    def test_selected_language_and_beam_reach_backend_and_detected_languages_are_reported(self):
        clock = ManualClock()
        recognizer = InspectingRecognizer(clock, languages=["zh", "en", "zh"])
        result = transcribe_samples(
            np.zeros(3 * SAMPLE_RATE), recognizer, mode="causal", window_ms=1000,
            language="zh", beam_size=5, clock=clock,
        )
        self.assertEqual(result.metadata["detected_languages"], ["zh", "en"])
        self.assertTrue(all(kwargs["language"] == "zh" and kwargs["beam_size"] == 5
                            for _, kwargs in recognizer.calls))

    def test_invalid_input_and_options_fail_before_backend_is_called(self):
        cases = [
            ([], {}), (np.zeros((16, 2)), {}), ([float("nan")], {}), ([float("inf")], {}),
            ([0.0], {"mode": "live"}), ([0.0], {"window_ms": 0}),
            ([0.0], {"window_ms": 30001}), ([0.0], {"window_ms": True}),
            ([0.0], {"window_ms": 4000.0}), ([0.0], {"beam_size": 0}),
            ([0.0], {"beam_size": 11}), ([0.0], {"beam_size": True}),
            ([0.0], {"media_offset_ms": -1}), ([0.0], {"media_offset_ms": True}),
        ]
        for samples, options in cases:
            with self.subTest(options=options, shape=np.asarray(samples).shape):
                clock = ManualClock()
                recognizer = InspectingRecognizer(clock)
                with self.assertRaises(ValueError):
                    transcribe_samples(samples, recognizer, clock=clock, **options)
                self.assertEqual(recognizer.calls, [])

    def test_bad_processing_clock_is_rejected(self):
        for times in ((1.0, 0.9), (1.0, float("nan")), (1.0, float("inf"))):
            with self.subTest(times=times):
                clock = ManualClock()
                recognizer = InspectingRecognizer(clock)
                measurements = iter(times)
                with self.assertRaisesRegex(ValueError, "clock"):
                    transcribe_samples(np.zeros(SAMPLE_RATE), recognizer, clock=lambda: next(measurements))


if __name__ == "__main__":
    unittest.main()

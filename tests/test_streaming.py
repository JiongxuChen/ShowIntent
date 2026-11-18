"""Incremental LocalAgreement contracts; fake words are not ASR accuracy data."""

from dataclasses import replace
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace
import unittest

import numpy as np

from src.speech.streaming import AgreementRecognizer, SAMPLE_RATE


def word(start, end, text):
    return SimpleNamespace(start=start, end=end, word=text)


class WordRecognizer:
    def __init__(self, batches, *, no_speech_prob=0.0, compression_ratio=None):
        self.batches = batches
        self.no_speech_prob = no_speech_prob
        self.compression_ratio = compression_ratio
        self.calls = []

    def transcribe(self, audio, **kwargs):
        index = len(self.calls)
        self.calls.append((audio.copy(), kwargs))
        batch = self.batches[index] if index < len(self.batches) else []
        item = SimpleNamespace(words=batch, no_speech_prob=self.no_speech_prob)
        if self.compression_ratio is not None:
            item.compression_ratio = self.compression_ratio
        segments = [item]
        return iter(segments), SimpleNamespace(language="en")


def silence(milliseconds=2000):
    return np.zeros(milliseconds * SAMPLE_RATE // 1000, dtype=np.float32)


class AgreementRecognizerTests(unittest.TestCase):
    def test_revised_suffix_is_withheld_until_two_hypotheses_agree(self):
        first = [word(0, 0.4, "Look"), word(0.4, 0.6, " at"), word(0.6, 1.0, " apple")]
        revised = [word(0, 0.4, "Look"), word(0.4, 0.6, " at"), word(0.6, 1.0, " cup")]
        model = WordRecognizer([first, revised, revised])
        extracted = []

        def extractor(event):
            extracted.append(event.text)
            return replace(event, mentioned_classes=["cup"] if "cup" in event.text else [])

        session = AgreementRecognizer(model, extractor=extractor)
        self.assertEqual(session.push(silence()), [])
        self.assertEqual(extracted, [])
        second = session.push(silence())
        self.assertEqual([event.text for event in second], ["Look at"])
        self.assertEqual(session.pending_text, "cup")
        third = session.push(silence())
        self.assertEqual([event.text for event in third], ["cup"])
        self.assertEqual(extracted, ["Look at", "cup"])
        self.assertEqual(third[0].mentioned_classes, ["cup"])
        self.assertNotIn("apple", " ".join(extracted))
        self.assertEqual(session.pending_text, "")

    def test_overlap_deduplicates_committed_words_while_preserving_new_words(self):
        model = WordRecognizer([
            [word(0.0, 0.4, "Hello"), word(0.4, 0.8, " world")],
            [word(0.0, 0.4, "Hello"), word(0.4, 0.8, " world"), word(0.8, 1.2, " again")],
            [word(0.75, 0.95, " world"), word(1.0, 1.3, " again")],
        ])
        session = AgreementRecognizer(model)
        events = []
        for _ in range(3):
            events.extend(session.push(silence()))
        events.extend(session.push([], finish=True))
        self.assertEqual([event.text for event in events], ["Hello world", "again"])
        self.assertEqual(" ".join(event.text for event in events).count("world"), 1)

    def test_genuine_repeated_words_are_retained_at_distinct_times(self):
        repeated = [word(0, 0.3, "Very"), word(0.3, 0.6, " very"), word(0.6, 1.0, " good")]
        model = WordRecognizer([repeated, repeated])
        session = AgreementRecognizer(model)
        self.assertEqual(session.push(silence()), [])
        events = session.push(silence())
        self.assertEqual([event.text for event in events], ["Very very good"])

    def test_empty_eof_flushes_latest_pending_once_without_new_asr_call(self):
        model = WordRecognizer([[word(0.1, 0.7, "An apple.")]])
        session = AgreementRecognizer(model, media_offset_ms=500)
        self.assertEqual(session.push(silence(1000)), [])
        events = session.push([], finish=True)
        self.assertEqual(len(model.calls), 1)
        self.assertEqual([(e.start_ms, e.end_ms, e.available_at_ms, e.text) for e in events],
                         [(600, 1200, 1500, "An apple.")])
        self.assertTrue(session.finished)
        self.assertEqual(session.last_update["eof_flushed_word_count"], 1)
        self.assertEqual(session.push([], finish=True), [])
        self.assertEqual(session.push([]), [])
        with self.assertRaisesRegex(RuntimeError, "EOF"):
            session.push(silence())

    def test_eof_with_final_audio_commits_stable_prefix_and_flushes_suffix(self):
        model = WordRecognizer([
            [word(0, 0.4, "Look")],
            [word(0, 0.4, "Look"), word(1, 1.3, " here.")],
        ])
        session = AgreementRecognizer(model)
        self.assertEqual(session.push(silence()), [])
        events = session.push(silence(), finish=True)
        self.assertEqual([event.text for event in events], ["Look", "here."])
        self.assertEqual(session.last_update["confirmed_word_count"], 1)
        self.assertEqual(session.last_update["eof_flushed_word_count"], 1)
        self.assertEqual(len(model.calls), 2)

    def test_backend_only_receives_arriving_audio_and_real_arrival_is_provisional(self):
        model = WordRecognizer([[], [], []])
        session = AgreementRecognizer(model, language="zh", beam_size=5, media_offset_ms=700)
        samples = np.arange(5 * SAMPLE_RATE, dtype=np.float32)
        blocks = [samples[:32000], samples[32000:64000], samples[64000:]]
        for index, block in enumerate(blocks):
            session.push(block)
            expected = samples[:sum(len(item) for item in blocks[:index + 1])]
            np.testing.assert_array_equal(model.calls[-1][0], expected)
            options = model.calls[-1][1]
            self.assertTrue(options["word_timestamps"])
            self.assertTrue(options["condition_on_previous_text"])
            self.assertTrue(options["vad_filter"])
            self.assertEqual(options["language"], "zh")
            self.assertEqual(options["beam_size"], 5)
        self.assertEqual(session.last_update["received_until_ms"], 5700)
        self.assertTrue(session.last_update["provisional_availability"])
        session.push([], received_until_ms=6000, finish=True)
        self.assertEqual(session.last_update["received_until_ms"], 6000)

    def test_audio_and_hypothesis_buffers_stay_bounded_when_nothing_stabilizes(self):
        model = WordRecognizer([
            [word(0, 0.5, "unstable1")],
            [word(0, 0.5, "unstable2")],
            [word(0, 0.5, "unstable3")],
            [],
        ])
        session = AgreementRecognizer(model, buffer_limit_ms=4000)
        seen = 0
        for _ in range(4):
            seen += 3 * SAMPLE_RATE
            self.assertEqual(session.push(silence(3000)), [])
            self.assertLessEqual(session.buffered_samples, 4 * SAMPLE_RATE)
            self.assertLessEqual(len(model.calls[-1][0]), 4 * SAMPLE_RATE)
            self.assertEqual(session.buffer_offset_samples + session.buffered_samples, seen)
            self.assertTrue(all(word[1] * SAMPLE_RATE > session.buffer_offset_samples
                                for word in session.hypotheses.complete()))
        self.assertEqual(session.total_trimmed_samples, 8 * SAMPLE_RATE)
        self.assertGreater(session.total_unconfirmed_trimmed_samples, 0)
        self.assertGreater(session.total_dropped_pending_words, 0)

    def test_rolling_trim_uses_only_previously_committed_prompt_and_keeps_pending_words(self):
        model = WordRecognizer([
            [word(0, 1, "alpha")],
            [word(0, 1, "alpha"), word(2, 3, " beta")],
            [word(0, 1, " beta"), word(2, 3, " gamma")],
            [word(0, 1, " gamma"), word(2, 3, " delta")],
        ])
        session = AgreementRecognizer(model, buffer_limit_ms=4000)
        events = []
        for _ in range(4):
            events.extend(session.push(silence()))
        events.extend(session.push([], finish=True))
        self.assertEqual([event.text for event in events], ["alpha", "beta", "gamma", "delta"])
        self.assertEqual(model.calls[0][1]["initial_prompt"], None)
        self.assertEqual(model.calls[1][1]["initial_prompt"], None)
        self.assertEqual(model.calls[2][1]["initial_prompt"], "alpha")
        self.assertEqual(model.calls[3][1]["initial_prompt"], "alpha beta")
        self.assertEqual(session.total_dropped_pending_words, 0)
        self.assertEqual([(event.start_ms, event.end_ms) for event in events],
                         [(0, 1000), (2000, 3000), (4000, 5000), (6000, 7000)])

    def test_chinese_tokens_are_concatenated_without_artificial_word_spaces(self):
        chinese = [word(0, 0.2, "这"), word(0.2, 0.4, "个"), word(0.4, 0.8, "苹果")]
        session = AgreementRecognizer(WordRecognizer([chinese, chinese]))
        self.assertEqual(session.push(silence()), [])
        self.assertEqual(session.push(silence())[0].text, "这个苹果")

    def test_silence_empty_session_and_unusable_word_timestamps_are_safe(self):
        session = AgreementRecognizer(WordRecognizer([]))
        self.assertEqual(session.push([], finish=True), [])
        self.assertTrue(session.finished)
        invalid = [word(float("nan"), 0.4, "invalid"), word(0, float("inf"), "invalid"),
                   word(0.5, 0.2, "invalid"), word(0, 0.2, "   "), word(0.1, 0.7, "valid")]
        session = AgreementRecognizer(WordRecognizer([invalid]))
        self.assertEqual(session.push(silence()), [])
        self.assertEqual(session.last_update["discarded_word_timestamps"], 3)
        self.assertEqual(session.push([], finish=True)[0].text, "valid")
        quiet = AgreementRecognizer(WordRecognizer(
            [[word(0, 1, "hallucination")]], no_speech_prob=0.95, compression_ratio=20.0,
        ))
        self.assertEqual(quiet.push(silence(), finish=True), [])

    def test_combined_hallucination_quality_guard_runs_before_word_agreement(self):
        model = WordRecognizer(
            [[word(0, 1, "重复字幕")]] * 2, no_speech_prob=0.7828, compression_ratio=22.74,
        )
        extracted = []
        session = AgreementRecognizer(model, extractor=lambda event: extracted.append(event.text) or event)
        for _ in range(2):
            self.assertEqual(session.push(silence()), [])
            self.assertEqual(session.last_update["rejected_quality_segments"], 1)
            self.assertEqual(session.pending_text, "")
        self.assertEqual(session.push([], finish=True), [])
        self.assertEqual(session.last_update["rejected_quality_segments"], 0)
        self.assertEqual(session.last_update["total_rejected_quality_segments"], 2)
        self.assertEqual(session.last_update["quality_policy"], "reject_high_no_speech_and_repetition_v1")
        self.assertEqual(extracted, [])

    def test_high_repetition_or_high_no_speech_alone_does_not_reject_valid_words(self):
        repeated = [word(0, 0.3, "Very"), word(0.3, 0.6, " very"), word(0.6, 1.0, " good")]
        for options in ({"no_speech_prob": 0.05, "compression_ratio": 20.0},
                        {"no_speech_prob": 0.95},
                        {"no_speech_prob": 0.95, "compression_ratio": 1.1}):
            with self.subTest(options=options):
                session = AgreementRecognizer(WordRecognizer([repeated, repeated], **options))
                self.assertEqual(session.push(silence()), [])
                self.assertEqual(session.push(silence())[0].text, "Very very good")
                self.assertEqual(session.last_update["total_rejected_quality_segments"], 0)

    def test_invalid_pushes_do_not_advance_the_session(self):
        session = AgreementRecognizer(WordRecognizer([]))
        cases = [
            ([float("nan")], {}), (np.zeros((2, 2)), {}),
            (silence(), {"received_until_ms": 1000}),
            (silence(), {"received_until_ms": True}),
            (silence(), {"finish": 1}),
        ]
        for samples, options in cases:
            with self.subTest(options=options), self.assertRaises(ValueError):
                session.push(samples, **options)
            self.assertEqual(session.received_samples, 0)
            self.assertEqual(len(session.model.calls), 0)
        session.push(silence(), received_until_ms=2500)
        with self.assertRaisesRegex(ValueError, "monotonic"):
            session.push([], received_until_ms=2400)

    def test_invalid_constructor_settings_are_rejected(self):
        for options in ({"beam_size": True}, {"beam_size": 0}, {"buffer_limit_ms": 3999},
                        {"buffer_limit_ms": 30001}, {"buffer_limit_ms": True},
                        {"context_ms": -1}, {"context_ms": 15000},
                        {"media_offset_ms": -1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                AgreementRecognizer(WordRecognizer([]), **options)

    def test_vendored_localagreement_primitive_matches_recorded_integrity(self):
        root = Path(__file__).resolve().parents[1]
        manifest = json.loads((root / "external/streaming_source.json").read_text(encoding="utf-8"))
        contents = (root / manifest["vendored"]).read_bytes()
        self.assertEqual(hashlib.sha256(contents).hexdigest(), manifest["vendored_sha256"])
        self.assertEqual(manifest["license"], "MIT")
        self.assertIn("extracted verbatim", manifest["reuse"])


if __name__ == "__main__":
    unittest.main()

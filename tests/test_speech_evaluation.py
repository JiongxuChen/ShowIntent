"""Reference reading and hand-computable ASR diagnostics."""

import importlib.util
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest

from src.speech.evaluation import event_summary, read_reference, score_transcript
from src.speech.events import SpeechEvent


HAS_JIWER = importlib.util.find_spec("jiwer") is not None
HAS_OPENCC = importlib.util.find_spec("opencc") is not None
ROOT = Path(__file__).resolve().parents[1]


class ReferenceAndSummaryTests(unittest.TestCase):
    def setUp(self):
        self.folder = tempfile.TemporaryDirectory()
        self.addCleanup(self.folder.cleanup)

    def file(self, name, content):
        path = Path(self.folder.name) / name
        path.write_text(content, encoding="utf-8")
        return path

    def test_srt_extracts_text_not_timestamp_or_language_evidence(self):
        path = self.file("reference.srt", "\ufeff1\n00:00:00,100 --> 00:00:01,000\n<i>Apple</i> &amp; cup.\n多行文本\n\n2\n00:00:01,100 --> 00:00:02,000\nDo not show it.\n")
        self.assertEqual(read_reference(path), "Apple & cup. 多行文本 Do not show it.")

    def test_simple_vtt_without_indexes_reuses_upstream_parser(self):
        path = self.file("reference.vtt", "WEBVTT\n\n00:00:00.100 --> 00:00:01.000 align:start\nA cup.\n\n00:00:01.100 --> 00:00:02.000\nAn apple.\n")
        self.assertEqual(read_reference(path), "A cup. An apple.")

    def test_plain_text_and_empty_reference(self):
        self.assertEqual(read_reference(self.file("reference.txt", "\ufeff  杯子\n苹果  ")), "杯子\n苹果")
        self.assertEqual(read_reference(self.file("empty.txt", " \n")), "")
        self.assertEqual(read_reference(self.file("empty.srt", "")), "")

    def test_unsupported_or_malformed_reference_is_not_silently_skipped(self):
        for name, content in (("bad.json", "{}"), ("bad.srt", "not captions"), ("bad.vtt", "not captions")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                read_reference(self.file(name, content))

    def test_unsupported_webvtt_blocks_cannot_be_absorbed_as_caption_text(self):
        for block in ("NOTE private note", "STYLE\n::cue { color: red; }", "intro\n00:00:01.100 --> 00:00:02.000\napple"):
            with self.subTest(block=block), self.assertRaises(ValueError):
                read_reference(self.file("unsupported.vtt", "WEBVTT\n\n00:00:00.100 --> 00:00:01.000\ncup\n\n" + block + "\n"))

    def test_summary_counts_and_nearest_rank_latency(self):
        events = [SpeechEvent(100, 500, 500 + delay, "cup", ["cup"],
                              ["PRESENTATION_LANGUAGE"] if index % 2 else ["NEGATED_LANGUAGE"])
                  for index, delay in enumerate(range(10, 210, 10))]
        summary = event_summary(events)
        self.assertEqual(summary["event_count"], 20)
        self.assertEqual(summary["class_counts"], {"cup": 20})
        self.assertEqual(summary["presentation_count"], 10)
        self.assertEqual(summary["negative_count"], 10)
        self.assertEqual(summary["latency_ms"], {"min": 10, "median": 105.0, "p95": 190, "max": 200})
        self.assertEqual(event_summary(event.to_dict() for event in events), summary)

    def test_empty_summary_does_not_claim_zero_latency(self):
        summary = event_summary([])
        self.assertEqual(summary["event_count"], 0)
        self.assertEqual(summary["class_counts"], {})
        self.assertTrue(all(value is None for value in summary["latency_ms"].values()))

    def test_summary_rejects_invalid_events(self):
        with self.assertRaises(ValueError):
            event_summary([{"text": "not a full event"}])
        with self.assertRaises(ValueError):
            event_summary(["not an event"])

    def test_optional_evaluation_imports_stay_lazy(self):
        command = [sys.executable, "-S", "-c", "from src.speech.evaluation import event_summary; print(event_summary([])['event_count'])"]
        completed = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        self.assertEqual(completed.stdout.strip(), "0")


@unittest.skipUnless(HAS_JIWER, "optional jiwer dependency is not installed")
class EnglishScoreTests(unittest.TestCase):
    def test_hand_computable_deletion_and_substitution_counts(self):
        deleted = score_transcript("the red cup", "the cup", "en")
        self.assertAlmostEqual(deleted["wer"], 1 / 3)
        self.assertEqual((deleted["hits"], deleted["deletions"], deleted["substitutions"], deleted["insertions"]), (2, 1, 0, 0))
        replaced = score_transcript("the red cup", "the blue cup", "en")
        self.assertAlmostEqual(replaced["wer"], 1 / 3)
        self.assertEqual((replaced["reference_units"], replaced["substitutions"]), (3, 1))

    def test_unicode_punctuation_case_and_apostrophe_normalisation(self):
        score = score_transcript("ＩＴ’S a cup, bottle!", "it's a cup bottle", "en-US")
        self.assertEqual(score["wer"], 0)
        self.assertEqual(score["reference_units"], 4)
        self.assertNotIn("reference", score)
        self.assertNotIn("hypothesis", score)

    def test_english_empty_reference_policy_is_explicit(self):
        self.assertEqual(score_transcript("", "", "en")["wer"], 0)
        inserted = score_transcript("", "apple cup", "en")
        self.assertEqual(inserted["wer"], 2)
        self.assertEqual((inserted["reference_units"], inserted["insertions"]), (0, 2))
        self.assertIn("empty reference", inserted["empty_reference_policy"])
        self.assertEqual(score_transcript("apple cup", "", "en")["wer"], 1)

    def test_language_and_text_validation(self):
        for reference, hypothesis, language in (("apple", "apple", "ja"), (None, "apple", "en"), ("apple", 1, "en"), ("apple", "apple", None)):
            with self.subTest(language=language), self.assertRaises(ValueError):
                score_transcript(reference, hypothesis, language)


@unittest.skipUnless(HAS_JIWER and HAS_OPENCC, "optional jiwer/OpenCC dependencies are not installed")
class ChineseScoreTests(unittest.TestCase):
    def test_raw_script_errors_remain_separate_from_simplified_score(self):
        score = score_transcript("蘋果在這裡", "苹果在这里", "zh")
        self.assertAlmostEqual(score["cer"], 3 / 5)
        self.assertEqual(score["substitutions"], 3)
        self.assertEqual(score["t2s"]["cer"], 0)
        self.assertEqual(score["t2s"]["substitutions"], 0)

    def test_script_normalisation_does_not_hide_real_word_error(self):
        score = score_transcript("蘋果", "苹果杯", "zh-TW")
        self.assertEqual(score["substitutions"], 1)
        self.assertEqual(score["insertions"], 1)
        self.assertEqual(score["cer"], 1)
        self.assertEqual(score["t2s"]["cer"], 0.5)
        self.assertEqual(score["t2s"]["insertions"], 1)

    def test_chinese_whitespace_punctuation_and_full_width_normalisation(self):
        score = score_transcript(" 苹果，杯子！Ａ１ ", "苹果杯子a1", "zh_CN")
        self.assertEqual(score["cer"], 0)
        self.assertEqual(score["reference_units"], 6)

    def test_chinese_empty_reference_uses_insertion_count(self):
        self.assertEqual(score_transcript("", "", "zh")["cer"], 0)
        score = score_transcript("", "苹果", "zh")
        self.assertEqual(score["cer"], 2)
        self.assertEqual(score["insertions"], 2)
        self.assertEqual(score["t2s"]["cer"], 2)


if __name__ == "__main__":
    unittest.main()

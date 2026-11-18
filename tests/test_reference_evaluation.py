"""Human-reference evaluation uses synthetic annotations in temporary folders.

No actual recording is transcribed, labelled or treated as human-verified here.
"""

from contextlib import redirect_stderr
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

from scripts.evaluate_speech_references import main as evaluate_main
from src.speech.events import SpeechEvent
from src.speech.reference_evaluation import evaluate_reference_session, score_verified_clip
from src.speech.references import make_draft


HAS_SCORES = importlib.util.find_spec("jiwer") is not None and importlib.util.find_spec("opencc") is not None
ROOT = Path(__file__).resolve().parents[1]


def clip(case_id="T01", *, audio_offset_ms=0, language="en"):
    return {"case_id": case_id, "source_path": "synthetic-not-a-real-recording.wav",
            "source_sha256": "a" * 64, "duration_ms": 4000,
            "audio_path": "synthetic.wav", "audio_offset_ms": audio_offset_ms,
            "draft_path": "not-a-reference.jsonl", "language": language}


def row(segment_id="T01-s001", start_ms=0, end_ms=4000, *, text="a cup", kind="speech",
        classes=None, cues=None):
    return {"segment_id": segment_id, "start_ms": start_ms, "end_ms": end_ms,
            "kind": kind, "text": text, "mentioned_classes": classes or [], "cue_types": cues or [],
            "timing_verified": True, "semantic_verified": True, "notes": ""}


def verified(media, segments=None, *, method="independent_listening"):
    reference = make_draft(media)
    reference.update(review_status="verified", human_confirmed=True, coverage_complete=True,
                     reviewer="Synthetic unit-test reviewer", reference_method=method,
                     segments=segments or [row(media["case_id"] + "-s001", classes=["cup"])])
    return reference


class ReferenceFixture:
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.references = self.root / "references"
        self.events = self.root / "events"
        self.references.mkdir()
        self.events.mkdir()

    def save(self, name, value):
        path = self.references / f"{name}.json"
        path.write_text(json.dumps(value), encoding="utf-8")
        return path

    def session(self, *clips):
        return {"schema_version": 1, "session_id": "synthetic-tests", "scope": "development", "clips": list(clips)}


class ReferenceReadinessTests(ReferenceFixture, unittest.TestCase):
    def test_missing_and_draft_references_never_produce_scores(self):
        first, second = clip("T01"), clip("T02")
        self.save("T02", make_draft(second))
        with patch("src.speech.reference_evaluation.score_transcript") as scorer:
            report = evaluate_reference_session(self.session(first, second), self.references, self.events)
            scorer.assert_not_called()
        self.assertEqual(report["status"], "PENDING_HUMAN_REVIEW")
        self.assertEqual(report["results"], [])
        self.assertEqual(report["cross_mode_aggregate"], {})
        self.assertEqual(report["eligible_verified_cases"], [])
        self.assertEqual(len(report["pending_references"]), 2)

    def test_unusable_reference_is_excluded_without_becoming_empty_speech_truth(self):
        media = clip()
        reference = make_draft(media)
        reference["review_status"] = "unusable"
        self.save("T01", reference)
        report = evaluate_reference_session(self.session(media), self.references, self.events)
        self.assertEqual(report["results"], [])
        self.assertEqual(report["excluded_unusable_cases"], [{"case_id": "T01", "reason": "HUMAN_MARKED_UNUSABLE"}])

    def test_unconfirmed_incomplete_wrong_source_or_uncertain_refs_cannot_score(self):
        media = clip()
        changes = ({"human_confirmed": False}, {"coverage_complete": False}, {"reviewer": ""},
                   {"source_sha256": "b" * 64}, {"reference_method": "automatic_asr"})
        for change in changes:
            with self.subTest(change=change):
                reference = verified(media)
                reference.update(change)
                self.save("T01", reference)
                with patch("src.speech.reference_evaluation.score_transcript") as scorer:
                    report = evaluate_reference_session(self.session(media), self.references, self.events)
                    scorer.assert_not_called()
                self.assertEqual(report["results"], [])
                self.assertEqual(report["eligible_verified_cases"], [])
        uncertain = verified(media)
        uncertain["segments"][0]["kind"] = "uncertain"
        self.save("T01", uncertain)
        self.assertEqual(evaluate_reference_session(self.session(media), self.references, self.events)["results"], [])

    def test_direct_scorer_refuses_unreviewed_or_unusable_records(self):
        media = clip()
        for status in ("draft", "unusable"):
            reference = make_draft(media)
            reference["review_status"] = status
            with self.subTest(status=status), self.assertRaises(ValueError):
                score_verified_clip(reference, media, [])

    def test_session_identifiers_modes_and_versions_are_validated(self):
        for session in (self.session(clip(), clip()), self.session(clip("../outside")),
                        {"schema_version": True, "clips": []}):
            with self.subTest(session=session), self.assertRaises(ValueError):
                evaluate_reference_session(session, self.references, self.events)
        with self.assertRaises(ValueError):
            evaluate_reference_session(self.session(), self.references, self.events, modes=("offline", "offline"))

    def test_cli_missing_session_writes_pending_report_without_optional_packages(self):
        output = self.root / "report-output"
        completed = subprocess.run([
            sys.executable, "-S", str(ROOT / "scripts/evaluate_speech_references.py"),
            "--session", str(self.root / "absent-session.json"), "--output-dir", str(output),
        ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PENDING_HUMAN_REVIEW")
        self.assertEqual(report["results"], [])
        self.assertEqual((output / "results.jsonl").read_text(encoding="utf-8"), "")
        self.assertNotIn("transcript_score", report)

    def test_cli_cannot_overwrite_reference_or_event_inputs(self):
        for output in (self.references, self.events):
            with self.subTest(output=output):
                stderr = io.StringIO()
                with redirect_stderr(stderr), self.assertRaises(SystemExit) as error:
                    evaluate_main(["--session", str(self.root / "session.json"),
                                   "--references-dir", str(self.references), "--events-dir", str(self.events),
                                   "--output-dir", str(output)])
                self.assertEqual(error.exception.code, 2)
                self.assertIn("overwrite", stderr.getvalue())
                self.assertFalse((output / "report.json").exists())

    def test_cli_existing_unreviewed_session_also_stays_pending_without_packages(self):
        media = clip()
        self.save("T01", make_draft(media))
        session_path = self.root / "session.json"
        session_path.write_text(json.dumps(self.session(media)), encoding="utf-8")
        output = self.root / "report-output"
        completed = subprocess.run([
            sys.executable, "-S", str(ROOT / "scripts/evaluate_speech_references.py"),
            "--session", str(session_path), "--events-dir", str(self.events),
            "--output-dir", str(output),
        ], cwd=ROOT, capture_output=True, text=True, encoding="utf-8", timeout=10)
        self.assertEqual(completed.returncode, 0, completed.stderr)
        report = json.loads((output / "report.json").read_text(encoding="utf-8"))
        self.assertEqual(report["status"], "PENDING_HUMAN_REVIEW")
        self.assertEqual(report["results"], [])
        self.assertEqual(report["eligible_verified_cases"], [])


@unittest.skipUnless(HAS_SCORES, "optional jiwer/OpenCC packages are not installed")
class VerifiedReferenceMetricTests(ReferenceFixture, unittest.TestCase):
    def run_file(self, media, mode="offline", events=None, *, metadata=None):
        path = self.events / f"{media['case_id']}_{mode}.jsonl"
        path.write_text("".join(json.dumps(event.to_dict()) + "\n" for event in events or []), encoding="utf-8")
        meta = {"source_sha256": media["source_sha256"], "mode": mode,
                "media_offset_ms": 0, "audio_offset_ms": media["audio_offset_ms"],
                "model_revision": "fake-model", "requested_language": "en", "beam_size": 3,
                "quality_policy": "fake-quality-policy", "device": "cpu",
                "compute_type": "int8", "cpu_threads": 4}
        meta.update(metadata or {})
        path.with_suffix(".meta.json").write_text(json.dumps(meta), encoding="utf-8")

    def test_whole_clip_text_score_does_not_drop_poorly_timed_correct_words(self):
        media = clip()
        reference = verified(media, [
            row("T01-q1", 0, 2000, kind="no_speech", text=""),
            row("T01-s1", 2000, 3000, text="a red cup", classes=["cup"]),
            row("T01-q2", 3000, 4000, kind="no_speech", text=""),
        ])
        result = score_verified_clip(reference, media, [SpeechEvent(100, 800, 4300, "a red cup")], language="en")
        self.assertEqual(result["transcript_score"]["wer"], 0)
        self.assertEqual(result["semantic_metrics"]["mentioned_classes"]["fn"], 1)
        self.assertEqual(result["semantic_metrics"]["mentioned_classes"]["fp"], 1)
        self.assertEqual(result["temporal_assignment"]["human_speech_intervals_without_matched_event"], 1)
        self.assertIn("approximate", result["temporal_assignment"]["limitation"])

    def test_missed_speech_counts_deletions_and_false_negatives_with_no_latency(self):
        media = clip()
        result = score_verified_clip(verified(media), media, [], language="en")
        self.assertEqual(result["transcript_score"]["wer"], 1)
        self.assertEqual(result["transcript_score"]["deletions"], 2)
        self.assertEqual(result["semantic_metrics"]["mentioned_classes"]["fn"], 1)
        self.assertEqual(result["human_boundary_latency_ms"]["overall"]["count"], 0)
        self.assertIsNone(result["human_boundary_latency_ms"]["overall"]["median"])

    def test_human_nonspeech_truth_reports_insertions_and_false_speech_not_percentage(self):
        media = clip()
        reference = verified(media, [row("T01-q1", kind="no_speech", text="")])
        result = score_verified_clip(reference, media, [SpeechEvent(200, 800, 4500, "apple cup")], language="en")
        self.assertEqual(result["transcript_score"]["reference_units"], 0)
        self.assertEqual(result["transcript_score"]["insertions"], 2)
        self.assertEqual(result["no_speech_diagnostics"]["events_assigned_to_no_speech"], 1)
        self.assertEqual(result["no_speech_diagnostics"]["false_speech_interval_rate"], 1)
        self.assertEqual(result["semantic_metrics"]["mentioned_classes"]["fp"], 2)
        self.assertIn("inserted units", result["transcript_score"]["empty_reference_policy"])

    def test_verified_chinese_reference_reports_raw_and_script_normalized_cer_separately(self):
        media = clip(language="zh")
        reference = verified(media, [row(text="蘋果杯子", classes=["apple", "cup"])])
        result = score_verified_clip(reference, media, [SpeechEvent(0, 1000, 4500, "苹果杯子")], language="zh")
        self.assertEqual(result["transcript_score"]["reference_units"], 4)
        self.assertEqual(result["transcript_score"]["cer"], 0.25)
        self.assertEqual(result["transcript_score"]["t2s"]["cer"], 0)
        self.assertEqual(result["semantic_metrics"]["mentioned_classes"]["tp"], 2)

    def test_current_extractor_replaces_stale_stored_semantic_labels(self):
        media = clip()
        reference = verified(media, [row(classes=["apple"], text="apple")])
        event = SpeechEvent(0, 1000, 4500, "apple", ["cup"], ["PRESENTATION_LANGUAGE"])
        result = score_verified_clip(reference, media, [event], language="en")
        self.assertEqual(result["semantic_source"], "current_rules_on_published_asr_text")
        self.assertEqual(len(result["lexicon_sha256"]), 64)
        self.assertEqual(result["semantic_metrics"]["mentioned_classes"],
                         {"tp": 1, "fp": 0, "fn": 0, "precision": 1, "recall": 1, "f1": 1})
        self.assertEqual(result["semantic_metrics"]["cue_types"]["fp"], 0)
        self.assertEqual(event.mentioned_classes, ["cup"])

    def test_max_overlap_counts_unmatched_predictions_and_human_boundary_negative_lag(self):
        media = clip(audio_offset_ms=85)
        reference = verified(media, [
            row("T01-s1", 0, 1000, text="apple", classes=["apple"]),
            row("T01-q1", 1000, 2000, text="", kind="no_speech"),
            row("T01-s2", 2000, 3000, text="cup", classes=["cup"]),
            row("T01-q2", 3000, 4000, text="", kind="no_speech"),
        ])
        events = [SpeechEvent(100 + 85, 400 + 85, 600 + 85, "apple cup"),
                  SpeechEvent(3500 + 85, 3900 + 85, 4400 + 85, "apple")]
        result = score_verified_clip(reference, media, events, language="en")
        semantic = result["semantic_metrics"]["mentioned_classes"]
        self.assertEqual((semantic["tp"], semantic["fp"], semantic["fn"]), (1, 2, 1))
        self.assertEqual(result["human_boundary_latency_ms"]["mentioned_classes"]["min"], -400)
        self.assertEqual(result["human_boundary_latency_ms"]["mentioned_classes"]["negative_count"], 1)
        self.assertEqual(result["no_speech_diagnostics"]["intervals_with_predicted_speech"], 1)

    def test_one_broad_event_is_assigned_once_and_boundary_overlap_is_not_automatically_hallucination(self):
        media = clip()
        reference = verified(media, [row("T01-s1", 0, 1500, text="apple", classes=["apple"]),
                                     row("T01-s2", 1500, 3000, text="cup", classes=["cup"]),
                                     row("T01-q1", 3000, 4000, text="", kind="no_speech")])
        result = score_verified_clip(reference, media, [SpeechEvent(0, 3100, 4500, "apple cup")], language="en")
        self.assertEqual(result["temporal_assignment"]["ambiguous_multi_speech_overlap_events"], 1)
        self.assertEqual(result["interval_semantic_diagnostics"][0]["matched_event_count"], 1)
        self.assertEqual(result["interval_semantic_diagnostics"][1]["matched_event_count"], 0)
        self.assertEqual(result["no_speech_diagnostics"]["predicted_events_overlapping_no_speech"], 1)
        self.assertEqual(result["no_speech_diagnostics"]["events_assigned_to_no_speech"], 0)

    def test_missing_modes_are_na_and_cross_mode_aggregates_use_identical_cases(self):
        first, second = clip("T01"), clip("T02")
        for media in (first, second):
            self.save(media["case_id"], verified(media))
        self.run_file(first, events=[SpeechEvent(0, 1000, 4500, "a cup")])
        for mode in ("offline", "causal", "agreement"):
            self.run_file(second, mode, events=[SpeechEvent(0, 1000, 4500, "a cup")])
        report = evaluate_reference_session(self.session(first, second), self.references, self.events)
        self.assertEqual(report["status"], "PARTIAL")
        self.assertEqual(report["common_case_ids_for_cross_mode_aggregate"], ["T02"])
        self.assertEqual(len([result for result in report["results"] if result["status"] == "N/A"]), 2)
        for aggregate in report["cross_mode_aggregate"].values():
            self.assertEqual(aggregate["case_ids"], ["T02"])
            self.assertEqual(aggregate["transcript_scores_by_language"]["en"]["reference_units"], 2)
        self.assertEqual(report["reference_method_counts"], {"independent_listening": 2})

    def test_source_clock_or_mode_mismatch_cannot_produce_accuracy(self):
        media = clip(audio_offset_ms=85)
        self.save("T01", verified(media))
        for changes in ({"source_sha256": "b" * 64}, {"audio_offset_ms": 0}, {"mode": "causal"}):
            with self.subTest(changes=changes):
                self.run_file(media, metadata=changes)
                report = evaluate_reference_session(self.session(media), self.references, self.events, modes=("offline",))
                self.assertEqual(report["status"], "NO_VALID_RUNS")
                self.assertEqual(report["results"][0]["status"], "INVALID_RUN")
                self.assertNotIn("transcript_score", report["results"][0])

    def test_assisted_and_exposed_references_never_get_an_independent_label(self):
        first, second = clip("T01"), clip("T02")
        self.save("T01", verified(first, method="asr_assisted"))
        self.save("T02", verified(second))
        (self.references.parent / "draft_exposure.json").write_text('{"T02": true}', encoding="utf-8")
        for media in (first, second):
            self.run_file(media, events=[])
        report = evaluate_reference_session(self.session(first, second), self.references, self.events, modes=("offline",))
        self.assertEqual(report["reference_method_counts"], {"asr_assisted": 2})
        self.assertTrue(all(result["reference_method"] == "asr_assisted" for result in report["results"]))
        self.assertTrue(report["results"][1]["asr_draft_exposed"])

    def test_full_clip_text_and_notes_are_not_republished_in_metric_result(self):
        media = clip()
        reference = verified(media, [row(text="The synthetic private dialogue is not an output field.", classes=[])])
        reference["notes"] = "A private reviewer note"
        result = score_verified_clip(reference, media, [], language="en")
        encoded = json.dumps(result)
        self.assertNotIn(reference["segments"][0]["text"], encoded)
        self.assertNotIn(reference["notes"], encoded)
        self.assertNotIn("Synthetic unit-test reviewer", encoded)

    def test_all_nonspeech_common_cases_have_null_aggregate_percentages(self):
        media = clip()
        self.save("T01", verified(media, [row(kind="no_speech", text="")]))
        self.run_file(media, events=[SpeechEvent(0, 1000, 4500, "apple")])
        report = evaluate_reference_session(self.session(media), self.references, self.events, modes=("offline",))
        score = report["cross_mode_aggregate"]["offline"]["transcript_scores_by_language"]["en"]
        self.assertIsNone(score["wer"])
        self.assertEqual(score["reference_units"], 0)
        self.assertEqual(score["insertions"], 1)

    def test_different_model_language_beam_or_quality_policy_prevents_paired_aggregate(self):
        media = clip()
        self.save("T01", verified(media))
        for field, changed in (("model_revision", "another-model"), ("requested_language", "zh"),
                               ("beam_size", 5), ("quality_policy", "another-quality-policy")):
            with self.subTest(field=field):
                for mode in ("offline", "causal", "agreement"):
                    self.run_file(media, mode, [SpeechEvent(0, 1000, 4500, "a cup")],
                                  metadata={field: changed} if mode == "causal" else {})
                report = evaluate_reference_session(self.session(media), self.references, self.events)
                self.assertEqual([result["status"] for result in report["results"]], ["SCORED"] * 3)
                self.assertEqual(report["common_case_ids_with_valid_runs"], ["T01"])
                self.assertEqual(report["common_case_ids_for_cross_mode_aggregate"], [])
                self.assertEqual(report["cross_mode_aggregate"], {})
                exclusion = report["cross_mode_configuration_exclusions"][0]
                self.assertEqual(exclusion["reason"], "CONFIGURATION_MISMATCH")
                self.assertIn(field, exclusion["accuracy_setting_differences"])
                self.assertEqual(exclusion["differing_settings"][field]["causal"], changed)

    def test_device_precision_or_thread_differences_are_flagged_and_not_paired(self):
        media = clip()
        self.save("T01", verified(media))
        for field, changed in (("device", "cuda"), ("compute_type", "float16"), ("cpu_threads", 8)):
            with self.subTest(field=field):
                for mode in ("offline", "causal", "agreement"):
                    self.run_file(media, mode, [], metadata={field: changed} if mode == "agreement" else {})
                report = evaluate_reference_session(self.session(media), self.references, self.events)
                self.assertEqual(report["cross_mode_aggregate"], {})
                exclusion = report["cross_mode_configuration_exclusions"][0]
                self.assertIn(field, exclusion["performance_setting_differences"])
                self.assertEqual(len(report["results"]), 3)

    def test_mode_specific_windows_are_allowed_but_unverified_settings_are_not_assumed_equal(self):
        media = clip()
        self.save("T01", verified(media))
        for mode in ("offline", "causal", "agreement"):
            settings = ({"window_ms": None} if mode == "offline" else
                        {"window_ms": 4000} if mode == "causal" else
                        {"step_ms": 2000, "buffer_ms": 15000})
            self.run_file(media, mode, [], metadata={"requested_language": None, **settings})
        report = evaluate_reference_session(self.session(media), self.references, self.events)
        self.assertEqual(report["common_case_ids_for_cross_mode_aggregate"], ["T01"])
        self.assertEqual(report["cross_mode_configuration_exclusions"], [])
        for mode in ("offline", "causal", "agreement"):
            meta_path = (self.events / f"T01_{mode}.jsonl").with_suffix(".meta.json")
            metadata = json.loads(meta_path.read_text(encoding="utf-8"))
            del metadata["quality_policy"]
            meta_path.write_text(json.dumps(metadata), encoding="utf-8")
        report = evaluate_reference_session(self.session(media), self.references, self.events)
        self.assertEqual(report["cross_mode_aggregate"], {})
        exclusion = report["cross_mode_configuration_exclusions"][0]
        self.assertEqual(exclusion["reason"], "UNVERIFIED_CONFIGURATION")
        self.assertEqual(exclusion["missing_or_unverified_settings"]["quality_policy"],
                         ["offline", "causal", "agreement"])


if __name__ == "__main__":
    unittest.main()

"""Human-reference guards use synthetic records, never real user approvals."""

from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

from src.speech.references import ReferenceStore, atomic_json, make_draft, validate_reference


class SpeechReferenceTests(unittest.TestCase):
    def setUp(self):
        self.clip = {"case_id": "T01", "duration_ms": 1000, "source_sha256": "a" * 64}

    def verified(self):
        record = make_draft(self.clip)
        record.update(review_status="verified", reviewer="synthetic-test", human_confirmed=True, coverage_complete=True)
        record["segments"][0].update(kind="speech", text="hello", timing_verified=True, semantic_verified=True)
        return record

    def test_generated_reference_is_ineligible_until_explicit_human_assertion(self):
        record = make_draft(self.clip)
        self.assertEqual(record["review_status"], "draft")
        self.assertEqual(record["segments"][0]["kind"], "uncertain")
        self.assertFalse(record["human_confirmed"])
        self.assertFalse(record["coverage_complete"])
        with self.assertRaisesRegex(ValueError, "not human-verified"):
            validate_reference(record, self.clip, require_verified=True)

    def test_every_human_confirmation_is_required(self):
        for field in ("human_confirmed", "coverage_complete", "reviewer"):
            record = self.verified()
            record[field] = "" if field == "reviewer" else False
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_reference(record, self.clip)
        for field in ("timing_verified", "semantic_verified"):
            record = self.verified()
            record["segments"][0][field] = False
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_reference(record, self.clip)

    def test_verified_intervals_must_be_ordered_disjoint_and_complete_asserted(self):
        record = self.verified()
        row = deepcopy(record["segments"][0])
        row.update(segment_id="T01-s002", start_ms=900, end_ms=1000)
        record["segments"].append(row)
        with self.assertRaisesRegex(ValueError, "重叠"):
            validate_reference(record, self.clip)
        record["review_status"] = "draft"
        validate_reference(record, self.clip)
        record["segments"][0]["end_ms"] = 900
        record["review_status"] = "verified"
        validate_reference(record, self.clip)

    def test_no_speech_must_have_empty_transcript_and_labels(self):
        record = self.verified()
        record["segments"][0].update(kind="no_speech", text="")
        validate_reference(record, self.clip, require_verified=True)
        for changes in ({"text": "hello"}, {"mentioned_classes": ["apple"]}, {"cue_types": ["ENDING_LANGUAGE"]}):
            value = deepcopy(record)
            value["segments"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_reference(value, self.clip)

    def test_uncertain_and_blank_speech_cannot_be_verified(self):
        for changes in ({"kind": "uncertain"}, {"text": " "}):
            record = self.verified()
            record["segments"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_reference(record, self.clip)

    def test_exact_fields_source_and_types_are_validated(self):
        for change in ({"source_sha256": "b" * 64}, {"case_id": "T02"},
                       {"schema_version": True}, {"human_confirmed": 1}, {"review_status": "auto"},
                       {"reference_method": "model_generated"}):
            value = self.verified()
            value.update(change)
            with self.subTest(change=change), self.assertRaises(ValueError):
                validate_reference(value, self.clip)
        for changes in ({"start_ms": True}, {"end_ms": 1001}, {"start_ms": 1000},
                        {"mentioned_classes": ["apple", "apple"]}, {"cue_types": ["UNKNOWN"]}):
            value = self.verified()
            value["segments"][0].update(changes)
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                validate_reference(value, self.clip)

    def test_store_preserves_snapshots_and_records_draft_exposure_permanently(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session.json"
            atomic_json(session, {"schema_version": 1, "session_id": "synthetic", "scope": "development", "clips": [self.clip]})
            store = ReferenceStore(session)
            self.assertEqual(store.progress(), {"verified": 0, "total": 1, "unusable": 0})
            value = self.verified()
            saved = store.save("T01", value)
            saved["segments"][0]["text"] = "external mutation"
            self.assertEqual(store.get("T01")["segments"][0]["text"], "hello")
            store.mark_draft_exposed("T01")
            self.assertEqual(ReferenceStore(session).get("T01")["reference_method"], "asr_assisted")
            value["reference_method"] = "independent_listening"
            self.assertEqual(store.save("T01", value)["reference_method"], "asr_assisted")
            audit = (Path(temp) / "review_audit.jsonl").read_text()
            self.assertNotIn("hello", audit)
            self.assertNotIn("synthetic-test", audit)

    def test_invalid_save_does_not_replace_previous_reference(self):
        with tempfile.TemporaryDirectory() as temp:
            session = Path(temp) / "session.json"
            atomic_json(session, {"schema_version": 1, "session_id": "synthetic", "scope": "development", "clips": [self.clip]})
            store = ReferenceStore(session)
            store.save("T01", self.verified())
            before = (Path(temp) / "references/T01.json").read_bytes()
            bad = self.verified()
            bad["human_confirmed"] = False
            with self.assertRaises(ValueError):
                store.save("T01", bad)
            self.assertEqual((Path(temp) / "references/T01.json").read_bytes(), before)


if __name__ == "__main__":
    unittest.main()

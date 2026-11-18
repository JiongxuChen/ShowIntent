"""Agreement CLI publishes flushed journal lines during a mocked ASR run."""

from contextlib import redirect_stderr, redirect_stdout
import io
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

import run_speech
from src.speech.asr import ASRRun
from src.speech.events import SpeechEvent


class StreamCLITests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.audio = self.root / "source.mp4"
        self.audio.write_bytes(b"fake media; decoding is mocked")
        self.model = self.root / "model"
        self.model.mkdir()
        (self.model / "model.bin").write_bytes(b"fake weights; no model is loaded")
        self.output = self.root / "events.jsonl"
        self.journal = self.root / "incremental.jsonl"

    def arguments(self, *extra):
        return ["--backend", "faster-whisper", "--mode", "agreement", "--audio", str(self.audio),
                "--model", str(self.model), "--output", str(self.output), *map(str, extra)]

    def invoke(self, args):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = run_speech.main(args)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def records(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def result(self, events):
        return ASRRun(
            events=list(events), windows=[{"window_index": 0, "event_count": len(events)}],
            transcripts=[{"text": event.text} for event in events],
            metadata={"mode": "agreement", "compute_rtf": 0.1, "event_count": len(events),
                      "timing_policy": "local_agreement_incremental"},
        )

    def assert_cli_error(self, response):
        self.assertEqual(response[0], 2, response[2])
        self.assertIn("error:", response[2])
        self.assertNotIn("Traceback", response[2])

    def test_journal_contains_flushed_lines_before_backend_returns_and_completes(self):
        first = SpeechEvent(100, 500, 1200, "大家看看杯子", ["cup"], ["PRESENTATION_LANGUAGE"])
        second = SpeechEvent(700, 1000, 2200, "Here is the apple.", ["apple"])
        status_path = self.journal.with_suffix(".status.json")

        def backend(*args, **kwargs):
            self.assertEqual(self.records(status_path)[0]["status"], "RUNNING")
            self.assertEqual(self.journal.read_bytes(), b"")
            callback = kwargs["on_events"]
            callback([first])
            self.assertEqual(self.records(self.journal), [first.to_dict()])
            self.assertFalse(self.output.exists())
            callback([second])
            self.assertEqual(self.records(self.journal), [first.to_dict(), second.to_dict()])
            self.assertFalse(self.output.exists())
            return self.result([first, second])

        lexicon = self.root / "lexicon.json"
        lexicon.write_text("{}", encoding="utf-8")
        with patch("src.speech.incremental.run_incremental_asr", side_effect=backend) as mocked:
            response = self.invoke(self.arguments("--journal", self.journal, "--pace", "--step-ms", 1000,
                                                  "--buffer-ms", 8000, "--language", "en",
                                                  "--media-offset-ms", 300, "--lexicon", lexicon,
                                                  "--no-language-cues"))
        self.assertEqual(response[0], 0, response[2])
        self.assertIn("ASR agreement", response[1])
        options = mocked.call_args.kwargs
        self.assertTrue(options["pace"])
        self.assertEqual(options["step_ms"], 1000)
        self.assertEqual(options["buffer_ms"], 8000)
        self.assertEqual(options["media_offset_ms"], 300)
        self.assertEqual(options["lexicon_path"], lexicon)
        self.assertFalse(options["extract_language"])
        self.assertEqual(self.records(self.output), [first.to_dict(), second.to_dict()])
        status = self.records(status_path)[0]
        self.assertEqual((status["status"], status["event_count"]), ("COMPLETE", 2))
        self.assertEqual(status["availability"], "in_memory_publication_before_transport")

    def test_backend_failure_preserves_partial_journal_and_explicit_failed_sidecar(self):
        event = SpeechEvent(0, 500, 1000, "committed before failure")
        self.output.write_text("old final output", encoding="utf-8")

        def backend(*args, **kwargs):
            kwargs["on_events"]([event])
            self.assertEqual(self.records(self.journal), [event.to_dict()])
            raise RuntimeError("fake inference failed after first commit")

        with patch("src.speech.incremental.run_incremental_asr", side_effect=backend):
            response = self.invoke(self.arguments("--journal", self.journal))
        self.assert_cli_error(response)
        self.assertEqual(self.records(self.journal), [event.to_dict()])
        status = self.records(self.journal.with_suffix(".status.json"))[0]
        self.assertEqual((status["status"], status["event_count"]), ("FAILED", 1))
        self.assertIn("fake inference failed", status["error"])
        self.assertEqual(self.output.read_text(encoding="utf-8"), "old final output")

    def test_existing_journal_or_status_is_never_truncated_and_backend_is_not_called(self):
        for existing in (self.journal, self.journal.with_suffix(".status.json")):
            with self.subTest(existing=existing.name):
                existing.write_bytes(b"keep previous run")
                with patch("src.speech.incremental.run_incremental_asr") as backend:
                    response = self.invoke(self.arguments("--journal", self.journal))
                self.assert_cli_error(response)
                self.assertIn("already exists", response[2])
                backend.assert_not_called()
                self.assertEqual(existing.read_bytes(), b"keep previous run")
                existing.unlink()

    def test_journal_and_sidecar_paths_cannot_collide_with_input_or_final_outputs(self):
        for journal in (self.audio, self.output, self.output.with_suffix(".meta.json"),
                        self.output.with_suffix(".transcript.jsonl"),
                        self.output.with_suffix(".windows.jsonl")):
            with self.subTest(journal=journal.name), patch("src.speech.incremental.run_incremental_asr") as backend:
                response = self.invoke(self.arguments("--journal", journal))
                self.assert_cli_error(response)
                backend.assert_not_called()
                self.assertEqual(self.audio.read_bytes(), b"fake media; decoding is mocked")
        # A sidecar can collide with a separately selected source too.
        source = self.journal.with_suffix(".status.json")
        source.write_bytes(b"source bytes")
        args = self.arguments("--journal", self.journal)
        args[args.index("--audio") + 1] = str(source)
        with patch("src.speech.incremental.run_incremental_asr") as backend:
            self.assert_cli_error(self.invoke(args))
            backend.assert_not_called()
        self.assertEqual(source.read_bytes(), b"source bytes")

    def test_journal_cannot_pollute_model_or_overwrite_lexicon(self):
        for journal in (self.model / "model.bin", self.model / "nested/journal.jsonl"):
            with self.subTest(journal=journal), patch("src.speech.incremental.run_incremental_asr") as backend:
                self.assert_cli_error(self.invoke(self.arguments("--journal", journal)))
                backend.assert_not_called()
        self.assertEqual((self.model / "model.bin").read_bytes(), b"fake weights; no model is loaded")
        lexicon = self.journal.with_suffix(".status.json")
        lexicon.write_text('{"keep":"dictionary"}', encoding="utf-8")
        with patch("src.speech.incremental.run_incremental_asr") as backend:
            self.assert_cli_error(self.invoke(self.arguments("--journal", self.journal, "--lexicon", lexicon)))
            backend.assert_not_called()
        self.assertEqual(lexicon.read_text(encoding="utf-8"), '{"keep":"dictionary"}')

    def test_optional_journal_absent_passes_no_callback_and_no_pacing(self):
        with patch("src.speech.incremental.run_incremental_asr", return_value=self.result([])) as backend:
            response = self.invoke(self.arguments())
        self.assertEqual(response[0], 0, response[2])
        self.assertIsNone(backend.call_args.kwargs["on_events"])
        self.assertFalse(backend.call_args.kwargs["pace"])
        self.assertFalse(self.journal.exists())
        self.assertFalse(self.journal.with_suffix(".status.json").exists())

    def test_journal_and_pace_require_agreement_asr_mode(self):
        cases = (
            self.arguments("--mode", "causal", "--journal", self.journal),
            self.arguments("--mode", "offline", "--pace"),
            ["--audio", str(self.audio), "--journal", str(self.journal)],
            ["--audio", str(self.audio), "--pace"],
        )
        for args in cases:
            with self.subTest(args=args), patch("src.speech.incremental.run_incremental_asr") as backend:
                self.assert_cli_error(self.invoke(args))
                backend.assert_not_called()
        self.assertFalse(self.journal.exists())


if __name__ == "__main__":
    unittest.main()

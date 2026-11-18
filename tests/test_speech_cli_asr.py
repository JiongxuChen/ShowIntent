"""CLI integration around a fake ASR result; no model/network dependency."""

from contextlib import redirect_stderr, redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import run_speech
from src.speech.asr import ASRRun
from src.speech.events import SpeechEvent


class ASRCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.source = self.root / "demonstration.mp4"
        self.source.write_bytes(b"fake media; backend is mocked")
        self.model = self.root / "local-model"
        self.model.mkdir()
        (self.model / "model.bin").write_bytes(b"fake model; never loaded")
        self.output = self.root / "events.jsonl"

    def result(self):
        return ASRRun(
            events=[SpeechEvent(100, 800, 1300, "Look at this cup.", ["cup"], ["PRESENTATION_LANGUAGE"])],
            windows=[{
                "window_index": 0, "start_sample": 0, "end_sample": 16000,
                "received_until_ms": 1000, "processing_start_ms": 1000,
                "available_at_ms": 1300, "compute_ms": 300, "event_count": 1,
            }],
            transcripts=[{"window_index": 0, "start_ms": 100, "end_ms": 800,
                          "text": "Look at this cup."}],
            metadata={"stage": "D2-D4", "mode": "causal", "compute_rtf": 0.3,
                      "source_kind": "pretrained_asr_from_media", "event_count": 1},
        )

    def arguments(self, *extra):
        return ["--backend", "faster-whisper", "--audio", str(self.source),
                "--model", str(self.model), "--output", str(self.output), *map(str, extra)]

    def invoke(self, arguments):
        stdout, stderr = io.StringIO(), io.StringIO()
        with redirect_stdout(stdout), redirect_stderr(stderr):
            try:
                code = run_speech.main(arguments)
            except SystemExit as exc:
                code = exc.code
        return code, stdout.getvalue(), stderr.getvalue()

    def records(self, suffix=None):
        path = self.output if suffix is None else self.output.with_suffix(suffix)
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def assert_error(self, response):
        code, _, stderr = response
        self.assertEqual(code, 2, stderr)
        self.assertIn("error:", stderr)
        self.assertNotIn("Traceback", stderr)

    def test_asr_cli_exports_events_transcripts_windows_and_measured_provenance(self):
        lexicon = self.root / "custom-lexicon.json"
        lexicon.write_text("{}", encoding="utf-8")
        with patch("src.speech.asr.run_asr", return_value=self.result()) as backend:
            response = self.invoke(self.arguments(
                "--mode", "causal", "--window-ms", 6000, "--language", "en",
                "--media-offset-ms", 700, "--beam-size", 5, "--device", "cpu",
                "--compute-type", "int8", "--cpu-threads", 2, "--lexicon", lexicon,
            ))
        self.assertEqual(response[0], 0, response[2])
        self.assertIn("ASR causal", response[1])
        backend.assert_called_once_with(
            self.source, self.model, mode="causal", window_ms=6000,
            media_offset_ms=700, language="en", beam_size=5, device="cpu",
            compute_type="int8", cpu_threads=2, extract_language=True, lexicon_path=lexicon,
        )
        events = self.records()
        self.assertEqual(SpeechEvent.from_dict(events[0]), self.result().events[0])
        self.assertEqual(self.records(".transcript.jsonl"), self.result().transcripts)
        self.assertEqual(self.records(".windows.jsonl"), self.result().windows)
        metadata = self.records(".meta.json")[0]
        self.assertEqual(metadata["source_kind"], "pretrained_asr_from_media")
        self.assertEqual(metadata["source_sha256"], hashlib.sha256(self.source.read_bytes()).hexdigest())
        self.assertNotIn("simulated_delay_ms", metadata)
        self.assertEqual(metadata["lexicon_sha256"], hashlib.sha256(lexicon.read_bytes()).hexdigest())

    def test_asr_language_extraction_can_be_disabled_for_ablation(self):
        with patch("src.speech.asr.run_asr", return_value=self.result()) as backend:
            response = self.invoke(self.arguments("--no-language-cues"))
        self.assertEqual(response[0], 0, response[2])
        self.assertFalse(backend.call_args.kwargs["extract_language"])
        self.assertIsNone(self.records(".meta.json")[0]["lexicon_sha256"])

    def test_backend_failures_are_actionable_and_preserve_existing_outputs(self):
        paths = [self.output] + [self.output.with_suffix(suffix) for suffix in
                                 (".meta.json", ".transcript.jsonl", ".windows.jsonl")]
        for path in paths:
            path.write_text(f"existing {path.name}", encoding="utf-8")
        originals = {path: path.read_bytes() for path in paths}
        for error in (ValueError("bad model"), FileNotFoundError("missing media"),
                      RuntimeError("unsupported compute type"), ImportError("faster_whisper missing")):
            with self.subTest(error=type(error).__name__), patch(
                "src.speech.asr.run_asr", side_effect=error,
            ):
                response = self.invoke(self.arguments())
                self.assert_error(response)
                for path, original in originals.items():
                    self.assertEqual(path.read_bytes(), original)
                if isinstance(error, ImportError):
                    self.assertIn("requirements-asr.txt", response[2])

    def test_asr_rejects_mock_or_reader_options_before_calling_backend(self):
        for extra in (("--subtitles", self.source), ("--audio-log", self.root / "log.jsonl")):
            with self.subTest(extra=extra), patch("src.speech.asr.run_asr") as backend:
                self.assert_error(self.invoke(self.arguments(*extra)))
                backend.assert_not_called()
        with patch("src.speech.asr.run_asr") as backend:
            self.assert_error(self.invoke(["--backend", "faster-whisper", "--subtitles", str(self.source)]))
            backend.assert_not_called()

    def test_all_asr_output_files_are_protected_against_input_collisions(self):
        for collision in (self.output, self.output.with_suffix(".meta.json"),
                          self.output.with_suffix(".transcript.jsonl"),
                          self.output.with_suffix(".windows.jsonl")):
            with self.subTest(collision=collision.name), patch("src.speech.asr.run_asr") as backend:
                collision.write_bytes(b"input audio bytes")
                args = self.arguments()
                args[args.index("--audio") + 1] = str(collision)
                self.assert_error(self.invoke(args))
                backend.assert_not_called()
                self.assertEqual(collision.read_bytes(), b"input audio bytes")

    def test_asr_outputs_cannot_overwrite_lexicon_input(self):
        lexicon = self.output.with_suffix(".meta.json")
        lexicon.write_text('{"config": "keep me"}', encoding="utf-8")
        with patch("src.speech.asr.run_asr") as backend:
            self.assert_error(self.invoke(self.arguments("--lexicon", lexicon)))
            backend.assert_not_called()
        self.assertEqual(lexicon.read_text(encoding="utf-8"), '{"config": "keep me"}')

    def test_asr_outputs_cannot_overwrite_or_pollute_model_directory(self):
        original = (self.model / "model.bin").read_bytes()
        for output in (self.model / "model.bin", self.model / "events.jsonl",
                       self.model / "nested/events.jsonl"):
            with self.subTest(output=output), patch("src.speech.asr.run_asr") as backend:
                args = self.arguments()
                args[args.index("--output") + 1] = str(output)
                response = self.invoke(args)
                self.assert_error(response)
                self.assertIn("model", response[2])
                backend.assert_not_called()
                self.assertEqual((self.model / "model.bin").read_bytes(), original)
        self.assertEqual(list(self.model.iterdir()), [self.model / "model.bin"])

    def test_existing_d1_subtitles_never_invoke_asr(self):
        source = Path(__file__).resolve().parents[1] / "fixtures/member_d/demo.srt"
        with patch("src.speech.asr.run_asr") as backend:
            response = self.invoke(["--subtitles", str(source), "--output", str(self.output)])
            backend.assert_not_called()
        self.assertEqual(response[0], 0, response[2])
        self.assertTrue(all(row["mentioned_classes"] == [] for row in self.records()))
        self.assertEqual(self.records(".meta.json")[0]["source_kind"], "handwritten_subtitle_mock")


if __name__ == "__main__":
    unittest.main()

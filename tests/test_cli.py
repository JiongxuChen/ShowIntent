"""End-to-end CLI checks against checked fixtures and temporary recordings."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
import wave


ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "fixtures" / "member_d"


class SpeechCLITests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def run_cli(self, *arguments, no_site=False):
        command = [sys.executable]
        if no_site:
            command.append("-S")
        command.extend([str(ROOT / "run_speech.py"), *(str(arg) for arg in arguments)])
        env = dict(os.environ, PYTHONUTF8="1")
        return subprocess.run(
            command, cwd=ROOT, env=env, text=True, encoding="utf-8",
            capture_output=True, timeout=20, check=False,
        )

    def read_jsonl(self, path):
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

    def write_wav(self, frames=4000):
        path = self.directory / "recording.wav"
        with wave.open(str(path), "wb") as recording:
            recording.setnchannels(1)
            recording.setsampwidth(2)
            recording.setframerate(16000)
            recording.writeframes(b"\0\0" * frames)
        return path

    def assert_cli_error(self, result):
        self.assertEqual(result.returncode, 2, result.stdout + result.stderr)
        self.assertIn("error:", result.stderr)
        self.assertNotIn("Traceback", result.stderr)

    def test_subtitle_modes_match_checked_fixtures_and_record_mock_provenance(self):
        for source_name, expected_name in (
            ("demo.srt", "expected_from_srt.jsonl"),
            ("manual_annotations.jsonl", "speech_events.jsonl"),
        ):
            with self.subTest(source=source_name):
                source = FIXTURES / source_name
                output = self.directory / (source_name + ".events.jsonl")
                result = self.run_cli("--subtitles", source, "--output", output)
                self.assertEqual(result.returncode, 0, result.stderr)
                self.assertEqual(self.read_jsonl(output), self.read_jsonl(FIXTURES / expected_name))
                metadata = json.loads(output.with_suffix(".meta.json").read_text(encoding="utf-8"))
                self.assertEqual(metadata["stage"], "D1")
                self.assertEqual(metadata["source_kind"], "handwritten_subtitle_mock")
                self.assertEqual(metadata["timing_kind"], "simulated_availability_not_measured_asr")
                self.assertEqual(metadata["semantic_annotations"], "manual_passthrough_or_empty_no_extractor")
                self.assertEqual(metadata["source_sha256"], hashlib.sha256(source.read_bytes()).hexdigest())
                self.assertEqual(metadata["event_count"], 4)
                self.assertEqual(metadata["simulated_delay_ms"], 200)

    def test_srt_runs_without_installed_site_packages(self):
        output = self.directory / "no_deps.jsonl"
        result = self.run_cli(
            "--subtitles", FIXTURES / "demo.srt", "--output", output, no_site=True,
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(self.read_jsonl(output), self.read_jsonl(FIXTURES / "expected_from_srt.jsonl"))

    def test_audio_only_decodes_all_frames_with_tail_and_shared_offset(self):
        audio = self.write_wav()
        log = self.directory / "audio.jsonl"
        result = self.run_cli("--audio", audio, "--audio-log", log, "--media-offset-ms", 300)
        self.assertEqual(result.returncode, 0, result.stderr)
        rows = self.read_jsonl(log)
        self.assertEqual([row["frames"] for row in rows], [1600, 1600, 800])
        self.assertEqual(sum(row["frames"] for row in rows), 4000)
        self.assertEqual([(row["start_ms"], row["end_ms"]) for row in rows], [(300, 400), (400, 500), (500, 550)])
        self.assertEqual(rows[-1]["end_sample"], 4000)
        self.assertTrue(all(row["channels"] == 1 and row["sample_rate"] == 16000 for row in rows))
        self.assertEqual(list(self.directory.glob("*.meta.json")), [])

    def test_subtitle_audio_duration_mismatch_leaves_existing_outputs_untouched(self):
        audio = self.write_wav()
        output = self.directory / "events.jsonl"
        log = self.directory / "audio.jsonl"
        output.write_text("existing events", encoding="utf-8")
        log.write_text("existing audio", encoding="utf-8")
        result = self.run_cli(
            "--subtitles", FIXTURES / "demo.srt", "--audio", audio,
            "--output", output, "--audio-log", log,
        )
        self.assert_cli_error(result)
        self.assertIn("beyond audio duration", result.stderr)
        self.assertEqual(output.read_text(encoding="utf-8"), "existing events")
        self.assertEqual(log.read_text(encoding="utf-8"), "existing audio")
        self.assertFalse(output.with_suffix(".meta.json").exists())

    def test_malformed_input_does_not_overwrite_existing_event_or_metadata(self):
        output = self.directory / "events.jsonl"
        metadata = output.with_suffix(".meta.json")
        output.write_text("existing events", encoding="utf-8")
        metadata.write_text("existing provenance", encoding="utf-8")
        for name, content in (("bad.srt", "broken subtitle"), ("bad.jsonl", '{"start_ms":0}')):
            with self.subTest(name=name):
                source = self.directory / name
                source.write_text(content, encoding="utf-8")
                result = self.run_cli("--subtitles", source, "--output", output)
                self.assert_cli_error(result)
                self.assertEqual(output.read_text(encoding="utf-8"), "existing events")
                self.assertEqual(metadata.read_text(encoding="utf-8"), "existing provenance")

    def test_output_cannot_replace_input_or_another_output(self):
        source = self.directory / "source.srt"
        source.write_bytes((FIXTURES / "demo.srt").read_bytes())
        original = source.read_bytes()
        result = self.run_cli("--subtitles", source, "--output", source)
        self.assert_cli_error(result)
        self.assertIn("overwrite input", result.stderr)
        self.assertEqual(source.read_bytes(), original)
        audio = self.write_wav()
        output = self.directory / "same.jsonl"
        result = self.run_cli(
            "--subtitles", source, "--audio", audio, "--output", output, "--audio-log", output,
        )
        self.assert_cli_error(result)
        self.assertIn("distinct", result.stderr)
        self.assertFalse(output.exists())

    def test_event_validation_mode_accepts_fixture_and_rejects_bad_contract(self):
        result = self.run_cli("--validate-events", FIXTURES / "speech_events.jsonl")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("4 event(s)", result.stdout)
        bad = self.directory / "bad.jsonl"
        bad.write_text('{"start_ms":1}\n', encoding="utf-8")
        self.assert_cli_error(self.run_cli("--validate-events", bad))

    def test_incompatible_or_missing_options_give_actionable_cli_errors(self):
        for arguments in (
            (), ("--subtitles", FIXTURES / "demo.srt", "--block-ms", 0),
            ("--audio", self.write_wav(), "--output", self.directory / "events.jsonl"),
            ("--subtitles", FIXTURES / "demo.srt", "--audio-log", self.directory / "audio.jsonl"),
            ("--validate-events", FIXTURES / "speech_events.jsonl", "--subtitles", FIXTURES / "demo.srt"),
        ):
            with self.subTest(arguments=arguments):
                self.assert_cli_error(self.run_cli(*arguments))


if __name__ == "__main__":
    unittest.main()

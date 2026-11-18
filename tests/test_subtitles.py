"""Input parsing, manual annotations and media-clock conversion checks."""

import json
from pathlib import Path
import tempfile
import unittest

from src.speech.events import SpeechEvent
from src.speech.subtitles import load_mock_events, read_events


class SubtitleTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)

    def write(self, name, content, encoding="utf-8"):
        path = self.directory / name
        path.write_text(content, encoding=encoding)
        return path

    def jsonl(self, records, name="input.jsonl"):
        return self.write(name, "\n".join(json.dumps(item, ensure_ascii=False) for item in records))

    def test_bom_multiline_srt_preserves_words_and_millisecond_clock(self):
        path = self.write("中文.SRT", "1\n00:00:01,125 --> 00:00:02,750\n大家看看\n这个杯子。\n", "utf-8-sig")
        events = load_mock_events(path, simulated_delay_ms=140, media_offset_ms=2000)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0].to_dict(), {
            "start_ms": 3125, "end_ms": 4750, "available_at_ms": 4890,
            "text": "大家看看\n这个杯子。", "mentioned_classes": [], "cue_types": [],
            "is_final": True,
        })

    def test_bad_srt_is_rejected_instead_of_silently_skipped(self):
        for content in ("broken subtitle", "garbage\n1\n00:00:00,100 --> 00:00:00,200\nhello\n"):
            path = self.write("bad.srt", content)
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "bad.srt"):
                load_mock_events(path)

    def test_manual_annotations_are_passed_through_without_extraction(self):
        path = self.jsonl([
            {"start_ms": 100, "end_ms": 200, "text": "大家看看这个苹果"},
            {"start_ms": 300, "end_ms": 400, "text": "人工测试标签",
             "mentioned_classes": ["cup"], "cue_types": ["PRESENTATION_LANGUAGE"]},
        ])
        first, second = load_mock_events(path, simulated_delay_ms=0)
        self.assertEqual(first.mentioned_classes, [])
        self.assertEqual(first.cue_types, [])
        self.assertEqual(second.mentioned_classes, ["cup"])
        self.assertEqual(second.cue_types, ["PRESENTATION_LANGUAGE"])
        self.assertEqual(first.available_at_ms, first.end_ms)

    def test_overlapping_cues_are_sorted_by_actual_availability(self):
        path = self.jsonl([
            {"start_ms": 100, "end_ms": 1000, "text": "long cue"},
            {"start_ms": 300, "end_ms": 400, "text": "short cue"},
        ])
        events = load_mock_events(path)
        self.assertEqual([event.text for event in events], ["short cue", "long cue"])
        self.assertEqual([event.available_at_ms for event in events], [600, 1200])

    def test_jsonl_blank_lines_and_utf8_bom(self):
        content = '\n{"start_ms":0,"end_ms":1,"text":"你好"}\n\n'
        path = self.write("bom.jsonl", content, "utf-8-sig")
        self.assertEqual(load_mock_events(path)[0].text, "你好")

    def test_invalid_jsonl_does_not_accept_extra_or_incomplete_records(self):
        contents = (
            "not json", "[]", '"cue"', '{"start_ms":0}',
            '{"start_ms":0,"end_ms":1,"text":"x","target_id":"x"}',
        )
        for content in contents:
            path = self.write("bad.jsonl", content)
            with self.subTest(content=content), self.assertRaisesRegex(ValueError, "bad.jsonl:1"):
                load_mock_events(path)

    def test_invalid_cue_values_are_rejected_with_location(self):
        valid = {"start_ms": 1, "end_ms": 3, "text": "x"}
        for changes in (
            {"start_ms": True}, {"end_ms": 1}, {"start_ms": -1},
            {"text": " "}, {"mentioned_classes": ["cup", "cup"]},
            {"cue_types": "PRESENTATION_LANGUAGE"},
        ):
            path = self.jsonl([dict(valid, **changes)], "bad.jsonl")
            with self.subTest(changes=changes), self.assertRaisesRegex(ValueError, "bad.jsonl:cue 1"):
                load_mock_events(path)

    def test_empty_inputs_and_wrong_extension_are_rejected(self):
        for name, content in (("empty.srt", ""), ("empty.jsonl", " \n"), ("cues.txt", "hello")):
            with self.subTest(name=name), self.assertRaises(ValueError):
                load_mock_events(self.write(name, content))

    def test_options_reject_negative_and_non_integer_values(self):
        path = self.jsonl([{"start_ms": 0, "end_ms": 1, "text": "x"}])
        for option in ("simulated_delay_ms", "media_offset_ms"):
            for value in (-1, True, 0.5):
                with self.subTest(option=option, value=value), self.assertRaises(ValueError):
                    load_mock_events(path, **{option: value})

    def test_event_reader_accepts_valid_stream_and_rejects_wrong_order(self):
        first = SpeechEvent(0, 10, 20, "first")
        second = SpeechEvent(5, 15, 30, "second", is_final=False)
        path = self.jsonl([first.to_dict(), second.to_dict()], "events.jsonl")
        self.assertEqual(read_events(path), [first, second])
        path = self.jsonl([second.to_dict(), first.to_dict()], "unordered.jsonl")
        with self.assertRaisesRegex(ValueError, "ordered"):
            read_events(path)

    def test_event_reader_rejects_invalid_contract_with_line_number(self):
        path = self.write("events.jsonl", '\n{"start_ms":1}\n')
        with self.assertRaisesRegex(ValueError, "event line 2"):
            read_events(path)


if __name__ == "__main__":
    unittest.main()

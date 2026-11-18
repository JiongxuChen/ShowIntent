"""Contract and causality checks for member D's independent event stream."""

import unittest

from src.speech.events import SpeechEvent, available_events, require_ms


class SpeechEventTests(unittest.TestCase):
    def event(self, **changes):
        values = dict(start_ms=100, end_ms=300, available_at_ms=500, text="看看杯子")
        values.update(changes)
        return SpeechEvent(**values)

    def test_exact_wire_schema_round_trip(self):
        event = self.event(mentioned_classes=["cup"], cue_types=["PRESENTATION_LANGUAGE"])
        data = event.to_dict()
        self.assertEqual(set(data), {
            "start_ms", "end_ms", "available_at_ms", "text", "mentioned_classes",
            "cue_types", "is_final",
        })
        self.assertEqual(SpeechEvent.from_dict(data), event)
        data["mentioned_classes"].append("apple")
        self.assertEqual(event.mentioned_classes, ["cup"])

    def test_no_default_semantics(self):
        event = self.event(text="大家看看这个苹果")
        self.assertEqual(event.mentioned_classes, [])
        self.assertEqual(event.cue_types, [])
        self.assertTrue(event.is_final)

    def test_reject_non_integer_or_negative_times(self):
        for field in ("start_ms", "end_ms", "available_at_ms"):
            for value in (True, False, 0.5, "100", None, -1):
                with self.subTest(field=field, value=value):
                    with self.assertRaises(ValueError):
                        self.event(**{field: value})

    def test_reject_invalid_interval_or_early_result(self):
        for changes in (
            {"start_ms": 300}, {"start_ms": 301}, {"available_at_ms": 299},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValueError):
                self.event(**changes)
        self.assertEqual(self.event(available_at_ms=300).available_at_ms, 300)

    def test_reject_invalid_text_and_labels(self):
        for text in ("", " \n", 123, None):
            with self.subTest(text=text), self.assertRaises(ValueError):
                self.event(text=text)
        for field in ("mentioned_classes", "cue_types"):
            for labels in (None, "cup", ("cup",), [""], [" "], [1], [["cup"]], ["cup", "cup"]):
                with self.subTest(field=field, labels=labels), self.assertRaises(ValueError):
                    self.event(**{field: labels})

    def test_is_final_requires_boolean(self):
        self.assertFalse(self.event(is_final=False).is_final)
        for value in (0, 1, "false", None):
            with self.subTest(value=value), self.assertRaises(ValueError):
                self.event(is_final=value)

    def test_from_dict_rejects_missing_extra_and_nonobject(self):
        valid = self.event().to_dict()
        for field in valid:
            missing = {key: value for key, value in valid.items() if key != field}
            with self.subTest(missing=field), self.assertRaises(ValueError):
                SpeechEvent.from_dict(missing)
        for value in (None, [], "event", dict(valid, target_id="forbidden")):
            with self.subTest(value=value), self.assertRaises(ValueError):
                SpeechEvent.from_dict(value)

    def test_availability_uses_return_time_inclusively(self):
        first = self.event()
        delayed = self.event(start_ms=200, end_ms=400, available_at_ms=800, text="后返回")
        self.assertEqual(available_events([first, delayed], 300), [])
        self.assertEqual(available_events([first, delayed], 499), [])
        self.assertEqual(available_events([first, delayed], 500), [first])
        self.assertEqual(available_events([first, delayed], 800), [first, delayed])
        for value in (-1, True, 1.0):
            with self.subTest(value=value), self.assertRaises(ValueError):
                available_events([first], value)

    def test_require_ms_accepts_zero(self):
        self.assertEqual(require_ms(0, "time"), 0)

    def test_mutating_nested_labels_cannot_export_invalid_event(self):
        event = self.event(mentioned_classes=["cup"])
        event.mentioned_classes.append("cup")
        with self.assertRaises(ValueError):
            event.to_dict()


if __name__ == "__main__":
    unittest.main()

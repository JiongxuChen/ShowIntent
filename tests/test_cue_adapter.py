"""D5/D6 category-only evidence: no future words or implied instance choice."""

import unittest

from src.speech.cue_adapter import SpeechCueAdapter, speech_cue_adapter
from src.speech.events import SpeechEvent


class SpeechCueAdapterTests(unittest.TestCase):
    def packet(self, classes=("cup", "apple", "bottle"), now=1000):
        return {"timestamp_ms": now, "frame_id": 30,
                "objects": [{"track_id": f"obj_{n}", "class_name": name}
                            for n, name in enumerate(classes)]}

    def event(self, text="大家看看这个水杯", **changes):
        values = dict(start_ms=100, end_ms=500, available_at_ms=800, text=text)
        values.update(changes)
        return SpeechEvent(**values)

    def cue(self, events=None, packet=None, **options):
        return speech_cue_adapter(
            [self.event()] if events is None else events,
            self.packet() if packet is None else packet, enabled=True, **options,
        )

    def test_defaults_off_and_empty_optional_speech_are_safe(self):
        cue = speech_cue_adapter([self.event()], self.packet())
        self.assertFalse(cue["has_evidence"])
        self.assertEqual(cue["reason_codes"], ["DISABLED"])
        for events in (None, []):
            cue = speech_cue_adapter(events, self.packet(), enabled=True)
            self.assertFalse(cue["has_evidence"])
            self.assertEqual(cue["reason_codes"], ["NO_AVAILABLE_RELEVANT_SPEECH"])

    def test_disabled_or_absent_speech_does_not_load_optional_lexicon(self):
        for enabled, events in ((False, [self.event()]), (True, [])):
            with self.subTest(enabled=enabled):
                cue = speech_cue_adapter(events, self.packet(), enabled=enabled,
                                         lexicon_path="a-lexicon-that-does-not-exist.json")
                self.assertFalse(cue["has_evidence"])

    def test_single_visible_category_yields_only_lexical_evidence(self):
        cue = self.cue()
        self.assertTrue(cue["has_evidence"])
        self.assertEqual(cue["suggested_classes"], ["cup"])
        self.assertTrue(cue["presentation_language"])
        self.assertEqual(cue["source_available_at_ms"], 800)
        for forbidden in ("target_id", "track_id", "state", "zoom", "mode"):
            self.assertNotIn(forbidden, cue)

    def test_plain_mention_does_not_become_presentation(self):
        cue = self.cue([self.event("桌上有杯子")])
        self.assertTrue(cue["has_evidence"])
        self.assertEqual(cue["suggested_classes"], ["cup"])
        self.assertFalse(cue["presentation_language"])
        self.assertEqual(cue["cue_types"], [])

    def test_future_event_is_not_available_until_actual_return(self):
        event = self.event(available_at_ms=1500)
        cue = self.cue([event], timestamp_ms=1499)
        self.assertFalse(cue["has_evidence"])
        self.assertIsNone(cue["source_available_at_ms"])
        self.assertTrue(self.cue([event], timestamp_ms=1500)["has_evidence"])

    def test_ttl_uses_spoken_content_age_not_delayed_return(self):
        event = self.event(available_at_ms=4000)
        cue = self.cue([event], timestamp_ms=4000, ttl_ms=3000)
        self.assertEqual(cue["reason_codes"], ["EXPIRED_SPEECH"])
        self.assertFalse(cue["has_evidence"])
        self.assertTrue(self.cue(timestamp_ms=3500, ttl_ms=3000)["has_evidence"])
        self.assertFalse(self.cue(timestamp_ms=3501, ttl_ms=3000)["has_evidence"])

    def test_multiple_same_class_abstains_from_instance_assignment(self):
        cue = self.cue(packet=self.packet(("cup", "cup", "apple")))
        self.assertEqual(cue["reason_codes"], ["AMBIGUOUS_INSTANCES"])
        self.assertEqual(cue["suggested_classes"], [])
        self.assertFalse(cue["has_evidence"])

    def test_missing_or_multiple_mentioned_categories_abstain(self):
        for events, packet, reason in (
            ([self.event()], self.packet(("apple",)), "NO_VISIBLE_MATCH"),
            ([self.event()], self.packet(()), "NO_VISIBLE_MATCH"),
            ([self.event("大家看看杯子和苹果")], self.packet(), "CONFLICTING_CLASSES"),
            ([self.event("看这里")], self.packet(), "NO_CLASS_MENTION"),
        ):
            with self.subTest(reason=reason):
                cue = self.cue(events, packet)
                self.assertFalse(cue["has_evidence"])
                self.assertEqual(cue["reason_codes"], [reason])

    def test_latest_negative_or_topic_change_cancels_older_positive(self):
        first = self.event()
        for text, reason in (("不要看这个苹果", "NEGATED_LANGUAGE"), ("先放下杯子", "TOPIC_CHANGE")):
            latest = self.event(text, start_ms=600, end_ms=800, available_at_ms=950)
            with self.subTest(text=text):
                cue = self.cue([latest, first])  # input need not be chronological
                self.assertFalse(cue["has_evidence"])
                self.assertEqual(cue["reason_codes"], [reason])
                self.assertEqual(cue["source_end_ms"], 800)

    def test_delayed_old_result_does_not_replace_new_spoken_topic(self):
        old = self.event("展示杯子", available_at_ms=1500)
        new = self.event("展示苹果", start_ms=700, end_ms=1000, available_at_ms=1200)
        cue = self.cue([old, new], timestamp_ms=1500)
        self.assertEqual(cue["suggested_classes"], ["apple"])
        self.assertEqual(cue["source_end_ms"], 1000)

    def test_final_result_wins_exact_timestamp_tie_with_partial(self):
        from dataclasses import replace
        final = self.event()
        partial = replace(final, is_final=False)
        for events in ([partial, final], [final, partial]):
            with self.subTest(events=events):
                result = self.cue(events)
                self.assertTrue(result["has_evidence"])
                self.assertNotEqual(result["reason_codes"], ["PARTIAL_TRANSCRIPT"])

    def test_latest_available_partial_abstains_instead_of_using_old_positive(self):
        partial = self.event("看看苹果", start_ms=600, end_ms=800, available_at_ms=950, is_final=False)
        cue = self.cue([self.event(), partial])
        self.assertEqual(cue["reason_codes"], ["PARTIAL_TRANSCRIPT"])
        self.assertFalse(cue["has_evidence"])

    def test_manual_positive_annotation_cannot_override_negative_text(self):
        event = self.event("不要看这个杯子", mentioned_classes=["cup"], cue_types=["PRESENTATION_LANGUAGE"])
        cue = self.cue([event])
        self.assertEqual(cue["reason_codes"], ["NEGATED_LANGUAGE"])
        self.assertFalse(cue["presentation_language"])

    def test_closing_returns_rejection_even_without_an_object_mention(self):
        cue = self.cue([self.event("今天就先給大家展示到這裡")])
        self.assertFalse(cue["has_evidence"])
        self.assertFalse(cue["presentation_language"])
        self.assertEqual(cue["cue_types"], ["ENDING_LANGUAGE"])
        self.assertEqual(cue["reason_codes"], ["ENDING_LANGUAGE"])
        self.assertEqual(cue["suggested_classes"], [])
        for forbidden in ("state", "target_id", "track_id", "zoom"):
            self.assertNotIn(forbidden, cue)

    def test_latest_ending_blocks_previous_presentation_after_available_time(self):
        first = self.event()
        closing = self.event("蘋果的展示結束了。", start_ms=600, end_ms=800, available_at_ms=1050)
        self.assertTrue(self.cue([first, closing], timestamp_ms=1049)["has_evidence"])
        result = self.cue([first, closing], timestamp_ms=1050)
        self.assertEqual(result["reason_codes"], ["ENDING_LANGUAGE"])
        self.assertEqual(result["suggested_classes"], [])
        self.assertEqual(result["source_end_ms"], 800)

    def test_stale_manual_positive_label_cannot_override_closing_text(self):
        event = self.event("展示這個杯子到這裡，今天的展示結束了。",
                           mentioned_classes=["cup"], cue_types=["PRESENTATION_LANGUAGE"])
        result = self.cue([event])
        self.assertEqual(result["reason_codes"], ["ENDING_LANGUAGE"])
        self.assertEqual(result["cue_types"], ["ENDING_LANGUAGE"])
        self.assertFalse(result["has_evidence"])
        self.assertFalse(result["presentation_language"])

    def test_negation_has_priority_over_manual_ending_or_positive_labels(self):
        event = self.event("不要结束展示这个杯子", mentioned_classes=["cup"],
                           cue_types=["ENDING_LANGUAGE", "PRESENTATION_LANGUAGE"])
        result = self.cue([event])
        self.assertEqual(result["reason_codes"], ["NEGATED_LANGUAGE"])
        self.assertEqual(result["cue_types"], ["NEGATED_LANGUAGE"])
        self.assertFalse(result["has_evidence"])

    def test_mixed_closing_and_presentation_clauses_never_supply_category_hint(self):
        event = self.event("今天展示到这里，给大家看看下一个杯子。")
        result = self.cue([event])
        self.assertEqual(result["reason_codes"], ["ENDING_LANGUAGE"])
        self.assertEqual(result["suggested_classes"], [])
        self.assertFalse(result["presentation_language"])

    def test_malformed_or_future_perception_abstains(self):
        for packet, reason in (
            ({"timestamp_ms": 1000}, "INVALID_PERCEPTION"),
            ({"timestamp_ms": 1000, "objects": [{"class_name": 1}]}, "INVALID_PERCEPTION"),
            (self.packet(now=1001), "FUTURE_PERCEPTION"),
        ):
            with self.subTest(reason=reason):
                cue = self.cue(packet=packet, timestamp_ms=1000)
                self.assertEqual(cue["reason_codes"], [reason])
                self.assertFalse(cue["has_evidence"])

    def test_invalid_time_and_options_are_rejected(self):
        for options in ({"ttl_ms": -1}, {"ttl_ms": True}, {"enabled": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SpeechCueAdapter(**options)
        for now in (-1, True, 1.2):
            with self.subTest(now=now), self.assertRaises(ValueError):
                self.cue(timestamp_ms=now)


if __name__ == "__main__":
    unittest.main()

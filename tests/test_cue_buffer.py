"""Incremental optional evidence stays bounded, causal and category-only."""

import unittest

from src.speech.cue_buffer import SpeechCueBuffer
from src.speech.events import SpeechEvent


class SpeechCueBufferTests(unittest.TestCase):
    def packet(self, now=1000, classes=("cup", "apple", "bottle")):
        return {"timestamp_ms": now, "objects": [{"class_name": name} for name in classes]}

    def event(self, text="大家看看这个杯子", **changes):
        values = dict(start_ms=100, end_ms=500, available_at_ms=800, text=text)
        values.update(changes)
        return SpeechEvent(**values)

    def buffer(self, events=(), **options):
        buffer = SpeechCueBuffer(enabled=True, **options)
        for event in events:
            buffer.push(event)
        return buffer

    def test_disabled_default_and_missing_speech_need_no_optional_dictionary(self):
        disabled = SpeechCueBuffer(lexicon_path="not-a-dictionary.json")
        disabled.push(self.event())
        self.assertEqual(disabled.evidence(self.packet())["reason_codes"], ["DISABLED"])
        absent = self.buffer(lexicon_path="not-a-dictionary.json")
        self.assertFalse(absent.evidence(self.packet())["has_evidence"])
        self.assertEqual(absent.diagnostics["buffered_count"], 0)

    def test_event_is_gated_at_actual_availability_and_output_has_no_decision(self):
        buffer = self.buffer([self.event(available_at_ms=1500)])
        self.assertFalse(buffer.evidence(self.packet(1499))["has_evidence"])
        cue = buffer.evidence(self.packet(1500))
        self.assertTrue(cue["has_evidence"])
        self.assertEqual(cue["suggested_classes"], ["cup"])
        for forbidden in ("state", "target_id", "track_id", "zoom", "mode"):
            self.assertNotIn(forbidden, cue)

    def test_future_negative_or_partial_does_not_invalidate_current_evidence(self):
        for text, is_final, reason in (
            ("不要看这个苹果", True, "NEGATED_LANGUAGE"),
            ("看看苹果", False, "PARTIAL_TRANSCRIPT"),
            ("先放下杯子", True, "TOPIC_CHANGE"),
        ):
            with self.subTest(reason=reason):
                buffer = self.buffer([
                    self.event(),
                    self.event(text, start_ms=600, end_ms=900, available_at_ms=1500,
                               is_final=is_final),
                ])
                self.assertTrue(buffer.evidence(self.packet(1400))["has_evidence"])
                cue = buffer.evidence(self.packet(1500))
                self.assertFalse(cue["has_evidence"])
                self.assertEqual(cue["reason_codes"], [reason])

    def test_out_of_order_delayed_old_result_does_not_replace_latest_spoken_topic(self):
        newer = self.event("展示苹果", start_ms=700, end_ms=1000, available_at_ms=1200)
        old = self.event(available_at_ms=1500)
        for deliveries in ((newer, old), (old, newer)):
            with self.subTest(deliveries=deliveries):
                buffer = self.buffer(deliveries)
                self.assertEqual(buffer.evidence(self.packet(1200))["suggested_classes"], ["apple"])
                cue = buffer.evidence(self.packet(1500))
                self.assertEqual(cue["suggested_classes"], ["apple"])
                self.assertEqual(cue["source_end_ms"], 1000)

    def test_lost_speech_expires_at_content_end_not_query_or_return_time(self):
        buffer = self.buffer([self.event()], ttl_ms=3000)
        self.assertTrue(buffer.evidence(self.packet(3500))["has_evidence"])
        self.assertFalse(buffer.evidence(self.packet(3501))["has_evidence"])
        self.assertEqual(buffer.diagnostics["expired_count"], 1)
        self.assertEqual(buffer.diagnostics["buffered_count"], 0)
        self.assertFalse(buffer.evidence(self.packet(4000))["has_evidence"])
        buffer.push(self.event(available_at_ms=4000))
        self.assertFalse(buffer.evidence(self.packet(4000))["has_evidence"])
        self.assertEqual(buffer.diagnostics["expired_count"], 2)

    def test_pending_old_content_is_not_purged_until_return_available(self):
        buffer = self.buffer([self.event(available_at_ms=5000)], ttl_ms=100)
        self.assertFalse(buffer.evidence(self.packet(4900))["has_evidence"])
        self.assertEqual(buffer.diagnostics["pending_count"], 1)
        self.assertEqual(buffer.diagnostics["expired_count"], 0)
        self.assertFalse(buffer.evidence(self.packet(5000))["has_evidence"])
        self.assertEqual(buffer.diagnostics["buffered_count"], 0)
        self.assertEqual(buffer.diagnostics["expired_count"], 1)

    def test_duplicates_and_external_mutation_cannot_change_retained_event(self):
        event = self.event(text="manual label", mentioned_classes=["cup"])
        buffer = self.buffer([event, event])
        self.assertEqual(buffer.diagnostics["buffered_count"], 1)
        self.assertEqual(buffer.diagnostics["duplicate_count"], 1)
        event.mentioned_classes.append("apple")
        self.assertEqual(buffer.evidence(self.packet())["suggested_classes"], ["cup"])
        snapshot = buffer.diagnostics
        snapshot["dropped_count"] = 100
        self.assertEqual(buffer.diagnostics["dropped_count"], 0)

    def test_same_timestamp_final_supersedes_partial_without_duplicate_suppression(self):
        for deliveries in ((False, True), (True, False)):
            with self.subTest(deliveries=deliveries):
                buffer = self.buffer([self.event(is_final=final) for final in deliveries])
                self.assertTrue(buffer.evidence(self.packet())["has_evidence"])
                self.assertEqual(buffer.diagnostics["buffered_count"], 2)

    def test_overflow_rejects_farfuture_instead_of_evicting_current_speech(self):
        buffer = self.buffer([self.event()], max_events=1)
        self.assertTrue(buffer.evidence(self.packet())["has_evidence"])
        buffer.push(self.event("不要看这个苹果", start_ms=2000, end_ms=2100,
                               available_at_ms=10000))
        self.assertTrue(buffer.evidence(self.packet(1200))["has_evidence"])
        self.assertEqual(buffer.diagnostics["buffered_count"], 1)
        self.assertEqual(buffer.diagnostics["dropped_count"], 1)

    def test_pending_overflow_retains_earliest_availability_before_first_query(self):
        buffer = self.buffer(max_events=2)
        for availability in (3000, 1500, 2000):
            buffer.push(self.event(available_at_ms=availability))
        self.assertEqual(buffer.diagnostics["pending_count"], 2)
        self.assertEqual(buffer.diagnostics["dropped_count"], 1)
        self.assertTrue(buffer.evidence(self.packet(1500))["has_evidence"])
        self.assertEqual(buffer.evidence(self.packet(2000))["source_available_at_ms"], 2000)

    def test_available_overflow_keeps_latest_spoken_cue_including_negation(self):
        buffer = self.buffer(max_events=1)
        buffer.evidence(self.packet(1000))
        buffer.push(self.event())
        buffer.push(self.event("不要看这个苹果", start_ms=600, end_ms=800, available_at_ms=950))
        cue = buffer.evidence(self.packet())
        self.assertEqual(cue["reason_codes"], ["NEGATED_LANGUAGE"])
        self.assertEqual(buffer.diagnostics["dropped_count"], 1)

    def test_many_pushes_stay_bounded_and_live_session_does_not_keep_history(self):
        buffer = self.buffer(max_events=4, ttl_ms=1000)
        for index in range(2000):
            now = 1000 + index * 100
            buffer.evidence(self.packet(now))
            buffer.push(self.event(start_ms=now - 200, end_ms=now - 100, available_at_ms=now))
            self.assertLessEqual(buffer.diagnostics["buffered_count"], 4)
        self.assertEqual(buffer.diagnostics["pushed_count"], 2000)
        self.assertGreater(buffer.diagnostics["dropped_count"], 0)
        self.assertTrue(buffer.evidence(self.packet(now))["has_evidence"])
        self.assertFalse(buffer.evidence(self.packet(now + 2000))["has_evidence"])
        self.assertEqual(buffer.diagnostics["buffered_count"], 0)

    def test_query_clock_can_repeat_but_cannot_regress_or_accept_invalid_time(self):
        buffer = self.buffer([self.event()])
        first = buffer.evidence(self.packet())
        self.assertEqual(buffer.evidence(self.packet()), first)
        for now in (999, -1, True, 0.5):
            with self.subTest(now=now), self.assertRaises(ValueError):
                buffer.evidence(self.packet(), timestamp_ms=now)
        self.assertEqual(buffer.diagnostics["last_timestamp_ms"], 1000)
        self.assertTrue(buffer.evidence(self.packet(now=900), timestamp_ms=1100)["has_evidence"])

    def test_invalid_configuration_and_delivery_are_rejected(self):
        for options in ({"max_events": 0}, {"max_events": -1}, {"max_events": True},
                        {"max_events": 1.5}, {"ttl_ms": -1}, {"enabled": 1}):
            with self.subTest(options=options), self.assertRaises(ValueError):
                SpeechCueBuffer(**options)
        buffer = self.buffer()
        for event in (None, {}, "words"):
            with self.subTest(event=event), self.assertRaises(ValueError):
                buffer.push(event)
        self.assertEqual(buffer.diagnostics["pushed_count"], 0)


if __name__ == "__main__":
    unittest.main()

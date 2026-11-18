"""Driver timing and publication use a fake session, never a model/network."""

import math
import unittest

import numpy as np

from src.speech.asr import SAMPLE_RATE
from src.speech.events import SpeechEvent
from src.speech.incremental import recognize_windows
from src.speech.media_stream import MediaWindow


class ManualClock:
    def __init__(self):
        self.nanoseconds = 0
        self.sleeps = []

    def __call__(self):
        return self.nanoseconds / 1_000_000_000

    def advance(self, seconds):
        self.nanoseconds += round(seconds * 1_000_000_000)

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.advance(seconds)


class ArrivingSource:
    def __init__(self, lengths, clock, *, decode_time=0):
        self.lengths = lengths
        self.clock = clock
        self.decode_time = decode_time
        self.yielded = 0
        self.eof = False

    def __iter__(self):
        offset = 0
        for length in self.lengths:
            self.clock.advance(self.decode_time)
            values = np.full(length, self.yielded + 1, dtype=np.float32)
            self.yielded += 1
            yield MediaWindow(values, offset, offset + length)
            offset += length
        self.eof = True


class InspectingSession:
    def __init__(self, clock, *, work_time=0.125, media_origin=0, commit_each=True):
        self.clock = clock
        self.work_time = work_time
        self.media_origin = media_origin
        self.commit_each = commit_each
        self.last_update = {}
        self.calls = []
        self.received_samples = 0

    def push(self, samples, *, received_until_ms, finish=False):
        begin = self.received_samples
        self.received_samples += len(samples)
        self.calls.append({"samples": samples.copy(), "received_until_ms": received_until_ms,
                           "finish": finish, "wall_at_call": self.clock()})
        self.clock.advance(self.work_time)
        self.last_update = {"pending_text": "" if finish else "unconfirmed", "test_session": True}
        if (finish and self.commit_each) or (not finish and not self.commit_each):
            return []
        start = self.media_origin + (begin * 1000 // SAMPLE_RATE if not finish else 0)
        end = self.media_origin + math.ceil(self.received_samples * 1000 / SAMPLE_RATE)
        return [SpeechEvent(start, end, received_until_ms, "committed EOF" if finish else "committed window")]


class IncrementalDriverTests(unittest.TestCase):
    def test_commits_publish_before_source_eof_and_never_read_a_future_window(self):
        clock = ManualClock()
        source = ArrivingSource([16000, 16000, 8001], clock)
        session = InspectingSession(clock)
        publications = []

        def callback(events):
            publications.append((source.yielded, source.eof, events))
            self.assertEqual(len(session.calls), source.yielded)
            self.assertLessEqual(events[0].end_ms, session.calls[-1]["received_until_ms"])

        result = recognize_windows(source, session, on_events=callback, clock=clock, sleep=clock.sleep)
        self.assertEqual([(count, eof) for count, eof, _ in publications], [(1, False), (2, False), (3, False)])
        self.assertTrue(source.eof)
        self.assertEqual([len(call["samples"]) for call in session.calls], [16000, 16000, 8001, 0])
        self.assertEqual([call["finish"] for call in session.calls], [False, False, False, True])
        self.assertEqual([call["received_until_ms"] for call in session.calls], [1000, 2000, 2501, 2501])
        for index, call in enumerate(session.calls[:-1], 1):
            self.assertTrue(np.all(call["samples"] == index))
        self.assertEqual([event.available_at_ms for event in result.events], [1125, 2125, 2626])
        self.assertEqual(result.metadata["first_published_at_ms"], 1125)
        self.assertEqual(result.metadata["total_compute_ms"], 500)
        self.assertEqual(result.metadata["duration_ms"], 2501)
        self.assertAlmostEqual(result.metadata["compute_rtf"], 500 / 2501)
        self.assertEqual(clock.sleeps, [])
        self.assertFalse(result.metadata["full_file_predecode"])
        self.assertFalse(result.metadata["live_microphone"])

    def test_slow_single_worker_backlog_cannot_backdate_publication(self):
        clock = ManualClock()
        session = InspectingSession(clock, work_time=2.5)
        result = recognize_windows(ArrivingSource([16000] * 3, clock), session, clock=clock)
        self.assertEqual([w["processing_start_ms"] for w in result.windows], [1000, 3500, 6000, 8500])
        self.assertEqual([e.available_at_ms for e in result.events], [3500, 6000, 8500])
        self.assertEqual(result.metadata["total_compute_ms"], 10000)

    def test_paced_replay_waits_for_media_then_stamps_real_compute_completion(self):
        clock = ManualClock()
        session = InspectingSession(clock)
        source = ArrivingSource([16000, 16000], clock, decode_time=0.25)
        callback_wall = []
        result = recognize_windows(source, session, pace=True, clock=clock, sleep=clock.sleep,
                                   on_events=lambda events: callback_wall.append((clock(), events[0].available_at_ms)))
        self.assertEqual([c["wall_at_call"] for c in session.calls], [1.0, 2.0, 2.125])
        self.assertEqual(callback_wall, [(1.125, 1125), (2.125, 2125)])
        self.assertTrue(clock.sleeps)
        self.assertTrue(all(0 < seconds <= 0.1 for seconds in clock.sleeps))
        self.assertEqual(result.metadata["incremental_decode_ms"], 500)
        self.assertEqual(result.metadata["timing_kind"], "wall_clock_file_replay")

    def test_paced_source_decode_backlog_is_reflected_in_wall_clock_availability(self):
        clock = ManualClock()
        session = InspectingSession(clock)
        result = recognize_windows(ArrivingSource([16000, 16000], clock, decode_time=1.5),
                                   session, pace=True, clock=clock, sleep=clock.sleep)
        self.assertEqual([w["processing_start_ms"] for w in result.windows], [1500, 3125, 3250])
        self.assertEqual([e.available_at_ms for e in result.events], [1625, 3250])
        self.assertEqual(clock.sleeps, [])

    def test_accelerated_replay_includes_decode_work_in_scheduled_readiness(self):
        clock = ManualClock()
        session = InspectingSession(clock)
        result = recognize_windows(ArrivingSource([16000, 16000], clock, decode_time=0.25),
                                   session, clock=clock)
        self.assertEqual([w["decode_ms"] for w in result.windows], [250, 250, 0])
        self.assertEqual([w["processing_start_ms"] for w in result.windows], [1250, 2250, 2375])
        self.assertEqual([e.available_at_ms for e in result.events], [1375, 2375])
        self.assertEqual(result.metadata["incremental_decode_ms"], 500)
        self.assertEqual(result.metadata["total_compute_ms"], 375)
        self.assertEqual(result.metadata["compute_rtf"], 375 / 2000)
        self.assertEqual(result.metadata["compute_and_decode_rtf"], 875 / 2000)
        self.assertTrue(all(w["provisional_availability"] is False for w in result.windows))

    def test_explicit_media_and_audio_offsets_use_common_clock_and_pacing(self):
        clock = ManualClock()
        session = InspectingSession(clock, media_origin=1250)
        result = recognize_windows(ArrivingSource([16000], clock), session,
                                   media_offset_ms=1000, audio_offset_ms=250,
                                   pace=True, clock=clock, sleep=clock.sleep)
        event = result.events[0]
        self.assertEqual((event.start_ms, event.end_ms, event.available_at_ms), (1250, 2250, 2375))
        self.assertEqual(session.calls[0]["wall_at_call"], 1.25)
        self.assertEqual(result.metadata["media_offset_ms"], 1000)
        self.assertEqual(result.metadata["audio_offset_ms"], 250)
        self.assertEqual(result.metadata["duration_ms"], 1000)

    def test_eof_flush_publishes_unconfirmed_tail_only_after_explicit_finish(self):
        clock = ManualClock()
        source = ArrivingSource([16000, 8000], clock)
        session = InspectingSession(clock, commit_each=False)
        callbacks = []
        result = recognize_windows(source, session, clock=clock,
                                   on_events=lambda events: callbacks.append((source.eof, events)))
        self.assertEqual(len(result.events), 1)
        self.assertEqual(result.events[0].text, "committed EOF")
        self.assertEqual(result.events[0].available_at_ms, 1750)
        self.assertEqual(callbacks[0][0], True)
        self.assertTrue(result.windows[-1]["eof"])
        self.assertEqual(result.metadata["eof_flushed_event_count"], 1)
        self.assertEqual(result.transcripts[0]["commit_policy"], "eof_flush")

    def test_empty_or_discontinuous_or_misdeclared_sources_are_rejected(self):
        bad_sources = (
            [], [MediaWindow(np.zeros(1), 1, 2)], [MediaWindow(np.zeros(0), 0, 0)],
            [MediaWindow(np.zeros(1), 0, 2)],
            [MediaWindow(np.zeros(16000), 0, 16000), MediaWindow(np.zeros(1), 16001, 16002)],
        )
        for source in bad_sources:
            clock = ManualClock()
            with self.subTest(source=source), self.assertRaises(ValueError):
                recognize_windows(source, InspectingSession(clock), clock=clock)

    def test_session_cannot_publish_words_from_unreceived_future_audio(self):
        clock = ManualClock()

        class FutureSession(InspectingSession):
            def push(self, samples, *, received_until_ms, finish=False):
                self.clock.advance(0.125)
                return [SpeechEvent(0, received_until_ms + 1, received_until_ms + 1, "future")]

        with self.assertRaisesRegex(ValueError, "beyond the received window"):
            recognize_windows(ArrivingSource([16000], clock), FutureSession(clock), clock=clock)

    def test_invalid_options_and_nonmonotonic_processing_clock_are_rejected(self):
        for options in ({"media_offset_ms": -1}, {"media_offset_ms": True},
                        {"audio_offset_ms": -1}, {"audio_offset_ms": 0.5}, {"pace": 1}):
            clock = ManualClock()
            with self.subTest(options=options), self.assertRaises(ValueError):
                recognize_windows([], InspectingSession(clock), clock=clock, **options)
        clock = ManualClock()
        with self.assertRaisesRegex(ValueError, "clock"):
            recognize_windows(ArrivingSource([16000], clock),
                              InspectingSession(clock, work_time=-0.125), clock=clock)


if __name__ == "__main__":
    unittest.main()

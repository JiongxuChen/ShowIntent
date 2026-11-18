"""Bounded incremental storage for B's optional speech evidence input.

This is a single-session media clock: evidence queries must be nondecreasing.
The adapter still owns the language/category gates, and B owns its final visual
decision. No historical scan grows with the duration of a live session.
"""

from __future__ import annotations

from pathlib import Path
from threading import RLock

from .cue_adapter import SpeechCueAdapter
from .events import SpeechEvent, require_ms


class SpeechCueBuffer:
    """Keep at most ``max_events`` immutable snapshots of delivered events.

    Delivery order may differ from spoken or result-availability order. Events
    are gated by ``available_at_ms`` at query time, and TTL is measured from
    ``end_ms``. A future negative/topic cue cannot invalidate current evidence.

    Capacity policy uses the last queried media time, or zero before the first
    query: retain available, unexpired events before pending events. Among the
    former retain the latest spoken cues; among the latter retain the earliest
    availability times. A full queue rejects a far-future delivery rather than
    evicting current speech. Overflow is an explicitly reported approximation:
    preloading more than the capacity is not equivalent to retaining a session.

    Duplicate detection covers currently retained identical wire events only;
    it does not retain an unbounded set of past delivery IDs. A repeated expired
    event is harmless and is counted as expired instead. Calls are synchronized
    so an event producer and the consumer can share one buffer.
    """

    def __init__(
        self, *, enabled: bool = False, ttl_ms: int = 3000,
        lexicon_path: str | Path | None = None, max_events: int = 256,
    ):
        if type(max_events) is not int or max_events < 1:
            raise ValueError("max_events must be a positive integer")
        self._adapter = SpeechCueAdapter(
            enabled=enabled, ttl_ms=ttl_ms, lexicon_path=lexicon_path,
        )
        self._max_events = max_events
        self._events: dict[tuple, SpeechEvent] = {}
        self._last_timestamp_ms: int | None = None
        self._pushed_count = 0
        self._dropped_count = 0
        self._expired_count = 0
        self._duplicate_count = 0
        self._lock = RLock()

    @property
    def max_events(self) -> int:
        return self._max_events

    @property
    def enabled(self) -> bool:
        return self._adapter.enabled

    @property
    def ttl_ms(self) -> int:
        return self._adapter.ttl_ms

    @staticmethod
    def _identity(event: SpeechEvent) -> tuple:
        return (
            event.start_ms, event.end_ms, event.available_at_ms, event.text,
            tuple(event.mentioned_classes), tuple(event.cue_types), event.is_final,
        )

    def _expired(self, event: SpeechEvent, now: int) -> bool:
        # Do not use a pending result to change retention before it is available.
        return event.available_at_ms <= now and now - event.end_ms > self.ttl_ms

    def _purge_expired(self, now: int) -> None:
        expired = [key for key, event in self._events.items() if self._expired(event, now)]
        for key in expired:
            del self._events[key]
        self._expired_count += len(expired)

    def _retention_priority(self, event: SpeechEvent, now: int) -> tuple:
        if event.available_at_ms <= now:
            return (1, event.end_ms, event.start_ms, event.available_at_ms, event.is_final)
        return (0, -event.available_at_ms, event.end_ms, event.start_ms, event.is_final)

    def push(self, event: SpeechEvent) -> None:
        """Validate and snapshot one event; never expose its mutable input lists."""
        if not isinstance(event, SpeechEvent):
            raise ValueError("event must be a SpeechEvent instance")
        snapshot = SpeechEvent.from_dict(event.to_dict())
        key = self._identity(snapshot)
        with self._lock:
            self._pushed_count += 1
            now = self._last_timestamp_ms if self._last_timestamp_ms is not None else 0
            self._purge_expired(now)
            if key in self._events:
                self._duplicate_count += 1
                return
            if self._expired(snapshot, now):
                self._expired_count += 1
                return
            self._events[key] = snapshot
            if len(self._events) > self.max_events:
                victim = min(
                    self._events,
                    key=lambda item: self._retention_priority(self._events[item], now),
                )
                del self._events[victim]
                self._dropped_count += 1

    def evidence(self, perception_packet: dict, timestamp_ms: int | None = None) -> dict:
        """Return category-only adapter evidence at a nondecreasing media time.

        Invalid or regressing query times raise without advancing the clock.
        Expired, already-available records are removed on a successful query;
        once removed, an otherwise empty buffer reports the adapter's ordinary
        ``NO_AVAILABLE_RELEVANT_SPEECH`` reason. Pending records remain gated.
        """
        if timestamp_ms is None:
            if not isinstance(perception_packet, dict):
                raise ValueError("perception_packet must be an object")
            timestamp_ms = perception_packet.get("timestamp_ms")
        now = require_ms(timestamp_ms, "timestamp_ms")
        with self._lock:
            if self._last_timestamp_ms is not None and now < self._last_timestamp_ms:
                raise ValueError("timestamp_ms must not regress within a SpeechCueBuffer session")
            candidates = [event for event in self._events.values() if not self._expired(event, now)]
            result = self._adapter.evidence(
                candidates, perception_packet, timestamp_ms=now,
            ).to_dict()
            self._last_timestamp_ms = now
            self._purge_expired(now)
            return result

    @property
    def diagnostics(self) -> dict:
        """A detached counter snapshot; no text, target IDs or control decisions."""
        with self._lock:
            now = self._last_timestamp_ms if self._last_timestamp_ms is not None else 0
            available_count = sum(event.available_at_ms <= now for event in self._events.values())
            return {
                "max_events": self.max_events,
                "buffered_count": len(self._events),
                "available_count": available_count,
                "pending_count": len(self._events) - available_count,
                "pushed_count": self._pushed_count,
                "dropped_count": self._dropped_count,
                "expired_count": self._expired_count,
                "duplicate_count": self._duplicate_count,
                "last_timestamp_ms": self._last_timestamp_ms,
            }

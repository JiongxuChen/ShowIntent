"""Bounded incremental Faster-Whisper adapter using upstream LocalAgreement-2.

The unchanged HypothesisBuffer primitive comes from ufal/whisper_streaming. This
module supplies arriving-sample bookkeeping, bounded audio and ShowIntent events.
Only common-prefix words are committed during a session. An explicit EOF may
flush the remaining latest hypothesis; no uncommitted words enter cue extraction.
Returned availability is provisional input-arrival time. The driver must stamp
actual completion after push returns and before publishing the events.
"""

from __future__ import annotations

import math

import numpy as np

from third_party.whisper_streaming.hypothesis import HypothesisBuffer

from .events import SpeechEvent, require_ms
from .quality import POLICY as QUALITY_POLICY, segment_quality

SAMPLE_RATE = 16000
UPSTREAM_COMMIT = "6da90b44b7e50d79695e68166d2a2c7609c75abb"


class AgreementRecognizer:
    def __init__(
        self, model, *, language=None, beam_size=3, media_offset_ms=0,
        buffer_limit_ms=15000, context_ms=1000, extractor=None,
    ):
        require_ms(media_offset_ms, "media_offset_ms")
        if type(beam_size) is not int or not 1 <= beam_size <= 10:
            raise ValueError("beam_size must be an integer between 1 and 10")
        if type(buffer_limit_ms) is not int or not 4000 <= buffer_limit_ms <= 30000:
            raise ValueError("buffer_limit_ms must be an integer between 4000 and 30000")
        if type(context_ms) is not int or not 0 <= context_ms < buffer_limit_ms:
            raise ValueError("context_ms must be an integer below buffer_limit_ms")
        self.model = model
        self.language = language
        self.beam_size = beam_size
        self.media_offset_ms = media_offset_ms
        self.buffer_limit_ms = buffer_limit_ms
        self.context_ms = context_ms
        self.extractor = extractor
        self.finished = False
        self.received_samples = 0
        self.buffer_offset_samples = 0
        self.audio_buffer = np.empty(0, dtype=np.float32)
        self.hypotheses = HypothesisBuffer()
        self.hypotheses.last_commited_time = media_offset_ms / 1000
        self._last_arrival_ms = media_offset_ms
        self._prompt = ""
        self._last_emitted_end = media_offset_ms / 1000
        self.last_update = {}
        self.total_trimmed_samples = 0
        self.total_unconfirmed_trimmed_samples = 0
        self.total_dropped_pending_words = 0
        self.total_rejected_quality_segments = 0

    @property
    def pending_text(self):
        return "".join(word[2] for word in self.hypotheses.complete()).strip()

    @property
    def buffered_samples(self):
        return len(self.audio_buffer)

    def _bound_buffer(self):
        maximum = self.buffer_limit_ms * SAMPLE_RATE // 1000
        if len(self.audio_buffer) <= maximum:
            return 0, 0
        required_cut = self.received_samples - maximum
        committed_sample = round(
            (self.hypotheses.last_commited_time - self.media_offset_ms / 1000) * SAMPLE_RATE
        )
        context_samples = self.context_ms * SAMPLE_RATE // 1000
        # Prefer cutting inside confirmed audio while retaining context. If
        # nothing stabilizes, enforce the hard bound and disclose the loss.
        safe_cut = max(self.buffer_offset_samples, committed_sample - context_samples)
        cut_sample = max(required_cut, safe_cut)
        cut_sample = min(cut_sample, self.received_samples)
        removed = cut_sample - self.buffer_offset_samples
        cutoff = self.media_offset_ms / 1000 + cut_sample / SAMPLE_RATE
        discarded = [word for word in self.hypotheses.complete() if word[1] <= cutoff]
        old_committed = [word for word in self.hypotheses.commited_in_buffer if word[1] <= cutoff]
        self._prompt = (self._prompt + "".join(word[2] for word in old_committed))[-200:]
        self.hypotheses.pop_commited(cutoff)
        self.hypotheses.buffer = [word for word in self.hypotheses.buffer if word[1] > cutoff]
        self.audio_buffer = self.audio_buffer[removed:].copy()
        self.buffer_offset_samples = cut_sample
        unconfirmed = max(0, cut_sample - max(committed_sample, cut_sample - removed))
        self.total_trimmed_samples += removed
        self.total_unconfirmed_trimmed_samples += unconfirmed
        self.total_dropped_pending_words += len(discarded)
        return removed, len(discarded)

    def _event(self, words, available_at_ms):
        if not words:
            return None
        # Original faster-whisper word tokens include spaces. Preserve those
        # spaces, and do not introduce spaces between Chinese character tokens.
        words = [word for word in words if word[1] > self._last_emitted_end]
        if not words:
            return None
        text = "".join(word[2] for word in words).strip()
        start_ms = max(self.media_offset_ms, round(words[0][0] * 1000))
        end_ms = min(available_at_ms, round(words[-1][1] * 1000))
        if not text or end_ms <= start_ms:
            return None
        event = SpeechEvent(start_ms, end_ms, available_at_ms, text)
        if self.extractor:
            event = self.extractor(event)
        self._last_emitted_end = words[-1][1]
        return event

    def push(self, samples16k, *, received_until_ms=None, finish=False):
        """Process only newly arrived mono 16 kHz samples, or explicitly end.

        Repeating an empty EOF is idempotent. Nonempty pushes after EOF fail.
        Input should normally be small capture blocks; a push larger than the
        audio cap deliberately drops its oldest excess and reports that loss.
        received_until_ms is the arrival clock, not recognition completion.
        """
        if type(finish) is not bool:
            raise ValueError("finish must be boolean")
        samples = np.asarray(samples16k, dtype=np.float32)
        if samples.ndim != 1 or not np.isfinite(samples).all():
            raise ValueError("samples must be a finite mono 16 kHz array")
        if self.finished:
            if len(samples):
                raise RuntimeError("cannot push audio after EOF")
            return []
        next_count = self.received_samples + len(samples)
        sample_end_ms = self.media_offset_ms + math.ceil(next_count * 1000 / SAMPLE_RATE)
        arrival = sample_end_ms if received_until_ms is None else require_ms(received_until_ms, "received_until_ms")
        if arrival < sample_end_ms or arrival < self._last_arrival_ms:
            raise ValueError("received_until_ms must cover all samples and be monotonic")
        self.received_samples = next_count
        self._last_arrival_ms = arrival
        if len(samples):
            self.audio_buffer = np.concatenate((self.audio_buffer, samples))
        removed, dropped = self._bound_buffer()
        output = []
        committed = []
        discarded = 0
        rejected_quality = 0
        detected_language = self.language
        if len(samples) and len(self.audio_buffer):
            segments, info = self.model.transcribe(
                self.audio_buffer.copy(), language=self.language, beam_size=self.beam_size,
                temperature=0.0, condition_on_previous_text=True, vad_filter=True,
                word_timestamps=True, initial_prompt=self._prompt or None,
            )
            detected_language = getattr(info, "language", self.language)
            words = []
            buffer_seconds = len(self.audio_buffer) / SAMPLE_RATE
            for segment in segments:
                if not segment_quality(segment)["accepted"]:
                    rejected_quality += 1
                    continue
                for word in getattr(segment, "words", None) or ():
                    if not isinstance(word.word, str) or not word.word.strip():
                        continue
                    if not math.isfinite(word.start) or not math.isfinite(word.end):
                        discarded += 1
                        continue
                    start = min(buffer_seconds, max(0.0, word.start))
                    end = min(buffer_seconds, max(0.0, word.end))
                    if end <= start:
                        discarded += 1
                        continue
                    words.append((start, end, word.word))
            offset = self.media_offset_ms / 1000 + self.buffer_offset_samples / SAMPLE_RATE
            self.hypotheses.insert(words, offset)
            committed = self.hypotheses.flush()
            event = self._event(committed, arrival)
            if event:
                output.append(event)
        eof_words = []
        if finish:
            # EOF is the only path that emits a hypothesis without agreement.
            eof_words = list(self.hypotheses.complete())
            event = self._event(eof_words, arrival)
            if event:
                output.append(event)
            self.hypotheses.buffer = []
            self.hypotheses.new = []
            self.finished = True
        self.total_rejected_quality_segments += rejected_quality
        self.last_update = {
            "received_until_ms": arrival, "input_samples": len(samples),
            "received_samples": self.received_samples,
            "buffer_start_sample": self.buffer_offset_samples,
            "buffer_end_sample": self.received_samples,
            "buffered_samples": len(self.audio_buffer),
            "confirmed_word_count": len(committed), "pending_word_count": len(self.hypotheses.complete()),
            "eof_flushed_word_count": len(eof_words), "eof": finish,
            "trimmed_samples": removed, "dropped_pending_words": dropped,
            "total_trimmed_samples": self.total_trimmed_samples,
            "total_unconfirmed_trimmed_samples": self.total_unconfirmed_trimmed_samples,
            "total_dropped_pending_words": self.total_dropped_pending_words,
            "rejected_quality_segments": rejected_quality,
            "total_rejected_quality_segments": self.total_rejected_quality_segments,
            "quality_policy": QUALITY_POLICY,
            "discarded_word_timestamps": discarded, "detected_language": detected_language,
            "provisional_availability": True,
            "agreement_policy": "upstream_hypothesis_buffer_localagreement_2",
            "upstream_commit": UPSTREAM_COMMIT,
        }
        return output

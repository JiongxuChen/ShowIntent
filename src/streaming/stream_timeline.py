"""Media-clock sampling and playback state helpers; no inference or labels."""
import math


def next_sample_time(timestamp_ms, scheduled_ms, rate):
    """Advance a fixed clock past this frame, avoiding per-frame rounding drift."""
    period=1000/rate
    return scheduled_ms+(math.floor(max(0,timestamp_ms-scheduled_ms)/period)+1)*period


def playback_state(intent, latest_timestamp, timestamp_ms, stale_ms=500):
    if latest_timestamp is None or timestamp_ms-latest_timestamp>stale_ms:
        return dict(timestamp_ms=timestamp_ms,state='NONE',target_id=None,reason_codes=['STALE_OR_MISSING_PERCEPTION'])
    return dict(intent,timestamp_ms=timestamp_ms)


def visual_feed_stalled(latest_timestamp,timestamp_ms,limit_ms=2000):
    """A packet with no objects is healthy; an absent packet stream is not."""
    return timestamp_ms-(latest_timestamp if latest_timestamp is not None else 0)>limit_ms

"""Conservative joint diagnostic gate for noise-driven repetitive ASR output."""

import math

POLICY = "reject_high_no_speech_and_repetition_v1"


def segment_quality(segment) -> dict:
    """Use upstream segment statistics; this is not a calibrated confidence score.

    Repetition alone may be genuine speech. Reject only when the recognizer also
    reports high no-speech probability. Original rejected text remains available
    in the private transcript diagnostics; it is not sent as SpeechEvent evidence.
    """
    values = {}
    for key in ("no_speech_prob", "compression_ratio"):
        value = getattr(segment, key, 0.0)
        try:
            number = float(value)
        except (ValueError, TypeError):
            number = 0.0
        values[key] = number if math.isfinite(number) else 0.0
    suspicious = values["no_speech_prob"] >= 0.6 and values["compression_ratio"] >= 2.4
    return {"accepted": not suspicious,
            "quality_reason": "HIGH_NO_SPEECH_AND_REPETITION" if suspicious else None,
            "quality_policy": POLICY, **values}

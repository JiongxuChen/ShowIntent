from src.speech.bridge import SpeechEvidenceBridge
from src.speech.events import SpeechEvent
from src.speech.language import extract_cues


def packet(classes=("apple",), timestamp_ms=1200):
    return {"timestamp_ms": timestamp_ms, "objects": [{"class_name": name} for name in classes]}


def test_speech_event_requires_causal_availability():
    try:
        SpeechEvent(0, 200, 199, "苹果")
    except ValueError:
        pass
    else:
        raise AssertionError("future speech must be rejected")


def test_lexical_cues_do_not_treat_a_closing_as_presentation():
    cues = extract_cues("今天就先給大家展示到這裡")
    assert cues.cue_types == ["ENDING_LANGUAGE"]


def test_bridge_abstains_for_ambiguous_instances_and_never_returns_intent_fields():
    event = SpeechEvent(100, 500, 800, "给大家看苹果")
    evidence = SpeechEvidenceBridge([event], enabled=True).evidence_for(packet(("apple", "apple")))
    assert evidence["reason_codes"] == ["AMBIGUOUS_INSTANCES"]
    assert not evidence["has_evidence"]
    assert not {"state", "target_id", "track_id", "zoom"} & set(evidence)


def test_disabled_bridge_is_a_noop():
    event = SpeechEvent(100, 500, 800, "给大家看苹果")
    evidence = SpeechEvidenceBridge([event]).evidence_for(packet())
    assert evidence["reason_codes"] == ["DISABLED"]

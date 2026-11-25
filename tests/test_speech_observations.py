"""Language timing and context regressions, without an ASR model."""
from src.speech.bridge import SpeechEvidenceBridge
from src.speech.events import SpeechEvent
from src.speech.language import extract_cues


def observe(events,now):
    return SpeechEvidenceBridge(events,enabled=True).evidence_for(dict(timestamp_ms=now,objects=[dict(class_name='apple')]))


def test_four_second_window_does_not_expire_a_recently_available_short_utterance():
    result=observe([SpeechEvent(0,600,4100,'给大家看苹果')],4100)
    assert result['has_evidence'] and result['presentation_language']


def test_unreasonably_delayed_result_is_expired_even_if_just_available():
    assert observe([SpeechEvent(0,600,10000,'给大家看苹果')],10000)['reason_codes']==['EXPIRED_SPEECH']


def test_result_expires_after_availability_and_keeps_no_positive_cues():
    result=observe([SpeechEvent(0,600,1000,'给大家看苹果')],4100)
    assert result['reason_codes']==['EXPIRED_SPEECH']
    assert not result['cue_types'] and not result['has_evidence']


def test_continuation_and_non_current_speech_have_explicit_types():
    assert 'CONTINUATION_LANGUAGE' in extract_cues('还没展示完这个苹果').cue_types
    assert 'NEGATED_LANGUAGE' not in extract_cues('还没展示完这个苹果').cue_types
    assert 'NON_CURRENT_LANGUAGE' in extract_cues('明天给大家展示苹果').cue_types


def test_old_continuation_annotation_does_not_keep_its_historical_negation():
    e=SpeechEvent(100,500,800,'还没展示完苹果',cue_types=['NEGATED_LANGUAGE'])
    assert 'NEGATED_LANGUAGE' not in observe([e],1000)['cue_types']

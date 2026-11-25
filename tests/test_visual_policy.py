"""Reducing visual dominance preserves silence and reports the same-policy ablation."""
import json
from src.intent.temporal import TemporalIntentStateMachine
from src.intent.fusion import FusionConfig
from src.intent.replay_real import replay_clip
from src.speech.events import SpeechEvent
from src.speech.bridge import SpeechEvidenceBridge
from src.speech.intent_language import PresentationLanguage


def rows():
    result=[]
    for t in range(0,4001,100):
        y=.8 if t<1000 else .45
        result.append(dict(timestamp_ms=t,frame_id=t//100,
          objects=[dict(track_id='a',class_name='apple',bbox_norm=[.3,y-.05,.4,y+.05],score=.9)],
          hands=[dict(hand_id='h',bbox_norm=[.3,y-.05,.4,y+.05],score=.9)],
          relations=[dict(hand_id='h',object_id='a',type='near_or_moving_together',score=.9)] if t>=1000 else []))
    return result


def replay(events=(),legacy=False):
    cfg=FusionConfig(visual_full_strength=1.,visual_confirmation_ms=600) if legacy else FusionConfig()
    engine=TemporalIntentStateMachine(fusion_config=cfg)
    bridge=SpeechEvidenceBridge(list(events),enabled=True)
    return [engine.update(p,speech_evidence=bridge.evidence_for(p)) for p in rows()]


def first_show(result):return next(r['timestamp_ms'] for r in result if r['state']=='SHOWING')


def test_silent_clear_presentation_remains_independent_but_more_conservative():
    assert first_show(replay())-first_show(replay(legacy=True))==200


def test_surface_language_recovers_confirmation_without_a_strong_single_modality():
    raw=SpeechEvent(1000,1200,1300,'这个苹果的表面很光滑')
    event=SpeechEvent(1000,1200,1300,raw.text,semantics=PresentationLanguage().analyze_event(raw))
    combined=replay([event])
    assert first_show(combined)<first_show(replay())
    assert 'JOINT_EVIDENCE' in next(r for r in combined if r['state']=='SHOWING')['reason_codes']
    engine=TemporalIntentStateMachine(fusion_config=FusionConfig(),language_only=True)
    bridge=SpeechEvidenceBridge([event],enabled=True)
    assert not any(engine.update(p,speech_evidence=bridge.evidence_for(p))['state']=='SHOWING' for p in rows())


def test_reference_can_complement_full_visual_observation_without_being_sufficient_alone():
    assert first_show(replay([SpeechEvent(1000,1200,1300,'这是这个苹果')]))<first_show(replay())


def test_new_policy_ablation_uses_identical_inputs(tmp_path):
    source=tmp_path/'perception_A01.jsonl';speech=tmp_path/'speech.jsonl'
    source.write_text(''.join(json.dumps(r)+'\n' for r in rows()));speech.write_text('')
    out=tmp_path/'replay'
    report=replay_clip(source,out,{},speech_source=speech,include_speech_ablation=True)
    assert 'visual-policy-only' in report['methods']
    a=[json.loads(x) for x in (out/'intent_showintent.jsonl').read_text().splitlines()]
    b=[json.loads(x) for x in (out/'intent_visual-policy-only.jsonl').read_text().splitlines()]
    assert a==b

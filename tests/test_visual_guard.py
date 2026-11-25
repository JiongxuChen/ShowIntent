from src.intent.persistent import PersistentIntentStateMachine
from src.intent.semantic_priority import SemanticPriorityIntentStateMachine,SemanticBranchIntentStateMachine
from src.intent.fusion import FusionConfig
from src.speech.bridge import SpeechEvidenceBridge
from src.speech.events import SpeechEvent
from dataclasses import replace
from src.speech.presentation_language import ContextualPresentationLanguage

def language_events(events):
    language=ContextualPresentationLanguage()
    return [replace(e,available_at_ms=e.end_ms,semantics=language.analyze_event(e)) for e in sorted(events,key=lambda x:(x.end_ms,x.start_ms))]


def packet(t):
    y=.45 if 1000<=t<4000 else .8
    return dict(timestamp_ms=t,frame_id=t,objects=[dict(track_id='a',class_name='cup',bbox_norm=[.35,y-.05,.45,y+.05],score=.9)],hands=[dict(hand_id='h',bbox_norm=[.35,y-.05,.45,y+.05],score=.9)],relations=[dict(hand_id='h',object_id='a',type='near_or_moving_together',score=.9)])


def test_spoken_ending_does_not_erase_current_visual_show_and_putdown_still_exits():
    visual=PersistentIntentStateMachine();guard=SemanticPriorityIntentStateMachine()
    for t in range(0,5001,100):
        p=packet(t);v=visual.update(p)
        evidence=None
        if t>=2500:evidence=dict(timestamp_ms=t,enabled=True,cue_types=['ENDING_LANGUAGE'],source_text='今天的展示就到这里',source_start_ms=2200,source_end_ms=2500,source_available_at_ms=2500)
        a=guard.update(p,speech_evidence=evidence)
        if v['state']=='SHOWING':assert (a['state'],a['target_id'])==('SHOWING',v['target_id'])
        if t==3500:assert a['state']=='SHOWING' and 'VISUAL_EVIDENCE_PRESERVED' in a['reason_codes']
        if t>=4500:assert a['state']!='SHOWING'


def test_raw_audio_continuation_cannot_extend_visual_exit(monkeypatch):
    def raw(self,p,**kwargs):
        return dict(timestamp_ms=p['timestamp_ms'],state='SHOWING',target_id='a',score=1.,reason_codes=['SPEECH_REFERENCE'])
    monkeypatch.setattr(SemanticBranchIntentStateMachine,'update',raw)
    guard=SemanticPriorityIntentStateMachine()
    for t in range(0,5001,100):
        a=guard.update(packet(t))
        if t>=4500:assert a['state']!='SHOWING'


def test_visual_target_is_preserved_against_conflicting_audio_target(monkeypatch):
    def raw(self,p,**kwargs):
        return dict(timestamp_ms=p['timestamp_ms'],state='SHOWING',target_id='b',score=1.,reason_codes=['SPEECH_REFERENCE'])
    monkeypatch.setattr(SemanticBranchIntentStateMachine,'update',raw)
    guard=SemanticPriorityIntentStateMachine()
    for t in range(0,3001,100):
        p=packet(t);p['objects'].append(dict(track_id='b',class_name='book',bbox_norm=[.65,.75,.75,.85],score=.9))
        a=guard.update(p)
        if t>=2000:assert a['state']=='SHOWING' and a['target_id']=='a'


def test_audio_lead_below_configured_minimum_is_rejected():
    events=language_events([SpeechEvent(1000,1200,1200,'这个是一个杯子')])
    # This stream gives a 200ms audio lead; require 300ms for admission.
    cfg=FusionConfig(speech_weight=2,confirmation_ms=450,visual_confirmation_ms=600,visual_full_strength=1)
    engines=[SemanticPriorityIntentStateMachine(fusion_config=cfg,min_audio_lead_ms=m) for m in [0,300]]
    first=[]
    for engine in engines:
        bridge=SpeechEvidenceBridge(events,enabled=True);values=[]
        for t in range(0,3001,100):
            p=packet(t);values.append(engine.update(p,speech_evidence=bridge.evidence_for(p)))
        first.append(next(v['timestamp_ms'] for v in values if v['state']=='SHOWING'))
    assert first[0]<first[1]


def test_shadow_resets_on_media_gap():
    engine=SemanticPriorityIntentStateMachine()
    for t in range(0,3001,100):engine.update(packet(t))
    result=engine.update(dict(timestamp_ms=5000,frame_id=5000,objects=[],hands=[],relations=[]))
    assert result['state']=='NONE' and not engine._closed_targets and engine._admitted is None

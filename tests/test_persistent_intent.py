import pytest

from src.intent.persistent import PersistentIntentStateMachine,PresentationGrounder
from src.intent.fusion import FusionConfig
from src.speech.presentation_language import ContextualPresentationLanguage
from src.speech.events import SpeechEvent
from src.speech.semantics import validate_semantics


def packet(t,y=.8,contact=True):
    return dict(timestamp_ms=t,frame_id=t,objects=[dict(track_id='object',class_name='cup',bbox_norm=[.35,y-.05,.45,y+.05],score=.9)],
                hands=[dict(hand_id='h',bbox_norm=[.35,y-.05,.45,y+.05],score=.9)] if contact else [],
                relations=[dict(hand_id='h',object_id='object',type='near_or_moving_together',score=.9)] if contact else [])


def test_confirmed_hold_survives_occluded_hand_but_exits_on_putdown():
    engine=PersistentIntentStateMachine()
    values=[]
    for t in range(0,7001,100):
        values.append(engine.update(packet(t,y=.8 if t<1000 or t>=6000 else .45,contact=t<=1200)))
    assert all(v['state']=='SHOWING' for v in values if 2500<=v['timestamp_ms']<6000)
    assert any('HAND_HOLD_RETAINED' in v['reason_codes'] for v in values)
    assert not any(v['state']=='SHOWING' for v in values if v['timestamp_ms']>=6500)


def test_static_unhandled_item_never_gets_inferred_contact():
    engine=PersistentIntentStateMachine()
    assert all(engine.update(packet(t,y=.45,contact=False))['state']!='SHOWING' for t in range(0,5000,100))


def test_large_input_gap_clears_memory_and_keeps_guarded_grounder():
    engine=PersistentIntentStateMachine(fusion_config=FusionConfig())
    for t in range(0,2500,100):engine.update(packet(t,y=.8 if t<1000 else .45))
    assert engine.state=='SHOWING'
    assert engine.update(packet(5000,y=.45,contact=False))['state']!='SHOWING'
    assert not engine.interactions
    assert isinstance(engine.fusion.grounder,PresentationGrounder)


def test_group_reference_cannot_bind_one_arbitrary_item():
    grounder=PresentationGrounder()
    target,status,_=grounder.resolve(1000,dict(source_text='给大家看看这两件产品',cue_types=['PRESENTATION_LANGUAGE'],source_start_ms=0,source_end_ms=1000,source_available_at_ms=1000),{}, {})
    assert target is None and status=='GROUP_INTRODUCTION_NOT_SINGLE_TARGET'


@pytest.mark.parametrize('text,act,classes',[
    ('这个是一个陶瓷咖啡杯','INTRODUCE',['cup']),
    ('杯身上有一个特别的小图案','DESCRIBE',['cup']),
    ('这瓶是新出的果汁','INTRODUCE',['bottle']),
])
def test_natural_introduction_and_description(text,act,classes):
    event=SpeechEvent(0,1000,1000,text)
    value=ContextualPresentationLanguage().analyze_event(event)
    validate_semantics(value,text)
    assert value['act']==act and value['classes']==classes


def test_future_description_is_not_promoted_to_current_show():
    event=SpeechEvent(0,1000,1000,'明天再介绍这个咖啡杯')
    value=ContextualPresentationLanguage().analyze_event(event)
    assert not value['current'] and value['act']=='NONE'


def test_explicit_session_closing_cannot_reopen_while_item_stays_held():
    engine=PersistentIntentStateMachine(fusion_config=FusionConfig())
    for t in range(0,2500,100):engine.update(packet(t,y=.8 if t<1000 else .45))
    assert engine.state=='SHOWING'
    for t in range(2500,5000,100):
        speech=dict(timestamp_ms=t,enabled=True,cue_types=['ENDING_LANGUAGE'],source_text='今天的展示就到这里',source_start_ms=2200,source_end_ms=2500,source_available_at_ms=2500)
        result=engine.update(packet(t,y=.45),speech_evidence=speech)
        assert result['state']!='SHOWING'
    assert engine.fusion.session_stop_ms==2500

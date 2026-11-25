from src.intent.audio_timeline import AudioTimeline
from src.speech.events import SpeechEvent


def event(text,start=0,end=1000,available=3000):
    return SpeechEvent(start,end,available,text)


def test_audio_only_needs_no_visual_input_and_does_not_emit_camera_target():
    engine=AudioTimeline([event('大家看看这个苹果')])
    assert engine.update(999)['state']=='NONE'
    result=engine.update(1000)
    assert result['state']=='SHOWING' and result['target_id'] is None
    assert engine.update(9000)['state']=='NONE'


def test_measured_availability_is_respected():
    engine=AudioTimeline([event('大家看看这个苹果')],timing='causal')
    assert engine.update(1000)['state']=='NONE'
    assert engine.update(3000)['state']=='SHOWING'


def test_price_alone_does_not_start_and_stop_closes_episode():
    engine=AudioTimeline([event('这个苹果五块钱一斤')])
    assert engine.update(1000)['state']=='NONE'
    engine=AudioTimeline([event('大家看看这个苹果'),event('先放下',1100,2000,4000)])
    assert engine.update(1000)['state']=='SHOWING'
    assert engine.update(2000)['state']=='NONE'


def test_continuation_does_not_create_cold_start():
    engine=AudioTimeline([event('它的封面是蓝色的')])
    assert engine.update(1000)['state']=='NONE'

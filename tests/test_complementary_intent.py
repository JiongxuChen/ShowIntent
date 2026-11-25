"""Complementarity, scoped controls, instance identity and causal semantics."""

from dataclasses import replace
import json
from pathlib import Path

import pytest

from src.intent.fusion import FusionConfig
from src.intent.temporal import TemporalIntentStateMachine
from src.speech.bridge import SpeechEvidenceBridge
from src.speech.events import SpeechEvent
from src.speech.language import extract_cues
from src.speech.semantics import validate_semantics
from src.perception.appearance_tracker import AppearanceTracker
from scripts.enrich_speech import enrich_events
from scripts.run_complementary import checked_stage,plans_for


def obj(tid='a',color='red',cls='unlisted_item',x=.3,y=.5):
    return dict(track_id=tid,class_name=cls,bbox_norm=[x,y,x+.1,y+.1],score=.9,attributes={'colors':[color]})


def packet(t,objects,related=()):
    return dict(timestamp_ms=t,frame_id=t,objects=objects,
                hands=[dict(hand_id='h',bbox_norm=[.3,.5,.4,.6],score=.9)] if related else [],
                relations=[dict(hand_id='h',object_id=tid,type='near_or_moving_together',score=.9) for tid in related])


def replay(rows,events=(),modality='joint'):
    bridge=SpeechEvidenceBridge(list(events),enabled=True)
    engine=TemporalIntentStateMachine(fusion_config=None if modality=='visual' else FusionConfig(),
                                      language_only=modality=='speech')
    result=[]
    for p in rows:
        result.append(engine.update(p) if modality=='visual' else engine.update(p,speech_evidence=bridge.evidence_for(p)))
    return result


@pytest.mark.parametrize('text',['不妨看看这本书','如果你刚学微积分，看看这本书','It is not only useful; look at this one'])
def test_positive_suggestion_and_local_condition_are_not_global_negations(text):
    cues=extract_cues(text).cue_types
    assert 'PRESENTATION_LANGUAGE' in cues
    assert not set(cues)&{'NEGATED_LANGUAGE','NON_CURRENT_LANGUAGE'}


def test_old_asr_annotations_are_recomputed_after_a_parser_fix():
    event=SpeechEvent(100,500,800,'不妨看看这本书',cue_types=['NEGATED_LANGUAGE'])
    e=SpeechEvidenceBridge([event],enabled=True).evidence_for(packet(1000,[obj(cls='book')]))
    assert 'NEGATED_LANGUAGE' not in e['cue_types']
    assert 'PRESENTATION_LANGUAGE' in e['cue_types']


def test_fragmentary_see_verb_without_a_referent_is_not_an_introduction():
    assert 'INTRODUCTION_LANGUAGE' not in extract_cues('我才先看，才哭得哭').cue_types


@pytest.mark.parametrize('text',['这个苹果不太好吃看看这个','这本书没有介绍完','给大家看看不是很漂亮的这个'])
def test_negative_quality_or_incomplete_description_is_not_an_instruction_to_stop(text):
    assert not set(extract_cues(text).cue_types)&{'NEGATED_LANGUAGE','ENDING_LANGUAGE'}


def test_partial_hold_and_introduction_jointly_trigger_but_each_ablation_abstains():
    # Recording starts after pickup. Neither a lift nor a strong language order
    # exists; category is deliberately absent from the naming lexicon.
    rows=[packet(t,[obj()],['a']) for t in range(0,3001,100)]
    events=[SpeechEvent(200,700,900,'首先是这个物品')]
    assert not any(r['state']=='SHOWING' for r in replay(rows,events,'visual'))
    assert not any(r['state']=='SHOWING' for r in replay(rows,events,'speech'))
    assert any(r['state']=='SHOWING' and 'JOINT_EVIDENCE' in r['reason_codes'] for r in replay(rows,events))


def test_ordinary_table_hold_and_product_name_still_do_not_trigger():
    rows=[packet(t,[obj(y=.75)],['a']) for t in range(0,3001,100)]
    assert not any(r['state']=='SHOWING' for r in replay(rows,[SpeechEvent(200,700,900,'这个物品很实惠')]))


def test_measured_color_resolves_two_interacted_instances_without_a_new_class_rule():
    rows=[packet(t,[obj('a','red'),obj('b','blue',x=.6)],['a','b']) for t in range(0,3001,100)]
    result=replay(rows,[SpeechEvent(200,700,900,'给大家看看蓝色的这个')])
    showing=[r for r in result if r['state']=='SHOWING']
    assert showing and all(r['target_id']=='b' for r in showing)


def test_delayed_ending_of_first_book_cannot_stop_the_second_book():
    rows=[]
    for t in range(0,5001,100):
        first=obj('first',cls='book',y=.8 if t<500 or t>=2400 else .4)
        second=obj('second',cls='book',x=.65,y=.8 if t<2600 else .4)
        related=['first'] if 500<=t<2400 else ['second'] if t>=2600 else []
        rows.append(packet(t,[first,second],related))
    delayed=SpeechEvent(1800,2200,3800,'这本书展示结束')
    out=replay(rows,[delayed])
    assert any(r['state']=='SHOWING' and r['target_id']=='second' for r in out if r['timestamp_ms']>=4000)
    assert not any('SPEECH_STOP' in r['reason_codes'] and r['target_id']=='second' for r in out)


def test_ambiguous_stop_is_not_broadcast_to_every_visible_instance():
    rows=[packet(t,[obj('a',cls='book'),obj('b',cls='book',x=.6)]) for t in range(0,2001,100)]
    bridge=SpeechEvidenceBridge([SpeechEvent(100,500,800,'不要展示书')],enabled=True)
    engine=TemporalIntentStateMachine(fusion_config=FusionConfig())
    for p in rows:engine.update(p,speech_evidence=bridge.evidence_for(p))
    assert not engine.fusion._blocked


def test_appearance_identity_survives_brief_loss_without_inventing_boxes():
    tracker=AppearanceTracker()
    a=dict(obj(cls='book'),appearance=[1.,0.])
    first=tracker.update(0,[a])[0]['track_id']
    assert tracker.update(100,[])==[]
    assert tracker.update(200,[dict(a,bbox_norm=[.31,.5,.41,.6])])[0]['track_id']==first
    # Different book at exactly the same position must not inherit the first ID.
    assert tracker.update(300,[dict(a,appearance=[0.,1.])])[0]['track_id']!=first


def test_semantic_observation_requires_a_current_source_quote_and_known_shape():
    value=dict(act='SHOW',current=True,referent_scope='mentioned',classes=[],object_phrase='这个',
               colors=[],quote='看看这个',confidence='high')
    validate_semantics(value,'请看看这个')
    with pytest.raises(ValueError):validate_semantics(value,'今天不错')
    with pytest.raises(ValueError):validate_semantics(dict(value,target_id='a'),'看看这个')


def test_semantic_compute_is_added_to_availability_and_context_uses_only_previous_words():
    class Model:
        def analyze(self,text,context,classes):
            assert context==([] if text=='第一句' else ['第一句'])
            return dict(act='NONE',current=True,referent_scope='unspecified',classes=[],object_phrase='',
                        colors=[],quote='',confidence='low')
    times=iter([0.,.1,.1,.3])
    events=[SpeechEvent(0,500,800,'第一句'),SpeechEvent(600,700,800,'第二句')]
    result,_=enrich_events(events,Model(),[],clock=lambda:next(times))
    assert result[0].available_at_ms>=900
    assert result[1].available_at_ms>=1100


def test_future_semantic_event_cannot_affect_the_earlier_prefix():
    rows=[packet(t,[obj()]) for t in range(0,2001,100)]
    event=SpeechEvent(100,500,3000,'看看这个',semantics=dict(act='SHOW',current=True,
             referent_scope='mentioned',classes=[],object_phrase='这个',colors=[],quote='看看这个',confidence='high'))
    assert replay(rows,[event])==replay(rows,[])


def test_resume_verifies_stage_inputs_and_outputs(tmp_path):
    out=tmp_path/'out';(out/'checkpoints').mkdir(parents=True);(out/'logs').mkdir()
    source=tmp_path/'source';source.write_text('source')
    product=out/'product'
    checked_stage(out,'clip','stage',['fake'],[product],{},action=lambda:product.write_text('done'),inputs=[source])
    checked_stage(out,'clip','stage',['fake'],[product],{},action=lambda:pytest.fail('recomputed'),inputs=[source])
    source.write_text('changed')
    with pytest.raises(ValueError):checked_stage(out,'clip','stage',['fake'],[product],{},inputs=[source])


def test_full_pipeline_has_no_clip_name_or_category_specific_model_arguments(tmp_path):
    plans=plans_for(tmp_path/'arbitrary.mp4',tmp_path/'run',Path('python'),vision_model=Path('v'),speech_model=Path('s'),semantic_model=Path('l'))
    assert 'src.perception.enhanced' in plans['perception'][0]
    assert 'scripts.enrich_speech' in plans['semantics'][0]
    assert '--extra-class' not in plans['perception'][0]
    assert '--context-ms' in plans['speech_raw'][0]

"""Joint decisions, causal language use, grounding, and modality conflicts."""

import copy
import json
import math
from pathlib import Path

import pytest

from src.intent.fusion import FusionConfig
from src.intent.run import run
from src.intent.temporal import TemporalIntentStateMachine
from src.speech.bridge import SpeechEvidenceBridge
from src.speech.events import SpeechEvent


def packet(t, *, y=.7, related=False, objects=None):
    objects = objects if objects is not None else [dict(track_id='a', class_name='apple',
                      bbox_norm=[.3,y-.05,.4,y+.05], score=.95)]
    return dict(timestamp_ms=t, frame_id=t, objects=objects,
                hands=[dict(hand_id='h',bbox_norm=[.3,.5,.4,.6],score=.9)] if related else [],
                relations=[dict(hand_id='h',object_id='a',type='near_or_moving_together',score=.9)] if related and objects else [])


def event(text, start=100, end=500, available=800, final=True):
    return SpeechEvent(start,end,available,text,is_final=final)


def replay(rows, events, *, fused=True):
    # Preserve regressions for the explicitly legacy evidence policy; the new
    # conservative policy has separate timing/ablation tests.
    engine = TemporalIntentStateMachine(fusion_config=FusionConfig(visual_full_strength=1.,visual_confirmation_ms=600) if fused else None)
    bridge = SpeechEvidenceBridge(events,enabled=True)
    outputs=[]
    for p in rows:
        outputs.append(engine.update(p,speech_evidence=bridge.evidence_for(p)) if fused else engine.update(p))
    return outputs


def static(until=3000, **options):
    return [packet(t,**options) for t in range(0,until+1,100)]


def lifted(until=3000):
    return [packet(t,y=.8 if t<1000 else .45,related=t>=1000) for t in range(0,until+1,100)]


def test_clear_speech_can_initiate_without_a_visual_lift_or_hand_detection():
    rows=static()
    assert not any(r['state']=='SHOWING' for r in replay(rows,[],fused=False))
    outputs=replay(rows,[event('给大家展示这个苹果')])
    assert any(r['state']=='SHOWING' for r in outputs)
    first=next(r for r in outputs if r['state']=='SHOWING')
    assert first['timestamp_ms']>=1400
    assert 'SPEECH_PRESENTATION' in first['reason_codes']
    assert first['target_id']=='a'


def test_visual_is_independently_sufficient_and_empty_language_is_identical():
    rows=lifted()
    assert replay(rows,[])==replay(rows,[],fused=False)
    assert any(r['state']=='SHOWING' for r in replay(rows,[]))


@pytest.mark.parametrize('text',['苹果很好吃，买一个吧','这是苹果','明天给大家展示苹果','如果展示苹果','刚才展示了苹果'])
def test_names_marketing_and_non_current_language_do_not_initiate_an_ordinary_hold(text):
    assert not any(r['state']=='SHOWING' for r in replay(static(related=True),[event(text)]))


def test_two_partial_observations_can_jointly_confirm_without_old_visual_acceptance():
    # Slow horizontal transport prevents the old pure-visual hold rule.
    rows=[]
    for t in range(0,3001,100):
        p=packet(t,y=.8 if t<1000 else .45,related=t>=1000)
        x=.3 + .06*max(0,t-1000)/1000
        p['objects'][0]['bbox_norm']=[x,.75 if t<1000 else .4,x+.1,.85 if t<1000 else .5]
        rows.append(p)
    events=[event('首先是这个苹果',start=1100,end=1400,available=1500)]
    assert not any(r['state']=='SHOWING' for r in replay(rows,[],fused=False))
    assert any(r['state']=='SHOWING' and 'JOINT_EVIDENCE' in r['reason_codes'] for r in replay(rows,events))


def test_deictic_speech_without_naming_is_resolved_by_a_unique_visible_instance():
    assert any(r['state']=='SHOWING' for r in replay(static(),[event('大家看看这个')]))


def test_same_category_ambiguity_abstains_but_unique_interaction_can_ground():
    objects=packet(0)['objects']
    objects=objects+[dict(objects[0],track_id='b',bbox_norm=[.6,.65,.7,.75])]
    assert not any(r['state']=='SHOWING' for r in replay(static(objects=objects),[event('给大家看苹果')]))
    outputs=replay(static(objects=objects,related=True),[event('给大家看苹果')])
    showing=[r for r in outputs if r['state']=='SHOWING']
    assert showing and all(r['target_id']=='a' for r in showing)


def test_missing_named_category_never_falls_back_to_an_unrelated_object():
    assert not any(r['state']=='SHOWING' for r in replay(static(),[event('给大家看橘子')]))


def test_no_visible_target_cannot_create_a_camera_target():
    assert not any(r['state']=='SHOWING' for r in replay(static(objects=[]),[event('给大家看苹果')]))


def test_future_and_partial_language_cannot_change_prefix_decisions():
    rows=static()
    base=replay(rows,[])
    assert replay(rows,[event('给大家看苹果',available=4000)])==base
    assert replay(rows,[event('给大家看苹果',final=False)])==base


def test_unrelated_chat_does_not_erase_an_available_presentation_statement():
    events=[event('给大家看苹果'),event('谢谢大家',start=600,end=900,available=1000)]
    assert any(r['state']=='SHOWING' for r in replay(static(),events))


def test_latest_control_cancels_earlier_positive_and_does_not_resurrect_after_ttl():
    events=[event('不要展示苹果',start=2100,end=2300,available=2400)]
    rows=replay(lifted(8000),events)
    assert any(r['state']=='SHOWING' for r in rows if r['timestamp_ms']<2400)
    assert not any(r['state']=='SHOWING' for r in rows if r['timestamp_ms']>=2400)
    assert any('CROSS_MODAL_CONFLICT' in r['reason_codes'] for r in rows)


def test_ending_statement_and_topic_change_can_end_an_ongoing_visual_show():
    for text in ('苹果展示结束','换下一个'):
        rows=replay(lifted(5000),[event(text,start=2100,end=2300,available=2400)])
        assert not any(r['state']=='SHOWING' for r in rows if r['timestamp_ms']>=2400)


def test_ending_a_different_category_does_not_end_the_current_one():
    rows=replay(lifted(),[event('橘子展示结束',start=2100,end=2300,available=2400)])
    assert rows[-1]['state']=='SHOWING'


def test_continuation_is_not_mistaken_for_negation_or_ending():
    rows=replay(lifted(),[event('苹果还没展示完',start=2100,end=2300,available=2400)])
    assert rows[-1]['state']=='SHOWING'


def test_observed_putdown_supersedes_an_older_positive_statement():
    rows=lifted(2200)+[packet(t,y=.8,related=True) for t in range(2300,5001,100)]
    outputs=replay(rows,[event('给大家展示苹果',start=1100,end=1500,available=1700)])
    assert outputs[-1]['state']!='SHOWING'
    assert not any(r['state']=='SHOWING' for r in outputs if r['timestamp_ms']>=2600)


def test_a_fresh_explicit_statement_can_start_a_new_episode_after_ending():
    events=[event('苹果展示结束',start=2100,end=2300,available=2400),
            event('给大家展示苹果',start=3000,end=3300,available=3500)]
    rows=replay(lifted(5000),events)
    assert any(r['state']=='SHOWING' for r in rows if r['timestamp_ms']>=4100)


def test_inputs_are_not_modified():
    rows=static();events=[event('给大家看苹果')]
    before=copy.deepcopy((rows,events))
    replay(rows,events)
    assert (rows,events)==before


@pytest.mark.parametrize('kwargs',[{'visual_weight':0},{'speech_weight':float('nan')},{'confirmation_ms':True},
                                    {'support_threshold':2},{'candidate_threshold':2}])
def test_invalid_fusion_configuration_is_rejected(kwargs):
    with pytest.raises(ValueError):FusionConfig(**kwargs)


def test_cli_api_produces_aligned_intent_and_evidence_and_protects_inputs(tmp_path):
    source=tmp_path/'p.jsonl';speech=tmp_path/'s.jsonl';out=tmp_path/'i.jsonl';trace=tmp_path/'trace.jsonl'
    source.write_text(''.join(json.dumps(p)+'\n' for p in static()))
    speech.write_text(json.dumps(event('给大家看苹果').to_dict())+'\n')
    assert run(source,out,'showintent',speech_path=speech,evidence_out=trace)==31
    intents=[json.loads(s) for s in out.read_text().splitlines()]
    evidence=[json.loads(s) for s in trace.read_text().splitlines()]
    assert [r['timestamp_ms'] for r in intents]==[r['timestamp_ms'] for r in evidence]
    assert any(r['state']=='SHOWING' for r in intents)
    for output in (source,speech,out):
        with pytest.raises(ValueError):run(source,output,'showintent',speech_path=speech)
    with pytest.raises(ValueError):run(source,tmp_path/'b.jsonl','pickup-baseline',speech_path=speech)


def test_non_current_negative_does_not_cancel_a_current_visual_show():
    rows=replay(lifted(),[event('明天不要展示苹果',start=2100,end=2300,available=2400)])
    assert rows[-1]['state']=='SHOWING'


def test_brief_relation_dropout_does_not_undo_an_explicit_stop():
    rows=lifted(6500)
    for p in rows:
        if 3000<=p['timestamp_ms']<=3200:p['relations']=[]
    outputs=replay(rows,[event('苹果展示结束',start=2100,end=2300,available=2400)])
    assert not any(r['state']=='SHOWING' for r in outputs if r['timestamp_ms']>=2400)


def test_batch_replay_compares_the_same_perception_with_and_without_language(tmp_path):
    from src.intent.replay_real import replay_clip
    source=tmp_path/'perception_A01.jsonl';speech=tmp_path/'speech_events_A01.jsonl'
    source.write_text(''.join(json.dumps(p)+'\n' for p in static()))
    speech.write_text(json.dumps(event('给大家展示苹果').to_dict())+'\n')
    out=tmp_path/'replay'
    report=replay_clip(source,out,{},speech_source=speech)
    assert set(report['methods'])=={'visual-only','showintent','pickup-baseline'}
    assert report['methods']['showintent']['showing_segments']
    assert not report['methods']['visual-only']['showing_segments']
    assert report['speech_sha256']
    assert (out/'evidence_showintent.jsonl').is_file()
    with pytest.raises(FileExistsError):replay_clip(source,out,{},speech_source=speech)


@pytest.mark.parametrize('text',['这个苹果展示效果很好','他说给大家看苹果'])
def test_describing_or_quoting_a_presentation_does_not_count_as_performing_one(text):
    assert not any(r['state']=='SHOWING' for r in replay(static(related=True),[event(text)]))


def test_language_decision_does_not_fabricate_visual_reason_codes():
    outputs=replay(static(),[event('给大家看苹果')])
    showing=[r for r in outputs if r['state']=='SHOWING']
    assert showing
    for row in showing:
        assert 'HAND_OBJECT_ASSOCIATED' not in row['reason_codes']
        assert 'RAISED_IN_FRAME' not in row['reason_codes']


def test_joint_cli_fixture_and_existing_camera_interface_work_together(tmp_path):
    import subprocess
    import sys
    output=tmp_path/'intent.jsonl';trace=tmp_path/'trace.jsonl';shot=tmp_path/'shot.jsonl'
    command=[sys.executable,'-B','-m','src.intent.run','--perception','tests/fixtures/demo_object_show/perception.jsonl',
             '--speech-events','tests/fixtures/demo_object_show/speech_events.jsonl',
             '--fusion-config','configs/evidence_fusion.json','--out',str(output),'--evidence-out',str(trace)]
    subprocess.run(command,check=True,capture_output=True,text=True)
    subprocess.run([sys.executable,'-B','-m','src.camera.run','--perception','tests/fixtures/demo_object_show/perception.jsonl',
                    '--intent',str(output),'--out',str(shot)],check=True,capture_output=True,text=True)
    assert len(output.read_text().splitlines())==len(shot.read_text().splitlines())==61
    assert any(json.loads(line)['state']=='SHOWING' for line in output.read_text().splitlines())


def test_equal_strong_modalities_naming_different_targets_do_not_get_a_language_tiebreak():
    rows=[]
    for t in range(0,3001,100):
        p=packet(t,y=.8 if t<1000 else .45,related=t>=1000)
        p['objects'][0]['class_name']='orange'
        p['objects'].append(dict(track_id='b',class_name='apple',bbox_norm=[.6,.75,.7,.85],score=.95))
        rows.append(p)
    outputs=replay(rows,[event('给大家展示苹果',start=600,end=900,available=1100)])
    assert not any(r['state']=='SHOWING' for r in outputs)
    assert any('CROSS_MODAL_TARGET_CONFLICT' in r['reason_codes'] for r in outputs)


def test_deictic_statement_cannot_hop_to_the_last_remaining_category():
    rows=[]
    for t in range(0,4501,100):
        apple=dict(track_id='a',class_name='apple',bbox_norm=[.3,.4,.4,.5],score=.95)
        orange=dict(track_id='b',class_name='orange',bbox_norm=[.7,.75,.8,.85],score=.95)
        rows.append(packet(t,objects=[apple,orange] if t<1800 else [orange],related=t<1800))
    outputs=replay(rows,[event('给大家看看这个',start=100,end=500,available=800)])
    assert any(r['state']=='SHOWING' and r['target_id']=='a' for r in outputs)
    assert not any(r['state']=='SHOWING' and r['target_id']=='b' for r in outputs)


def test_delayed_pronoun_is_not_bound_to_an_unconfirmed_stale_candidate():
    rows=[]
    for t in range(0,6001,100):
        p=packet(t,y=.8,related=True)
        y=.8 if t<2500 else .74
        x=.65 if t<2500 else .65-.08*min(t-2500,2000)/1000
        p['objects'].append(dict(track_id='b',class_name='orange',bbox_norm=[x-.05,y-.05,x+.05,y+.05],score=.95))
        p['relations'].append(dict(hand_id='h',object_id='b',type='near_or_moving_together',score=.7))
        rows.append(p)
    outputs=replay(rows,[event('给大家看看这个',start=1000,end=3500,available=4000)])
    showing=[r for r in outputs if r['state']=='SHOWING']
    assert showing and all(r['target_id']=='b' for r in showing)


def test_old_attribute_description_cannot_hold_a_returning_object_above_its_noisy_table_anchor():
    rows=[]
    for t in range(0,6501,100):
        y=.8 if t<1000 else .5 if t<3000 else min(.76,.5+(t-3000)*.0002)
        p=packet(t,y=y,related=True)
        rows.append(p)
    outputs=replay(rows,[event('这是这个苹果',start=2000,end=2800,available=3300)])
    assert any(r['state']=='SHOWING' for r in outputs if r['timestamp_ms']<3300)
    assert not any(r['state']=='SHOWING' for r in outputs if r['timestamp_ms']>=5000)

"""Guard trigger precision against the failure modes found on real inputs."""

import copy

from src.intent.temporal import TemporalIntentStateMachine
from src.intent.mock_scenarios import make_scenarios
from src.intent.evaluate_review import score_intervals


def obj(tid, y, x=.4, cls='cup'):
    return {'track_id':tid,'class_name':cls,'bbox_norm':[x-.05,y-.05,x+.05,y+.05],'score':.95}


def packet(t, objects, related=(), hand_y=.8):
    return {'timestamp_ms':t,'frame_id':t,'objects':objects,
            'hands':[{'hand_id':'h','bbox_norm':[.35,hand_y-.05,.45,hand_y+.05],'score':.9}],
            'relations':[{'hand_id':'h','object_id':tid,'type':'near_or_moving_together','score':.9} for tid in related]}


def replay(rows):
    sm=TemporalIntentStateMachine()
    return [sm.update(p) for p in rows]


def test_hand_wave_does_not_turn_a_static_shelf_object_into_presentation():
    rows=[]
    for t in range(0,4001,100):
        rows.append(packet(t,[obj('static',.715),obj('table',.8,x=.7)],['static'],
                           hand_y=.8 if t<1000 else .65))
    assert not any(r['state']=='SHOWING' for r in replay(rows))


def test_early_hand_wave_cannot_bootstrap_an_object_already_visible_at_start():
    rows=[packet(t,[obj('static',.65),obj('table',.8,.7)],['static'],
                 hand_y=.8 if t<500 else .65) for t in range(0,4001,100)]
    assert not any(r['state']=='SHOWING' for r in replay(rows))


def test_scale_growth_supports_a_moving_presentation_but_is_not_a_trigger_alone():
    for lifted in (False,True):
        rows=[]
        for t in range(0,3501,100):
            y=.8 if lifted and t<1000 else .6
            x=.4 if t<1000 else .4+.08*(t-1000)/1000
            target=obj('a',y,x)
            if t>=1000:
                target['bbox_norm']=[x-.075,y-.075,x+.075,y+.075]
            rows.append(packet(t,[target,obj('table',.8,.85)],['a'],hand_y=.6))
        assert any(r['state']=='SHOWING' for r in replay(rows)) == lifted


def test_handedness_duplicates_and_array_order_do_not_change_decisions():
    original=next(s for s in make_scenarios(30) if s.name=='two_targets').packets
    altered=copy.deepcopy(original)
    for p in altered:
        p['hands'].reverse()
        for h in p['hands']:h['hand_id']='duplicate'
        for r in p['relations']:r['hand_id']='duplicate'
    assert replay(original)==replay(altered)


def test_candidate_can_keep_observed_lift_across_an_unambiguous_id_change():
    rows=[]
    for t in range(0,3001,100):
        tid='old' if t<1500 else 'new'
        rows.append(packet(t,[obj(tid,.8 if t<1000 else .65)],[] if t<1000 else [tid]))
    outputs=replay(rows)
    assert any(r['state']=='SHOWING' and r['target_id']=='new' for r in outputs)


def test_ambiguous_same_class_tracks_are_not_relinked_as_a_confirmed_target():
    rows=[packet(t,[obj('a',.8 if t<1000 else .45)],[] if t<1000 else ['a']) for t in range(0,2501,100)]
    rows += [packet(t,[obj('b',.46,.39),obj('c',.46,.41)],['b','c']) for t in range(2600,4501,100)]
    outputs=replay(rows)
    assert any(r['state']=='SHOWING' and r['target_id']=='a' for r in outputs)
    assert not any('TARGET_RELINKED' in r['reason_codes'] for r in outputs)


def test_tolerates_box_jitter_without_treating_slow_transport_as_hold():
    rows=[packet(t,[obj('a',.8 if t<1000 else .65,x=.4+(.004 if i%2 else -.004))],
                 [] if t<1000 else ['a']) for i,t in enumerate(range(0,4001,33))]
    assert replay(rows)[-1]['state']=='SHOWING'


def test_review_uses_timestamp_intersections_and_reports_wrong_class():
    perception=[packet(t,[obj('a',.6,cls='apple'),obj('b',.6,.7,cls='bottle')]) for t in (0,100,400,1000)]
    intents=[{'timestamp_ms':t,'state':state,'target_id':tid}
             for t,state,tid in ((0,'NONE',None),(100,'SHOWING','a'),(400,'SHOWING','b'),(1000,'NONE',None))]
    result=score_intervals(perception,intents,[{'start_ms':50,'end_ms':700,'label':'SHOWING','allowed_classes':'apple'},
                                              {'start_ms':700,'end_ms':1000,'label':'NON_SHOWING','allowed_classes':''}])
    assert result[0]['covered_ms']==650
    assert result[0]['correct_class_showing_ms']==300
    assert result[0]['wrong_class_showing_ms']==300
    assert result[1]['showing_ms']==300

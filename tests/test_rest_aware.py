from types import SimpleNamespace
from src.intent.grounding import Binding,UtteranceGrounder
from src.intent.rest_aware import RestAwareGrounder,RestAwareFusion
from src.intent.fusion import FusionConfig


def binding(y=.70):return Binding('x','cup',.5,y,y,0)


def test_adjustment_while_raised_does_not_expire_show_language():
    rows=[dict(y=.79,rest_y=.87,lift_seen=True),dict(y=.71,rest_y=.87,lift_seen=True)]
    b=binding()
    assert UtteranceGrounder()._returned_between(b,rows)
    assert not RestAwareGrounder()._returned_between(b,rows)


def test_true_putdown_and_old_statement_remain_invalidated():
    rows=[dict(y=.68,rest_y=.87,lift_seen=True),dict(y=.86,rest_y=.87,lift_seen=True)]
    assert RestAwareGrounder()._returned_between(binding(),rows)


def test_statement_before_pickup_can_still_expire_after_real_return():
    rows=[dict(y=.70,rest_y=.87,lift_seen=True),dict(y=.87,rest_y=.87,lift_seen=True)]
    assert RestAwareGrounder()._returned_between(binding(.87),rows)


def test_tiny_rest_jitter_without_lift_does_not_invent_return():
    rows=[dict(y=.86,rest_y=.87,lift_seen=False),dict(y=.88,rest_y=.87,lift_seen=False)]
    assert not RestAwareGrounder()._returned_between(binding(.87),rows)


def test_fusion_reset_retains_rest_policy_and_custom_margin():
    f=RestAwareFusion(FusionConfig(),putdown_margin=.03);f.reset()
    assert isinstance(f.grounder,RestAwareGrounder) and f.grounder.putdown_margin==.03


def test_actual_grounding_remains_sticky_only_after_real_putdown():
    g=RestAwareGrounder()
    speech=dict(source_start_ms=0,source_end_ms=50,source_available_at_ms=50,suggested_classes=['cup'])
    returned=[]
    for ts,y in [(0,.70),(100,.79),(200,.70),(300,.86),(400,.65)]:
        obj=dict(class_name='cup',bbox_norm=[.45,y-.04,.55,y+.04])
        track=SimpleNamespace(points=[(ts,.5,y)],inherited_from=None,anchor_y=.87,lift_seen=True)
        g.observe(ts,{'x':obj},{'x':track},{'x':1},None)
        if ts:returned.append(g.resolve(ts,speech,{'x':obj},{'x':1})[2])
    assert returned==[False,False,True,True]


def test_prior_v2_showing_and_target_are_preserved(monkeypatch):
    from src.intent.rest_aware import RestAwareIntentStateMachine,RestAwareBranchIntentStateMachine
    engine=RestAwareIntentStateMachine()
    baseline=dict(timestamp_ms=100,state='SHOWING',target_id='original',score=1,reason_codes=[])
    engine.reference=SimpleNamespace(update=lambda *a,**k:baseline)
    def candidate(self,*a,**k):return dict(baseline,state='NONE',target_id=None)
    monkeypatch.setattr(RestAwareBranchIntentStateMachine,'update',candidate)
    assert engine.update(dict(timestamp_ms=100))==baseline

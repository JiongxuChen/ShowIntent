from src.intent.delayed_adaptation import DelayedIntentStateMachine
from src.intent.safe_early import SafeEarlyIntentStateMachine
from src.intent.streaming import SparseVisualIntentStateMachine
from src.intent.persistent import PersistentIntentStateMachine
def packet(t,*,kind='remote',with_history=True,raised=True,visible=True):
    # The lift precedes the first object detection and includes hand occlusion.
    y=.84 if t<600 else .55
    hands=[] if 900<=t<2000 or (not with_history and t<2000) else [dict(hand_id='h',bbox_norm=[.45,y-.03,.55,y+.03],score=.9)]
    objects=[dict(track_id='support',class_name='bottle',bbox_norm=[.7,.82,.9,.94],score=.9)]
    if t>=2000 and visible:
        cy=.55 if raised else .80
        objects.append(dict(track_id='held',class_name=kind,bbox_norm=[.42,cy-.05,.58,cy+.05],score=.9))
    return dict(timestamp_ms=t,frame_id=t//100,objects=objects,hands=hands,
                relations=[dict(hand_id='h',object_id='held',type='near_or_moving_together',score=.95)] if len(objects)>1 and hands else [])


def test_adaptation_inherits_safe_b_and_injects_only_visual_observation_recovery():
    adapted=DelayedIntentStateMachine();offline=SafeEarlyIntentStateMachine()
    assert isinstance(adapted,SafeEarlyIntentStateMachine)
    assert isinstance(adapted.visual_engine,SparseVisualIntentStateMachine)
    assert type(offline.visual_engine) is PersistentIntentStateMachine
    old=[];new=[]
    for ts in range(0,3900,100):
        p=packet(ts);old.append(offline.update(p));new.append(adapted.update(p))
    assert not any(r['state']=='SHOWING' for r in old)
    assert any(r['state']=='SHOWING' for r in new)


def test_no_observed_hand_lift_remains_negative_in_adapted_b():
    e=DelayedIntentStateMachine()
    assert not any(e.update(packet(ts,with_history=False))['state']=='SHOWING' for ts in range(0,3900,100))


def test_recovery_only_keeps_every_offline_timing_parameter():
    from src.intent.delayed_adaptation import RecoveryOnlyIntentStateMachine
    config=dict(observation_gap_ms=750,evidence_gap_ms=250,missing_tolerance_ms=600,presentation_exit_ms=600)
    offline=SafeEarlyIntentStateMachine(**config)
    recovered=RecoveryOnlyIntentStateMachine(**config)
    full=DelayedIntentStateMachine(**config)
    assert vars(offline.config)==vars(recovered.config)
    changed={k for k,v in vars(full.config).items() if v!=vars(offline.config)[k]}
    assert changed==set(config)
    assert full.config.showing_min_ms==offline.config.showing_min_ms
    assert full.config.putdown_min_ms==offline.config.putdown_min_ms

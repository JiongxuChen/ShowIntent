from src.intent.streaming import SparseVisualIntentStateMachine

def sequence(x,raised_y,person=True,bilateral=False):
    frames=[]
    for t in range(0,4200,100):
        y=.84 if t<600 else .84+(raised_y-.84)*min(1,(t-600)/600)
        objects=[dict(track_id='table_item',class_name='bottle',bbox_norm=[.7,.84,.8,.94],score=.9),dict(track_id='held',class_name='apple',bbox_norm=[x-.04,y-.04,x+.04,y+.04],score=.9)]
        if bilateral:objects.append(dict(track_id='other_held',class_name='bottle',bbox_norm=[.76,.66,.84,.74],score=.9))
        if person:objects.append(dict(track_id='presenter',class_name='person',bbox_norm=[.1,.1,.9,.9],score=.9))
        frames.append(dict(timestamp_ms=t,frame_id=t//100,objects=objects,hands=[dict(hand_id='hand',bbox_norm=[x-.05,y-.05,x+.05,y+.05],score=1)],relations=[dict(hand_id='hand',object_id='held',type='near_or_moving_together',score=1)]))
    return frames

def showing(x,y,person=True,bilateral=False):
    engine=SparseVisualIntentStateMachine()
    return any(engine.update(f)['state']=='SHOWING' for f in sequence(x,y,person,bilateral))

def test_low_side_hold_rejected_but_central_and_high_side_presentations_allowed():
    assert not showing(.2,.70,bilateral=True)
    assert showing(.2,.70)
    assert showing(.5,.70)
    assert showing(.2,.4,bilateral=True)

def test_missing_person_retains_existing_visual_behavior():
    assert showing(.2,.70,False)

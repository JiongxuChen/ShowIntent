from src.perception.motion_tracker import MotionObjectTracker


def test_motion_retains_identity_after_gap_and_never_emits_hidden_box():
    tracker=MotionObjectTracker()
    old=tracker.update([('apple',(.2,.2,.3,.3),.9)])[0][0]
    tracker.update([('apple',(.22,.2,.32,.3),.9)])
    assert tracker.update([])==[]
    assert tracker.update([('apple',(.26,.2,.36,.3),.9)])[0][0]==old


def test_wrong_class_and_far_object_get_new_ids():
    tracker=MotionObjectTracker()
    old=tracker.update([('apple',(.2,.2,.3,.3),.9)])[0][0]
    tracker.update([])
    result=tracker.update([('orange',(.2,.2,.3,.3),.9),('apple',(.7,.7,.8,.8),.9)])
    assert all(row[0]!=old for row in result)
    assert len({row[0] for row in result})==2


def test_merge_covering_two_mature_objects_is_not_emitted():
    tracker=MotionObjectTracker()
    objects=[('apple',(.2,.2,.3,.3),.9),('apple',(.32,.2,.42,.3),.9)]
    for _ in range(10):
        tracker.update(objects)
    assert tracker.update([('apple',(.18,.18,.44,.32),.6)])==[]
    assert tracker.events[-1]['event']=='MERGED_BOX_REJECTED'
    assert len(tracker.update(objects))==2

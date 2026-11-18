from src.perception.relink_tracker import RelinkObjectTracker


def test_short_occlusion_restores_id_without_emitting_fake_box():
    tracker = RelinkObjectTracker()
    old = tracker.update([('orange', (.3,.3,.4,.4), .9)])[0][0]
    assert tracker.update([]) == []
    assert tracker.update([]) == []
    assert tracker.update([('orange', (.31,.3,.41,.4), .9)])[0][0] == old


def test_far_or_wrong_class_does_not_inherit_id():
    for name, box in [('apple',(.3,.3,.4,.4)), ('orange',(.7,.7,.8,.8))]:
        tracker = RelinkObjectTracker()
        old = tracker.update([('orange',(.3,.3,.4,.4),.9)])[0][0]
        tracker.update([])
        assert tracker.update([(name,box,.9)])[0][0] != old


def test_ambiguous_reappearance_gets_new_identity():
    tracker = RelinkObjectTracker()
    old = {x[0] for x in tracker.update([('orange',(.3,.3,.4,.4),.9),('orange',(.34,.3,.44,.4),.9)])}
    tracker.update([])
    assert tracker.update([('orange',(.32,.3,.42,.4),.9)])[0][0] not in old


def test_partial_reveal_recovers_only_with_overlap():
    tracker = RelinkObjectTracker()
    old = tracker.update([('orange',(.3,.3,.5,.5),.9)])[0][0]
    tracker.update([])
    assert tracker.update([('orange',(.35,.35,.45,.45),.9)])[0][0] == old
    assert tracker.update([('orange',(.3,.3,.5,.5),.9)])[0][0] == old


def test_small_nonoverlapping_box_cannot_inherit_hidden_id():
    tracker = RelinkObjectTracker()
    old = tracker.update([('orange',(.3,.3,.5,.5),.9)])[0][0]
    tracker.update([])
    assert tracker.update([('orange',(.51,.35,.61,.45),.9)])[0][0] != old

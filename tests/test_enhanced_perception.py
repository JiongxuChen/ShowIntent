"""CPU geometry/appearance checks; requires A's image dependencies, not models."""

import pytest

cv2=pytest.importorskip('cv2')
np=pytest.importorskip('numpy')
from src.perception.enhanced import visual_attributes,interaction_relations,merge_detections
from src.perception.appearance_tracker import appearance_distance,AppearanceTracker


def test_measured_achromatic_appearance_distinguishes_white_and_black_objects():
    white=np.full((40,40,3),255,dtype=np.uint8)
    black=np.zeros_like(white)
    a,aa=visual_attributes(white,(0,0,40,40));b,bb=visual_attributes(black,(0,0,40,40))
    assert aa['colors']==['white'] and bb['colors']==['black']
    assert appearance_distance(a,b)>.5
    tracker=AppearanceTracker()
    base=dict(class_name='book',bbox_norm=[.2,.3,.5,.6],score=.9)
    first=tracker.update(0,[dict(base,appearance=a)])[0]['track_id']
    assert tracker.update(100,[dict(base,appearance=b)])[0]['track_id']!=first


def test_grip_at_a_large_object_edge_is_not_rejected_by_object_center_distance():
    hands=[dict(hand_id='h',bbox_norm=[.05,.4,.15,.5])]
    objects=[dict(track_id='book',class_name='book',bbox_norm=[.1,.2,.7,.7]),
             dict(track_id='person',class_name='person',bbox_norm=[0.,0.,1.,1.]),
             dict(track_id='far',class_name='other',bbox_norm=[.8,.8,.9,.9])]
    relations=interaction_relations(hands,objects)
    assert len(relations)==1 and relations[0]['object_id']=='book' and relations[0]['score']>.8


def test_crop_and_global_duplicates_do_not_create_two_targets():
    rows=[('book',(10,10,30,30),.8),('book',(11,11,30,30),.9),('book',(60,60,80,80),.7)]
    assert len(merge_detections(rows))==2

from src.perception.local_detection import fuse_detections


def test_global_object_outside_local_view_is_kept():
    high=dict(class_name='apple',bbox_norm=[.2,.1,.3,.2],score=.8)
    low=dict(class_name='apple',bbox_norm=[.2,.6,.3,.7],score=.9)
    assert fuse_detections([high],[low])==[low,high]


def test_local_overlap_replaces_global_duplicate_only_for_same_class():
    apple=dict(class_name='apple',bbox_norm=[.2,.6,.3,.7],score=.8)
    orange=dict(class_name='orange',bbox_norm=[.2,.6,.3,.7],score=.8)
    assert fuse_detections([apple],[apple])==[apple]
    assert len(fuse_detections([orange],[apple]))==2

"""Detector-defined classes, exact PTS, and category-independent item decisions."""
import importlib.util
from fractions import Fraction
from pathlib import Path
import sys
from types import ModuleType,SimpleNamespace

import pytest

from src.perception.timing import load_timestamps,timestamps_from_frames,validate_timestamps
from src.intent.fusion import FusionConfig
from src.intent.temporal import TemporalIntentStateMachine
from src.speech.bridge import SpeechEvidenceBridge
from src.speech.events import SpeechEvent


def perception_module(monkeypatch):
    monkeypatch.setitem(sys.modules,'cv2',ModuleType('cv2'))
    monkeypatch.setitem(sys.modules,'numpy',ModuleType('numpy'))
    spec=importlib.util.spec_from_file_location('src.perception._category_test',Path('src/perception/run.py'))
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    return module


def test_a_retains_all_model_labels_without_a_product_whitelist(monkeypatch):
    module=perception_module(monkeypatch)
    labels=['spoon','toaster','laptop','book','person','model_specific_label']
    box=SimpleNamespace(tolist=lambda:[10,10,20,20])
    boxes=[SimpleNamespace(cls=[i],xyxy=[box],conf=[.9]) for i in range(len(labels))]
    result=SimpleNamespace(names=dict(enumerate(labels)),boxes=boxes)
    model=module.PerceptionModels.__new__(module.PerceptionModels)
    model.yolo=lambda *a,**k:[result];model.confidence=.35
    assert [r[0] for r in model.detect_objects(None,1280)]==labels


def test_pts_correctly_represents_the_repeated_opencv_start_without_inventing_time():
    frames=[SimpleNamespace(pts=p,time_base=Fraction(1,90000)) for p in (0,2999,5999)]
    assert timestamps_from_frames(frames)==[0,33,66]


def test_vfr_pts_are_not_replaced_by_a_fixed_fps_clock():
    frames=[SimpleNamespace(pts=p,time_base=Fraction(1,1000)) for p in (0,40,125,200)]
    assert timestamps_from_frames(frames)==[0,40,125,200]


@pytest.mark.parametrize('values',[[33,33,66],[0,-1],[0,True],[]])
def test_invalid_media_time_maps_remain_rejected(values):
    with pytest.raises(ValueError):validate_timestamps(values)


def test_timestamp_map_load_rejects_frame_count_mismatch(tmp_path):
    path=tmp_path/'time.json';path.write_text('{"frame_count":3,"timestamp_ms":[0,33]}')
    with pytest.raises(ValueError):load_timestamps(path)


@pytest.mark.parametrize('cls',['spoon','toaster','laptop','model_specific_label'])
def test_b_can_confirm_a_new_detector_class_without_adding_a_language_alias(cls):
    bridge=SpeechEvidenceBridge([SpeechEvent(100,500,800,'给大家看看这个物品')],enabled=True)
    engine=TemporalIntentStateMachine(fusion_config=FusionConfig())
    outputs=[]
    for t in range(0,2501,100):
        p=dict(timestamp_ms=t,frame_id=t,objects=[dict(track_id='new',class_name=cls,bbox_norm=[.3,.6,.4,.7],score=.9)],hands=[],relations=[])
        outputs.append(engine.update(p,speech_evidence=bridge.evidence_for(p)))
    assert any(r['state']=='SHOWING' and r['target_id']=='new' for r in outputs)


def test_a_person_is_not_selected_as_a_product_even_with_explicit_speech():
    bridge=SpeechEvidenceBridge([SpeechEvent(100,500,800,'给大家看看这个')],enabled=True)
    engine=TemporalIntentStateMachine(fusion_config=FusionConfig())
    for t in range(0,2501,100):
        p=dict(timestamp_ms=t,frame_id=t,objects=[dict(track_id='presenter',class_name='person',bbox_norm=[.1,.1,.9,.9],score=.9)],hands=[],relations=[])
        assert engine.update(p,speech_evidence=bridge.evidence_for(p))['state']!='SHOWING'


def test_model_runner_uses_pts_map_even_when_opencv_duplicates_the_start(tmp_path,monkeypatch):
    import json
    module=perception_module(monkeypatch);cv=module.cv2
    for name,value in dict(CAP_PROP_FPS=1,CAP_PROP_ORIENTATION_META=2,CAP_PROP_POS_MSEC=3,
                           ROTATE_90_CLOCKWISE=0,ROTATE_90_COUNTERCLOCKWISE=1,ROTATE_180=2).items():
        setattr(cv,name,value)
    frame=SimpleNamespace(shape=(480,640,3));frame.copy=lambda:frame
    class Capture:
        index=0
        def isOpened(self):return True
        def get(self,key):
            if key==cv.CAP_PROP_FPS:return 30
            if key==cv.CAP_PROP_ORIENTATION_META:return 0
            return (33,33,66)[self.index-1]
        def read(self):
            self.index+=1
            return (True,frame) if self.index<=3 else (False,None)
        def release(self):pass
    cv.VideoCapture=lambda _:Capture();cv.destroyAllWindows=lambda:None
    model=SimpleNamespace(detect_objects=lambda *a:[],detect_hands=lambda *a:[])
    monkeypatch.setattr(module,'PerceptionModels',lambda *a:model)
    timing=tmp_path/'pts.json';timing.write_text('{"frame_count":3,"timestamp_ms":[0,33,66]}')
    out=tmp_path/'p.jsonl'
    monkeypatch.setattr(sys,'argv',['perception','--input','video.mp4','--out',str(out),'--timestamp-map',str(timing),'--rotate','none'])
    module.main()
    assert [json.loads(s)['timestamp_ms'] for s in out.read_text().splitlines()]==[0,33,66]


def test_unrelated_high_background_does_not_take_an_interacted_item_reference():
    bridge=SpeechEvidenceBridge([SpeechEvent(100,500,800,'给大家看看这个')],enabled=True)
    engine=TemporalIntentStateMachine(fusion_config=FusionConfig())
    showing=[]
    for t in range(0,2501,100):
        objects=[dict(track_id='a',class_name='toaster',bbox_norm=[.3,.7,.4,.8],score=.9),
                 dict(track_id='wall',class_name='clock',bbox_norm=[.6,.1,.7,.2],score=.9)]
        p=dict(timestamp_ms=t,frame_id=t,objects=objects,hands=[dict(hand_id='h',bbox_norm=[.3,.65,.4,.75],score=.9)],
               relations=[dict(hand_id='h',object_id='a',type='near_or_moving_together',score=.9)])
        intent=engine.update(p,speech_evidence=bridge.evidence_for(p))
        if intent['state']=='SHOWING':showing.append(intent['target_id'])
    assert showing and set(showing)=={'a'}

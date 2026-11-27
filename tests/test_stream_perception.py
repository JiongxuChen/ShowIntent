import numpy as np
from src.streaming.stream_perception import EnhancedStreamPerception


class Models:
    visible=True
    def detect_hands(self,frame):
        return [('hand_left',[.4,.4,.6,.6],.9)]*2 if self.visible else []
    def detect_objects(self,frame,imgsz):
        if not self.visible or frame.shape[0]==200:return []
        return [('toothbrush',(20,20,80,80),.9),('person',(20,20,80,80),.9)]


def test_hand_crop_recovers_unlisted_class_and_unique_hands_without_fabrication():
    models=Models();adapter=EnhancedStreamPerception(models,640)
    frame=np.full((200,200,3),255,dtype=np.uint8)
    first=adapter.update(0,0,frame)
    assert len({h['hand_id'] for h in first['hands']})==2
    assert len(first['objects'])==1
    obj=first['objects'][0]
    assert obj['class_name']=='toothbrush' and obj['attributes']['colors']==['white']
    assert obj['appearance'] and all(0<=x<=1 for x in obj['bbox_norm'])
    assert any(r['object_id']==obj['track_id'] for r in first['relations'])
    second=adapter.update(100,1,frame)
    assert second['objects'][0]['track_id']==obj['track_id']
    models.visible=False
    assert adapter.update(200,2,frame)['objects']==[]


class BatchModels(Models):
    confidence=.25
    def detect_hands(self,frame):
        return [('left',[.2,.4,.4,.6],.9),('right',[.6,.4,.8,.6],.9)] if self.visible else []
    def yolo(self,crops,**kwargs):
        from types import SimpleNamespace
        assert kwargs['imgsz']==640 and kwargs['conf']==self.confidence
        results=[]
        for crop in crops:
            rows=Models.detect_objects(self,crop,kwargs['imgsz'])
            names={i:r[0] for i,r in enumerate(rows)}
            boxes=[SimpleNamespace(cls=np.array([i]),conf=np.array([r[2]]),xyxy=np.array([r[1]])) for i,r in enumerate(rows)]
            results.append(SimpleNamespace(names=names,boxes=boxes))
        return results


def test_batch_retains_all_regions_and_mapped_detections_with_fewer_calls():
    frame=np.full((200,200,3),255,dtype=np.uint8)
    serial=EnhancedStreamPerception(BatchModels(),640)
    batched=EnhancedStreamPerception(BatchModels(),640,batch_crops=True)
    assert batched.update(0,0,frame)==serial.update(0,0,frame)
    assert batched.crop_images==serial.crop_images==3
    assert serial.crop_calls==3 and batched.crop_calls==1
    batched.models.visible=False
    assert batched.update(100,1,frame)['objects']==[]
    assert batched.crop_calls==1


def test_all_batch_sizes_warmed_with_fixed_rect_policy_before_input():
    from src.streaming.stream_perception import warmup_crop_batches
    class WarmModels:
        confidence=.25
        def __init__(self):self.calls=[]
        def yolo(self,crops,**options):
            from types import SimpleNamespace
            self.calls.append((len(crops),options))
            return [SimpleNamespace(names={},boxes=[]) for _ in crops]
    models=WarmModels();warmup_crop_batches(models,960)
    assert [n for n,_ in models.calls]==[1,2,3]
    assert all(k['imgsz']==960 and k['rect'] is False for _,k in models.calls)

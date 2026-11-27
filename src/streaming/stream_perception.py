"""Streaming adapter reusing enhanced perception without changing member A's code."""
from src.perception.run import normalize_box
from src.perception.enhanced import hand_regions,merge_detections,visual_attributes,interaction_relations
from src.perception.appearance_tracker import AppearanceTracker


def batch_detect_crops(models,crops,imgsz):
    """Keep every ROI and inference size, but submit them in one model call."""
    if not crops:return []
    results=models.yolo(crops,verbose=False,conf=models.confidence,imgsz=imgsz,rect=False)
    if len(results)!=len(crops):raise ValueError('Batched detector returned wrong number of ROI results')
    rows=[]
    for result in results:
        rows.append([(result.names[int(box.cls[0])],tuple(float(x) for x in box.xyxy[0].tolist()),float(box.conf[0])) for box in result.boxes])
    return rows


def warmup_crop_batches(models,imgsz):
    """Exercise all supported ROI batch sizes before the media clock starts."""
    import numpy as np
    for count in (1,2,3):
        print(f'Vision worker: warming ROI batch {count}/3',flush=True)
        crops=[np.zeros((96+32*i,160,3),dtype=np.uint8) for i in range(count)]
        batch_detect_crops(models,crops,min(960,imgsz))


class EnhancedStreamPerception:
    def __init__(self,models,imgsz,*,batch_crops=False):
        self.models=models;self.imgsz=imgsz;self.tracker=AppearanceTracker();self.crop_calls=0;self.crop_images=0;self.batch_crops=batch_crops

    def update(self,ts,frame_id,frame):
        height,width=frame.shape[:2]
        hands=[dict(hand_id=f'{hid}_{i}',bbox_norm=box,score=round(score,4))
               for i,(hid,box,score) in enumerate(self.models.detect_hands(frame))]
        detections=list(self.models.detect_objects(frame,self.imgsz))
        regions=hand_regions(hands,width,height)
        self.crop_images+=len(regions)
        if self.batch_crops:
            crops=[frame[y1:y2,x1:x2] for x1,y1,x2,y2 in regions]
            predictions=batch_detect_crops(self.models,crops,min(960,self.imgsz))
            self.crop_calls+=bool(crops)
        else:
            predictions=[self.models.detect_objects(frame[y1:y2,x1:x2],min(960,self.imgsz)) for x1,y1,x2,y2 in regions]
            self.crop_calls+=len(regions)
        for (x1,y1,x2,y2),local in zip(regions,predictions):
            for cls,b,score in local:
                if cls=='person' or b[0]<3 or b[1]<3 or b[2]>x2-x1-3 or b[3]>y2-y1-3:continue
                detections.append((cls,(b[0]+x1,b[1]+y1,b[2]+x1,b[3]+y1),score))
        rows=[]
        for cls,box,score in merge_detections(detections):
            normalized=normalize_box(box,width,height)
            if normalized[0]>=normalized[2] or normalized[1]>=normalized[3]:continue
            appearance,attributes=visual_attributes(frame,box)
            rows.append(dict(class_name=cls,bbox_norm=normalized,score=round(score,4),appearance=appearance,attributes=attributes))
        objects=self.tracker.update(ts,rows)
        return dict(timestamp_ms=ts,frame_id=frame_id,objects=objects,hands=hands,relations=interaction_relations(hands,objects))

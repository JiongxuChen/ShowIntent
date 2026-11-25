"""B-owned detector adapter; original A PerceptionModels remains untouched."""
from pathlib import Path
from .run import PerceptionModels


class ClosedSetModels(PerceptionModels):
    def __init__(self, weights, confidence, use_hands, *, backend, device='0'):
        if backend not in {'yolo','rtdetr'}:raise ValueError('unsupported backend')
        if not Path(weights).is_file():raise FileNotFoundError(weights)
        from ultralytics import YOLO,RTDETR
        self.yolo=(YOLO if backend=='yolo' else RTDETR)(str(weights))
        self.yolo.overrides['device']=device
        self.confidence=confidence;self.hands=None
        if len(self.yolo.names)!=80:raise ValueError('benchmark requires an 80-class COCO detection checkpoint')
        if use_hands:
            import mediapipe as mp
            self.hands=mp.solutions.hands.Hands(static_image_mode=False,max_num_hands=2,min_detection_confidence=.5,min_tracking_confidence=.5)

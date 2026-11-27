"""Sparse-stream reacquisition using measured hand motion and current detections.

An object first detected after a lift must not require its own missing takeoff
frames. No class, video ID, labels, or synthetic visible objects are used.
"""
import math
from .persistent import PersistentIntentStateMachine
from .rest_aware import RestAwareIntentStateMachine
from .types import bbox_center


class SparseVisualIntentStateMachine(PersistentIntentStateMachine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.hand_history=[]

    def _observe_hands(self,ts,detections):
        super()._observe_hands(ts,detections)
        self.hand_history=[v for v in self.hand_history if ts-v[0]<=3000]
        self.hand_history.extend((ts,*bbox_center(h['bbox_norm'])) for h in detections)

    def _observe(self,ts,objects,related):
        super()._observe(ts,objects,related)
        # A support reference must actually be stable and lower in the frame.
        supports=[t.points[-1][2] for t in self._tracks.values()
                  if t.last_seen_ms==ts and len(t.points)>=4
                  and t.points[-1][2]>=self.config.display_region_max_y
                  and not t.moving]
        if not supports:return
        floor=max(supports)
        for tid,obj in objects.items():
            track=self._tracks.get(tid)
            if track is None or track.lift_seen or tid not in related:continue
            cx,cy=bbox_center(obj['bbox_norm'])
            # Reacquire only a stable, elevated object with sustained contact.
            if (cy>=min(.66,floor-.12) or track.moving
                    or track.related_since is None or ts-track.related_since<400):continue
            points=[p for p in track.points if ts-p[0]<=700]
            if len(points)<3 or points[-1][0]-points[0][0]<400:continue
            if max(p[1] for p in points)-min(p[1] for p in points)>.035:continue
            nearby=[(hx,hy) for hx,hy in self._hand_centers.values()
                    if math.hypot(cx-hx,cy-hy)<=.20]
            lifted=any(old_y-hy>=.08 and abs(old_x-hx)<.20
                       for hx,hy in nearby for t,old_x,old_y in self.hand_history
                       if 400<=ts-t<=3000)
            if lifted:
                track.anchor_y=max(track.anchor_y,floor)
                track.lift_seen=track.hand_context=True


class StreamIntentStateMachine:
    """Original v3 confirmations plus a conservative sparse visual branch."""
    def __init__(self,*args,fusion_config=None,**kwargs):
        self.base=RestAwareIntentStateMachine(*args,fusion_config=fusion_config,**kwargs)
        self.visual_engine=SparseVisualIntentStateMachine(*args,**kwargs)
        self.fusion=self.base.fusion

    def update(self,packet,*,speech_evidence=None):
        baseline=self.base.update(packet,speech_evidence=speech_evidence)
        visual=self.visual_engine.update(packet)
        result=baseline if baseline['state']=='SHOWING' else visual if visual['state']=='SHOWING' else baseline
        self.fusion.diagnostics.update(sparse_visual_intent=visual,
            stream_policy='SPARSE_VISUAL_REACQUISITION' if result is visual else 'REST_AWARE_BASE')
        return result

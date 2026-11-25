"""Short-term physical-instance association using location AND measured appearance.

It emits only current detections: retained state never fabricates a visible box.
The matching core is stdlib-only so identity safety can be tested without a GPU.
"""

from dataclasses import dataclass
import math

from .tracker import iou, center_distance


def appearance_distance(a, b):
    if not a or not b or len(a)!=len(b):
        return None
    return math.sqrt(max(0.,1.-sum(math.sqrt(max(0.,x*y)) for x,y in zip(a,b))))


@dataclass
class Instance:
    box: tuple
    class_name: str
    appearance: list
    last_seen_ms: int
    velocity: tuple = (0.,0.)


class AppearanceTracker:
    def __init__(self, ttl_ms=1500):
        self.ttl_ms=ttl_ms; self.instances={}; self.next_id=1; self.last_ts=None

    def update(self, timestamp_ms, detections):
        if self.last_ts is not None and timestamp_ms<=self.last_ts:
            raise ValueError('appearance tracker requires increasing media timestamps')
        self.last_ts=timestamp_ms
        self.instances={k:v for k,v in self.instances.items() if timestamp_ms-v.last_seen_ms<=self.ttl_ms}
        options=[]
        for i,d in enumerate(detections):
            for tid,old in self.instances.items():
                if old.class_name!=d['class_name']:continue
                dt=min((timestamp_ms-old.last_seen_ms)/1000,.3)
                predicted=tuple(x+old.velocity[j%2]*dt for j,x in enumerate(old.box))
                distance=center_distance(predicted,d['bbox_norm'])
                overlap=iou(old.box,d['bbox_norm'])
                appearance=appearance_distance(old.appearance,d.get('appearance',[]))
                if distance>.18 or (appearance is not None and appearance>.5):continue
                if overlap<.05 and distance>.08 and (appearance is None or appearance>.35):continue
                cost=distance*2+(1-overlap)*.25+(appearance or 0.)*.7
                options.append((cost,tid,i))
        matches={};used=set()
        for _,tid,i in sorted(options):
            if i not in matches and tid not in used:
                matches[i]=tid;used.add(tid)
        result=[]
        for i,d in enumerate(detections):
            tid=matches.get(i)
            if tid is None:
                tid=f'item_{self.next_id}';self.next_id+=1
            old=self.instances.get(tid);box=tuple(d['bbox_norm']);velocity=(0.,0.)
            if old:
                dt=(timestamp_ms-old.last_seen_ms)/1000
                velocity=tuple(((box[j]+box[j+2])-(old.box[j]+old.box[j+2]))/(2*dt) for j in (0,1))
                velocity=tuple(max(-1.,min(1.,x)) for x in velocity)
            self.instances[tid]=Instance(box,d['class_name'],d.get('appearance',[]),timestamp_ms,velocity)
            result.append(dict(d,track_id=tid))
        return result

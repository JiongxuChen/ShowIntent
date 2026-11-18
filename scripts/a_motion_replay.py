"""Replay cached full-frame detections with experimental motion association."""
import json
from pathlib import Path
from src.perception.motion_tracker import MotionObjectTracker


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'outputs/candidate_fullfps/perception_C03.jsonl'
    target=root/'outputs/motion_guarded'
    target.mkdir(parents=True,exist_ok=True)
    tracker=MotionObjectTracker()
    with (target/'perception_C03.jsonl').open('w',encoding='utf-8') as out:
        for line in source.read_text(encoding='utf-8').splitlines():
            packet=json.loads(line)
            detections=[(o['class_name'],tuple(o['bbox_norm']),o['score']) for o in packet['objects']]
            packet['objects']=[dict(track_id=t,class_name=n,bbox_norm=list(b),score=s) for t,n,b,s in tracker.update(detections)]
            packet['relations']=[] # Cached diagnostic has no hands or relations.
            out.write(json.dumps(packet)+'\n')
    print(target/'perception_C03.jsonl')
    (target/'tracking_audit.json').write_text(json.dumps(tracker.events,indent=2),encoding='utf-8')


if __name__=='__main__':
    main()

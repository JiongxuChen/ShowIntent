"""Unified offline candidate: cached detections -> motion -> duplicate filter.

Full-frame constant-30FPS development replay only; does not benchmark inference.
"""
import argparse
import json
from pathlib import Path
from src.perception.motion_tracker import MotionObjectTracker
from src.perception.relations import infer_relations
from src.perception.types import Detection, HandDetection
from scripts.a_duplicate_candidate import filter_stream
from dataclasses import asdict


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out-dir',type=Path,required=True)
    args=parser.parse_args()
    args.out_dir.mkdir(parents=True,exist_ok=True)
    tracker=MotionObjectTracker()
    rows=[]
    for line in args.input.read_text(encoding='utf-8').splitlines():
        packet=json.loads(line)
        detections=[(o['class_name'],tuple(o['bbox_norm']),o['score']) for o in packet['objects']]
        packet['objects']=[dict(track_id=t,class_name=n,bbox_norm=list(b),score=s) for t,n,b,s in tracker.update(detections)]
        # Recompute relations after changing IDs rather than retaining stale references.
        packet['relations']=[asdict(r) for r in infer_relations([HandDetection(**h) for h in packet['hands']],[Detection(**o) for o in packet['objects']])]
        rows.append(packet)
    audit=[]
    with (args.out_dir/'perception.jsonl').open('w',encoding='utf-8') as stream:
        for packet,audit in filter_stream(rows):
            stream.write(json.dumps(packet)+'\n')
    (args.out_dir/'audit.json').write_text(json.dumps(dict(tracking=tracker.events,suppression=audit),indent=2),encoding='utf-8')
    (args.out_dir/'provenance.json').write_text(json.dumps(dict(source=str(args.input.resolve()),frames=len(rows),hands_present=any(p['hands'] for p in rows),status='development_candidate_not_accepted'),indent=2),encoding='utf-8')
    print(f'{len(rows)} frames; {len(audit)} suppressed detection rows')


if __name__=='__main__':
    main()

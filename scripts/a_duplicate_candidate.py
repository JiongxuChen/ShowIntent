"""Offline duplicate-suppression experiment; never modifies source JSONL."""
import json
from pathlib import Path
from src.perception.tracker import iou, center_distance


def filter_stream(rows):
    hits = {}; audit = []
    for packet in rows:
        objects = packet['objects']
        for obj in objects:
            tid = obj['track_id']
            hits[tid] = hits.get(tid,0)+1
        hidden = set()
        for candidate in objects:
            tid = candidate['track_id']
            if hits[tid] >= 12:
                continue
            for anchor in objects:
                aid = anchor['track_id']
                if aid == tid or anchor['class_name'] != candidate['class_name'] or hits[aid] < 30:
                    continue
                a,b = tuple(anchor['bbox_norm']),tuple(candidate['bbox_norm'])
                if anchor['score'] >= .8 and candidate['score'] < anchor['score']*.7 and iou(a,b) >= .3 and center_distance(a,b) < .06:
                    hidden.add(tid)
                    audit.append(dict(timestamp_ms=packet['timestamp_ms'],hidden_id=tid,anchor_id=aid,score=candidate['score'],overlap=iou(a,b)))
                    break
        value=dict(packet)
        value['objects']=[o for o in objects if o['track_id'] not in hidden]
        value['relations']=[r for r in packet['relations'] if r['object_id'] not in hidden]
        yield value, audit


def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'outputs/motion_fullfps/perception_C03.jsonl'
    target=root/'outputs/duplicate_candidate'
    target.mkdir(parents=True,exist_ok=True)
    rows=[json.loads(x) for x in source.read_text(encoding='utf-8').splitlines()]
    audit=[]
    with (target/'perception_C03.jsonl').open('w',encoding='utf-8') as stream:
        for packet,audit in filter_stream(rows):
            stream.write(json.dumps(packet)+'\n')
    (target/'suppression_audit.json').write_text(json.dumps(audit,indent=2),encoding='utf-8')
    print(f'Hidden {len(audit)} detection rows; source unchanged')


if __name__=='__main__':
    main()

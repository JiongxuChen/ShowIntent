"""Explain cached tracker gates without rerunning detector or changing IDs."""
import json
from pathlib import Path
from src.perception.relink_tracker import RelinkObjectTracker
from src.perception.tracker import center_distance, iou


def main():
    root = Path(__file__).resolve().parents[1]
    source = root/'outputs/candidate_fullfps/perception_C03.jsonl'
    tracker = RelinkObjectTracker(max_missed=48)
    report = []
    for line in source.read_text(encoding='utf-8').splitlines():
        packet = json.loads(line)
        detections = [(o['class_name'],tuple(o['bbox_norm']),o['score']) for o in packet['objects']]
        if packet['timestamp_ms'] in (1966,2066,2132,12463):
            evidence = []
            for name, box, score in detections:
                if box[1] > .72:
                    continue
                for tid, old in tracker._tracks.items():
                    if old.class_name != name:
                        continue
                    area = (box[2]-box[0])*(box[3]-box[1])
                    old_area = (old.bbox[2]-old.bbox[0])*(old.bbox[3]-old.bbox[1])
                    evidence.append(dict(old_id=tid,missed=old.missed,distance=center_distance(old.bbox,box),iou=iou(old.bbox,box),area_ratio=area/max(old_area,1e-9),detected_box=box,old_box=old.bbox))
            report.append(dict(timestamp_ms=packet['timestamp_ms'],evidence=evidence))
        tracker.update(detections)
    target = root/'outputs/candidate_fullfps/relink_diagnosis.json'
    target.write_text(json.dumps(report,indent=2),encoding='utf-8')
    for event in report:
        print(event['timestamp_ms'])
        for e in event['evidence']:
            if e['distance'] < .2:
                print(e['old_id'], 'missing',e['missed'],'distance',round(e['distance'],3),'iou',round(e['iou'],3),'ratio',round(e['area_ratio'],3))


if __name__ == '__main__':
    main()

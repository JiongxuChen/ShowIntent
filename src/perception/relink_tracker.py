"""Conservative short-gap association, opt-in and independent of intent."""
from .tracker import SimpleObjectTracker, _Track, center_distance, iou


class RelinkObjectTracker(SimpleObjectTracker):
    """Retain hidden tracks, but reject ambiguous or large-scale reacquisition.

    Call at a fixed cadence. max_missed counts updates, not milliseconds.
    Missing boxes are never emitted as observed detections.
    """
    def __init__(self, max_missed=8, relink_distance=0.08, ambiguity_margin=0.025):
        super().__init__(max_missed=max_missed)
        self.relink_distance = relink_distance
        self.ambiguity_margin = ambiguity_margin

    def update(self, detections):
        candidates = []
        for index, (name, box, _) in enumerate(detections):
            for tid, track in self._tracks.items():
                if track.class_name != name:
                    continue
                distance = center_distance(track.bbox, box)
                area = max((box[2]-box[0])*(box[3]-box[1]), 1e-9)
                old_area = max((track.bbox[2]-track.bbox[0])*(track.bbox[3]-track.bbox[1]), 1e-9)
                ratio = area / old_area
                # A partially revealed object can shrink sharply after occlusion.
                # Only admit it when it still overlaps the retained box strongly.
                partial_reveal = track.missed > 0 and 0.2 <= ratio < 0.4 and iou(track.bbox, box) >= 0.2
                if not (0.4 <= ratio <= 2.5 or partial_reveal):
                    continue
                if distance <= self.relink_distance or (track.missed == 0 and iou(track.bbox, box) >= .15):
                    candidates.append((distance, tid, index))
        matches = {}
        used = set()
        for distance, tid, index in sorted(candidates):
            if tid in used or index in matches:
                continue
            if self._tracks[tid].missed:
                # A dormant track needs a unique match in both directions.
                alternatives = [d for d,t,i in candidates if (i == index and t != tid) or (t == tid and i != index)]
                if alternatives and min(alternatives) - distance < self.ambiguity_margin:
                    continue
            matches[index] = tid
            used.add(tid)
        for track in self._tracks.values():
            track.missed += 1
        output = []
        for index, (name, box, score) in enumerate(detections):
            tid = matches.get(index)
            if tid is None:
                tid = f'obj_{self._next_id}'
                self._next_id += 1
            reference = box
            previous = self._tracks.get(tid)
            if previous is not None:
                old_area = max((previous.bbox[2]-previous.bbox[0])*(previous.bbox[3]-previous.bbox[1]),1e-9)
                ratio = ((box[2]-box[0])*(box[3]-box[1])) / old_area
                if ratio < .4:
                    reference = previous.bbox
            self._tracks[tid] = _Track(tid, name, reference)
            output.append((tid, name, box, score))
        self._tracks = {t:v for t,v in self._tracks.items() if v.missed <= self.max_missed}
        return output

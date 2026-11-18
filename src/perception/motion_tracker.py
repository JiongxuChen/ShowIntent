"""Experimental causal tracker with bounded motion and track maturity."""
from dataclasses import dataclass
from math import log
from .tracker import center_distance, iou


@dataclass
class Track:
    name: str
    box: tuple
    hits: int = 1
    missed: int = 0
    velocity: tuple = (0., 0.)


class MotionObjectTracker:
    def __init__(self, max_missed=48):
        self.max_missed = max_missed
        self.tracks = {}
        self.next_id = 1
        self.events = []
        self.frame = -1

    def update(self, detections):
        self.frame += 1
        clean = []
        for name,box,score in detections:
            area=max((box[2]-box[0])*(box[3]-box[1]),1e-9)
            contained=[]
            for tid,track in self.tracks.items():
                if track.name!=name or track.hits<8 or track.missed>2:
                    continue
                old=track.box
                cx,cy=(old[0]+old[2])/2,(old[1]+old[3])/2
                old_area=max((old[2]-old[0])*(old[3]-old[1]),1e-9)
                if box[0]<=cx<=box[2] and box[1]<=cy<=box[3] and area/old_area>1.5:
                    contained.append(tid)
            if len(contained)>=2:
                self.events.append(dict(frame_id=self.frame,event='MERGED_BOX_REJECTED',tracks=contained,bbox=list(box)))
                continue
            clean.append((name,box,score))
        detections=clean
        edges = []
        for tid, track in self.tracks.items():
            # Extrapolate only a short horizon: long hidden motion is unknown.
            horizon = min(track.missed + 1, 5)
            dx,dy = (v*horizon for v in track.velocity)
            predicted = tuple(v + (dx if k%2==0 else dy) for k,v in enumerate(track.box))
            for index,(name,box,score) in enumerate(detections):
                if name != track.name:
                    continue
                area = max((box[2]-box[0])*(box[3]-box[1]),1e-9)
                old_area = max((track.box[2]-track.box[0])*(track.box[3]-track.box[1]),1e-9)
                ratio = area/old_area
                distance = min(center_distance(predicted,box),center_distance(track.box,box))
                overlap = max(iou(predicted,box),iou(track.box,box))
                partial = track.hits>=8 and .10<=ratio<.22 and overlap>=.10
                if not (.22 <= ratio <= 3 or partial) or distance > .10:
                    continue
                # Mature histories beat transient duplicate boxes, but geometry dominates.
                cost = distance + .015*abs(log(ratio)) - .025*overlap - .025*min(track.hits/8,1)
                edges.append((cost,tid,index))
        matches = {}; used = set()
        for cost,tid,index in sorted(edges):
            if tid in used or index in matches:
                continue
            track = self.tracks[tid]
            if track.missed:
                rivals = [c for c,t,i in edges if i==index and t!=tid and self.tracks[t].hits >= min(track.hits,8)]
                if rivals and min(rivals)-cost < .008:
                    continue
            matches[index]=tid; used.add(tid)
        for track in self.tracks.values():
            track.missed += 1
        output=[]
        for index,(name,box,score) in enumerate(detections):
            tid=matches.get(index)
            if tid is None:
                tid=f'obj_{self.next_id}'; self.next_id+=1
                self.tracks[tid]=Track(name,box)
            else:
                track=self.tracks[tid]
                if track.missed>1:
                    self.events.append(dict(frame_id=self.frame,event='TRACK_RECOVERED',track_id=tid,missing_frames=track.missed-1))
                steps=track.missed
                vx=((box[0]+box[2])-(track.box[0]+track.box[2]))/(2*steps)
                vy=((box[1]+box[3])-(track.box[1]+track.box[3]))/(2*steps)
                track.velocity=tuple(max(-.025,min(.025,v)) for v in (vx,vy))
                area=(box[2]-box[0])*(box[3]-box[1])
                old_area=max((track.box[2]-track.box[0])*(track.box[3]-track.box[1]),1e-9)
                # Avoid poisoning a mature reference with a transient merged/partial box.
                if track.hits < 3 or .5 <= area/old_area <= 1.8:
                    track.box=box
                track.hits+=1; track.missed=0
            output.append((tid,name,box,score))
        self.tracks={k:v for k,v in self.tracks.items() if v.missed<=self.max_missed}
        return output

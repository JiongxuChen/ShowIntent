"""Bind each linguistic observation to an utterance-time object instance.

Historical frames are already observed when a delayed final transcript arrives.
Localization uses interaction and geometry, not a completed visual intent verdict.
"""

from dataclasses import dataclass
import math
import re

from .types import bbox_center
from src.perception.appearance_tracker import appearance_distance


@dataclass
class Binding:
    track_id: str
    class_name: str
    x: float
    y: float
    source_y: float
    last_seen_ms: int
    returned: bool = False
    appearance: list = None


class UtteranceGrounder:
    def __init__(self, *, history_ms=10000, action_grace_ms=1000, margin=.02,
                 relink_ms=1500, relink_distance=.15, return_margin=.06):
        self.history_ms = history_ms
        self.action_grace_ms = action_grace_ms
        self.margin = margin
        self.relink_ms, self.relink_distance = relink_ms, relink_distance
        self.return_margin = return_margin
        self.history = []
        self.bindings = {}

    def observe(self, ts, objects, tracks, related, confirmed):
        floor = max((bbox_center(o['bbox_norm'])[1] for o in objects.values()), default=0.)
        entries = {}
        for tid, obj in objects.items():
            x, y = bbox_center(obj['bbox_norm'])
            points = [p for p in tracks[tid].points if ts-p[0] <= 1000]
            # Movement/elevation identify the referent, without imposing lift,
            # dwell, region or speed requirements on the linguistic intent.
            movement = max((math.hypot(x-p[1], y-p[2]) for p in points), default=0.)
            salience = max(0., floor-y) + movement + .05*related.get(tid, 0.)
            entries[tid] = dict(class_name=obj.get('class_name'), x=x, y=y,
                                salience=salience, related=tid in related, moved=movement > .02,
                                confirmed=tid == confirmed,
                                parent=tracks[tid].inherited_from,
                                colors=obj.get('attributes',{}).get('colors',[]),
                                appearance=obj.get('appearance',[]))
        self.history.append((ts, entries))
        self.history = [(t, rows) for t, rows in self.history if t >= ts-self.history_ms]
        self.bindings = {key:b for key,b in self.bindings.items() if key[1] >= ts-self.history_ms}

    def _select(self, frames, classes, source_end, *, attributes=None, control=False):
        latest, scores = {}, {}
        for ts, entries in frames:
            for tid, row in entries.items():
                if classes and row['class_name'] != classes[0]:
                    continue
                latest[tid] = (ts, row)
                scores[tid] = max(scores.get(tid, 0.), row['salience'])
        if not latest:
            return None
        if control:
            # A delayed end/switch refers to the episode active while spoken,
            # not the episode active when transcription eventually arrived.
            anchored = [(t,tid) for t,rows in frames if t<=source_end
                         for tid,row in rows.items() if tid in latest and row['confirmed']]
            if anchored:
                last=max(t for t,_ in anchored)
                chosen={tid for t,tid in anchored if t==last}
                latest={tid:r for tid,r in latest.items() if tid in chosen}
        # Known track fragments are one localization hypothesis, not competitors.
        parents = {r['parent'] for _,r in latest.values() if r['parent']}
        candidates = [tid for tid in latest if tid not in parents]
        if (attributes or {}).get('discourse',{}).get('new_item'):
            candidates=[tid for tid in candidates if not latest[tid][1]['confirmed']]
        if classes and (attributes or {}).get('discourse',{}).get('version')==2:
            competing_interaction=any(row['class_name']!=classes[0] and (row['related'] or row['moved'])
                                      for _,rows in frames for row in rows.values())
            if competing_interaction:
                candidates=[tid for tid in candidates if any(rows.get(tid,{}).get('related')
                    or rows.get(tid,{}).get('moved') or rows.get(tid,{}).get('confirmed') for _,rows in frames)]
        colors=set((attributes or {}).get('colors',[]))
        if colors:
            matched=[tid for tid in candidates if colors & set(latest[tid][1].get('colors',[]))]
            # When color was actually measured, contradicting it is not a match.
            measured=any(latest[tid][1].get('colors') for tid in candidates)
            if measured:
                candidates=matched
        if len(candidates) > 1:
            # Background geometry alone cannot resolve an ambiguous pronoun.
            # A potentially presented class may be anything the detector returns.
            engaged = {tid for _,rows in frames for tid,row in rows.items()
                       if row['related'] or row.get('moved') or row['confirmed']}
            candidates = [tid for tid in candidates if tid in engaged]
        phrase=(attributes or {}).get('object_phrase','').strip()
        if phrase and not classes and not colors and not re.fullmatch(
                r'(?:这个|這個|这个物品|這個物品|它|这件|這件|this(?:\s+one)?|it)',phrase,re.I):
            # An unmapped explicit noun is not permission to select the lone
            # unrelated background detection. Require actual interaction.
            candidates=[tid for tid in candidates if any(
                rows.get(tid,{}).get('related') or rows.get(tid,{}).get('moved') or rows.get(tid,{}).get('confirmed')
                for _,rows in frames)]
        if not candidates:
            return None
        if len(candidates) == 1:
            tid = candidates[0]
        else:
            ranked = sorted(candidates, key=lambda k:scores[k], reverse=True)
            if len(ranked)>1 and scores[ranked[0]]-scores[ranked[1]] > self.margin:
                tid = ranked[0]
            else:
                # Only a confirmed episode may resolve an otherwise ambiguous
                # pronoun; an arbitrary CANDIDATE must not become its antecedent.
                confirmed = [k for k in candidates if latest[k][1]['confirmed']]
                associated = [k for k in candidates if latest[k][1]['related']]
                if len(confirmed) == 1:
                    tid = confirmed[0]
                elif len(associated) == 1:
                    tid = associated[0]
                else:
                    return None
        ts, row = latest[tid]
        source_rows = [r[tid] for t,r in frames if t <= source_end and tid in r]
        if not source_rows:
            source_rows = [r[tid] for _,r in frames if tid in r][:1]
        source_y = source_rows[-1]['y']
        return Binding(tid,row['class_name'],row['x'],row['y'],source_y,ts,
                       appearance=row.get('appearance',[]))

    def _returned_between(self,binding,rows):
        return any(row['y']-binding.source_y>self.return_margin for row in rows)

    def _returned_now(self,binding,y):
        return y-binding.source_y>self.return_margin

    def resolve(self, ts, speech, objects, related):
        start, end = speech.get('source_start_ms'), speech.get('source_end_ms')
        if start is None or end is None:
            return None, 'NO_USABLE_SPEECH', False
        classes = speech.get('suggested_classes', [])
        if len(classes)>1:
            return None, 'CONFLICTING_CLASSES', False
        key = (start,end,speech.get('source_available_at_ms'))
        binding = self.bindings.get(key)
        if binding is None:
            frames = [(t,r) for t,r in self.history if max(0,start-250) <= t <= end+self.action_grace_ms]
            control=bool(set(speech.get('cue_types',[])) & {'NEGATED_LANGUAGE','ENDING_LANGUAGE','TOPIC_CHANGE','ACTIVITY_NONDISPLAY'})
            binding = self._select(frames,classes,end,attributes=speech.get('attributes'),control=control)
            if binding is None:
                return None, 'UTTERANCE_TARGET_UNRESOLVED', False
            self.bindings[key] = binding
        status = 'BOUND_TO_UTTERANCE'
        if speech.get('referent_scope') in ('previous','other'):
            # A historical/other-party statement is never a veto on a new shot.
            return None, 'NON_CURRENT_REFERENT', binding.returned
        later=[rows[binding.track_id] for t,rows in self.history if end<t<=ts and binding.track_id in rows]
        if self._returned_between(binding,later):
            binding.returned=True
        if binding.track_id not in objects:
            # Never rebind a deictic statement to a different visible class.
            options = []
            for tid, obj in objects.items():
                if binding.returned:break
                if obj.get('class_name') != binding.class_name or tid not in related:
                    continue
                visual_distance=appearance_distance(binding.appearance,obj.get('appearance',[]))
                if visual_distance is not None and visual_distance>.5:continue
                x,y = bbox_center(obj['bbox_norm'])
                distance = math.hypot(x-binding.x,y-binding.y)
                if ts-binding.last_seen_ms <= self.relink_ms and distance <= self.relink_distance:
                    options.append((distance,tid))
            options.sort()
            if not options or (len(options)>1 and options[1][0]-options[0][0] <= .04):
                return None, 'BOUND_TARGET_MISSING', binding.returned
            binding.track_id = options[0][1]
            status = 'BOUND_TARGET_RELINKED'
        x,y = bbox_center(objects[binding.track_id]['bbox_norm'])
        # A later clear lowering invalidates this statement, rather than
        # repeatedly letting a recent attribute description hold the shot open.
        if ts > end and self._returned_now(binding,y):
            binding.returned = True
        binding.x,binding.y,binding.last_seen_ms = x,y,ts
        binding.appearance=objects[binding.track_id].get('appearance',binding.appearance)
        return binding.track_id, status, binding.returned

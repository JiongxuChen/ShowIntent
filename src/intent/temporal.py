"""Shared visual/language intent transitions with optional peer evidence fusion.

Combines lift/hand context, presentation-region geometry, bounded track links,
robust motion and object-scale evidence. The public entry point is src.intent.run; no pretrained model is needed here.
"""

from __future__ import annotations

import math
from statistics import median
from dataclasses import asdict, dataclass, field, replace

from .types import IntentPacket, bbox_center
from .fusion import EvidenceFusion, FusionConfig
from src.perception.appearance_tracker import appearance_distance


@dataclass(frozen=True)
class IntentConfig:
    candidate_min_ms: float = 300
    showing_min_ms: float = 600
    missing_tolerance_ms: float = 600
    putdown_min_ms: float = 100
    ending_min_ms: float = 150
    ending_max_ms: float = 2000
    candidate_decay_ms: float = 300
    candidate_switch_ms: float = 300
    relink_max_ms: float = 1500
    observation_gap_ms: float = 750
    presentation_exit_ms: float = 600
    history_window_ms: float = 4000
    track_ttl_ms: float = 10000
    relation_min_score: float = .30
    raise_margin: float = .06
    putdown_margin: float = .025
    stable_speed_tolerance: float = .08
    stable_span_tolerance: float = .035
    relink_max_dist: float = .15
    motion_window_ms: float = 300
    hand_lift_margin: float = .05
    prelink_max_ms: float = 500
    prelink_max_dist: float = .18
    display_region_max_y: float = .72
    evidence_gap_ms: float = 250
    object_context_motion: float = .025
    cold_start_region_max_y: float = .69
    cold_start_window_ms: float = 1200
    presentation_scale_ratio: float = 1.3
    display_exit_y: float = .73

    def __post_init__(self) -> None:
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (float, int)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f"{name} must be a finite positive number")
        if self.ending_max_ms < self.ending_min_ms:
            raise ValueError("ending_max_ms must be >= ending_min_ms")
        if self.putdown_margin >= self.raise_margin:
            raise ValueError("putdown_margin must be < raise_margin (hysteresis)")
        if self.history_window_ms < self.showing_min_ms:
            raise ValueError("history_window_ms must cover showing_min_ms")
        if self.track_ttl_ms < max(self.observation_gap_ms, self.relink_max_ms):
            raise ValueError("track_ttl_ms must cover observation/relink windows")
        if self.relation_min_score > 1:
            raise ValueError("relation_min_score must be <= 1")
        if not 0 < self.cold_start_region_max_y <= self.display_region_max_y < self.display_exit_y <= 1:
            raise ValueError('invalid display region hysteresis')
        if self.presentation_scale_ratio <= 1:
            raise ValueError('presentation_scale_ratio must be > 1')


@dataclass
class _Track:
    class_name: str
    anchor_y: float
    last_seen_ms: int
    points: list[tuple[int, float, float]] = field(default_factory=list)
    related_since: int | None = None
    qualified_since: int | None = None
    lift_seen: bool = False
    moving: bool = False
    hand_context: bool = False
    inherited_from: str | None = None
    good_ms: float = 0
    bad_since: int | None = None
    evidence_ts: int | None = None
    was_good: bool = False
    first_seen_ms: int = 0
    approach: bool = False
    cold_start_allowed: bool = False
    appearance: list = field(default_factory=list)


class TemporalIntentStateMachine:
    def __init__(self, config: IntentConfig | None = None, *, fusion_config: FusionConfig | None = None,
                 language_only: bool = False, **overrides) -> None:
        self.config = replace(config or IntentConfig(), **overrides)
        self.fusion = EvidenceFusion(fusion_config) if fusion_config is not None else None
        if language_only and self.fusion is None:
            raise ValueError('language ablation requires speech fusion')
        self.language_only=language_only
        if self.fusion is not None:self.fusion.language_only=language_only
        self._track_ids = {}
        self._last_related = {}
        self.state = "NONE"
        self.target_id: str | None = None
        self._tracks: dict[str, _Track] = {}
        self._last_ts: int | None = None
        self._state_since = 0
        self._candidate_lost_since: int | None = None
        self._putdown_since: int | None = None
        self._weak_since: int | None = None
        self._last_score = 0.
        self._hand_anchors: dict[str, float] = {}
        self._hand_centers: dict[str, tuple[float, float]] = {}
        self._hand_tracks: dict[str, tuple[int, float, float, float]] = {}
        self._next_hand = 0
        self._size_references: dict[str, dict[str, tuple[int, float]]] = {}
        self._session_start_ts: int | None = None
        self.episode_id = None
        self.episode_started_ms = None
        self._episode_counter = 0

    def _observe_hands(self, ts: int, detections: list[dict]) -> None:
        # MediaPipe handedness is not an instance ID (both hands may be "left").
        # Use short geometric tracks for movement context, without altering A.
        self._hand_tracks = {k:v for k,v in self._hand_tracks.items() if ts-v[0] <= self.config.observation_gap_ms}
        centers = sorted(bbox_center(h['bbox_norm']) for h in detections)
        options = sorted((math.hypot(cx-v[1],cy-v[2]),i,k)
                         for i,(cx,cy) in enumerate(centers) for k,v in self._hand_tracks.items()
                         if math.hypot(cx-v[1],cy-v[2]) <= .20)
        used=set(); matches={}
        for _,i,k in options:
            if i not in matches and k not in used:
                matches[i]=k;used.add(k)
        self._hand_centers = {}; self._hand_anchors = {}
        for i,(cx,cy) in enumerate(centers):
            k=matches.get(i)
            if k is None:
                self._next_hand += 1;k=f'context_hand_{self._next_hand}'
                anchor=cy
            else:
                anchor=max(cy,self._hand_tracks[k][3])
            self._hand_tracks[k]=(ts,cx,cy,anchor)
            self._hand_centers[k]=(cx,cy);self._hand_anchors[k]=anchor

    def _packet(self, ts: int, score: float, reasons: list[str]) -> dict:
        if self.fusion is not None:
            self.fusion.diagnostics.update(episode_id=self.episode_id,episode_started_ms=self.episode_started_ms)
            track = self._tracks.get(self.target_id)
            if self.target_id not in self._last_related:
                reasons = [r for r in reasons if r != 'HAND_OBJECT_ASSOCIATED']
            if track is None or not self._raised(track):
                reasons = [r for r in reasons if r != 'RAISED_IN_FRAME']
            view = self.fusion.view(self.target_id)
            if (view.get('speech', 0) or view.get('blocked') or view.get('ambiguous') or view.get('semantic_conflict')
                    or 'SPEECH_REFERENT_RETURNED' in view.get('reasons', [])):
                reasons = list(dict.fromkeys(reasons + view.get('reasons', [])))
                score = 0. if view.get('blocked') or view.get('ambiguous') or view.get('semantic_conflict') else max(score, min(1., view.get('total', 0)))
        return IntentPacket(ts, self.state, self.target_id, score, reasons).to_dict()

    def _transition(self, state: str, ts: int, target: str | None = None) -> None:
        if state=='NONE':
            self.episode_id=self.episode_started_ms=None
        elif state=='SHOWING' and self.episode_id is None:
            self._episode_counter+=1
            self.episode_id=f'episode_{self._episode_counter}'
            track=self._tracks.get(target)
            starts=[self._state_since]
            if track is not None and track.related_since is not None:starts.append(track.related_since)
            self.episode_started_ms=min(starts)
        self.state, self.target_id, self._state_since = state, target, ts
        self._candidate_lost_since = self._putdown_since = self._weak_since = None

    def _raised(self, track: _Track) -> bool:
        margin = self.config.putdown_margin if track.lift_seen else self.config.raise_margin
        return bool(track.points) and track.points[-1][2] < track.anchor_y - margin

    def _ready(self, track: _Track, ts: int) -> bool:
        visual_ms=(max(self.config.showing_min_ms,self.fusion.config.visual_confirmation_ms)
                   if self.fusion is not None else self.config.showing_min_ms)
        visual_ready = (not self.language_only and track.qualified_since is not None
                        and track.good_ms >= visual_ms)
        if self.fusion is None:
            return visual_ready
        tid = self._track_ids.get(id(track))
        if self.fusion.blocked(tid):
            return False
        view = self.fusion.view(tid)
        if view.get('ambiguous'):
            return False
        return visual_ready or (view.get('speech', 0) > 0 and view.get('ready', False))

    def _language_support(self, tid):
        if self.fusion is None:
            return False
        view = self.fusion.view(tid)
        return view.get('speech', 0) > 0 and view.get('supported', False) and not view.get('ambiguous')

    def _observe(self, ts: int, objects: dict, related: dict) -> None:
        cfg = self.config
        # Scene support supplies a height reference, never an independent trigger.
        scene_floor = max([bbox_center(o['bbox_norm'])[1] for o in objects.values()]
                          + [t.anchor_y for t in self._tracks.values()] + [0.])
        claimed = set()
        for cls,refs in self._size_references.items():
            self._size_references[cls] = {k:v for k,v in refs.items() if ts-v[0]<=cfg.track_ttl_ms}
        for tid,obj in objects.items():
            b=obj['bbox_norm'];area=(b[2]-b[0])*(b[3]-b[1])
            if bbox_center(b)[1]>=cfg.display_region_max_y:
                self._size_references.setdefault(obj.get('class_name',''),{})[tid]=(ts,area)
        fresh = {}
        for tid, obj in objects.items():
            if tid in self._tracks:
                continue
            cx, cy = bbox_center(obj['bbox_norm'])
            options = []
            for old_id, old in self._tracks.items():
                if old_id in objects or old_id in claimed or not old.points or old.class_name != obj.get('class_name'):
                    continue
                if old_id == self.target_id and self.state in ('SHOWING', 'ENDING'):
                    continue
                visual_distance=appearance_distance(old.appearance,obj.get('appearance',[]))
                if visual_distance is not None and visual_distance>.5:continue
                distance = math.hypot(cx-old.points[-1][1], cy-old.points[-1][2])
                if ts-old.last_seen_ms <= cfg.prelink_max_ms and distance <= cfg.prelink_max_dist:
                    options.append((distance, old_id, old))
            options.sort(key=lambda x: x[0])
            if options and (len(options)==1 or options[1][0]-options[0][0]>.04):
                _, old_id, old = options[0]
                fresh[tid] = _Track(obj.get('class_name',''), old.anchor_y, old.last_seen_ms,
                                    list(old.points), old.related_since, None, old.lift_seen,
                                    inherited_from=old_id, first_seen_ms=ts)
                fresh[tid].qualified_since = old.qualified_since
                fresh[tid].good_ms = old.good_ms
                fresh[tid].cold_start_allowed = old.cold_start_allowed or old.lift_seen
                claimed.add(old_id)
        for tid, track in list(self._tracks.items()):
            if tid not in objects:
                track.related_since = None
                if ts-track.last_seen_ms > cfg.evidence_gap_ms:
                    track.qualified_since = None; track.good_ms = 0
                track.was_good = False
                if tid != self.target_id and ts - track.last_seen_ms > cfg.track_ttl_ms:
                    del self._tracks[tid]
        for tid, obj in objects.items():
            cx, cy = bbox_center(obj["bbox_norm"])
            track = self._tracks.get(tid) or fresh.get(tid)
            if track is None or ts - track.last_seen_ms > cfg.observation_gap_ms:
                # Reappearing detections cannot carry old confirmation streaks.
                # ENDING can explicitly inherit identity via the relink branch.
                old = track
                track = _Track(obj.get("class_name", ""), cy, ts, first_seen_ms=ts,
                               cold_start_allowed=ts > self._session_start_ts)
                if old is not None and tid == self.target_id and ts - old.last_seen_ms <= cfg.relink_max_ms:
                    track.anchor_y, track.lift_seen = old.anchor_y, old.lift_seen
                self._tracks[tid] = track
            else:
                self._tracks[tid] = track
            previous = track.points[-1] if track.points else None
            track.anchor_y = max(track.anchor_y, cy)
            track.class_name = obj.get("class_name", "")
            track.appearance=obj.get('appearance',[])
            track.last_seen_ms = ts
            track.points.append((ts, cx, cy))
            track.points[:] = [p for p in track.points if ts - p[0] <= cfg.history_window_ms]
            associated = tid in related
            b=obj['bbox_norm'];area=(b[2]-b[0])*(b[3]-b[1])
            sizes=[v[1] for v in self._size_references.get(track.class_name,{}).values()]
            track.approach=bool(sizes) and area >= median(sizes)*cfg.presentation_scale_ratio
            # Hand movement is supporting context, never a standalone trigger.
            hand_lift = any(
                math.hypot(cx-hx, cy-hy) <= .20
                and hy < self._hand_anchors[hid]-cfg.hand_lift_margin
                for hid, (hx,hy) in self._hand_centers.items())
            actual_motion = max((p[2] for p in track.points), default=cy)-cy >= cfg.object_context_motion
            cold_context = (track.cold_start_allowed and ts-track.first_seen_ms <= cfg.cold_start_window_ms
                            and (cy < cfg.cold_start_region_max_y or track.approach))
            if associated and hand_lift and (actual_motion or cold_context) and cy < scene_floor-cfg.raise_margin:
                track.anchor_y = max(track.anchor_y, scene_floor)
                track.lift_seen = track.hand_context = True
            track.related_since = (track.related_since if track.related_since is not None else ts) if associated else None
            if associated and previous is not None and self._raised(track):
                track.lift_seen = True
            if cy >= track.anchor_y - cfg.putdown_margin:
                track.lift_seen = False
            recent = [p for p in track.points if ts-p[0] <= cfg.motion_window_ms]
            if len(recent)>=4:
                split=len(recent)//2; left,right=recent[:split],recent[split:]
                dt=median(p[0] for p in right)-median(p[0] for p in left)
                speed=abs(median(p[1] for p in right)-median(p[1] for p in left))*1000/dt
                vertical_path=sum(abs(b[2]-a[2]) for a,b in zip(recent,recent[1:]))
                vertical_net=abs(recent[-1][2]-recent[0][2])
                if vertical_path and vertical_net/vertical_path>.85 and vertical_net>cfg.stable_span_tolerance:
                    speed=max(speed,vertical_net*1000/(recent[-1][0]-recent[0][0]))
            else:
                speed = math.inf if previous is None else math.hypot(cx - previous[1], cy - previous[2]) * 1000 / (ts - previous[0])
            track.moving = speed > cfg.stable_speed_tolerance
            spatial = track.lift_seen and self._raised(track) and cy < cfg.display_region_max_y
            eligible = associated and spatial and (not track.moving or track.approach)
            if not eligible:
                track.bad_since = track.bad_since if track.bad_since is not None else ts
                if not spatial or ts-track.bad_since >= cfg.evidence_gap_ms:
                    track.qualified_since = None; track.good_ms = 0
                track.was_good = False; track.evidence_ts = ts
                continue
            if track.qualified_since is None:
                track.qualified_since = ts
                track.good_ms = 0
            elif track.was_good and track.evidence_ts is not None:
                track.good_ms += ts-track.evidence_ts
            track.bad_since = None; track.evidence_ts = ts; track.was_good = True
            points = [p for p in track.points if p[0] >= max(track.qualified_since, ts - cfg.showing_min_ms)]
            # Horizontal sustained transport is a negative cue; vertical box
            # oscillation during turning is not by itself a presentation end.
            span = max(p[1] for p in points) - min(p[1] for p in points)
            if span > cfg.stable_span_tolerance and not track.approach:
                track.qualified_since = ts
                track.good_ms = 0

    def _best(self, ts: int, objects: dict, related: dict) -> str | None:
        candidates = [tid for tid in objects if tid in related and self._tracks[tid].related_since is not None
                      and ts - self._tracks[tid].related_since >= self.config.candidate_min_ms]
        if self.fusion is not None:
            candidates = [tid for tid in candidates if not self.fusion.blocked(tid)]
            candidates += [tid for tid in objects if tid not in candidates
                           and self.fusion.view(tid).get('speech', 0) > 0
                           and self.fusion.view(tid).get('candidate', False)]
        if not candidates:
            return None
        has_language=bool(self.fusion and any(self.fusion.view(tid).get('speech',0)>0 for tid in candidates))
        def rank(tid):
            track = self._tracks[tid]
            progress = 0 if track.qualified_since is None else ts - track.qualified_since
            support = self.fusion.view(tid).get('total', 0) if has_language else 0
            return (self._ready(track, ts), support, progress, track.lift_seen and self._raised(track), related.get(tid, 0.))
        return max(candidates, key=rank)

    def _candidate_reasons(self, track: _Track, ts: int) -> list[str]:
        reasons = ["HAND_OBJECT_ASSOCIATED"]
        if track.lift_seen:
            if track.hand_context:
                reasons.append("HAND_LIFT_CONTEXT")
            elif not track.inherited_from:
                reasons.append("LIFT_MOTION_OBSERVED")
            if track.approach:
                reasons.append("APPROACH_SCALE")
            if track.inherited_from:
                reasons.append("PRECONFIRM_TRACK_LINKED")
        else:
            reasons.append("NO_OBSERVED_LIFT")
        if self._raised(track):
            reasons.append("RAISED_IN_FRAME")
        if track.moving:
            reasons.append("TRANSPORT_MOTION")
        if self._ready(track, ts):
            reasons.append("TEMPORAL_HOLD")
        return reasons

    def _relink(self, ts: int, objects: dict, related: dict) -> str | None:
        if self.fusion is not None and self.fusion.blocked(self.target_id):
            return None
        original = self._tracks.get(self.target_id)
        if original is None or not original.points or ts - original.last_seen_ms > self.config.relink_max_ms:
            return None
        best, best_distance = None, self.config.relink_max_dist
        options = []
        _, ox, oy = original.points[-1]
        for tid, obj in objects.items():
            if tid == self.target_id or tid not in related or not original.class_name or obj.get("class_name") != original.class_name:
                continue
            visual_distance=appearance_distance(original.appearance,obj.get('appearance',[]))
            if visual_distance is not None and visual_distance>.5:continue
            cx, cy = bbox_center(obj["bbox_norm"])
            distance = math.hypot(cx - ox, cy - oy)
            if distance <= best_distance and cy < min(self.config.display_region_max_y, original.anchor_y - self.config.raise_margin):
                options.append((distance, tid))
        options.sort()
        if options and (len(options)==1 or options[1][0]-options[0][0]>.04):
            best=options[0][1]
        if best is not None:
            track = self._tracks[best]
            track.anchor_y, track.lift_seen = original.anchor_y, True
        return best

    def update(self, packet: dict, *, speech_evidence: dict | None = None) -> dict:
        ts = packet.get("timestamp_ms")
        if isinstance(ts, bool) or not isinstance(ts, int) or ts < 0:
            raise ValueError("timestamp_ms must be a non-negative integer")
        if self._last_ts is not None and ts <= self._last_ts:
            raise ValueError("timestamp_ms must strictly increase; create a new engine for a new clip")
        gap = self._last_ts is not None and ts - self._last_ts > self.config.observation_gap_ms
        if gap:
            self._transition("NONE", ts)
            if self.fusion is not None:
                self.fusion.reset()
                self.fusion.language_only=self.language_only
            self._tracks.clear()
            self._hand_anchors.clear()
            self._hand_tracks.clear()
            self._size_references.clear()
        self._last_ts = ts
        if self._session_start_ts is None or gap:
            self._session_start_ts = ts
        self._observe_hands(ts, packet.get('hands', []))
        # Person detections remain in A's facts, but a presenter is not an item target.
        objects = {o["track_id"]: o for o in packet.get("objects", [])
                   if o.get('class_name', '').casefold() != 'person'}
        related = {}
        for rel in packet.get("relations", []):
            score = rel.get("score", 0.)
            tid = rel.get("object_id")
            if (tid in objects and rel.get("type") == "near_or_moving_together"
                    and isinstance(score, (int, float)) and math.isfinite(score)
                    and self.config.relation_min_score <= score <= 1):
                related[tid] = max(related.get(tid, 0.), score)
        self._last_related = related
        self._observe(ts, objects, related)
        self._track_ids = {id(track): tid for tid, track in self._tracks.items()}
        if self.fusion is not None:
            self.fusion.update(ts, objects, self._tracks, related, self.target_id,
                               speech_evidence, self._raised,
                               confirmed=self.target_id if self.state in ('SHOWING', 'ENDING') else None,
                               episode_started_ms=self.episode_started_ms)
            if self.fusion.blocked(self.target_id) and self.state in ('SHOWING', 'CANDIDATE'):
                was_showing = self.state == 'SHOWING'
                self._transition('ENDING' if was_showing else 'NONE', ts,
                                 self.target_id if was_showing else None)
                reasons = (['CROSS_MODAL_CATEGORY_MISMATCH'] if self.fusion.view(self.target_id).get('semantic_conflict') else ['SPEECH_STOP'])
                if self.fusion.view(self.target_id).get('visual_full', False):
                    reasons.append('CROSS_MODAL_CONFLICT')
                return self._packet(ts, 0., reasons)
        elif speech_evidence is not None:
            raise ValueError('speech evidence requires fusion_config')
        if self.state == "NONE":
            best = self._best(ts, objects, related)
            if best is not None:
                self._transition("CANDIDATE", ts, best)
                return self._packet(ts, related.get(best, 0.), self._candidate_reasons(self._tracks[best], ts))
            return self._packet(ts, 0., ["EVIDENCE_GAP" if gap else "INSUFFICIENT_EVIDENCE"])
        if self.state == "CANDIDATE":
            best = self._best(ts, objects, related)
            current = self._tracks.get(self.target_id)
            # Re-evaluate before the old candidate's missing-relation branch.
            if (best is not None and best != self.target_id and self._ready(self._tracks[best], ts)
                    and (current is None or not self._ready(current, ts))
                    and ts - self._state_since >= self.config.candidate_switch_ms):
                self._transition("CANDIDATE", ts, best)
                return self._packet(ts, related.get(best, 0.), ["CANDIDATE_REPLACED"] + self._candidate_reasons(self._tracks[best], ts))
            if self.target_id not in related and not self._language_support(self.target_id):
                if self._candidate_lost_since is None:
                    self._candidate_lost_since = ts
                if ts - self._candidate_lost_since >= self.config.candidate_decay_ms:
                    self._transition("NONE", ts)
                return self._packet(ts, 0., ["INSUFFICIENT_EVIDENCE"])
            self._candidate_lost_since = None
            if current is not None and self._ready(current, ts):
                self._last_score = related.get(self.target_id, 0.)
                self._transition("SHOWING", ts, self.target_id)
                return self._packet(ts, self._last_score, self._candidate_reasons(current, ts))
            return self._packet(ts, related.get(self.target_id, 0.), self._candidate_reasons(current, ts))
        if self.state == "SHOWING":
            track = self._tracks[self.target_id]
            if self.target_id not in objects:
                relinked = self._relink(ts, objects, related)
                if relinked is not None:
                    self.target_id = relinked
                    self._last_score = related[relinked]
                    return self._packet(ts, self._last_score, ["HAND_OBJECT_ASSOCIATED", "TARGET_RELINKED"])
                if ts - track.last_seen_ms >= self.config.missing_tolerance_ms:
                    self._transition("ENDING", ts, self.target_id)
                return self._packet(ts, self._last_score, ["TARGET_LOST"])
            self._last_score = related.get(self.target_id, 0.)
            on_table = track.points[-1][2] >= track.anchor_y - self.config.putdown_margin
            if on_table and not self._language_support(self.target_id):
                if self._putdown_since is None:
                    self._putdown_since = ts
                if ts - self._putdown_since >= self.config.putdown_min_ms:
                    self._transition("ENDING", ts, self.target_id)
                    return self._packet(ts, self._last_score, ["BACK_ON_TABLE"])
            else:
                self._putdown_since = None
            weak = (self.target_id not in related or (track.moving and not track.approach)
                    or not self._raised(track) or track.points[-1][2]>=self.config.display_exit_y)
            weak = weak and not self._language_support(self.target_id)
            if weak:
                if self._weak_since is None:
                    self._weak_since = ts
                if ts - self._weak_since >= self.config.presentation_exit_ms:
                    self._transition("ENDING", ts, self.target_id)
                    return self._packet(ts, self._last_score, ["PRESENTATION_EVIDENCE_LOST"])
            else:
                self._weak_since = None
            return self._packet(ts, self._last_score, ["HAND_OBJECT_ASSOCIATED", "RAISED_IN_FRAME"] if not weak else ["INSUFFICIENT_EVIDENCE"])
        # ENDING: only explicit evidence can resume; always has a hard limit.
        if ts - self._state_since >= self.config.ending_max_ms:
            self._transition("NONE", ts)
            return self._packet(ts, 0., ["ENDING_TIMEOUT"])
        track = self._tracks.get(self.target_id)
        if (self.target_id in objects and (self.target_id in related or self._language_support(self.target_id))
                and track is not None and self._ready(track, ts)):
            self._last_score = related.get(self.target_id, 0.)
            self._transition("SHOWING", ts, self.target_id)
            return self._packet(ts, self._last_score, ["TEMPORAL_HOLD", "TARGET_RESUMED"])
        if self.target_id not in objects:
            relinked = self._relink(ts, objects, related)
            if relinked is not None:
                self._last_score = related[relinked]
                self._transition("SHOWING", ts, relinked)
                return self._packet(ts, self._last_score, ["HAND_OBJECT_ASSOCIATED", "TARGET_RELINKED"])
        if ts - self._state_since >= self.config.ending_min_ms:
            self._transition("NONE", ts)
            return self._packet(ts, 0., ["INSUFFICIENT_EVIDENCE"])
        return self._packet(ts, self._last_score, ["TARGET_LOST" if self.target_id not in objects else "PRESENTATION_EVIDENCE_LOST"])

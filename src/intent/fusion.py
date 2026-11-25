"""Peer visual/language evidence, before the shared intent state transitions.

Scores are inspectable rule strengths, not probabilities. Grounding identifies
an instance for C; it does not require a successful visual intent verdict.
"""

from dataclasses import asdict, dataclass
import math

from .grounding import UtteranceGrounder


@dataclass(frozen=True)
class FusionConfig:
    visual_weight: float = 1.0
    speech_weight: float = 1.0
    support_threshold: float = 1.0
    candidate_threshold: float = .5
    confirmation_ms: float = 600
    return_margin: float = .06
    visual_full_strength: float = .75
    visual_confirmation_ms: float = 800

    def __post_init__(self):
        for name, value in asdict(self).items():
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value <= 0:
                raise ValueError(f'{name} must be finite and positive')
        if self.candidate_threshold > self.support_threshold:
            raise ValueError('candidate_threshold must not exceed support_threshold')
        if self.support_threshold > min(self.visual_weight, self.speech_weight):
            raise ValueError('either strong modality must be able to reach support_threshold')
        if not .5<=self.visual_full_strength<=1:
            raise ValueError('visual_full_strength must be between .5 and 1')


class EvidenceFusion:
    def __init__(self, config: FusionConfig):
        self.config = config
        self.grounder = UtteranceGrounder(return_margin=config.return_margin)
        self.views = {}
        self._since = {}
        self._last_ts = None
        self._blocked = {}
        self._last_raised = {}
        self._released = {}
        self._last_control = None
        self.diagnostics = {}
        self.language_only = False

    def reset(self):
        self.__init__(self.config)

    def update(self, ts, objects, tracks, related, active, speech, raised, *, confirmed=None, episode_started_ms=None):
        cfg = self.config
        speech = speech or {}
        if speech.get('enabled'):
            if speech.get('timestamp_ms') != ts:
                raise ValueError('speech evidence timestamp must match perception')
            available = speech.get('source_available_at_ms')
            if available is not None and available > ts:
                raise ValueError('future speech cannot be fused')
        cues = set(speech.get('cue_types', ())) if speech.get('enabled') else set()
        classes = speech.get('suggested_classes', []) if cues or speech.get('has_evidence') else []
        source_end = speech.get('source_end_ms')
        source_key = (speech.get('source_start_ms'), source_end, speech.get('source_available_at_ms'))
        discourse=speech.get('attributes',{}).get('discourse',{})
        self.grounder.observe(ts, objects, tracks, related, confirmed)
        # Carry observations across an explicitly linked detector fragment.
        for tid,track in tracks.items():
            parent=track.inherited_from
            if parent:
                for mapping in (self._since,self._blocked,self._last_raised,self._released):
                    if parent in mapping and tid not in mapping:
                        value=mapping[parent]
                        mapping[tid]=dict(value) if isinstance(value,dict) else value
        if classes or cues:
            grounded, grounding_status, returned = self.grounder.resolve(ts, speech, objects, related)
        else:
            grounded, grounding_status, returned = None, 'NO_USABLE_SPEECH', False
        linguistic = 0.
        if not cues & {'NON_CURRENT_LANGUAGE','MODEL_NO_DISPLAY_INTENT','ACTIVITY_NONDISPLAY'}:
            if 'PRESENTATION_LANGUAGE' in cues:
                linguistic = 1.
            elif 'CONTINUATION_LANGUAGE' in cues and active is not None:
                linguistic = .5
            elif 'INTRODUCTION_LANGUAGE' in cues:
                linguistic = .5
            elif 'ATTRIBUTE_LANGUAGE' in cues:
                linguistic = .5
            elif 'REFERENCE_LANGUAGE' in cues:
                linguistic = .25
            elif classes:
                linguistic = .25
        controls = (cues & {'NEGATED_LANGUAGE', 'ENDING_LANGUAGE', 'TOPIC_CHANGE','ACTIVITY_NONDISPLAY'}
                    if 'NON_CURRENT_LANGUAGE' not in cues else set())
        if controls:
            linguistic = 0.
            if source_key != self._last_control:
                self._last_control = source_key
                # Positive and negative statements use the SAME instance/time
                # grounding. Never broadcast a veto across a whole category.
                older_episode=(grounded==confirmed and episode_started_ms is not None
                               and source_end is not None and source_end<episode_started_ms)
                affected = [grounded] if grounded is not None and not returned and not older_episode else []
                for tid in affected:
                    if tid is not None:
                        self._blocked[tid] = {'source': source_key, 'saw_release': False}
        # A model-recognized item change can close the old episode only when
        # the new instance is grounded AND a visible handoff supports it.
        if (discourse.get('new_item') and grounded is not None and active is not None and grounded!=active
                and grounded in related and active in tracks and (active not in related or not raised(tracks[active]))):
            self._blocked[active]={'source':source_key,'saw_release':False}
        live_ids = set(tracks)
        for mapping in (self._since, self._blocked, self._released, self._last_raised):
            for tid in set(mapping)-live_ids:
                del mapping[tid]
        views = {}
        for tid, obj in objects.items():
            track = tracks[tid]
            is_raised = raised(track)
            was_raised = self._last_raised.get(tid, False)
            if was_raised and not is_raised:
                self._released[tid] = ts
            self._last_raised[tid] = is_raised
            visual = (cfg.visual_full_strength if track.was_good else .5 if tid in related and track.lift_seen and is_raised
                      else .25 if tid in related else 0.)
            # A steady, salient manipulated item can complement an introduction
            # even if detection began after pickup. No complete lift is required.
            points=[p for p in track.points if ts-p[0]<=600]
            steady=(len(points)>=4 and max(p[1] for p in points)-min(p[1] for p in points)<.035
                    and max(p[2] for p in points)-min(p[2] for p in points)<.05)
            elevated=(track.points[-1][2]<.65) if track.points else False
            if tid in related and steady and elevated and not track.moving:
                visual=max(visual,.5)
            if self.language_only:visual=0.
            language = linguistic if tid == grounded and not returned else 0.
            conflict=(discourse.get('version')==2 and speech.get('referent_scope')=='current'
                      and linguistic>0 and source_end is not None and len(classes)==1 and track.class_name!=classes[0]
                      and tid in related and any(
                          rows.get(tid,{}).get('related') for t,rows in self.grounder.history
                          if speech.get('source_start_ms',ts)<=t<=source_end))
            # A cold, ordinary hold plus "this item" is not yet an intent.
            # A genuine lift can complement reference; an existing hold needs
            # display/intro/surface-description language instead of marketing.
            if ('REFERENCE_LANGUAGE' in cues and not cues & {'PRESENTATION_LANGUAGE','INTRODUCTION_LANGUAGE','ATTRIBUTE_LANGUAGE'}
                    and not track.lift_seen and confirmed!=tid):
                language=min(language,.25)
            if speech.get('referent_scope') in ('previous','other'):
                language=0.
            # A statement preceding a later observed put-down cannot resurrect it.
            if source_end is not None and source_end < self._released.get(tid, -1):
                language = 0.
            barrier = self._blocked.get(tid)
            if barrier:
                if not is_raised:
                    barrier['saw_release'] = True
                fresh_language = language >= 1 and source_key != barrier['source']
                if fresh_language or (barrier['saw_release'] and not was_raised and is_raised and visual >= .5):
                    del self._blocked[tid]
                    barrier = None
            total = cfg.visual_weight*visual + cfg.speech_weight*language
            supported = not barrier and not conflict and total >= cfg.support_threshold
            if supported:
                self._since.setdefault(tid, ts)
            else:
                self._since.pop(tid, None)
            reasons = []
            if visual:
                reasons.append('VISUAL_PRESENTATION' if track.was_good else 'VISUAL_PARTIAL')
            if tid == grounded and returned:
                reasons.append('SPEECH_REFERENT_RETURNED')
            if language:
                reasons.append('SPEECH_PRESENTATION' if language == 1 else
                               'SPEECH_INTRODUCTION' if 'INTRODUCTION_LANGUAGE' in cues else
                               'SPEECH_ATTRIBUTE' if 'ATTRIBUTE_LANGUAGE' in cues else 'SPEECH_REFERENCE')
            if visual and language:
                reasons.append('JOINT_EVIDENCE')
            if barrier:
                reasons.extend(('SPEECH_STOP', 'CROSS_MODAL_CONFLICT') if track.was_good and not self.language_only else ('SPEECH_STOP',))
            if conflict:reasons.append('CROSS_MODAL_CATEGORY_MISMATCH')
            views[tid] = dict(visual=visual, speech=language, total=total, blocked=bool(barrier),
                              semantic_conflict=conflict,
                              visual_full=track.was_good and not self.language_only,
                              visual_accumulated_ms=track.good_ms,
                              visual_confirmation_required_ms=cfg.visual_confirmation_ms,
                              candidate=not barrier and not conflict and total >= cfg.candidate_threshold,
                              supported=supported,
                              ready=supported and ts-self._since[tid] >= cfg.confirmation_ms,
                              reasons=reasons)
        # Equal strong observations for different targets are a conflict, not
        # a tie to break in favor of language or vision.
        supported = [(tid, v) for tid, v in views.items() if v['supported']]
        if supported:
            top = max(v['total'] for _, v in supported)
            tied = [(tid, v) for tid, v in supported if abs(v['total']-top) < 1e-9]
            if len(tied) > 1 and any(v['speech'] >= 1 for _, v in tied) and any(v['visual_full'] for _, v in tied):
                for tid, view in tied:
                    view.update(ambiguous=True, ready=False, candidate=False)
                    view['reasons'].append('CROSS_MODAL_TARGET_CONFLICT')
                    self._since.pop(tid, None)
        self.views = views
        self._last_ts = ts
        self.diagnostics = dict(timestamp_ms=ts, speech=speech, grounded_id=grounded,
                                grounding_status=grounding_status, referent_returned=returned,
                                control_applied_to=grounded if controls and grounded in self._blocked else None,
                                candidates=views)

    def view(self, tid):
        return self.views.get(tid, {})

    def blocked(self, tid):
        return tid in self._blocked or self.views.get(tid,{}).get('semantic_conflict',False)

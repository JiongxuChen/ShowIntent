"""B refinement: bounded interaction memory and presentation-aware speech grounding.

Consumes the same A facts; never synthesizes detections or consults labels.
The original engine remains the frozen baseline.
"""
import re

from .temporal import TemporalIntentStateMachine
from .fusion import EvidenceFusion
from .grounding import UtteranceGrounder
from .types import bbox_center
from src.perception.appearance_tracker import appearance_distance


class PresentationGrounder(UtteranceGrounder):
    def observe(self,ts,objects,tracks,related,confirmed):
        super().observe(ts,objects,tracks,related,confirmed)
        for tid,row in self.history[-1][1].items():
            track=tracks[tid]
            row['presented']=(track.lift_seen and len(track.points)>=4
                              and row['y']<min(.72,track.anchor_y-.025))

    def _select(self,frames,classes,source_end,*,attributes=None,control=False):
        if not control:
            frames=[(t,{k:v for k,v in rows.items() if v.get('presented')}) for t,rows in frames]
        return super()._select(frames,classes,source_end,attributes=attributes,control=control)

    def resolve(self,ts,speech,objects,related):
        text=speech.get('source_text','')
        controls=set(speech.get('cue_types',[])) & {'NEGATED_LANGUAGE','ENDING_LANGUAGE','TOPIC_CHANGE','ACTIVITY_NONDISPLAY'}
        if not controls and re.search(r'(?:这|這|那)?(?:两|兩|三|四|几|幾|多|[2-9])(?:个|個|款|瓶|杯|本|件)|\b(?:these|those|both|all three)\b',text,re.I):
            return None,'GROUP_INTRODUCTION_NOT_SINGLE_TARGET',False
        return super().resolve(ts,speech,objects,related)


class PresentationEvidenceFusion(EvidenceFusion):
    def __init__(self,config):
        super().__init__(config)
        self.grounder=PresentationGrounder(return_margin=config.return_margin)
        self.session_stop_ms=None

    def update(self,ts,objects,tracks,related,active,speech,raised,**kwargs):
        speech=speech or {};cues=set(speech.get('cue_types',[]));text=speech.get('source_text','')
        if 'NON_CURRENT_LANGUAGE' not in cues:
            if ('ENDING_LANGUAGE' in cues and re.search(r'(?:今天|本次|这次|這次|本场|本場).{0,12}(?:展示|演示|介绍|介紹).{0,6}(?:到这里|到這裡|结束|結束)',text)):
                self.session_stop_ms=speech.get('source_end_ms',ts)
            elif (self.session_stop_ms is not None and speech.get('source_start_ms',-1)>=self.session_stop_ms
                  and cues & {'PRESENTATION_LANGUAGE','INTRODUCTION_LANGUAGE'}):
                self.session_stop_ms=None
        super().update(ts,objects,tracks,related,active,speech,raised,**kwargs)
        # An explicit whole-presentation closing applies to the episode, so a
        # second still-held object cannot immediately reopen the same shot.
        if self.session_stop_ms is not None:
            for tid in tracks:
                self._blocked[tid]={'source':('session',self.session_stop_ms),'saw_release':False}
                self._since.pop(tid,None)
                if tid in self.views:
                    self.views[tid].update(blocked=True,candidate=False,supported=False,ready=False)
                    self.views[tid]['reasons'].append('PRESENTATION_SESSION_CLOSED')


class PersistentIntentStateMachine(TemporalIntentStateMachine):
    def __init__(self,*args,hand_memory_ms=3000,**kwargs):
        super().__init__(*args,**kwargs)
        if hand_memory_ms<=0:raise ValueError('hand_memory_ms must be positive')
        self.hand_memory_ms=hand_memory_ms;self.interactions={};self.retained=set()
        if self.fusion is not None:
            self.fusion=PresentationEvidenceFusion(self.fusion.config)
            self.fusion.language_only=self.language_only

    def _observe(self,ts,objects,related):
        direct=dict(related);self.retained=set()
        confirmed=self.target_id if self.state=='SHOWING' else None
        self.interactions={k:v for k,v in self.interactions.items() if ts-v[0]<=self.hand_memory_ms or k==confirmed}
        continuity={}
        for tid,obj in objects.items():
            track=self._tracks.get(tid);old=self.interactions.get(tid)
            if track is None or old is None:continue
            _,strength,appearance=old
            cy=bbox_center(obj['bbox_norm'])[1]
            distance=appearance_distance(appearance,obj.get('appearance',[]))
            # A prior real interaction and observed lift are necessary. A static
            # table item or a cold appearance cannot acquire inferred contact.
            if (track.lift_seen and cy<min(self.config.display_region_max_y,track.anchor_y-self.config.putdown_margin)
                    and ts-track.last_seen_ms<=self.config.observation_gap_ms
                    and (distance is None or distance<=.5)):
                if tid not in direct:related[tid]=strength;self.retained.add(tid)
                recent=[p for p in track.points if ts-p[0]<=1000]
                if (tid==confirmed and recent and max(p[1] for p in recent)-min(p[1] for p in recent)<.08
                        and max(p[2] for p in recent)-min(p[2] for p in recent)<.10):
                    continuity[tid]=(track.qualified_since,track.good_ms,track.last_seen_ms)
        for tid,strength in direct.items():
            self.interactions[tid]=(ts,strength,objects[tid].get('appearance',[]))
        super()._observe(ts,objects,related)
        for tid,(since,good,previous) in continuity.items():
            track=self._tracks.get(tid)
            if track is None or not track.lift_seen:continue
            track.qualified_since=since if since is not None else ts
            track.good_ms=good+ts-previous;track.was_good=True
            track.bad_since=None;track.evidence_ts=ts;track.moving=False

    def _packet(self,ts,score,reasons):
        if self.target_id in self.retained:
            reasons=[r for r in reasons if r!='HAND_OBJECT_ASSOCIATED']+['HAND_HOLD_RETAINED']
        return super()._packet(ts,score,reasons)

    def update(self,packet,*,speech_evidence=None):
        if self._last_ts is not None and packet['timestamp_ms']-self._last_ts>self.config.observation_gap_ms:
            self.interactions.clear()
        result=super().update(packet,speech_evidence=speech_evidence)
        return result

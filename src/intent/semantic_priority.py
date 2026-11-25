"""Selective language authority; keep visual confirmation and localization intact."""
from dataclasses import replace
from .fusion import FusionConfig
from .persistent import PersistentIntentStateMachine


def semantic_config(config,evidence):
    cues=set((evidence or {}).get('cue_types',[]))
    positive=cues & {'PRESENTATION_LANGUAGE','INTRODUCTION_LANGUAGE','ATTRIBUTE_LANGUAGE','CONTINUATION_LANGUAGE'}
    if positive and not cues & {'NON_CURRENT_LANGUAGE','MODEL_NO_DISPLAY_INTENT','ACTIVITY_NONDISPLAY'}:
        return config
    return replace(config,speech_weight=min(config.speech_weight,1.0))


class SemanticBranchIntentStateMachine(PersistentIntentStateMachine):
    def __init__(self,*args,fusion_config=None,**kwargs):
        self.priority_config=fusion_config or FusionConfig(speech_weight=2,confirmation_ms=350,visual_confirmation_ms=600,visual_full_strength=1)
        super().__init__(*args,fusion_config=self.priority_config,**kwargs)

    def update(self,packet,*,speech_evidence=None):
        self.fusion.config=semantic_config(self.priority_config,speech_evidence)
        return super().update(packet,speech_evidence=speech_evidence)


class SemanticPriorityIntentStateMachine(SemanticBranchIntentStateMachine):
    """Preserve an independent visual path; admit grounded semantic early starts.

    The guard reads only current perception and model states, never labels.
    It preserves visual SHOWING and targets, but cannot promise unseen-data
    precision: an added early start still needs independent evaluation.
    """
    def __init__(self,*args,fusion_config=None,hand_memory_ms=3000,min_audio_lead_ms=200,**kwargs):
        super().__init__(*args,fusion_config=fusion_config,hand_memory_ms=hand_memory_ms,**kwargs)
        if min_audio_lead_ms<0:raise ValueError('min_audio_lead_ms must be non-negative')
        self.visual_engine=PersistentIntentStateMachine(*args,hand_memory_ms=hand_memory_ms,**kwargs)
        self.min_audio_lead_ms=min_audio_lead_ms
        self._previous_visual=None;self._closed_targets=set();self._admitted=None

    def update(self,packet,*,speech_evidence=None):
        visual=self.visual_engine.update(packet)
        fused=super().update(packet,speech_evidence=speech_evidence)
        previous=self._previous_visual
        if previous and previous['state']=='SHOWING' and (visual['state']!='SHOWING' or visual['target_id']!=previous['target_id']):
            self._closed_targets.add(previous['target_id'])
        self._closed_targets={tid for tid in self._closed_targets if tid in self.visual_engine._tracks and self.visual_engine._tracks[tid].lift_seen}
        result=dict(visual);reason='VISUAL_PATH'
        if visual['state']=='SHOWING':
            self._admitted=None
            if fused['state']!='SHOWING' or fused['target_id']!=visual['target_id']:
                result['reason_codes']=list(result['reason_codes'])+['VISUAL_EVIDENCE_PRESERVED']
                reason='VISUAL_PRESERVED_OVER_LANGUAGE'
        else:
            tid=fused['target_id'];track=self.visual_engine._tracks.get(tid)
            cues=set((speech_evidence or {}).get('cue_types',[]))
            explicit=bool(cues & {'PRESENTATION_LANGUAGE','INTRODUCTION_LANGUAGE','ATTRIBUTE_LANGUAGE','CONTINUATION_LANGUAGE'})
            explicit=explicit and not bool(cues & {'NON_CURRENT_LANGUAGE','MODEL_NO_DISPLAY_INTENT','ACTIVITY_NONDISPLAY','NEGATED_LANGUAGE','ENDING_LANGUAGE','TOPIC_CHANGE'})
            candidate=(fused['state']=='SHOWING' and visual['state']=='CANDIDATE' and visual['target_id']==tid
                       and tid not in self._closed_targets and track is not None and track.last_seen_ms==packet['timestamp_ms']
                       and track.lift_seen and self.visual_engine._raised(track))
            remaining=max(0,self.visual_engine.config.showing_min_ms-(track.good_ms if track is not None else 0))
            if candidate and (self._admitted==tid or explicit and remaining>=self.min_audio_lead_ms):
                result=dict(fused);self._admitted=tid;reason='SEMANTIC_EARLY_CONFIRMATION'
                result['reason_codes']=list(result['reason_codes'])+['GROUNDED_SEMANTIC_EARLY_START']
            else:
                self._admitted=None
                if fused['state']=='SHOWING':reason='AUDIO_ONLY_CONTINUATION_REJECTED'
        self._previous_visual=dict(visual)
        self.fusion.diagnostics.update(visual_reference_intent=visual,raw_fused_intent=fused,output_policy=reason,
                                       min_audio_lead_ms=self.min_audio_lead_ms)
        return result

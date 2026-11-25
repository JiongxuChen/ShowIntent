"""Offline candidate: ground speech expiry in an observed return to rest."""
from .grounding import UtteranceGrounder
from .fusion import EvidenceFusion
from .semantic_priority import SemanticPriorityIntentStateMachine


class RestAwareGrounder(UtteranceGrounder):
    def __init__(self,*,putdown_margin=.025,**kwargs):
        super().__init__(**kwargs);self.putdown_margin=putdown_margin

    def observe(self,ts,objects,tracks,related,confirmed):
        super().observe(ts,objects,tracks,related,confirmed)
        for tid,row in self.history[-1][1].items():
            track=tracks[tid]
            row.update(rest_y=track.anchor_y,lift_seen=track.lift_seen)

    def _returned_between(self,binding,rows):
        minimum=binding.source_y;raised=False
        for row in rows:
            minimum=min(minimum,row['y'])
            rest=row.get('rest_y')
            if rest is None:continue
            if row.get('lift_seen') and row['y']<rest-self.putdown_margin:raised=True
            # Relative movement alone is not a return; it must reach the
            # observed rest region after a lift, with a clear downward change.
            if (raised or binding.source_y<rest-self.putdown_margin) and row['y']>=rest-self.putdown_margin and row['y']-minimum>self.return_margin:
                return True
        return False

    def _returned_now(self,binding,y):
        row=self.history[-1][1].get(binding.track_id) if self.history else None
        return bool(row and self._returned_between(binding,[row]))


class RestAwareFusion(EvidenceFusion):
    def __init__(self,config,putdown_margin=.025):
        super().__init__(config);self.putdown_margin=putdown_margin
        self.grounder=RestAwareGrounder(return_margin=config.return_margin,putdown_margin=putdown_margin)

    def reset(self):
        self.__init__(self.config,self.putdown_margin)


class RestAwareBranchIntentStateMachine(SemanticPriorityIntentStateMachine):
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.fusion=RestAwareFusion(self.priority_config,self.config.putdown_margin)


class RestAwareIntentStateMachine(RestAwareBranchIntentStateMachine):
    """Retain every confirmed v2 output while testing the corrected early path."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,**kwargs)
        self.reference=SemanticPriorityIntentStateMachine(*args,**kwargs)

    def update(self,packet,*,speech_evidence=None):
        baseline=self.reference.update(packet,speech_evidence=speech_evidence)
        candidate=super().update(packet,speech_evidence=speech_evidence)
        result=baseline if baseline['state']=='SHOWING' else candidate
        self.fusion.diagnostics.update(baseline_reference_intent=baseline,
            rest_candidate_intent=candidate,
            output_policy='BASELINE_V2_PRESERVED' if baseline['state']=='SHOWING' else self.fusion.diagnostics['output_policy'])
        return result

"""Require current presentation geometry for language-only early confirmation."""
from .rest_aware import RestAwareIntentStateMachine
from .types import bbox_center


class SafeEarlyIntentStateMachine(RestAwareIntentStateMachine):
    def update(self,packet,*,speech_evidence=None):
        result=super().update(packet,speech_evidence=speech_evidence)
        visual=self.fusion.diagnostics['visual_reference_intent']
        if visual['state']=='SHOWING':return result
        if result['state']=='SHOWING':
            obj=next((o for o in packet['objects'] if o['track_id']==result['target_id']),None)
            if obj is None or bbox_center(obj['bbox_norm'])[1]>=self.config.display_region_max_y:
                self.fusion.diagnostics['early_geometry_rejected']=True
                return dict(visual)
        return result

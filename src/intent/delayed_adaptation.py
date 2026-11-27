"""Offline safe B with one injectable sparse-observation recovery extension."""
from .safe_early import SafeEarlyIntentStateMachine
from .streaming import SparseVisualIntentStateMachine


class RecoveryOnlyIntentStateMachine(SafeEarlyIntentStateMachine):
    """Offline timing, adding only measured sparse lift recovery."""
    def __init__(self,*args,**kwargs):
        super().__init__(*args,visual_engine_cls=SparseVisualIntentStateMachine,**kwargs)


class DelayedIntentStateMachine(RecoveryOnlyIntentStateMachine):
    def __init__(self,*args,**kwargs):
        # Sparse observations can arrive >750ms apart. Keep the same evidence
        # requirements, allowing only bounded interruptions; putdown stays fast.
        for key,minimum in [('observation_gap_ms',1250),('evidence_gap_ms',500),
                            ('missing_tolerance_ms',1000),('presentation_exit_ms',900)]:
            kwargs[key]=max(kwargs.get(key,0),minimum)
        super().__init__(*args,**kwargs)

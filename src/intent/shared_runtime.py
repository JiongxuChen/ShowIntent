"""One B policy and parameter contract for offline and delayed execution."""
from .safe_early import SafeEarlyIntentStateMachine
from .persistent import PersistentIntentStateMachine


def shared_engines(visual_config,fusion_config,hand_memory_ms=3000,*,recovery=False):
    from .delayed_adaptation import RecoveryOnlyIntentStateMachine
    cls=RecoveryOnlyIntentStateMachine if recovery else SafeEarlyIntentStateMachine
    return (cls(**visual_config,fusion_config=fusion_config,hand_memory_ms=hand_memory_ms),
            PersistentIntentStateMachine(**visual_config,hand_memory_ms=hand_memory_ms))

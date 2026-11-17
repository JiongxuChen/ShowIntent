"""Optional member-D speech evidence for ShowIntent.

The package emits lexical evidence only. B retains responsibility for intent
state and target-instance selection, and C continues to consume B's packets.
"""

from .events import SpeechEvent
from .cue_adapter import CueEvidence, SpeechCueAdapter

__all__ = ["SpeechEvent", "CueEvidence", "SpeechCueAdapter"]

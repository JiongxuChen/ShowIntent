"""Audio-only temporal baseline. No object, hand, track or bounding-box input."""
from dataclasses import replace
import re
from src.speech.intent_language import PresentationLanguage


class AudioTimeline:
    def __init__(self, events, *, timing='content', hold_ms=8000):
        if timing not in {'content','causal'} or hold_ms<=0:raise ValueError('invalid audio timeline configuration')
        self.events=sorted([e for e in events if e.is_final],key=lambda e:e.end_ms if timing=='content' else e.available_at_ms)
        self.timing=timing;self.hold_ms=hold_ms;self.index=0;self.until=-1;self.last=-1;self.language=PresentationLanguage();self.classes=[]

    def update(self, timestamp_ms):
        if timestamp_ms<self.last:raise ValueError('timestamps must be monotonic')
        self.last=timestamp_ms;acts=[]
        while self.index<len(self.events):
            e=self.events[self.index];available=e.end_ms if self.timing=='content' else e.available_at_ms
            if available>timestamp_ms:break
            self.index+=1
            # Process all clauses in spoken order, rather than losing an earlier action.
            for text in filter(None,(x.strip() for x in re.split(r'[，。！？；,;!?\n]+',e.text))):
                s=self.language.analyze_event(replace(e,text=text,semantics=None));act=s['act'];acts.append(act)
                if not s['current'] or s['referent_scope'] in {'previous','other'}:continue
                if act in {'STOP','SWITCH'}:self.until=-1;self.classes=[]
                elif act in {'SHOW','INTRODUCE'}:
                    self.until=available+self.hold_ms;self.classes=s['classes']
                elif act in {'DESCRIBE','CONTINUE'} and available<=self.until:
                    self.until=available+self.hold_ms
        return dict(timestamp_ms=timestamp_ms,state='SHOWING' if timestamp_ms<self.until else 'NONE',target_id=None,
                    suggested_classes=self.classes if timestamp_ms<self.until else [],acts=acts,
                    scope='AUDIO_ONLY_TEMPORAL_NOT_CAMERA_PACKET',timing=self.timing)

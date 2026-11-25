"""Action-focused language observations with explicit, inspectable scope.

This backend intentionally does not trust the previous model's scope/act tags.
It reinterprets original ASR text; it never edits the saved model observations.
"""
import re
from .language import LanguageCueExtractor


class PresentationLanguage:
    def __init__(self):
        self.extractor=LanguageCueExtractor()
        self.context_class=[]
        self.context_end=-100000

    def analyze_event(self,event):
        text=event.text
        lexical=self.extractor.extract(text)
        clause=lexical.source_clause
        cues=set(lexical.cue_types)
        scope=lexical.referent_scope
        act='NONE'
        noncurrent='NON_CURRENT_LANGUAGE' in cues
        display=bool(re.search(r'(?:大家|你们|你們|各位).{0,4}看|(?:看一下|看一看|看看)(?:这|這|它|另一面|背面|侧面)|\b(?:look at|let me show|take a look)\b',clause,re.I))
        surface=bool(re.search(r'颜色|顏色|表面|纹理|紋理|封面|材质|材質|背面|侧面|另一面|光滑|粗糙|形状|形狀|尺寸|\b(?:texture|surface|cover|shape|colour|color)\b',clause,re.I))
        marketing=bool(re.search(r'优惠|優惠|福利|块钱|塊錢|一斤|两斤|兩斤|下单|下單|购买|購買|好吃|酸甜|美味|品质|品質|\b(?:buy|discount|price|delicious)\b',clause,re.I))
        ending=bool(re.search(r'先放(?:到|在|回)|先放下|换(?:下)?一个|換(?:下)?一個|\bput (?:it|this) (?:down|back)\b',clause,re.I))
        if not noncurrent:
            if 'NEGATED_LANGUAGE' in cues:act='STOP'
            elif 'ENDING_LANGUAGE' in cues or ending:act='STOP'
            elif 'TOPIC_CHANGE' in cues:act='SWITCH'
            elif 'PRESENTATION_LANGUAGE' in cues or display:act='SHOW'
            elif 'CONTINUATION_LANGUAGE' in cues:act='CONTINUE'
            elif surface and not marketing:act='DESCRIBE'
            elif 'INTRODUCTION_LANGUAGE' in cues and not marketing:act='INTRODUCE'
            elif 'REFERENCE_LANGUAGE' in cues:act='REFER'
        classes=list(lexical.mentioned_classes)
        # Description/pronoun clauses can carry a previously explicit referent,
        # but a plain greeting/price/quality statement cannot establish one.
        recent=event.start_ms-self.context_end<=12000
        if not classes and recent and act in {'DESCRIBE','CONTINUE','STOP'}:
            classes=list(self.context_class)
        if act in {'SHOW','INTRODUCE'}:
            self.context_class=classes;self.context_end=event.end_ms
        elif act=='DESCRIBE' and recent:self.context_end=event.end_ms
        elif act in {'STOP','SWITCH'}:
            self.context_class=[];self.context_end=-100000
        return dict(act=act,current=not noncurrent,referent_scope=scope,classes=classes,
                    object_phrase='',colors=(lexical.attributes or {}).get('colors',[]),
                    quote=clause if act!='NONE' else '',confidence='high')

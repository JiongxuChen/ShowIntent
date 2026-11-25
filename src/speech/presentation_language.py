"""Natural description additions, used only by the refined B replay."""
import re
from .intent_language import PresentationLanguage


class ContextualPresentationLanguage(PresentationLanguage):
    def analyze_event(self,event):
        value=super().analyze_event(event);text=event.text
        if not value['current'] or value['act'] in {'STOP','SWITCH'}:return value
        # Product words and parts contribute the corresponding detector concept;
        # they do not add a detector whitelist or require a scene configuration.
        if not value['classes']:
            if re.search(r'(?:咖啡|陶瓷|玻璃|随行|隨行|马克|馬克)杯|杯(?:身|套|盖|蓋|把)',text):value['classes']=['cup']
            elif re.search(r'(?:这|這|那|一|两|兩|三)瓶|瓶(?:身|盖|蓋)',text):value['classes']=['bottle']
        surface=re.search(r'杯身|瓶身|包装|包裝|图案|圖案|提手|杯套|绳子|繩子|盖子|蓋子|正面|前面|外面|大小|尺寸',text)
        advice=re.search(r'购买|購買|买之前|買之前|下单|下單|价格|價格|优惠|優惠|别光|別光',text)
        introduction=re.search(r'(?:这|這)(?:个|個|一款|款|瓶|本|支).{0,4}是|介绍(?:一下)?(?:这|這)',text)
        if not advice and value['act'] in {'NONE','REFER'}:
            if surface:value['act']='DESCRIBE'
            elif introduction:value['act']='INTRODUCE'
        if value['act']!='NONE':value['quote']=text
        if value['act'] in {'INTRODUCE','SHOW','DESCRIBE'}:
            if not value['classes'] and event.start_ms-self.context_end<=12000:value['classes']=list(self.context_class)
            if value['classes']:self.context_class=list(value['classes']);self.context_end=event.end_ms
        return value

from src.speech.intent_language import PresentationLanguage
from src.speech.events import SpeechEvent
import pytest

@pytest.mark.parametrize('text,act',[
 ('先给大家看一下这个苹果','SHOW'),('大家好','NONE'),('虽然','NONE'),
 ('为了优惠新老客户','NONE'),('此刻正推出福利奖','NONE'),
 ('今天桌上准备了苹果和橘子','NONE'),('把这个苹果挪过去','REFER'),
 ('它的表面有细小的纹理','DESCRIBE'),('它的品质非常好','NONE'),
 ('苹果就先介绍到这里','STOP'),('这个苹果先放到桌上','STOP'),
 ('不妨看看这本书','SHOW'),('明天给大家展示这个','NONE')])
def test_action_and_scope(text,act):
    e=SpeechEvent(0,1000,2000,text)
    result=PresentationLanguage().analyze_event(e)
    assert result['act']==act
    assert result['referent_scope']!='previous'
    SpeechEvent(0,1000,2000,text,semantics=result)

def test_previous_requires_actual_historical_words():
    p=PresentationLanguage()
    assert p.analyze_event(SpeechEvent(0,1000,2000,'上一个物品展示结束'))['referent_scope']=='previous'

def test_focused_interpreter_does_not_need_a_new_category_configuration():
    result=PresentationLanguage().analyze_event(SpeechEvent(0,1000,2000,'给大家看看这个螺丝刀的侧面'))
    assert result['act']=='SHOW' and result['referent_scope']=='mentioned'
    assert result['classes']==[]

"""Validated rule-language observations, independent of camera decisions."""

ACTS = {'SHOW', 'INTRODUCE', 'DESCRIBE', 'REFER', 'CONTINUE', 'STOP', 'SWITCH', 'NONE'}
SCOPES = {'mentioned', 'current', 'previous', 'other', 'unspecified'}
COLORS = {'red','orange','yellow','green','blue','purple','white','black','gray','brown','pink'}
FIELDS = {'act','current','referent_scope','classes','object_phrase','colors','quote','confidence'}


def validate_semantics(value, text):
    if not isinstance(value, dict) or not FIELDS<=set(value) or set(value)-FIELDS-{'discourse'}:
        raise ValueError('semantic observation has unexpected fields')
    if value['act'] not in ACTS or value['referent_scope'] not in SCOPES:
        raise ValueError('invalid semantic act or referent scope')
    if type(value['current']) is not bool or value['confidence'] not in {'high','medium','low'}:
        raise ValueError('invalid semantic current/confidence')
    for key in ('classes','colors'):
        if not isinstance(value[key], list) or any(not isinstance(x,str) or not x.strip() for x in value[key]):
            raise ValueError(f'invalid semantic {key}')
        if len(value[key]) != len(set(value[key])):
            raise ValueError(f'duplicate semantic {key}')
    if set(value['colors'])-COLORS or not isinstance(value['object_phrase'],str):
        raise ValueError('invalid semantic attributes')
    if not isinstance(value['quote'],str) or value['quote'] not in text:
        raise ValueError('semantic quote must come from the current transcript')
    if value['act'] != 'NONE' and not value['quote'].strip():
        raise ValueError('an actionable semantic observation requires source words')
    if 'discourse' in value:
        DISCOURSE_ACTS={'SHOW','INTRODUCE','DESCRIBE','REFER','CONTINUE','STOP','SWITCH','ARRANGE','SELL','PLAN','CHAT','UNKNOWN'}
        d=value['discourse']
        if (not isinstance(d,dict) or set(d)!={'version','activity','new_item','episode'} or d['version']!=2
                or d['activity'] not in DISCOURSE_ACTS or type(d['new_item']) is not bool
                or type(d['episode']) is not int or d['episode']<0):raise ValueError('invalid discourse observation')
    return value


def semantic_cues(value):
    mapping={'SHOW':'PRESENTATION_LANGUAGE','INTRODUCE':'INTRODUCTION_LANGUAGE',
             'DESCRIBE':'ATTRIBUTE_LANGUAGE','REFER':'REFERENCE_LANGUAGE',
             'CONTINUE':'CONTINUATION_LANGUAGE','STOP':'ENDING_LANGUAGE','SWITCH':'TOPIC_CHANGE'}
    act=value['act']
    cues={mapping[act]} if act in mapping and value['confidence']!='low' else set()
    if act in {'STOP','SWITCH'} and value['confidence']!='high':
        cues=set()  # An uncertain stop must not veto a clear ongoing display.
    if act=='SHOW' and value['confidence']=='medium':
        cues={'INTRODUCTION_LANGUAGE'}
    if not value['current'] and cues:
        cues.add('NON_CURRENT_LANGUAGE')
    discourse=value.get('discourse',{})
    if discourse.get('activity')=='ARRANGE' and value['referent_scope']=='current':cues.add('ACTIVITY_NONDISPLAY')
    if discourse.get('activity') in {'SELL','CHAT','PLAN','UNKNOWN'}:cues.add('MODEL_NO_DISPLAY_INTENT')
    return list(value['classes']), cues, {'colors':list(value['colors']),
        'object_phrase':value['object_phrase'],'discourse':discourse}, value['referent_scope']

from types import SimpleNamespace
import pytest
from src.intent.safe_early import SafeEarlyIntentStateMachine
from src.intent.rest_aware import RestAwareIntentStateMachine

@pytest.mark.parametrize('cy,expected',[(.6,'SHOWING'),(.72,'CANDIDATE'),(.8,'CANDIDATE')])
def test_language_only_confirmation_requires_current_display_geometry(monkeypatch,cy,expected):
    e=SafeEarlyIntentStateMachine()
    visual=dict(timestamp_ms=100,state='CANDIDATE',target_id='x')
    e.fusion.diagnostics['visual_reference_intent']=visual
    monkeypatch.setattr(RestAwareIntentStateMachine,'update',lambda *a,**k:dict(visual,state='SHOWING'))
    p=dict(objects=[dict(track_id='x',bbox_norm=[.4,cy-.03,.6,cy+.03])])
    assert e.update(p)['state']==expected

def test_confirmed_visual_output_is_preserved_even_below_display_region(monkeypatch):
    e=SafeEarlyIntentStateMachine();v=dict(state='SHOWING',target_id='x')
    e.fusion.diagnostics['visual_reference_intent']=v
    monkeypatch.setattr(RestAwareIntentStateMachine,'update',lambda *a,**k:v)
    assert e.update(dict(objects=[])) is v

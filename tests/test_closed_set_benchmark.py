import json
from scripts.run_refined_b import offline_streams,normalized_score
def test_current_three_modalities_do_not_trigger_on_empty_negative():
 per=[dict(timestamp_ms=t,frame_id=i,objects=[],hands=[],relations=[]) for i,t in enumerate([0,100,200,300])]
 streams=offline_streams(per,[],{}, {})
 assert set(streams)=={'showintent','visual-only','audio-only'}
 for states in streams.values():
  assert all(v['state']!='SHOWING' for v in states)

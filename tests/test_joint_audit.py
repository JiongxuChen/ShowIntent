"""A completed-run audit must reject mismatched inputs and future evidence."""
import copy
import json
from pathlib import Path

import pytest

from src.intent.analyze_joint_run import analyze_run
from src.intent.replay_real import replay_clip
from src.speech.events import SpeechEvent


def prepare(tmp_path):
    perception=tmp_path/'perception';perception.mkdir()
    run=tmp_path/'run';(run/'speech').mkdir(parents=True)
    source=perception/'perception_A01.jsonl'
    packets=[dict(timestamp_ms=t,frame_id=n,objects=[dict(track_id='a',class_name='apple',
                    bbox_norm=[.3,.65,.4,.75],score=.9)],hands=[],relations=[]) for n,t in enumerate(range(0,3001,100))]
    source.write_text(''.join(json.dumps(p)+'\n' for p in packets))
    speech=run/'speech/speech_events_A01.jsonl'
    speech.write_text(json.dumps(SpeechEvent(100,500,800,'给大家看苹果').to_dict())+'\n')
    speech.with_suffix('.meta.json').write_text(json.dumps(dict(event_count=1,mode='causal',compute_rtf=.1)))
    (run/'run_config.json').write_text(json.dumps(dict(clips=['A01'],source_sha256={})))
    replay_clip(source,run/'results/A01',{},speech_source=speech)
    review=tmp_path/'review.csv'
    review.write_text('clip_id,start_ms,end_ms,label,allowed_classes\nA01,0,3000,SHOWING,apple\n')
    return run,perception,review


def test_audit_summarizes_the_recorded_outputs_without_reinterpreting_them(tmp_path):
    run,perception,review=prepare(tmp_path)
    before={p:p.read_bytes() for p in run.rglob('*') if p.is_file()}
    report,queue=analyze_run(run,perception,review)
    assert report['clip_count']==1 and report['total_packets']==31
    assert report['summary']['visual-only']['correct_class_coverage']==0
    assert report['summary']['showintent']['correct_class_coverage']>0
    assert queue and queue[0]['reason']=='ADDITIONAL_JOINT_SHOWING'
    assert {p:p.read_bytes() for p in before}==before


def test_audit_rejects_replaced_perception_input(tmp_path):
    run,perception,review=prepare(tmp_path)
    with (perception/'perception_A01.jsonl').open('a') as stream:stream.write('\n')
    with pytest.raises(ValueError,match='hash'):analyze_run(run,perception,review)


def test_audit_rejects_future_speech_in_trace(tmp_path):
    run,perception,review=prepare(tmp_path)
    path=run/'results/A01/evidence_showintent.jsonl'
    rows=[json.loads(s) for s in path.read_text().splitlines()]
    rows[0]['speech']['source_available_at_ms']=1000
    path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    with pytest.raises(ValueError,match='future'):analyze_run(run,perception,review)


def test_audit_can_read_a_new_b_replay_while_reusing_original_speech(tmp_path):
    run,perception,review=prepare(tmp_path)
    destination=tmp_path/'fixed_replay'
    (run/'results').rename(destination)
    report,queue=analyze_run(run,perception,review,replay_dir=destination)
    assert report['total_packets']==31 and queue


def test_unlabeled_new_videos_report_no_coverage_instead_of_zero_accuracy(tmp_path):
    run,perception,review=prepare(tmp_path)
    review.write_text('clip_id,start_ms,end_ms,label,allowed_classes\n')
    report,queue=analyze_run(run,perception,review)
    for method in report['summary'].values():
        assert method['correct_class_coverage'] is None
        assert method['evaluation_status']=='NO_MATCHING_REVIEW_INTERVALS'
        assert method['review_interval_count']==0

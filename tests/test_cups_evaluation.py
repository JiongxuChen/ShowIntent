import csv
import json
from pathlib import Path

import pytest

from src.evaluation.collection import check_annotations, check_timebase, evaluate
from scripts.run_complementary import digest


def fixture_collection(tmp_path):
    dataset=tmp_path/'dataset';dataset.mkdir()
    video=tmp_path/'original.mp4';video.write_bytes(b'independent original source')
    manifest=[dict(clip_id='negative',source_path=str(video),eval_start_ms=0,eval_end_ms=2000)]
    labels=[dict(clip_id='negative',event_id='n1',start_ms=0,end_ms=2000,physical_target_id='',class_name='',label='NON_SHOWING',confidence='HIGH',review_status='PENDING',annotator='assistant_original_video_review',reviewer='')]
    for name,rows in [('dataset_manifest.csv',manifest),('annotations.csv',labels)]:
        with (dataset/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    (dataset/'annotation_record.json').write_text(json.dumps(dict(annotations_sha256=digest(dataset/'annotations.csv'),video_sha256={'negative':digest(video)})))
    return dataset


def test_frozen_annotation_edit_is_rejected(tmp_path):
    dataset=fixture_collection(tmp_path)
    check_annotations(dataset)
    path=dataset/'annotations.csv';path.write_text(path.read_text().replace('NON_SHOWING','SHOWING'))
    with pytest.raises(ValueError,match='changed after freeze'):
        check_annotations(dataset)


def test_original_annotation_rounding_is_verified_against_every_frame():
    saved=[0,33,66];packets=[{'timestamp_ms':t} for t in saved]
    row=dict(clip_id='rounded',eval_start_ms=0,eval_end_ms=67)
    assert check_timebase(packets,saved,row,[0,33,67])['rounding_verified']
    with pytest.raises(ValueError,match='without original annotation PTS'):
        check_timebase(packets,saved,row)
    with pytest.raises(ValueError,match='round-versus-floor'):
        check_timebase(packets,saved,row,[0,35,67])


@pytest.mark.parametrize('timestamps',[[0,33],[0,34,66],[1,34,67]])
def test_rounding_exception_does_not_hide_missing_or_shifted_perception(timestamps):
    row=dict(clip_id='rounded',eval_start_ms=0,eval_end_ms=67)
    with pytest.raises(ValueError,match='Incomplete or shifted perception'):
        check_timebase([{'timestamp_ms':t} for t in timestamps],[0,33,66],row,[0,33,67])


def test_negative_clip_reports_false_time_without_fake_audio_target(tmp_path):
    dataset=fixture_collection(tmp_path)
    raw=tmp_path/'raw';run=tmp_path/'stable'
    for path in [raw/'timestamps',raw/'speech_raw',run/'perception',run/'results/negative']:path.mkdir(parents=True)
    (raw/'timestamps/negative.json').write_text(json.dumps({'timestamp_ms':[0,1000,2000]}))
    (raw/'speech_raw/speech_events_negative.jsonl').write_text('')
    (run/'run_config.json').write_text(json.dumps({'original_run':str(raw)}))
    packets=[dict(timestamp_ms=t,objects=[dict(track_id='new_id')]) for t in [0,1000,2000]]
    (run/'perception/perception_negative.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in packets))
    predictions=[dict(timestamp_ms=t,state='SHOWING' if t==0 else 'NONE',target_id='new_id' if t==0 else None) for t in [0,1000,2000]]
    for method in ['visual-only','showintent','pickup-baseline']:
        (run/f'results/negative/intent_{method}.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in predictions))
    out=tmp_path/'evaluation/evaluation.json';evaluate(dataset,run,out)
    report=json.loads(out.read_text())
    assert report['summary']['showintent']['false_seconds']==1
    assert report['summary']['showintent']['temporal_coverage'] is None
    assert report['negative_clips']['negative']['showintent']['triggered'] is True
    assert report['negative_clips']['negative']['audio-only']['triggered'] is False
    assert report['physical_metrics'] is None
    assert report['status']=='PROVISIONAL_ASSISTANT_LABELS_PENDING_USER_REVIEW'
    mapping=list(csv.DictReader((out.parent/'track_mapping_pending.csv').open()))
    assert mapping[0]['physical_target_id']==''
    assert mapping[0]['review_status']=='PENDING'

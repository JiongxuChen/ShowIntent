"""Optional short regressions over supplied ASR/A logs, without running ASR.

These development checks are not independent evaluation or reviewed event truth.
"""
import json
import os
from pathlib import Path

import pytest

from src.intent.fusion import FusionConfig
from src.intent.run import build_engine
from src.speech.bridge import SpeechEvidenceBridge


def replay_clip(clip):
    root=Path(__file__).resolve().parents[1]
    run=Path(os.environ.get('SHOWINTENT_JOINT_RUN',root/'outputs/joint_all_run'))
    directory=Path(os.environ.get('SHOWINTENT_PERCEPTION_DIR',root/'outputs/perception_real'))
    source=directory/f'perception_{clip}.jsonl';speech=run/'speech'/f'speech_events_{clip}.jsonl'
    if not source.is_file() or not speech.is_file():
        pytest.skip('supplied A/ASR development logs are not available')
    inputs=[json.loads(line) for line in source.read_text().splitlines() if line.strip()]
    classes={o['track_id']:o['class_name'] for p in inputs for o in p['objects']}
    bridge=SpeechEvidenceBridge.from_jsonl(speech,enabled=True)
    engine=build_engine('showintent',fusion_config=FusionConfig())
    outputs=[engine.update(p,speech_evidence=bridge.evidence_for(p)) for p in inputs]
    return outputs,classes


@pytest.mark.parametrize('clip,expected_class',[('A01','apple'),('A02','orange')])
def test_supplied_deictic_events_do_not_switch_to_the_other_fruit(clip,expected_class):
    outputs,classes=replay_clip(clip)
    showing=[row for row in outputs if row['state']=='SHOWING']
    assert showing
    assert all(classes[r['target_id']]==expected_class for r in showing)


def test_old_attribute_description_does_not_hold_into_the_selected_negative_core():
    outputs,_=replay_clip('D01')
    assert not any(r['state']=='SHOWING' for r in outputs if 31000<=r['timestamp_ms']<33000)

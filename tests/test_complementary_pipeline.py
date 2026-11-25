"""Exercise the complete runner/resume with fake model stages and real B replay."""

import json
from pathlib import Path
import shutil
from types import SimpleNamespace
import sys

from scripts import run_complementary as runner
from src.intent.complementarity import analyze
from src.speech.events import SpeechEvent


def test_full_command_produces_aligned_peer_ablations_and_resumes(tmp_path,monkeypatch):
    root=tmp_path/'project';root.mkdir();(root/'configs').mkdir()
    for name in ('presentation_intent.json','stable_fusion.json','speech_lexicon.json'):
        shutil.copyfile(Path('configs')/name,root/'configs'/name)
    inputs=root/'Example_Data1';inputs.mkdir();(inputs/'novel_scene.mp4').write_bytes(b'fake media')
    python=root/'python';python.touch()
    vision=root/'vision.pt';vision.write_bytes(b'fake weights')
    speech=root/'speech_model';speech.mkdir();(speech/'model.bin').touch()
    for name in ['config.json','tokenizer.json','vocabulary.json']:(speech/name).touch()
    out=root/'outputs/run'
    calls=[]
    def write(path,rows):
        path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
        path.write_text(''.join(json.dumps(r)+'\n' for r in rows))
    def fake(command,**kwargs):
        def arg(flag):return command[command.index(flag)+1]
        if '-c' in command:
            return SimpleNamespace(stdout='',returncode=0)
        stage=arg('-m');calls.append(stage)
        if stage=='scripts.extract_video_timestamps':
            target=Path(arg('--out'));target.parent.mkdir(parents=True)
            target.write_text(json.dumps(dict(frame_count=21,timestamp_ms=list(range(0,2001,100)))))
        elif stage=='src.perception.enhanced':
            rows=[]
            for t in range(0,2001,100):
                rows.append(dict(timestamp_ms=t,frame_id=t//100,
                  objects=[dict(track_id='physical_1',class_name='unlisted_item',bbox_norm=[.3,.5,.4,.6],score=.9)],
                  hands=[dict(hand_id='h',bbox_norm=[.3,.5,.4,.6],score=.9)],
                  relations=[dict(hand_id='h',object_id='physical_1',type='near_or_moving_together',score=.9)]))
            write(arg('--out'),rows)
            Path(arg('--metrics-out')).write_text(json.dumps(dict(frames_processed=21)))
        elif stage=='scripts.run_speech':
            target=Path(arg('--out'));write(target,[SpeechEvent(200,700,900,'这是这个物品').to_dict()])
            write(target.with_suffix('.windows.jsonl'),[])
            target.with_suffix('.meta.json').write_text(json.dumps(dict(event_count=1,mode='causal',compute_rtf=.1)))
        elif stage=='scripts.enrich_speech':
            target=Path(arg('--out'));write(target,[SpeechEvent(200,700,1000,'这是这个物品',semantics=dict(
                 act='INTRODUCE',current=True,referent_scope='mentioned',classes=[],object_phrase='这个物品',colors=[],
                 quote='这是这个物品',confidence='high')).to_dict()])
            write(target.with_suffix('.semantic.jsonl'),[])
            target.with_suffix('.meta.json').write_text(json.dumps(dict(event_count=1,mode='causal',compute_rtf=.1)))
        elif stage=='src.intent.complementarity':
            result,queue=analyze(out,review=None);target=Path(arg('--out'));target.mkdir()
            (target/'complementarity.json').write_text(json.dumps(result))
            (target/'review_windows.csv').write_text('review_status\nPENDING\n')
        else:
            raise AssertionError(stage)
        return SimpleNamespace(returncode=0)
    monkeypatch.setattr(runner,'ROOT',root)
    monkeypatch.setattr(runner.subprocess,'run',fake)
    argv=['run','--python',str(python),'--input-dir',str(inputs),'--out',str(out),
          '--vision-model',str(vision),'--speech-model',str(speech)]
    monkeypatch.setattr(sys,'argv',argv)
    runner.main()
    result=json.loads((out/'analysis/complementarity.json').read_text())
    assert result['status']=='DIAGNOSTICS_NOT_PROOF_OF_IMPROVEMENT'
    methods=result['clips'][0]['methods']
    assert methods['showintent']['showing_ms']>0
    assert methods['visual-only']['showing_ms']==0
    assert methods['speech-with-localization']['showing_ms']==0
    assert methods['showintent']['selected_correct_class_coverage'] is None
    count=len(calls)
    monkeypatch.setattr(sys,'argv',argv+['--resume']);runner.main()
    assert len(calls)==count  # All complete model stages and B outputs are reused.

"""Resumable user-launched stronger perception/ASR/semantics and peer B replay."""

import argparse
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

from src.intent.fusion import FusionConfig
from src.intent.temporal import IntentConfig
from src.intent.replay_real import replay_clip,load_and_check

ROOT=Path(__file__).resolve().parents[1]


def digest(path):
    h=hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda:stream.read(4*1024*1024),b''):h.update(chunk)
    return h.hexdigest()


def plans_for(video,out,python,*,vision_model,speech_model,semantic_model=None):
    clip=video.stem
    per=out/'perception'/f'perception_{clip}.jsonl'
    raw=out/'speech_raw'/f'speech_events_{clip}.jsonl'
    speech=out/'speech'/f'speech_events_{clip}.jsonl'
    timing=out/'timestamps'/f'{clip}.json'
    metrics=per.with_name(per.stem+'_metrics.json')
    return {
        'timestamps':([str(python),'-B','-m','scripts.extract_video_timestamps','--input',str(video),'--out',str(timing)],[timing]),
        'perception':([str(python),'-B','-m','src.perception.enhanced','--input',str(video),'--timestamp-map',str(timing),
                       '--out',str(per),'--metrics-out',str(metrics),'--yolo-model',str(vision_model)],[per,metrics]),
        'speech_raw':([str(python),'-B','-m','scripts.run_speech','--audio',str(video),'--model',str(speech_model),
                      '--out',str(raw),'--mode','causal','--window-ms','6000','--context-ms','1500',
                      '--device','cuda','--compute-type','float16','--beam-size','5','--language','zh'],
                      [raw,raw.with_suffix('.windows.jsonl'),raw.with_suffix('.meta.json')]),
        'semantics':([str(python),'-B','-m','scripts.enrich_speech','--events',str(raw),'--perception',str(per),
                      '--backend','action','--out',str(speech)],
                     [speech,speech.with_suffix('.semantic.jsonl'),speech.with_suffix('.meta.json')]),
    }


def checked_stage(out,clip,stage,command,artifacts,env,action=None,inputs=()):
    marker=out/'checkpoints'/f'{clip}_{stage}.json'
    input_hashes={str(p):digest(p) for p in inputs}
    for flag in ('--input','--timestamp-map','--audio','--events','--perception'):
        if flag in command:
            p=Path(command[command.index(flag)+1])
            input_hashes[str(p)]=digest(p)
    if marker.exists():
        recorded=json.loads(marker.read_text())
        if (recorded['command']!=command or recorded.get('inputs')!=input_hashes or
                recorded['outputs']!={str(p.relative_to(out)):digest(p) for p in artifacts if p.is_file()}):
            raise ValueError(f'{clip}/{stage}: completed artifacts changed; use a new run directory')
        print(f'{clip}: reuse verified {stage}',flush=True);return
    partial=[p for p in artifacts if p.exists()]
    if partial:
        quarantine=out/'incomplete'/f'{clip}_{stage}_{time.time_ns()}'
        quarantine.mkdir(parents=True)
        for p in partial:shutil.move(str(p),str(quarantine/p.name))
    log=out/'logs'/f'{clip}_{stage}_{time.time_ns()}.log'
    print(f'{clip}: {stage}; log={log}',flush=True)
    if action is None:
        with log.open('w',encoding='utf-8') as stream:
            subprocess.run(command,cwd=ROOT,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
    else:
        action()
    if any(not p.is_file() for p in artifacts):raise ValueError(f'{clip}/{stage}: missing outputs')
    record=dict(command=command,inputs=input_hashes,outputs={str(p.relative_to(out)):digest(p) for p in artifacts})
    temp=marker.with_suffix('.partial');temp.write_text(json.dumps(record,indent=2)+'\n');temp.replace(marker)


def main():
    parser=argparse.ArgumentParser(description='以物体实例和展示过程组织视觉/语言，默认处理 Example_Data1 全部视频')
    parser.add_argument('--input-dir',type=Path,default=Path('Example_Data1'))
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--python',type=Path,default=Path(sys.executable))
    parser.add_argument('--vision-model',type=Path,default=Path('models/yolo11m.pt'))
    parser.add_argument('--speech-model',type=Path,default=Path('models/faster-whisper-large-v3'))
    parser.add_argument('--semantic-model',type=Path,default=None,help=argparse.SUPPRESS)
    parser.add_argument('--clips',nargs='+',help='Optional stems; otherwise every MP4, without category/scenario configuration')
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--dry-run',action='store_true')
    args=parser.parse_args()
    out=args.out.absolute();source=args.input_dir.absolute();python=args.python.absolute()
    if not out.resolve().is_relative_to((ROOT/'outputs').resolve()):parser.error('--out must be under project outputs/')
    videos=sorted(source.glob('*.mp4'))
    if args.clips:
        requested=set(args.clips);videos=[v for v in videos if v.stem in requested]
        if requested-{v.stem for v in videos}:parser.error('requested clip not found')
    if not videos:parser.error('no MP4 videos found')
    models=dict(vision_model=args.vision_model.absolute(),speech_model=args.speech_model.absolute())
    plans={v.stem:plans_for(v,out,python,**models) for v in videos}
    if args.dry_run:
        print(json.dumps({clip:{stage:command for stage,(command,_) in stages.items()} for clip,stages in plans.items()},indent=2));return
    if out.exists() and not args.resume:parser.error('output exists; use --resume or choose a new directory')
    if not python.is_file():parser.error('joint interpreter missing; prepare the separate environment first')
    for p in (models['vision_model'],models['speech_model']/'model.bin'):
        if not p.is_file():parser.error(f'local weights missing: {p}')
    required=[]
    required += [models['speech_model']/name for name in ('config.json','tokenizer.json','vocabulary.json')]
    if any(not p.is_file() for p in required):
        parser.error('model download is incomplete; wait for verified download completion before starting inference')
    env=os.environ.copy();env.update(HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',YOLO_AUTOINSTALL='false',
                                    YOLO_CONFIG_DIR=str(out/'ultralytics_config'))
    # CTranslate2 needs CUDA libraries at process startup, independent of torch.
    lib_query=subprocess.run([str(python),'-c',
          'import importlib.util,pathlib; print(":".join(str(pathlib.Path(s.origin).parent/"lib") for x in ("nvidia.cublas","nvidia.cudnn") if (s:=importlib.util.find_spec(x)) and s.origin))'],
          capture_output=True,text=True,check=True,cwd=ROOT)
    env['LD_LIBRARY_PATH']=lib_query.stdout.strip()+':'+env.get('LD_LIBRARY_PATH','')
    subprocess.run([str(python),'-B','-c',
          'import torch,cv2,mediapipe,av,faster_whisper,ctranslate2; '
          'assert torch.cuda.is_available(), "CUDA device unavailable; launch on your allocated compute node"; '
          'assert "float16" in ctranslate2.get_supported_compute_types("cuda"), "CTranslate2 CUDA/float16 unavailable"; '
          'print("GPU:",torch.cuda.get_device_name(0))'],cwd=ROOT,env=env,check=True)
    config=json.loads((ROOT/'configs/presentation_intent.json').read_text());IntentConfig(**config)
    fusion=FusionConfig(**json.loads((ROOT/'configs/stable_fusion.json').read_text()))
    provenance=dict(version='complementary_instances_v1',clips=[v.stem for v in videos],mode='causal',
            visual_config=config,fusion_config=asdict(fusion),
            source_sha256={str(p.relative_to(ROOT)):digest(p) for folder in ('src','scripts')
                           for p in sorted((ROOT/folder).rglob('*.py'))},
            speech_lexicon_sha256=digest(ROOT/'configs/speech_lexicon.json'),
            video_sha256={v.name:digest(v) for v in videos},
            model_sha256={str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else p.name:digest(p)
              for m in models.values() for p in ([m] if m.is_file() else sorted(m.glob('*')))
              if p.is_file() and not p.name.startswith('download_manifest') and not p.name.endswith('.partial')},
            plans={c:{s:cmd for s,(cmd,_) in stages.items()} for c,stages in plans.items()})
    run_config=out/'run_config.json'
    if out.exists():
        if not run_config.is_file() or json.loads(run_config.read_text())!=provenance:
            parser.error('run inputs/code/config changed; choose a new directory instead of resuming')
    else:
        out.mkdir(parents=True)
        for folder in ('logs','checkpoints','perception','results','results_rules'): (out/folder).mkdir()
        run_config.write_text(json.dumps(provenance,ensure_ascii=False,indent=2)+'\n')
    for video in videos:
        clip=video.stem
        for stage,(command,artifacts) in plans[clip].items():
            checked_stage(out,clip,stage,command,artifacts,env)
        speech_meta=json.loads((out/'speech'/f'speech_events_{clip}.meta.json').read_text())
        print(f'{clip}: {speech_meta["event_count"]} ASR events; '
              f'{speech_meta.get("semantic_fallback_count",0)} semantic rule fallbacks',flush=True)
        per=out/'perception'/f'perception_{clip}.jsonl'
        rows,quality=load_and_check(per)
        metrics=json.loads(per.with_name(per.stem+'_metrics.json').read_text())
        if len(rows)!=metrics['frames_processed']:raise ValueError('perception completion count mismatch')
        timing=json.loads((out/'timestamps'/f'{clip}.json').read_text())
        if [p['timestamp_ms'] for p in rows]!=timing['timestamp_ms']:raise ValueError('perception differs from decoded PTS')
        marker=out/'perception'/f'completion_{clip}.json'
        if not marker.exists():marker.write_text(json.dumps(dict(source_kind='real_video_full_clip',quality=quality,
            perception_sha256=digest(per),video_sha256=provenance['video_sha256'][video.name]),indent=2)+'\n')
        for kind,speech_folder,ablate in (('results','speech',True),('results_rules','speech_raw',False)):
            destination=out/kind/clip
            modes=['visual-only','showintent','pickup-baseline']+(['speech-with-localization','visual-policy-only'] if ablate else [])
            artifacts=[destination/f'intent_{m}.jsonl' for m in modes]+[destination/'report.json',destination/'evidence_showintent.jsonl']
            checked_stage(out,clip,kind,['B-replay',kind],artifacts,env,action=lambda d=destination,s=speech_folder,a=ablate:
                replay_clip(per,d,config,speech_source=out/s/f'speech_events_{clip}.jsonl',fusion_config=fusion,
                            include_speech_ablation=a),inputs=[per,out/speech_folder/f'speech_events_{clip}.jsonl'])
    command=[str(python),'-B','-m','src.intent.complementarity','--run-dir',str(out),'--out',str(out/'analysis')]
    checked_stage(out,'all','analysis',command,[out/'analysis/complementarity.json',out/'analysis/review_windows.csv'],env,
                  inputs=[p for p in (out/'results').rglob('*.jsonl')])
    print(f'Completed: {out}. Extra triggers are pending review, not proven gains.',flush=True)


if __name__=='__main__':main()

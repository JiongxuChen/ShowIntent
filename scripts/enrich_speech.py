"""User-launched local semantic inference, preserving measured availability."""

import argparse
from dataclasses import replace
import hashlib
import json
import math
from pathlib import Path
import time

from src.speech.bridge import read_events_jsonl
from scripts.run_speech import write_jsonl


def enrich_events(events, model, classes, *, clock=time.perf_counter):
    result=[]; diagnostics=[]; context=[]; ready=0
    for e in events:
        start=clock(); error=None
        try:
            vocabulary=classes(e) if callable(classes) else classes
            semantics=(model.analyze_event(e) if hasattr(model,'analyze_event') else
                       model.analyze(e.text,context,vocabulary))
        except (ValueError, KeyError, TypeError) as exc:
            semantics=None; error=str(exc)
        compute=math.ceil((clock()-start)*1000)
        if compute<0:raise ValueError('semantic processing clock went backwards')
        ready=max(e.available_at_ms,ready)+compute
        result.append(replace(e,semantics=semantics,available_at_ms=ready))
        diagnostics.append(dict(start_ms=e.start_ms,end_ms=e.end_ms,asr_available_at_ms=e.available_at_ms,
                                available_at_ms=ready,compute_ms=compute,error=error,semantics=semantics,
                                raw_model_output=getattr(model,'last_output',None)))
        context.append(e.text)
    return result,diagnostics


def main():
    parser=argparse.ArgumentParser(description='规则语言对照入口；模型讲解理解使用 scripts.run_semantic_dialogue')
    parser.add_argument('--events',type=Path,required=True)
    parser.add_argument('--perception',type=Path,required=True)
    parser.add_argument('--model',type=Path,default=None,help=argparse.SUPPRESS)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--device',choices=('cuda','cpu'),default='cuda')
    parser.add_argument('--backend',choices=('action',),default='action')
    args=parser.parse_args()
    targets=[args.out,args.out.with_suffix('.semantic.jsonl'),args.out.with_suffix('.meta.json')]
    if len(set(p.resolve() for p in targets))!=3 or any(p.exists() for p in targets):
        parser.error('semantic outputs already exist or are not distinct; choose a new .jsonl path')
    events=read_events_jsonl(args.events)
    packets=[json.loads(line) for line in args.perception.read_text().splitlines() if line.strip()]
    metrics=args.perception.with_name(args.perception.stem+'_metrics.json')
    vocabulary=json.loads(metrics.read_text()).get('detector_classes') if metrics.is_file() else None
    def classes_at(e):
        # A declared model vocabulary is time independent. With older logs use
        # only classes already observed when this ASR result became available.
        return sorted(set(vocabulary or [o['class_name'] for p in packets if p['timestamp_ms']<=e.available_at_ms
                                         for o in p['objects']])-{'person'})
    if events:
        from src.speech.intent_language import PresentationLanguage
        model=PresentationLanguage()
        enriched,diagnostics=enrich_events(events,model,classes_at)
    else:
        enriched,diagnostics=[],[]
    write_jsonl(args.out,(e.to_dict() for e in enriched))
    write_jsonl(targets[1],diagnostics)
    meta=json.loads(args.events.with_suffix('.meta.json').read_text())
    meta.update(semantic_model=None,semantic_observation_version=1,
                language_backend=args.backend,
                semantic_input_sha256=hashlib.sha256(args.events.read_bytes()).hexdigest(),
                semantic_fallback_count=sum(d['error'] is not None for d in diagnostics),
                total_semantic_compute_ms=sum(d['compute_ms'] for d in diagnostics),
                event_count=len(enriched))
    targets[2].write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(f'{len(enriched)} semantic events; {meta["semantic_fallback_count"]} rule fallbacks',flush=True)


if __name__=='__main__':main()

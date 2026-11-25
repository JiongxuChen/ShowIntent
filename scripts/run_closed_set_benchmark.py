"""Controlled detector comparison: fixed A postprocessing, B and saved D; user launches GPU work."""
import argparse,csv,hashlib,json,os,subprocess,sys
from pathlib import Path
from scripts.run_complementary import checked_stage,digest
from src.intent.replay_real import replay_clip,load_and_check
from src.intent.fusion import FusionConfig
from src.intent.formal_evaluation import read_csv,validate_labels,score_clip,prf
from src.evaluation.collection import check_annotations,summarize

ROOT=Path(__file__).resolve().parents[1]


def inputs_for(row,raw_speech=False):
    source=Path({'fruit_original':'outputs/cache/fruit','books_original':'outputs/cache/books',
                 'cups_new':'outputs/cache/cups','drinks_new':'outputs/cache/drinks'}[row['source_collection']])
    speech=(source/'speech_raw')/f'speech_events_{row["clip_id"]}.jsonl'
    return source/'timestamps'/f'{row["clip_id"]}.json',speech


def load_collections(names):
    rows=[];labels=[];splits={};files=set()
    for name in names:
        dataset=Path('evaluation')/name
        manifest=read_csv(dataset/'dataset_manifest.csv');annotations=read_csv(dataset/'annotations.csv');splitrows=read_csv(dataset/'split_manifest.csv')
        if name in {'cups','drinks'}:check_annotations(dataset)
        else:
            errors=validate_labels(annotations,[],manifest,splitrows)
            if errors:raise ValueError(str(errors))
        kind={'fruit':'fruit_original','books':'books_original','cups':'cups_new','drinks':'drinks_new'}[name]
        selected=[dict(r,benchmark_collection=name) for r in manifest if r['source_collection']==kind]
        if not selected:raise ValueError('Collection contains no clips: '+name)
        ids={r['clip_id'] for r in selected}
        if ids & set(splits):raise ValueError('Duplicate clip IDs across collections')
        rows.extend(selected);labels.extend(r for r in annotations if r['clip_id'] in ids)
        splits.update({r['clip_id']:r['split'] for r in splitrows if r['clip_id'] in ids})
        files.update(dataset/f for f in ['dataset_manifest.csv','annotations.csv','split_manifest.csv','protocol.json'])
        if (dataset/'annotation_record.json').is_file():files.add(dataset/'annotation_record.json')
    return rows,labels,splits,sorted(files)


def replay_persistent(per,target,visual,speech,fusion,memory):
    from scripts.run_refined_b import replay
    from src.speech.bridge import read_events_jsonl
    packets,_=load_and_check(per)
    streams,observations,traces=replay(packets,read_events_jsonl(speech),visual,fusion,memory)
    target.mkdir(parents=True,exist_ok=True)
    for method,values in streams.items():
        (target/f'intent_{method}.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in values))
    (target/'speech_used.jsonl').write_text(''.join(json.dumps(v.to_dict(),ensure_ascii=False)+'\n' for v in observations))
    (target/'evidence_showintent.jsonl').write_text(''.join(json.dumps(v,ensure_ascii=False)+'\n' for v in traces))
    (target/'report.json').write_text(json.dumps(dict(profile='persistent_B_v1',hand_memory_ms=memory,frames=len(packets),timing='RETROSPECTIVE_CONTENT',speech_sha256=digest(speech)),indent=2)+'\n')


def command_for(row,model,config,out,python,device):
    c=row['clip_id'];timestamps,_=inputs_for(row);dest=out/'perception'
    args=[str(python),'-B','-m','src.perception.enhanced','--input',str(Path(row['source_path']).resolve()),'--timestamp-map',str(timestamps.resolve()),'--out',str(dest/f'perception_{c}.jsonl'),'--metrics-out',str(dest/f'perception_{c}_metrics.json'),'--yolo-model',str(Path(model['weights']).resolve()),'--detector-backend',model['backend'],'--device',device,'--imgsz',str(config['imgsz']),'--crop-imgsz',str(config['crop_imgsz']),'--confidence',str(config['confidence'])]
    if not config['hand_crops']:args.append('--no-hand-crops')
    return args


def evaluate_model(out,rows,labels,splits,*,profile='baseline',hand_memory_ms=None):
    reports={};maps=[];processing={}
    for row in rows:
        c=row['clip_id'];packets,_=load_and_check(out/'perception'/f'perception_{c}.jsonl');processing[c]=json.loads((out/'perception'/f'perception_{c}_metrics.json').read_text());reports[c]={}
        mapping_ids=set();lookup={}
        for method in ['visual-only','showintent']:
            path=out/'results'/c/f'intent_{method}.jsonl';intents=[json.loads(l) for l in path.read_text().splitlines()]
            mapping_ids.update(i['target_id'] for i in intents if i['state']=='SHOWING')
            temporal=[dict(timestamp_ms=i['timestamp_ms'],state=i['state'],target_id='TEMPORAL_ONLY' if i['state']=='SHOWING' else None) for i in intents]
            score=score_clip(packets,temporal,[r for r in labels if r['clip_id']==c],[])
            reports[c][method]={k:score[k] for k in ['temporal_events','positive_ms','negative_ms','false_showing_ms','showing_ms','matched_boundary_errors']}
        for p in packets:
            for o in p['objects']:
                if o['track_id'] in mapping_ids:
                    key=o['track_id'];v=lookup.setdefault(key,dict(clip_id=c,track_id=key,start_ms=p['timestamp_ms'],end_ms=p['timestamp_ms'],physical_target_id='',review_status='PENDING',annotator='',reviewer='',notes='New detector ID; independently verify physical identity, including switches'))
                    v['end_ms']=min(int(row['eval_end_ms']),p['timestamp_ms']+1)
        maps.extend(lookup.values())
    with (out/'track_mapping.csv').open('w',newline='') as f:
        fields=['clip_id','track_id','start_ms','end_ms','physical_target_id','review_status','annotator','reviewer','notes'];w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(maps)
    summary={}
    for split in ['development','test']:
        summary[split]={}
        for method in ['visual-only','showintent']:
            selected=[reports[c][method] for c in reports if splits[c]==split]
            tp=sum(r['temporal_events']['tp'] for r in selected);fp=sum(r['temporal_events']['fp'] for r in selected);fn=sum(r['temporal_events']['fn'] for r in selected)
            pos=sum(r['positive_ms'] for r in selected);neg=sum(r['negative_ms'] for r in selected);false=sum(r['false_showing_ms'] for r in selected)
            errors=[e for r in selected for e in r['matched_boundary_errors']]
            summary[split][method]=dict(clips=len(selected),events=prf(tp,tp+fp,tp+fn),temporal_coverage=sum(r['showing_ms']-r['false_showing_ms'] for r in selected)/pos if pos else None,false_seconds=false/1000,false_seconds_per_negative_minute=60*false/neg if neg else None,
                boundary_mae_ms={kind:sum(abs(e[kind+'_error_ms']) for e in errors if e[kind+'_error_ms'] is not None)/max(1,sum(e[kind+'_error_ms'] is not None for e in errors)) if any(e[kind+'_error_ms'] is not None for e in errors) else None for kind in ['start','end']})
    by_collection={}
    for name in sorted({r.get('benchmark_collection',r.get('source_collection','unspecified')) for r in rows}):
        ids=[r['clip_id'] for r in rows if r.get('benchmark_collection',r.get('source_collection','unspecified'))==name]
        by_collection[name]={m:dict(clips=len(ids),**summarize([reports[c][m] for c in ids])) for m in ['visual-only','showintent']}
    by_collection['pooled']={m:dict(clips=len(reports),**summarize([r[m] for r in reports.values()])) for m in ['visual-only','showintent']}
    report=dict(scope='BINARY_TEMPORAL_DETECTOR_COMPARISON_NOT_mAP_NOT_INSTANCE_ACCURACY',test_independence='Previously used for development; new scenes have assistant labels pending user review',clips=reports,summary=summary,by_collection=by_collection,processing_metrics=processing,physical_metrics=None,physical_metric_status='NEW_DETECTOR_MAPPING_PENDING')
    report['annotation_review_status']={c:sorted({r.get('review_status','UNSPECIFIED') for r in labels if r['clip_id']==c}) for c in reports}
    report['b_profile']=profile
    report['hand_memory_ms']=hand_memory_ms
    (out/'evaluation.json').write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n');return summary


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True);p.add_argument('--models',nargs='+');p.add_argument('--clips',nargs='+');p.add_argument('--collections',nargs='+',choices=['fruit','books','cups','drinks'],default=['fruit','books']);p.add_argument('--intent-profile',choices=['baseline','persistent']);p.add_argument('--hand-memory-ms',type=int,default=3000);p.add_argument('--device',default='0');p.add_argument('--resume',action='store_true');p.add_argument('--dry-run',action='store_true');p.add_argument('--check-only',action='store_true');args=p.parse_args()
    os.chdir(ROOT)
    if len(set(args.collections))!=len(args.collections) or args.hand_memory_ms<=0:p.error('invalid collections/memory')
    profile=args.intent_profile or ('persistent' if set(args.collections)&{'cups','drinks'} else 'baseline')
    cfg=json.loads(Path('configs/closed_set_benchmark.json').read_text());selected=args.models or list(cfg['models']);rows,labels,splits,dataset_files=load_collections(args.collections)
    if not selected or len(set(selected))!=len(selected) or set(selected)-set(cfg['models']):p.error('invalid model selection')
    if args.clips and (len(set(args.clips))!=len(args.clips) or set(args.clips)-{r['clip_id'] for r in rows}):p.error('invalid clips; excluded clips cannot be selected')
    rows=[r for r in rows if not args.clips or r['clip_id'] in args.clips]
    out=args.out.resolve();python=Path(sys.executable)
    if not out.is_relative_to(ROOT/'outputs'):p.error('output must be under outputs')
    plans={m:{r['clip_id']:command_for(r,cfg['models'][m],cfg,out/m,python,args.device) for r in rows} for m in selected}
    if args.dry_run:print(json.dumps(plans,ensure_ascii=False,indent=2));return
    for m in selected:
        if not Path(cfg['models'][m]['weights']).is_file():p.error('missing weight: '+m)
    for row in rows:
        if not Path(row['source_path']).is_file() or not all(x.is_file() for x in inputs_for(row,raw_speech=profile=='persistent')):p.error('video/timestamp/saved speech missing: '+row['clip_id'])
    if args.check_only:
        from src.perception.closed_set import ClosedSetModels
        import ultralytics
        for m in selected:
            model=cfg['models'][m];det=ClosedSetModels(model['weights'],cfg['confidence'],False,backend=model['backend'],device='cpu');assert len(det.yolo.names)==80;assert det.yolo.names[73]=='book';print(m,'weight load and 80 COCO classes OK',flush=True);del det
        print('ultralytics',ultralytics.__version__,';',len(rows),'clips; B profile:',profile,'; no GPU inference performed');return
    env=os.environ.copy();env.update(YOLO_AUTOINSTALL='false',YOLO_CONFIG_DIR=str(out/'ultralytics_config'),HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1')
    subprocess.run([str(python),'-c','import torch; assert torch.cuda.is_available(), "GPU required"'],check=True,env=env)
    import ultralytics
    record=dict(config=cfg,models=selected,collections=args.collections,intent_profile=profile,hand_memory_ms=args.hand_memory_ms if profile=='persistent' else None,clips=[r['clip_id'] for r in rows],device=args.device,ultralytics=ultralytics.__version__,source_sha256={str(x):digest(x) for folder in ['src','scripts'] for x in Path(folder).rglob('*.py')},config_sha256={str(x):digest(x) for x in [Path('configs/presentation_intent.json'),Path('configs/stable_fusion.json'),Path('configs/speech_lexicon.json')]},dataset_sha256={str(x):digest(x) for x in dataset_files},model_sha256={m:digest(Path(cfg['models'][m]['weights'])) for m in selected},video_sha256={r['clip_id']:digest(Path(r['source_path'])) for r in rows},speech_sha256={r['clip_id']:digest(inputs_for(r,raw_speech=profile=='persistent')[1]) for r in rows},timestamp_sha256={r['clip_id']:digest(inputs_for(r)[0]) for r in rows})
    if out.exists():
        if not args.resume or not (out/'run_config.json').is_file() or json.loads((out/'run_config.json').read_text())!=record:p.error('existing output or changed inputs; choose a new directory')
    else:out.mkdir(parents=True);(out/'run_config.json').write_text(json.dumps(record,ensure_ascii=False,indent=2)+'\n')
    visual=json.loads(Path('configs/presentation_intent.json').read_text());fusion=FusionConfig(**json.loads(Path('configs/stable_fusion.json').read_text()));all_summary={};scene_summary={}
    for m in selected:
        dest=out/m
        for folder in ['logs','checkpoints','perception','results']:(dest/folder).mkdir(parents=True,exist_ok=True)
        for row in rows:
            c=row['clip_id'];per=dest/'perception'/f'perception_{c}.jsonl';metrics=dest/'perception'/f'perception_{c}_metrics.json'
            checked_stage(dest,c,'perception',plans[m][c],[per,metrics],env,inputs=[Path(cfg['models'][m]['weights'])])
            packets,_=load_and_check(per);timing=json.loads(inputs_for(row)[0].read_text())
            if [v['timestamp_ms'] for v in packets]!=timing['timestamp_ms']:raise ValueError('incomplete or shifted perception')
            target=dest/'results'/c;speech=inputs_for(row,raw_speech=profile=='persistent')[1]
            if profile=='persistent':
                artifacts=[target/f'intent_{x}.jsonl' for x in ['visual-only','showintent','audio-only']]+[target/'report.json',target/'speech_used.jsonl',target/'evidence_showintent.jsonl']
                action=lambda:replay_persistent(per,target,visual,speech,fusion,args.hand_memory_ms)
            else:
                artifacts=[target/f'intent_{x}.jsonl' for x in ['visual-only','showintent','pickup-baseline']]+[target/'report.json',target/'evidence_showintent.jsonl']
                action=lambda:replay_clip(per,target,visual,speech_source=speech,fusion_config=fusion)
            checked_stage(dest,c,'intent',['fixed-B-D-replay',profile,str(args.hand_memory_ms) if profile=='persistent' else ''],artifacts,env,action=action,inputs=[per,speech])
        all_summary[m]=evaluate_model(dest,rows,labels,splits,profile=profile,hand_memory_ms=args.hand_memory_ms if profile=='persistent' else None)
        scene_summary[m]=json.loads((dest/'evaluation.json').read_text())['by_collection']
        (out/'summary.json').write_text(json.dumps(all_summary,ensure_ascii=False,indent=2)+'\n')
        (out/'scenes_summary.json').write_text(json.dumps(scene_summary,ensure_ascii=False,indent=2)+'\n')
        print(m,'completed and scored',flush=True)

if __name__=='__main__':main()

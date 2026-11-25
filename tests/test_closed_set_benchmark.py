import csv,json
from pathlib import Path
from scripts.run_closed_set_benchmark import command_for,evaluate_model,inputs_for,replay_persistent
from src.intent.fusion import FusionConfig


def test_detector_plan_preserves_shared_controls_and_selects_native_backend(tmp_path):
    row=dict(clip_id='demo',source_path='Example_Data1/demo.mp4',source_collection='fruit_original',result_root='outputs/stable_fruit_current')
    cfg=dict(imgsz=1280,crop_imgsz=960,confidence=.25,hand_crops=True)
    for backend in ['yolo','rtdetr']:
        cmd=command_for(row,dict(weights='models/demo.pt',backend=backend),cfg,tmp_path,Path('/python'),'0')
        assert cmd[cmd.index('--detector-backend')+1]==backend
        assert cmd[cmd.index('--imgsz')+1]=='1280' and cmd[cmd.index('--crop-imgsz')+1]=='960'
        assert '--no-hand-crops' not in cmd
    cfg['hand_crops']=False
    assert '--no-hand-crops' in command_for(row,dict(weights='models/demo.pt',backend='yolo'),cfg,tmp_path,Path('/python'),'0')


def test_model_evaluation_keeps_temporal_match_distinct_from_instance_accuracy(tmp_path):
    for folder in ['perception','results/demo']:(tmp_path/folder).mkdir(parents=True)
    frames=[dict(timestamp_ms=t,frame_id=i,objects=[],hands=[],relations=[]) for i,t in enumerate([0,100,200,300])]
    (tmp_path/'perception/perception_demo.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in frames))
    (tmp_path/'perception/perception_demo_metrics.json').write_text('{}')
    intents=[dict(timestamp_ms=t,state='SHOWING' if t==100 else 'NONE',target_id='unverified_detector_id' if t==100 else None) for t in [0,100,200,300]]
    for method in ['visual-only','showintent']:(tmp_path/f'results/demo/intent_{method}.jsonl').write_text(''.join(json.dumps(p)+'\n' for p in intents))
    labels=[dict(clip_id='demo',event_id=str(a),start_ms=a,end_ms=b,label=kind,physical_target_id='real_book' if kind=='SHOWING' else '',class_name='book' if kind=='SHOWING' else '') for a,b,kind in [(0,100,'NON_SHOWING'),(100,200,'SHOWING'),(200,300,'NON_SHOWING')]]
    summary=evaluate_model(tmp_path,[dict(clip_id='demo',eval_end_ms='300')],labels,{'demo':'development'})
    assert summary['development']['showintent']['events']['tp']==1
    report=json.loads((tmp_path/'evaluation.json').read_text())
    assert report['physical_metrics'] is None and report['physical_metric_status']=='NEW_DETECTOR_MAPPING_PENDING'
    assert list(csv.DictReader((tmp_path/'track_mapping.csv').open()))==[]


def test_new_scene_inputs_use_their_own_timestamps_and_raw_speech():
    for collection,name in [('cups_new','cups'),('drinks_new','drinks')]:
        row=dict(clip_id='unique',source_collection=collection,result_root=f'outputs/cache/{name}')
        timestamps,speech=inputs_for(row,raw_speech=True)
        assert timestamps==Path(f'outputs/cache/{name}/timestamps/unique.json')
        assert speech==Path(f'outputs/cache/{name}/speech_raw/speech_events_unique.jsonl')
        assert inputs_for(row)[1]==Path(f'outputs/cache/{name}/speech_raw/speech_events_unique.jsonl')


def test_persistent_replay_uses_shared_engine_and_preserves_no_trigger_negative(tmp_path):
    per=tmp_path/'perception.jsonl';speech=tmp_path/'speech.jsonl';target=tmp_path/'results'
    frames=[dict(timestamp_ms=t,frame_id=i,objects=[],hands=[],relations=[]) for i,t in enumerate([0,100,200,300])]
    per.write_text(''.join(json.dumps(p)+'\n' for p in frames));speech.write_text('')
    replay_persistent(per,target,{},speech,FusionConfig(),3000)
    for method in ['visual-only','showintent','audio-only']:
        values=[json.loads(l) for l in (target/f'intent_{method}.jsonl').read_text().splitlines()]
        assert [v['timestamp_ms'] for v in values]==[0,100,200,300]
        assert all(v['state']!='SHOWING' for v in values)
    assert json.loads((target/'report.json').read_text())['profile']=='persistent_B_v1'


def test_evaluation_reports_scene_groups_without_requiring_human_approval(tmp_path):
    rows=[];labels=[];splits={}
    for c,name,state in [('a','cups','SHOWING'),('b','drinks','NON_SHOWING')]:
        (tmp_path/'perception').mkdir(exist_ok=True);dest=tmp_path/'results'/c;dest.mkdir(parents=True)
        frames=[dict(timestamp_ms=t,frame_id=i,objects=[],hands=[],relations=[]) for i,t in enumerate([0,100,200])]
        (tmp_path/'perception'/f'perception_{c}.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in frames))
        (tmp_path/'perception'/f'perception_{c}_metrics.json').write_text('{}')
        values=[dict(timestamp_ms=t,state='SHOWING' if state=='SHOWING' else 'NONE',target_id='unmapped' if state=='SHOWING' else None) for t in [0,100,200]]
        for m in ['visual-only','showintent']:(dest/f'intent_{m}.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
        rows.append(dict(clip_id=c,eval_end_ms=200,benchmark_collection=name));splits[c]='test'
        labels.append(dict(clip_id=c,event_id=c,start_ms=0,end_ms=200,label=state,physical_target_id='real_cup' if state=='SHOWING' else '',class_name='cup' if state=='SHOWING' else '',review_status='PENDING'))
    evaluate_model(tmp_path,rows,labels,splits,profile='persistent',hand_memory_ms=3000)
    r=json.loads((tmp_path/'evaluation.json').read_text())
    assert r['by_collection']['cups']['showintent']['events']['tp']==1
    assert r['by_collection']['drinks']['showintent']['temporal_coverage'] is None
    assert r['by_collection']['pooled']['showintent']['clips']==2
    assert r['b_profile']=='persistent' and r['hand_memory_ms']==3000
    assert r['annotation_review_status']=={'a':['PENDING'],'b':['PENDING']}

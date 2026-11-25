import pytest
from src.intent.formal_evaluation import score_clip,match_events,iou,validate_labels


def fixture(states,ids=None):
    p=[dict(timestamp_ms=t,objects=[]) for t in range(0,len(states)*100,100)]
    i=[dict(timestamp_ms=r['timestamp_ms'],state=s,target_id=(ids[n] if ids else 'x')) for n,(r,s) in enumerate(zip(p,states))]
    return p,i


def label(a,b,kind='SHOWING',target='real'):
    return dict(clip_id='c',event_id=str(a),start_ms=a,end_ms=b,label=kind,physical_target_id=target if kind=='SHOWING' else '',class_name='book' if kind=='SHOWING' else '',
        annotator='one',reviewer='two',confidence='HIGH',review_status='REVIEWED')


def mapping(track,target,a=0,b=1000):return dict(track_id=track,physical_target_id=target,start_ms=a,end_ms=b,review_status='REVIEWED')


def test_wrong_physical_target_is_not_correct_even_same_category():
    p,i=fixture(['NONE','SHOWING','SHOWING','SHOWING','NONE'])
    r=score_clip(p,i,[label(0,100,'NON_SHOWING'),label(100,400)], [mapping('x','other')])
    assert r['temporal_events']['tp']==1
    assert r['physical_events']['tp']==0 and r['wrong_target_ms']==300


def test_mapping_joins_id_fragments_but_splits_real_item_switch():
    p,i=fixture(['SHOWING']*5,['a','b','c','c','c'])
    labels=[label(0,200),label(200,400,target='second')]
    r=score_clip(p,i,labels,[mapping('a','real'),mapping('b','real'),mapping('c','second')])
    assert r['physical_events']['tp']==2 and len(r['predicted_episodes'])==2


def test_uncertain_and_last_packet_not_extrapolated():
    p,i=fixture(['SHOWING']*4)
    r=score_clip(p,i,[label(0,100,'UNCERTAIN'),label(100,300,'NON_SHOWING')],[])
    assert r['excluded_uncertain_ms']==100 and r['false_showing_ms']==200
    assert r['physical_events'] is None and r['unmapped_target_ms']==200


def test_fragments_are_false_positives_and_single_event_not_recalled_twice():
    p,i=fixture(['SHOWING','NONE','SHOWING','SHOWING','SHOWING'])
    r=score_clip(p,i,[label(0,400)],[mapping('x','real')],.2)
    assert r['temporal_events']['tp']==1 and r['temporal_events']['fp']==1
    assert r['extra_fragments']==1


def test_no_predictions_is_not_fake_perfect_precision():
    p,i=fixture(['NONE']*4)
    r=score_clip(p,i,[label(0,300)],[])
    assert r['temporal_events']['recall']==0 and r['temporal_events']['precision'] is None


def test_matching_is_max_cardinality_not_greedy():
    gt=[dict(start_ms=0,end_ms=5),dict(start_ms=4,end_ms=6)]
    pred=[dict(start_ms=0,end_ms=6),dict(start_ms=0,end_ms=2)]
    assert len(match_events(gt,pred,lambda a,b:iou(a,b)>.1))==2


def test_pending_labels_and_development_holdout_rejected():
    manifest=[dict(clip_id='c',eval_start_ms=0,eval_end_ms=1000,used_for_development='true',time_alignment_reviewed='false')]
    splits=[dict(clip_id='c',split='test',participant_id='p',session_id='s')]
    errors=validate_labels([],[],manifest,splits)
    assert any('incomplete' in e for e in errors) and any('development data' in e for e in errors)
    assert any('alignment' in e for e in errors)


def test_formal_cli_only_scores_reviewed_aligned_data(tmp_path,monkeypatch):
    import csv,json,hashlib,sys,shutil
    from pathlib import Path
    from scripts import evaluate_formal as runner
    data=tmp_path/'evaluation';data.mkdir()
    shutil.copyfile('evaluation/fruit/protocol.json',data/'protocol.json')
    p,i=fixture(['NONE','SHOWING','NONE','NONE'])
    per=tmp_path/'per.jsonl';per.write_text(''.join(json.dumps(r)+'\n' for r in p))
    result=tmp_path/'run/results/c';result.mkdir(parents=True)
    (result/'intent_showintent.jsonl').write_text(''.join(json.dumps(r)+'\n' for r in i))
    def write(name,rows):
        with (data/name).open('w',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    write('dataset_manifest.csv',[dict(clip_id='c',eval_start_ms=0,eval_end_ms=300,used_for_development='true',
        player_offset_ms=0,time_alignment_reviewed='true',perception_path=str(per),result_root=str(tmp_path/'run'),
        perception_sha256=hashlib.sha256(per.read_bytes()).hexdigest())])
    write('annotations.csv',[label(0,100,'NON_SHOWING'),label(100,200),label(200,300,'NON_SHOWING')])
    write('track_mapping.csv',[dict(mapping('x','real',0,300),clip_id='c',annotator='one',reviewer='two')])
    write('split_manifest.csv',[dict(clip_id='c',split='development',participant_id='p',session_id='s')])
    output=tmp_path/'scored.json'
    monkeypatch.setattr(sys,'argv',['run','--dataset',str(data),'--out',str(output)])
    runner.main();report=json.loads(output.read_text())
    assert report['status']=='SCORED'
    assert report['micro_summary']['physical_events']['tp']==1
    assert report['micro_summary']['start_mae_ms']==0
    monkeypatch.setattr(sys,'argv',['run','--dataset',str(data),'--temporal-only','--out',str(tmp_path/'temporal.json')])
    runner.main();temporal=json.loads((tmp_path/'temporal.json').read_text())
    assert temporal['micro_summary']['physical_events'] is None
    assert temporal['micro_summary']['correct_target_coverage'] is None
    assert temporal['clips']['c']['correct_target_ms'] is None
    assert temporal['micro_summary']['temporal_showing_coverage']==1.0
    annotations=[label(0,300,'UNCERTAIN')];annotations[0]['review_status']='PENDING'
    write('annotations.csv',annotations)
    monkeypatch.setattr(sys,'argv',['run','--dataset',str(data),'--out',str(tmp_path/'blocked.json')])
    with pytest.raises(SystemExit):runner.main()
    blocked=json.loads((tmp_path/'blocked.json').read_text())
    assert blocked['status']=='NOT_READY' and 'clips' not in blocked


def test_verified_original_pts_does_not_claim_browser_alignment():
    manifest=[dict(clip_id='c',eval_start_ms=0,eval_end_ms=100,used_for_development='true',
        time_alignment_reviewed='false',player_offset_ms='',annotation_timebase='original_media_pts',annotation_timebase_verified='true')]
    splits=[dict(clip_id='c',split='development',participant_id='',session_id='')]
    assert validate_labels([label(0,100)],[],manifest,splits)==[]
    manifest[0]['annotation_timebase_verified']='false'
    assert any('alignment' in e for e in validate_labels([label(0,100)],[],manifest,splits))


def test_assistant_identity_draft_is_opt_in_and_not_human_review():
    p,i=fixture(['SHOWING']*4)
    draft=dict(mapping('x','real'),review_status='PENDING',annotator='assistant_identity_review',reviewer='')
    assert score_clip(p,i,[label(0,300)],[draft])['physical_events'] is None
    assert score_clip(p,i,[label(0,300)],[draft],include_assistant_mappings=True)['physical_events']['tp']==1
    assert draft['review_status']=='PENDING' and draft['reviewer']==''


def test_user_designated_previously_developed_test_is_explicit():
    manifest=[dict(clip_id='c',eval_start_ms=0,eval_end_ms=100,used_for_development='true',annotation_timebase='original_media_pts',annotation_timebase_verified='true')]
    splits=[dict(clip_id='c',split='test',participant_id='',session_id='',allow_development_test='true')]
    assert validate_labels([label(0,100)],[],manifest,splits)==[]
    splits[0]['allow_development_test']='false'
    assert any('development data' in e for e in validate_labels([label(0,100)],[],manifest,splits))

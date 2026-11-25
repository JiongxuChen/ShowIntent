"""Human-reviewed event evaluation. Never reads labels during B inference."""
import csv,json
from pathlib import Path


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))


def validate_labels(annotations,mappings,manifest,splits):
    errors=[];byclip={r['clip_id']:[] for r in manifest};lookup={r['clip_id']:r for r in manifest}
    for r in annotations:
        c=r['clip_id']
        if c not in lookup:errors.append(f'unknown clip {c}');continue
        try:a,b=int(r['start_ms']),int(r['end_ms'])
        except ValueError:errors.append(f'{c}: noninteger times');continue
        if not int(lookup[c]['eval_start_ms'])<=a<b<=int(lookup[c]['eval_end_ms']):errors.append(f'{c}: annotation outside evaluation interval')
        if r['label'] not in {'SHOWING','NON_SHOWING','UNCERTAIN'}:errors.append(f'{c}: invalid label')
        if r['review_status']!='REVIEWED' or not r.get('annotator') or not r.get('reviewer') or r.get('annotator')==r.get('reviewer'):
            errors.append(f'{c}: independent annotator/reviewer and REVIEWED required')
        if r['label']=='SHOWING' and (not r['physical_target_id'] or r['physical_target_id']=='NOT_TARGET' or not r['class_name']):errors.append(f'{c}: physical target/class required')
        if r['confidence'] not in {'HIGH','MEDIUM','LOW'}:errors.append(f'{c}: invalid confidence')
        if r['label']!='SHOWING' and r['physical_target_id']:errors.append(f'{c}: non-showing interval must not claim a target')
        byclip[c].append(r)
    for c,rows in byclip.items():
        rows.sort(key=lambda r:int(r['start_ms']));cursor=int(lookup[c]['eval_start_ms'])
        if len({r['event_id'] for r in rows})!=len(rows):errors.append(f'{c}: duplicate event IDs')
        for r in rows:
            if int(r['start_ms'])!=cursor:errors.append(f'{c}: gaps/overlaps in full annotation partition')
            cursor=int(r['end_ms'])
        if cursor!=int(lookup[c]['eval_end_ms']):errors.append(f'{c}: annotation incomplete')
    if annotations and not any(r['label']!='UNCERTAIN' for r in annotations):errors.append('no evaluable reviewed intervals')
    groups={};splitrows={r['clip_id']:r for r in splits}
    for c in lookup:
        r=splitrows.get(c)
        if r is None:errors.append(f'{c}: missing split');continue
        if r['split'] not in {'development','validation','test'}:errors.append(f'{c}: invalid split')
        if lookup[c].get('used_for_development')=='true' and r['split']!='development' and r.get('allow_development_test')!='true':errors.append(f'{c}: development data cannot be relabeled holdout')
        if r['split']!='development' and r.get('allow_development_test')!='true' and (not r['participant_id'] or not r['session_id']):errors.append(f'{c}: holdout needs participant and recording session')
        for field in ['session_id']:
            if r[field]:groups.setdefault((field,r[field]),set()).add(r['split'])
    if any(len(v)>1 for v in groups.values()):errors.append('recording-session leakage across splits')
    for c in lookup:
        media_pts_verified=(lookup[c].get('annotation_timebase')=='original_media_pts'
                            and lookup[c].get('annotation_timebase_verified')=='true')
        if not media_pts_verified:
            if lookup[c].get('time_alignment_reviewed')!='true':errors.append(f'{c}: video/media time alignment unverified')
            try:int(lookup[c]['player_offset_ms'])
            except (KeyError,ValueError):errors.append(f'{c}: verified player/media offset missing')
        tracks={}
        for r in [r for r in mappings if r['clip_id']==c and r['review_status']=='REVIEWED']:
            try:a,b=int(r['start_ms']),int(r['end_ms'])
            except ValueError:errors.append(f'{c}: invalid mapping times');continue
            if not int(lookup[c]['eval_start_ms'])<=a<b<=int(lookup[c]['eval_end_ms']):errors.append(f'{c}: mapping outside evaluation interval')
            if r['review_status']!='REVIEWED' or not r['physical_target_id'] or not r.get('annotator') or not r.get('reviewer') or r.get('annotator')==r.get('reviewer'):
                errors.append(f'{c}: mapping needs reviewed physical identity (or NOT_TARGET)')
            tracks.setdefault(r['track_id'],[]).append((a,b))
        if any(any(a[1]>b[0] for a,b in zip(sorted(v),sorted(v)[1:])) for v in tracks.values()):errors.append(f'{c}: overlapping mappings for same track')
    return errors


def match_events(gt,pred,eligible):
    """Maximum-cardinality bipartite matching; descending IoU edge order."""
    edges={i:sorted([j for j,p in enumerate(pred) if eligible(g,p)],key=lambda j:-iou(g,pred[j])) for i,g in enumerate(gt)}
    owners={}
    def visit(i,seen):
        for j in edges[i]:
            if j in seen:continue
            seen.add(j)
            if j not in owners or visit(owners[j],seen):owners[j]=i;return True
        return False
    for i in range(len(gt)):visit(i,set())
    return sorted((i,j) for j,i in owners.items())


def iou(a,b):
    overlap=max(0,min(a['end_ms'],b['end_ms'])-max(a['start_ms'],b['start_ms']))
    return overlap/(max(a['end_ms'],b['end_ms'])-min(a['start_ms'],b['start_ms']))


def prf(tp,np,ng):
    return dict(tp=tp,fp=np-tp,fn=ng-tp,precision=tp/np if np else None,recall=tp/ng if ng else None,
                f1=2*tp/(np+ng) if np+ng else None)


def score_clip(packets,intents,labels,mappings,threshold=.3,include_assistant_mappings=False):
    if len(packets)<2:raise ValueError('need at least two perception packets')
    if any(i['state'] not in {'NONE','CANDIDATE','SHOWING','ENDING'} or i['state']=='SHOWING' and i['target_id'] is None for i in intents):raise ValueError('invalid intent packet')
    if len(packets)!=len(intents) or any(p['timestamp_ms']!=i['timestamp_ms'] for p,i in zip(packets,intents)):
        raise ValueError('prediction must match perception frame count and timestamps exactly')
    if any(b['timestamp_ms']<=a['timestamp_ms'] for a,b in zip(packets,packets[1:])):raise ValueError('nonmonotonic timestamps')
    labels=sorted(labels,key=lambda r:int(r['start_ms']));gt=[dict(r,start_ms=int(r['start_ms']),end_ms=int(r['end_ms'])) for r in labels if r['label']=='SHOWING']
    maps={}
    for r in mappings:
        if r['review_status']!='REVIEWED' and not (include_assistant_mappings and r.get('annotator')=='assistant_identity_review' and r.get('physical_target_id')):continue
        maps.setdefault(r['track_id'],[]).append((int(r['start_ms']),int(r['end_ms']),r['physical_target_id']))
    pred=[];spans=[];negative_ms=false_ms=positive_ms=correct_ms=wrong_ms=unknown_ms=showing_ms=excluded_ms=missing_bbox_ms=0
    for n,(p,it) in enumerate(zip(packets[:-1],intents[:-1])):
        a,b=p['timestamp_ms'],packets[n+1]['timestamp_ms']
        if it['state'] not in {'NONE','CANDIDATE','SHOWING','ENDING'}:raise ValueError('invalid intent state')
        for r in labels:
            lo,hi=max(a,int(r['start_ms'])),min(b,int(r['end_ms']))
            if lo>=hi:continue
            if r['label']=='UNCERTAIN':excluded_ms+=hi-lo;continue
            if r['label']=='SHOWING':positive_ms+=hi-lo
            else:negative_ms+=hi-lo
            if it['state']!='SHOWING':continue
            # Split at every mapping boundary so partial identities aren't guessed.
            boundaries={lo,hi}
            for x,y,_ in maps.get(it['target_id'],[]):
                if lo<x<hi:boundaries.add(x)
                if lo<y<hi:boundaries.add(y)
            cuts=sorted(boundaries)
            for x,y in zip(cuts,cuts[1:]):
                identity=next((v for l,h,v in maps.get(it['target_id'],[]) if l<=x and y<=h),None)
                spans.append(dict(start_ms=x,end_ms=y,identity=identity));showing_ms+=y-x
                if it['target_id'] not in {o['track_id'] for o in p['objects']}:missing_bbox_ms+=y-x
                if identity is None:unknown_ms+=y-x
                if r['label']=='NON_SHOWING':false_ms+=y-x
                elif identity==r['physical_target_id']:correct_ms+=y-x
                elif identity is not None:wrong_ms+=y-x
                identity_key=identity if identity is not None else 'UNMAPPED:'+str(it['target_id'])
                if pred and pred[-1]['end_ms']==x and pred[-1]['identity_key']==identity_key:pred[-1]['end_ms']=y
                else:pred.append(dict(start_ms=x,end_ms=y,identity_key=identity_key))
    temporal=match_events(gt,pred,lambda g,p:iou(g,p)>=threshold)
    def identity_overlap(g,p):
        return sum(max(0,min(s['end_ms'],g['end_ms'],p['end_ms'])-max(s['start_ms'],g['start_ms'],p['start_ms']))
                   for s in spans if s['identity']==g['physical_target_id'])
    physical=match_events(gt,pred,lambda g,p:iou(g,p)>=threshold and identity_overlap(g,p)>=.5*(min(g['end_ms'],p['end_ms'])-max(g['start_ms'],p['start_ms'])))
    errors=[]
    uncertain=[r for r in labels if r['label']=='UNCERTAIN']
    left,right=packets[0]['timestamp_ms'],packets[-1]['timestamp_ms']
    for i,j in temporal:
        g,p=gt[i],pred[j]
        errors.append(dict(event_id=g['event_id'],start_error_ms=None if g['start_ms']==left or any(int(r['end_ms'])==g['start_ms'] for r in uncertain) else p['start_ms']-g['start_ms'],
            end_error_ms=None if g['end_ms']==right or any(int(r['start_ms'])==g['end_ms'] for r in uncertain) else p['end_ms']-g['end_ms']))
    return dict(temporal_events=prf(len(temporal),len(pred),len(gt)),
        physical_events=prf(len(physical),len(pred),len(gt)) if unknown_ms==0 else None,
        physical_metric_status='COMPLETE_MAPPING' if unknown_ms==0 else 'UNMAPPED_PREDICTED_TARGETS',
        false_showing_ms=false_ms,negative_ms=negative_ms,false_seconds_per_negative_minute=60*false_ms/negative_ms if negative_ms else None,
        showing_ms=showing_ms,missing_visible_target_ms=missing_bbox_ms,positive_ms=positive_ms,correct_target_ms=correct_ms,wrong_target_ms=wrong_ms,unmapped_target_ms=unknown_ms,
        correct_target_coverage=correct_ms/positive_ms if positive_ms else None,excluded_uncertain_ms=excluded_ms,
        extra_fragments=sum(max(0,sum(min(g['end_ms'],p['end_ms'])>max(g['start_ms'],p['start_ms']) for p in pred)-1) for g in gt),
        matched_boundary_errors=errors,predicted_episodes=pred)

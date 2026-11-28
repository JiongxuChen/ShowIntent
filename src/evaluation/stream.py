"""Rescore the released actual playback logs without videos or models."""
import argparse,json,math
from dataclasses import replace
from pathlib import Path
from src.intent.formal_evaluation import read_csv
from src.intent.audio_timeline import AudioTimeline
from src.speech.events import SpeechEvent
from src.speech.presentation_language import ContextualPresentationLanguage
from src.evaluation.collection import summarize
from src.streaming.evaluate_delayed_stream import hold
from scripts.run_refined_b import read,normalized_score

def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--root',type=Path,default=Path('outputs/stream'));p.add_argument('--out',type=Path,required=True);a=p.parse_args()
 if a.out.exists():p.error('Choose a new output directory')
 scores={}
 for group in ['fruit','books','cups','drinks']:
  labels=read_csv(Path('evaluation')/group/'annotations.csv')
  for row in read_csv(Path('evaluation')/group/'dataset_manifest.csv'):
   c=row['clip_id'];base=a.root/group/c;old=json.loads((base/'evaluation.json').read_text());runtime=old['runtime']
   if runtime['status']!='finished' or runtime['worker_exitcodes']!=[0,0] or runtime['input_frames']!=runtime['output_frames']:raise ValueError('Incomplete stream '+c)
   lo,hi=old['horizon_ms'];clock=read(base/('holding-intent.jsonl' if (base/'holding-intent.jsonl').exists() else 'playback-intent.jsonl'));times=sorted({lo,hi,*[r['timestamp_ms'] for r in clock if lo<r['timestamp_ms']<hi]});per=[dict(timestamp_ms=t) for t in times]
   cropped=[dict(r,start_ms=max(lo,int(r['start_ms'])),end_ms=min(hi,int(r['end_ms']))) for r in labels if r['clip_id']==c and int(r['end_ms'])>lo and int(r['start_ms'])<hi]
   events=[replace(SpeechEvent.from_dict({k:v for k,v in r.items() if k in SpeechEvent.__dataclass_fields__}),available_at_ms=max(r['end_ms'],math.ceil(r['wall_ready_ms']-runtime['delay_seconds']*1000))) for r in read(base/'speech.jsonl')]
   audio=AudioTimeline(events,timing='causal');audio.language=ContextualPresentationLanguage();avs=hold(clock,times);vs=hold(read(base/('holding-visual.jsonl' if (base/'holding-visual.jsonl').exists() else 'playback-visual.jsonl')),times);aud=[dict(audio.update(t),timestamp_ms=t) for t in times]
   scores['yolo11m/'+group+'/'+c]={mode:normalized_score(per,values,cropped) for mode,values in [('showintent',avs),('visual-only',vs),('audio-only',aud)]}
 summary={'yolo11m':{g:{mode:summarize([v[mode] for k,v in scores.items() if g=='all' or k.split('/')[1]==g]) for mode in ['showintent','visual-only','audio-only']} for g in ['all','fruit','books','cups','drinks']}}
 a.out.mkdir(parents=True);(a.out/'evaluation.json').write_text(json.dumps(dict(clips=scores,summary=summary,validation_kind='Controlled B replay on measured input timing; no new end-to-end execution'),indent=2)+'\n');print('Completed',len(scores),'clips')
if __name__=='__main__':main()

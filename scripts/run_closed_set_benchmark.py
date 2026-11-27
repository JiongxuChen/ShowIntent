"""Five-detector offline benchmark; DINO receives predeclared scene priors."""
import argparse,json,subprocess,sys
from pathlib import Path
from scripts.run_refined_b import collection,process
from src.intent.formal_evaluation import read_csv
from src.evaluation.collection import summarize
MODELS=['yolo11n','yolo11m','yolo11l','rtdetr-l','grounding-dino-tiny-prior']
def main():
 p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,required=True);p.add_argument('--models',nargs='+',choices=MODELS,default=MODELS);p.add_argument('--clips',nargs='+');p.add_argument('--device',default='cuda:0');p.add_argument('--collections',nargs='+',default=['fruit','books','cups','drinks']);a=p.parse_args()
 if a.out.exists():p.error('Choose a new output directory')
 scores={};prior=json.loads(Path('configs/grounding_dino_scenario_prior.json').read_text())
 for group in a.collections:
  dataset,rows,cache=collection(group);labels=read_csv(dataset/'annotations.csv')
  for r in rows:
   c=r['clip_id']
   if a.clips and c not in a.clips:continue
   for model in a.models:
    dest=a.out/group/model/c;dest.mkdir(parents=True);per=dest/'perception.jsonl';timing=cache/'timestamps'/(c+'.json')
    cmd=[sys.executable,'-m','src.perception.enhanced','--input',r['source_path'],'--timestamp-map',str(timing),'--out',str(per),'--metrics-out',str(dest/'perception_metrics.json'),'--device',a.device,'--imgsz','1280','--crop-imgsz','960']
    if model==MODELS[-1]:
     cfg={k:v for k,v in prior.items() if k!='scene_queries'};cfg['queries']=prior['scene_queries'][group];pf=dest/'prompt.json';pf.write_text(json.dumps(cfg));cmd+=['--detector-backend','grounding-dino','--yolo-model',cfg['model'],'--prompt-config',str(pf)]
    else:cmd+=['--detector-backend','rtdetr' if model=='rtdetr-l' else 'yolo','--yolo-model','models/'+model+'.pt']
    subprocess.run(cmd,check=True);scores[model+'/'+group+'/'+c]=process(per,cache/'speech_raw'/('speech_events_'+c+'.jsonl'),timing,[l for l in labels if l['clip_id']==c],dest/'predictions')
 if not scores:p.error('No matching clips')
 summary={model:{mode:summarize([v[mode] for k,v in scores.items() if k.startswith(model+'/')]) for mode in ['showintent','visual-only','audio-only']} for model in a.models}
 (a.out/'evaluation.json').write_text(json.dumps(dict(clips=scores,summary=summary,scope='Four closed-set detectors and DINO with extra scene priors; original offline ASR reused'),indent=2)+'\n')
if __name__=='__main__':main()

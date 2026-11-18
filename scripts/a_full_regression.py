"""Reproducible local A candidate regression; originals are never modified."""
import json
import subprocess
import sys
from pathlib import Path


def run(*args):
    subprocess.run([sys.executable,'-m',*map(str,args)],check=True)


def main():
    root=Path(__file__).resolve().parents[1]
    out=root/'outputs/a_final_candidate'
    out.mkdir(parents=True,exist_ok=True)
    summary=[]
    for clip in ['B03','C01','C03']:
        video=next((root.parent/'received_media').rglob(clip+'.mp4'))
        dest=out/clip
        dest.mkdir(exist_ok=True)
        print('START '+clip,flush=True)
        raw=dest/'raw.jsonl'
        run('src.perception.run','--input',video,'--yolo-model',root/'yolo11s.pt','--out',raw,'--metrics-out',dest/'metrics.json')
        run('src.perception.validate_jsonl',raw)
        run('scripts.a_unified_replay','--input',raw,'--out-dir',dest/'unified')
        final=dest/'unified/perception.jsonl'
        run('src.perception.validate_jsonl',final)
        run('src.perception.review_overlay','--video',video,'--perception',final,'--out',dest/'review.mp4','--legend')
        rows=[json.loads(x) for x in final.read_text(encoding='utf-8').splitlines()]
        summary.append(dict(clip=clip,frames=len(rows),frames_with_hands=sum(bool(r['hands']) for r in rows),frames_with_relations=sum(bool(r['relations']) for r in rows),unique_ids=len({o['track_id'] for r in rows for o in r['objects']}),metrics=json.loads((dest/'metrics.json').read_text()),status='awaiting_visual_acceptance'))
        (out/'summary.json').write_text(json.dumps(summary,indent=2),encoding='utf-8')
        print('DONE '+clip,flush=True)


if __name__=='__main__':
    main()

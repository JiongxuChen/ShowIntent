"""Download official release assets via host network, without model inference."""
import argparse,hashlib,json,os
from pathlib import Path
from urllib.request import Request,urlopen

RELEASE='https://api.github.com/repos/ultralytics/assets/releases/tags/v8.3.0'

def get(url):return urlopen(Request(url,headers={'User-Agent':'ShowIntent-model-download'}),timeout=45)
def sha(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for b in iter(lambda:f.read(4*1024*1024),b''):h.update(b)
    return h.hexdigest()

def main():
    p=argparse.ArgumentParser();p.add_argument('--models',nargs='+');args=p.parse_args()
    cfg=json.loads(Path('configs/closed_set_benchmark.json').read_text());selected=args.models or list(cfg['models'])
    if set(selected)-set(cfg['models']):p.error('unknown model')
    with get(RELEASE) as response:release=json.load(response)
    assets={r['name']:r for r in release['assets']};record={}
    for name in selected:
        path=Path(cfg['models'][name]['weights']);row=assets[path.name];path.parent.mkdir(exist_ok=True)
        previous=Path('models/closed_set_download_manifest.json')
        old=json.loads(previous.read_text()).get('models',{}) if previous.exists() else {}
        if path.exists():
            if path.stat().st_size!=row['size']:raise ValueError(f'wrong existing file size: {path}; not overwritten')
            if name in old and sha(path)!=old[name]['sha256']:raise ValueError('existing weight changed')
        else:
            partial=path.with_suffix('.pt.partial');print(f'{name}: downloading {row["size"]/1e6:.1f} MB',flush=True)
            with get(row['browser_download_url']) as source,partial.open('wb') as target:
                while chunk:=source.read(1024*1024):target.write(chunk)
                target.flush();os.fsync(target.fileno())
            if partial.stat().st_size!=row['size']:raise ValueError('incomplete download')
            partial.replace(path)
        digest=sha(path)
        if row.get('digest') and row['digest']!='sha256:'+digest:raise ValueError('upstream digest mismatch')
        record[name]=dict(path=str(path),bytes=path.stat().st_size,sha256=digest,source=row['browser_download_url'],upstream_sha256=row.get('digest'),verification='HTTPS official release + asset byte size; local SHA256 recorded')
        print(f'{name}: ready',flush=True)
        old.update(record);previous.write_text(json.dumps(dict(release=RELEASE,models=old),indent=2)+'\n')

if __name__=='__main__':main()

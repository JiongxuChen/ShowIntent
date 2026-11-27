"""Download pinned official Grounding DINO files on the host, no inference."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import time
from urllib.request import Request,urlopen
from concurrent.futures import ThreadPoolExecutor,as_completed
import shutil

REPO='IDEA-Research/grounding-dino-tiny'
REVISION='a2bb814dd30d776dcf7e30523b00659f4f141c71'
FILES=['config.json','preprocessor_config.json','tokenizer.json','tokenizer_config.json','special_tokens_map.json','added_tokens.json','vocab.txt','model.safetensors']


def parallel_weight(path,row,endpoints):
    """Bounded host-network range requests, verified as a full file afterwards."""
    parts=path.parent/'.download_parts';parts.mkdir(exist_ok=True)
    size=row['size'];chunk=16*1024*1024;ranges=[(i,a,min(a+chunk,size)-1) for i,a in enumerate(range(0,size,chunk))]
    def fetch(spec):
        index,begin,end=spec;part=parts/f'{index:04d}.part';marker=part.with_suffix('.json')
        if part.exists() and marker.exists():
            m=json.loads(marker.read_text())
            if m['begin']==begin and m['end']==end and hashlib.sha256(part.read_bytes()).hexdigest()==m['sha256']:return part
        for host in endpoints:
            for attempt in range(2):
                try:
                    url=f'{host}/{REPO}/resolve/{REVISION}/model.safetensors?download=true&part={index}&attempt={attempt}&request={time.time_ns()}'
                    req=Request(url,headers={'User-Agent':'ShowIntent','Range':f'bytes={begin}-{end}'})
                    temporary=part.with_suffix('.tmp')
                    started=time.monotonic()
                    with urlopen(req,timeout=30) as response,temporary.open('wb') as f:
                        if response.status!=206 or response.headers.get('Content-Range')!=f'bytes {begin}-{end}/{size}':raise ValueError('Server ignored requested byte range')
                        while data:=response.read(65536):
                            f.write(data)
                            if time.monotonic()-started>120:raise TimeoutError('Range wall-time budget exceeded')
                    if temporary.stat().st_size!=end-begin+1:raise ValueError('Partial range')
                    sha=hashlib.sha256(temporary.read_bytes()).hexdigest();temporary.replace(part)
                    marker.write_text(json.dumps(dict(begin=begin,end=end,sha256=sha)))
                    return part
                except Exception as e:print('Range retry',index,host,type(e).__name__,flush=True)
        raise RuntimeError('Range unavailable: '+str(index))
    with ThreadPoolExecutor(max_workers=8) as pool:
        futures=[pool.submit(fetch,spec) for spec in ranges]
        for i,future in enumerate(as_completed(futures),1):future.result();print('Weight ranges',i,'/',len(ranges),flush=True)
    partial=path.with_suffix(path.suffix+'.partial')
    with partial.open('wb') as output:
        for index,_,_ in ranges:
            with (parts/f'{index:04d}.part').open('rb') as source:shutil.copyfileobj(source,output,4*1024*1024)
    verify(partial,row);partial.replace(path);shutil.rmtree(parts)


def verify(path,metadata):
    data_size=path.stat().st_size
    if data_size!=metadata['size']:raise ValueError('Incorrect file size: '+str(path))
    h=hashlib.sha256() if metadata.get('lfs') else hashlib.sha1()
    if not metadata.get('lfs'):h.update(f'blob {data_size}\0'.encode())
    with path.open('rb') as f:
        for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
    expected=metadata['lfs']['sha256'] if metadata.get('lfs') else metadata['blobId']
    if h.hexdigest()!=expected:raise ValueError('Upstream checksum mismatch: '+str(path))
    from scripts.run_complementary import digest
    return digest(path)


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--out',type=Path,default=Path('models/grounding-dino-tiny'));a=p.parse_args()
    endpoints=['https://huggingface.co','https://hf-mirror.com'];metadata=None
    for host in endpoints:
        try:
            with urlopen(Request(f'{host}/api/models/{REPO}/revision/{REVISION}?blobs=true',headers={'User-Agent':'ShowIntent'}),timeout=30) as response:metadata=json.load(response)
            if metadata['sha']!=REVISION:raise ValueError('Revision mismatch')
            break
        except Exception as e:print('Metadata endpoint unavailable:',host,type(e).__name__,flush=True)
    if metadata is None:raise RuntimeError('Cannot read pinned model metadata')
    entries={r['rfilename']:r for r in metadata['siblings']};a.out.mkdir(parents=True,exist_ok=True);record={}
    for name in FILES:
        target=a.out/name;row=entries[name]
        if not target.exists() and name=='model.safetensors':parallel_weight(target,row,endpoints)
        if not target.exists():
            partial=target.with_suffix(target.suffix+'.partial');success=False
            for host in endpoints:
                for attempt in range(2):
                    try:
                        url=f'{host}/{REPO}/resolve/{REVISION}/{name}'
                        print('Downloading',name,'from',host,flush=True)
                        with urlopen(Request(url,headers={'User-Agent':'ShowIntent'}),timeout=90) as response,partial.open('wb') as f:
                            count=0;last=0
                            while block:=response.read(4*1024*1024):
                                f.write(block);count+=len(block)
                                if count-last>=64*1024*1024:print(name,round(count/1e6),'MB',flush=True);last=count
                            f.flush();os.fsync(f.fileno())
                        verify(partial,row);partial.replace(target);success=True;break
                    except Exception as e:print('Download attempt failed:',name,type(e).__name__,flush=True)
                if success:break
            if not success:raise RuntimeError('Download incomplete: '+name+'; partial preserved, rerun downloader')
        record[name]=dict(bytes=target.stat().st_size,sha256=verify(target,row))
    (a.out/'download_manifest.json').write_text(json.dumps(dict(repo=REPO,revision=REVISION,files=record,verification='Pinned HTTPS metadata; LFS SHA256 / Git blob SHA1 and local SHA256'),indent=2)+'\n')
    print('Pinned local model verified:',a.out,flush=True)


if __name__=='__main__':main()

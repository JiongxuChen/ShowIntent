"""Download pinned official weights through the host network; verify every file."""

import argparse
import hashlib
import json
from pathlib import Path
import time
from urllib.request import Request, urlopen


MODELS = {
    'vision': ('Ultralytics/YOLO11', 'models', ['yolo11m.pt']),
    'speech': ('Systran/faster-whisper-large-v3', 'models/faster-whisper-large-v3',
               ['config.json', 'model.bin', 'tokenizer.json', 'vocabulary.json', 'preprocessor_config.json']),

}


def request(url,offset=0):
    headers={'User-Agent': 'Mozilla/5.0 ShowIntent-model-download'}
    if offset or '/resolve/' in url:headers['Range']=f'bytes={offset}-'
    return urlopen(Request(url, headers=headers), timeout=90)


def checksum(path, *, git_blob=False):
    h = hashlib.sha1() if git_blob else hashlib.sha256()
    if git_blob:
        h.update(f'blob {path.stat().st_size}\0'.encode())
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(4*1024*1024), b''):
            h.update(chunk)
    return h.hexdigest()


def download(kind, endpoint, only_files=None):
    repo, folder, wanted = MODELS[kind]
    if only_files is not None:
        if set(only_files)-set(wanted):raise ValueError('unknown file selection')
        wanted=list(only_files)
    with request(f'{endpoint}/api/models/{repo}?blobs=true') as response:
        info = json.load(response)
    revision = info['sha']
    files = {r['rfilename']: r for r in info['siblings']}
    destination = Path(folder)
    destination.mkdir(parents=True, exist_ok=True)
    manifest = dict(repo=repo, revision=revision, endpoint=endpoint, files={})
    for name in wanted:
        row = files[name]
        lfs = row.get('lfs')
        expected = lfs['sha256'] if lfs else row['blobId']
        target = destination/name
        if target.exists() and checksum(target, git_blob=not bool(lfs)) != expected:
            raise ValueError(f'existing model has a different checksum: {target}')
        if not target.exists():
            partial = target.with_suffix(target.suffix+('.parallel.partial' if only_files else '.partial'))
            print(f'{kind}: downloading {name}', flush=True)
            for attempt in range(4):
                try:
                    # A fresh signed redirect avoids stale mirror CDN redirects.
                    offset=partial.stat().st_size if partial.exists() else 0
                    with request(f'{endpoint}/{repo}/resolve/{revision}/{name}?showintent={time.time_ns()}',offset) as source:
                        resume=offset>0 and source.status==206
                        sink=partial.open('ab' if resume else 'wb')
                        total = offset if resume else 0
                        last = time.monotonic()
                        with sink:
                            while block := source.read(1024*1024):
                                sink.write(block); total += len(block)
                                if time.monotonic()-last > 25:
                                    print(f'{name}: {total/1e6:.0f} MB', flush=True); last=time.monotonic()
                    if checksum(partial, git_blob=not bool(lfs)) != expected:
                        partial.unlink()
                        raise ValueError(f'checksum mismatch: {name}')
                    partial.rename(target)
                    break
                except Exception as exc:
                    print(f'{name}: attempt {attempt+1} failed: {type(exc).__name__}: {exc}',flush=True)
                    if attempt == 3:
                        raise
                    time.sleep(2)
        manifest['files'][name] = dict(size=target.stat().st_size, sha256=checksum(target), hub_hash=expected)
        print(f'{kind}: verified {name}', flush=True)
    if only_files is None:
        (destination/f'download_manifest_{kind}.json').write_text(json.dumps(manifest, indent=2)+'\n')


def main():
    parser=argparse.ArgumentParser(description='Download official visual/ASR weights, without inference')
    parser.add_argument('--endpoint', default='https://hf-mirror.com')
    parser.add_argument('--models', nargs='+', choices=MODELS, default=['vision','speech'])
    parser.add_argument('--workers',type=int,default=1)
    args=parser.parse_args()
    for kind in args.models:
        if args.workers<1 or args.workers>4:parser.error('--workers must be 1..4')
        import fcntl
        folder=Path(MODELS[kind][1]);folder.mkdir(parents=True,exist_ok=True)
        with (folder/'.download.lock').open('a') as lock:
            try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
            except BlockingIOError:parser.error(f'{kind}: another download is already active')
            if args.workers>1:
                from concurrent.futures import ThreadPoolExecutor
                with ThreadPoolExecutor(max_workers=args.workers) as pool:
                    list(pool.map(lambda f:download(kind,args.endpoint.rstrip('/'),[f]),MODELS[kind][2]))
            download(kind, args.endpoint.rstrip('/'))
    print('All requested model files verified; download manifests written.',flush=True)


if __name__ == '__main__':
    main()

"""Prepare a low-resolution upright CFR input; does not run any model."""
import argparse
from pathlib import Path
import subprocess
import shutil
import json
import hashlib


def resolve_ffmpeg(explicit=None):
    candidates=[explicit] if explicit else [shutil.which("ffmpeg"), "/usr/bin/ffmpeg", "/bin/ffmpeg"]
    if not explicit:
        try:
            import imageio_ffmpeg
            candidates.append(imageio_ffmpeg.get_ffmpeg_exe())
        except (ImportError, RuntimeError):
            pass
    failures=[]
    for candidate in dict.fromkeys(candidates):
        if not candidate:continue
        try:
            subprocess.run([candidate,"-version"],check=True,capture_output=True,timeout=10)
            return candidate
        except (OSError,subprocess.SubprocessError) as exc:
            failures.append(f"{candidate}: {exc}")
    raise RuntimeError("No runnable FFmpeg; install imageio-ffmpeg in this Python environment, or use --ffmpeg with a path that exists on this node. " + "; ".join(failures))


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--video',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    p.add_argument('--height',type=int,default=960)
    p.add_argument('--fps',type=int,default=15)
    p.add_argument('--ffmpeg',help='Explicit executable path; otherwise probe PATH, system paths and bundled imageio-ffmpeg')
    a=p.parse_args()
    if not a.video.is_file() or a.out.exists() or a.height<128 or a.fps<5:p.error('Invalid input, size, fps or existing output')
    ffmpeg=resolve_ffmpeg(a.ffmpeg)
    print(f'Using FFmpeg: {ffmpeg}',flush=True)
    a.out.parent.mkdir(parents=True,exist_ok=True)
    subprocess.run([ffmpeg,'-nostdin','-n','-i',str(a.video),'-map','0:v:0','-map','0:a:0?',
        '-vf',f'scale=-2:{a.height},fps={a.fps}:start_time=0','-c:v','libx264','-preset','fast','-crf','23',
        '-c:a','aac','-metadata:s:v:0','rotate=0',str(a.out)],check=True)
    def digest(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for block in iter(lambda:f.read(1048576),b''):h.update(block)
        return h.hexdigest()
    a.out.with_suffix('.source.json').write_text(json.dumps(dict(source=str(a.video),source_sha256=digest(a.video),proxy_sha256=digest(a.out),fps=a.fps,height=a.height,timebase='original media zero; CFR start_time=0 pads leading video gap',version=2),indent=2)+'\n')



if __name__=='__main__':main()

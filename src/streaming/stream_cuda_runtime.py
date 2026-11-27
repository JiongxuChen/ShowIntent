"""Expose installed NVIDIA wheel libraries before loading native ASR runtimes."""
from pathlib import Path
import os
import subprocess
import sys


def runtime_library_path(search_paths, current=''):
    libraries=[]
    for entry in search_paths:
        for name in ('cublas','cudnn'):
            directory=Path(entry)/'nvidia'/name/'lib'
            if directory.is_dir():libraries.append(str(directory.resolve()))
    return ':'.join(dict.fromkeys(libraries+[x for x in current.split(':') if x]))


def prepare_cuda_runtime():
    existing=os.environ.get('LD_LIBRARY_PATH','')
    configured=runtime_library_path(sys.path,existing)
    if configured!=existing:
        env=dict(os.environ,LD_LIBRARY_PATH=configured)
        # The dynamic loader needs these paths at process startup, before imports.
        os.execve(sys.executable,[sys.executable,'-m','scripts.run_delayed_stream',*sys.argv[1:]],env)
    probe=subprocess.run([sys.executable,'-c',
        'import ctypes; [ctypes.CDLL(n) for n in ("libcublas.so.12","libcudnn.so.9","libcudnn_ops.so.9")]'],capture_output=True,text=True)
    if probe.returncode:
        raise RuntimeError('CUDA ASR library preflight failed before starting the stream. '
                           'Use --asr-device cpu to keep GPU vision with CPU speech. Details: '+probe.stderr[-2000:])

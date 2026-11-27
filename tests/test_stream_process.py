import multiprocessing as mp
from src.streaming.stream_process import isolated_entry


def send_large_result(channel):
    channel.put(b'x'*1_000_000)


def fail():
    raise ValueError('test worker failure')


def test_isolated_exit_flushes_all_results():
    ctx=mp.get_context('spawn');channel=ctx.Queue()
    worker=ctx.Process(target=isolated_entry,args=(send_large_result,(channel,),[channel]))
    worker.start()
    assert len(channel.get(timeout=10))==1_000_000
    worker.join(timeout=10)
    assert worker.exitcode==0
    channel.close()


def test_isolated_exit_does_not_hide_failure():
    ctx=mp.get_context('spawn')
    worker=ctx.Process(target=isolated_entry,args=(fail,(),[]))
    worker.start();worker.join(timeout=10)
    assert worker.exitcode==1


def test_child_created_semaphore_is_unlinked_before_explicit_exit(tmp_path):
    import os
    from pathlib import Path
    import subprocess
    import sys
    script=tmp_path/'semaphore_check.py'
    script.write_text('''import multiprocessing as mp
from src.streaming.stream_process import isolated_entry
keep=[]
def create_resource():
    keep.append(mp.get_context('spawn').Semaphore(1))
if __name__=='__main__':
    ctx=mp.get_context('spawn')
    p=ctx.Process(target=isolated_entry,args=(create_resource,(),[]))
    p.start();p.join(10)
    assert p.exitcode==0
''')
    env=dict(os.environ,PYTHONPATH=str(Path.cwd()))
    result=subprocess.run([sys.executable,str(script)],capture_output=True,text=True,timeout=20,env=env)
    assert result.returncode==0,result.stderr
    assert 'leaked semaphore' not in result.stderr

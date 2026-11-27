"""Exit dedicated native-model processes after explicitly flushing IPC.

Do not run Python/native global destructors after inference: some combinations
of CUDA and MediaPipe hang there. The OS releases process-owned native memory.
Errors still have nonzero exit codes. No parent process uses this exit path.
"""
import os
import sys
import traceback
import faulthandler
from multiprocessing import util


def isolated_entry(target,args,output_queues):
    faulthandler.enable()
    if os.environ.get('SHOWINTENT_STREAM_DEBUG')=='1':
        faulthandler.dump_traceback_later(25,repeat=True)
    code=0
    try:
        target(*args)
    except BaseException:
        traceback.print_exc();code=1
    try:
        for channel in output_queues:
            channel.close()
            channel.join_thread()
        # Run only multiprocessing resource finalizers (including child-created
        # semaphore unlink callbacks), not Python atexit or native destructors.
        print('Inference process: IPC flushed; cleaning multiprocessing resources',flush=True)
        util._run_finalizers()
        print('Inference process: cleanup complete',flush=True)
        sys.stdout.flush();sys.stderr.flush()
    except BaseException:
        traceback.print_exc();code=1
    faulthandler.cancel_dump_traceback_later()
    os._exit(code)

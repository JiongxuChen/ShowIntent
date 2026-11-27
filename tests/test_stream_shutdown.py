import multiprocessing as mp
from src.streaming.stream_shutdown import shutdown_workers


def fill_pipe(channel):
    channel.put(b'x'*2_000_000)


def test_shutdown_drains_results_before_joining_worker():
    ctx=mp.get_context('spawn');channel=ctx.Queue()
    process=ctx.Process(target=fill_pipe,args=(channel,));process.start()
    forced=shutdown_workers([process],[channel],timeout=5)
    assert forced==[]
    assert process.exitcode==0
    channel.close()


def test_one_cleanup_error_does_not_skip_remaining_resources():
    from src.streaming.stream_shutdown import close_channels
    class Channel:
        def __init__(self,fail=False):self.closed=False;self.joined=False;self.fail=fail
        def close(self):
            if self.fail:raise OSError('test close failure')
            self.closed=True
        def join_thread(self):self.joined=True
    broken=Channel(True);good=Channel()
    errors=close_channels([broken,good])
    assert len(errors)==1
    assert good.closed and good.joined

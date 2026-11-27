"""Drain IPC while workers exit, so feeder threads cannot deadlock on full pipes."""
import queue
import time


def shutdown_workers(workers, queues, timeout=30):
    deadline=time.monotonic()+timeout
    while any(w.is_alive() for w in workers) and time.monotonic()<deadline:
        for channel in queues:
            for _ in range(256):
                try:channel.get_nowait()
                except queue.Empty:break
        for worker in workers:worker.join(timeout=.02)
    forced=[]
    for worker in workers:
        # Recheck immediately before signalling: process exit may race polling.
        worker.join(timeout=0)
        if worker.is_alive():
            forced.append(worker.name);worker.terminate();worker.join(timeout=5)
            if worker.is_alive():worker.kill();worker.join(timeout=5)
    return forced


def close_channels(channels, *, discard=None):
    """Attempt every close even if one IPC resource fails during cleanup."""
    failures=[]
    for index,channel in enumerate(channels):
        try:
            if channel is discard:channel.cancel_join_thread()
            channel.close()
            if channel is not discard:channel.join_thread()
        except Exception as exc:failures.append(f'channel {index}: {exc!r}')
    return failures

import pytest
from scripts.export_delayed_split import schedule

def test_recorded_deadlines_preserve_delay_and_lateness():
    timing=[dict(timestamp_ms=0,written_at_ms=6022),dict(timestamp_ms=67,written_at_ms=6099)]
    assert schedule(timing,[dict(timestamp_ms=0),dict(timestamp_ms=67)],6000)==[6022,6099]

def test_future_or_misaligned_output_is_rejected():
    with pytest.raises(ValueError):schedule([dict(timestamp_ms=0,written_at_ms=5990)],[dict(timestamp_ms=0)],6000)
    with pytest.raises(ValueError):schedule([dict(timestamp_ms=0,written_at_ms=6000)],[dict(timestamp_ms=67)],6000)


def test_source_timestamps_and_buffer_placeholder():
    from scripts.export_delayed_split import source_time
    assert source_time(None)=='--:--.--'
    assert source_time(0)=='00:00.00'
    assert source_time(20000)=='00:20.00'
    assert source_time(59999)=='01:00.00'


def test_bundled_proxy_is_resolved_locally_and_tampering_rejected(tmp_path):
    import json,hashlib
    from types import SimpleNamespace
    from scripts.export_delayed_split import resolve_video
    run=tmp_path/'outputs/stream/fruit/demo';run.mkdir(parents=True)
    video=tmp_path/'outputs/stream_media/fruit/demo.mp4';video.parent.mkdir(parents=True);video.write_bytes(b'proxy')
    item=dict(proxy='outputs/stream_media/fruit/demo.mp4',proxy_sha256=hashlib.sha256(b'proxy').hexdigest())
    (run.parent.parent/'render_inputs.json').write_text(json.dumps(dict(clips={'fruit/demo':item})))
    args=SimpleNamespace(video=None,data_root=tmp_path)
    assert resolve_video(run,{},args)==video
    video.write_bytes(b'wrong')
    with pytest.raises(ValueError):resolve_video(run,{},args)


def test_absent_proxy_requires_verified_original_before_preparing(tmp_path,monkeypatch):
    import json,hashlib
    from types import SimpleNamespace
    from scripts.export_delayed_split import resolve_video
    run=tmp_path/'outputs/stream/fruit/demo';run.mkdir(parents=True)
    item=dict(proxy='outputs/stream_media/demo.mp4',proxy_sha256='unused',source='Example_Data1/demo.mp4',source_sha256=hashlib.sha256(b'original').hexdigest(),height=1280,fps=15)
    (run.parent.parent/'render_inputs.json').write_text(json.dumps(dict(clips={'fruit/demo':item})))
    args=SimpleNamespace(video=None,data_root=tmp_path)
    with pytest.raises(FileNotFoundError):resolve_video(run,{},args)
    source=tmp_path/item['source'];source.parent.mkdir();source.write_bytes(b'wrong')
    with pytest.raises(ValueError):resolve_video(run,{},args)
    source.write_bytes(b'original');calls=[]
    monkeypatch.setattr('scripts.export_delayed_split.subprocess.run',lambda command,**kwargs:calls.append(command))
    assert resolve_video(run,{},args)==tmp_path/item['proxy']
    assert calls[0][calls[0].index('--height')+1]=='1280'

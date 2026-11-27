import sys
from types import SimpleNamespace
from src.streaming.prepare_stream_proxy import resolve_ffmpeg


def test_bundled_fallback_when_system_executable_missing(monkeypatch):
    import src.streaming.prepare_stream_proxy as module
    monkeypatch.setattr(module.shutil,'which',lambda _:None)
    monkeypatch.setitem(sys.modules,'imageio_ffmpeg',SimpleNamespace(get_ffmpeg_exe=lambda:'/portable/ffmpeg'))
    def run(args,**kwargs):
        if args[0]!='/portable/ffmpeg':raise FileNotFoundError(args[0])
    monkeypatch.setattr(module.subprocess,'run',run)
    assert resolve_ffmpeg()=='/portable/ffmpeg'


def test_explicit_invalid_path_does_not_silently_change_executable(monkeypatch):
    import pytest
    import src.streaming.prepare_stream_proxy as module
    def run(*args,**kwargs):raise FileNotFoundError('missing')
    monkeypatch.setattr(module.subprocess,'run',run)
    with pytest.raises(RuntimeError,match='exists on this node'):resolve_ffmpeg('/missing')

"""Range resume is used only when the server confirms partial content."""
import hashlib,json,io
from scripts import download_joint_models as downloader
import pytest

@pytest.mark.parametrize('status',[200,206])
def test_resumed_download_is_verified_and_ignored_range_restarts(tmp_path,monkeypatch,status):
    payload=b'abcdefgh'
    monkeypatch.setitem(downloader.MODELS,'fixture',('example/repo',str(tmp_path),['weights']))
    (tmp_path/'weights.parallel.partial').write_bytes(payload[:3])
    info={'sha':'pinned','siblings':[{'rfilename':'weights','lfs':{'sha256':hashlib.sha256(payload).hexdigest()}}]}
    calls=[]
    def request(url,offset=0):
        calls.append(offset)
        response=io.BytesIO(json.dumps(info).encode() if '/api/' in url else payload[offset:] if status==206 else payload)
        response.status=status
        return response
    monkeypatch.setattr(downloader,'request',request)
    downloader.download('fixture','https://example.test',['weights'])
    assert (tmp_path/'weights').read_bytes()==payload and calls==[0,3]

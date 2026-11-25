"""Word ownership and repetition safeguards with a fake recognizer, no inference."""

from types import SimpleNamespace as Row

import pytest

np=pytest.importorskip('numpy')
from src.speech.asr import transcribe_samples


def test_overlap_uses_past_audio_without_repeating_already_committed_words():
    calls=[]
    class Model:
        def transcribe(self,audio,**kwargs):
            calls.append(len(audio))
            words=([Row(start=.2,end=.8,word='苹果')] if len(calls)==1 else
                   [Row(start=.1,end=.4,word='苹果'),Row(start=.6,end=1.,word=' 展示结束')])
            segment=Row(start=words[0].start,end=words[-1].end,text=''.join(w.word for w in words),words=words)
            return [segment],Row(language='zh')
    clock=iter([0.,.01,.01,.02])
    result=transcribe_samples(np.ones(32000,dtype=np.float32),Model(),mode='causal',window_ms=1000,
                             context_ms=500,clock=lambda:next(clock))
    assert calls==[16000,24000]
    assert [e.text for e in result.events]==['苹果','展示结束']
    assert result.events[1].start_ms==1100
    assert all(e.available_at_ms>=e.end_ms for e in result.events)


def test_extreme_transcript_loop_is_discarded_with_a_count():
    class Model:
        def transcribe(self,*args,**kwargs):
            return [Row(start=0.,end=1.,text='认识'*12)],Row(language='zh')
    clock=iter([0.,.01])
    result=transcribe_samples(np.ones(16000,dtype=np.float32),Model(),clock=lambda:next(clock))
    assert result.events==[] and result.metadata['discarded_segments']==1


def test_video_without_an_audio_stream_returns_empty_observations_without_loading_weights(tmp_path):
    cv2=pytest.importorskip('cv2')
    pytest.importorskip('faster_whisper')
    from src.speech.asr import run_asr
    video=tmp_path/'silent.avi'
    writer=cv2.VideoWriter(str(video),cv2.VideoWriter_fourcc(*'MJPG'),10,(32,32))
    assert writer.isOpened()
    for _ in range(3):writer.write(np.zeros((32,32,3),dtype=np.uint8))
    writer.release()
    model=tmp_path/'model';model.mkdir();(model/'model.bin').touch()
    result=run_asr(video,model,mode='causal')
    assert result.events==[] and result.metadata['source_kind']=='media_without_audio_stream'

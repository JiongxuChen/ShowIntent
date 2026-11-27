from types import SimpleNamespace
from src.streaming.stream_workers import warmup_recognizer


def test_warmup_exercises_real_search_and_vad_without_future_audio():
    calls=[]
    class Recognizer:
        def transcribe(self,samples,**kwargs):
            calls.append((len(samples),float(abs(samples).sum()),kwargs))
            return iter([]),SimpleNamespace(language='zh')
    warmup_recognizer(Recognizer(),4,5)
    assert len(calls)==2
    assert all(n==64000 and total==0 for n,total,_ in calls)
    assert [kw['vad_filter'] for _,_,kw in calls]==[False,True]
    assert all(kw['beam_size']==5 for _,_,kw in calls)

from scripts.run_delayed_stream import evidence_time


def test_buffer_cannot_reveal_future_utterance():
    assert evidence_time(4000,4500,6000)==4000


def test_slow_asr_is_unavailable_before_playback_deadline():
    assert evidence_time(4000,11000,6000)==5000
    assert evidence_time(4000,10000.1,6000)==4001

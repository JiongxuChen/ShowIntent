from scripts.a_duplicate_candidate import filter_stream


def test_adjacent_equal_strength_fruits_are_preserved():
    rows=[]
    for index in range(40):
        rows.append(dict(timestamp_ms=index*33,objects=[dict(track_id='a',class_name='apple',bbox_norm=[.2,.2,.4,.4],score=.9),dict(track_id='b',class_name='apple',bbox_norm=[.3,.2,.5,.4],score=.9)],relations=[]))
    assert all(len(packet['objects'])==2 for packet,_ in filter_stream(rows))


def test_new_weak_overlap_is_logged_and_source_is_preserved():
    anchor=dict(track_id='a',class_name='apple',bbox_norm=[.2,.2,.4,.4],score=.9)
    rows=[dict(timestamp_ms=i*33,objects=[anchor],relations=[]) for i in range(30)]
    rows.append(dict(timestamp_ms=1000,objects=[anchor,dict(track_id='b',class_name='apple',bbox_norm=[.22,.22,.42,.42],score=.4)],relations=[]))
    result=list(filter_stream(rows))
    assert len(result[-1][0]['objects'])==1
    assert len(rows[-1]['objects'])==2
    assert result[-1][1][-1]['hidden_id']=='b'

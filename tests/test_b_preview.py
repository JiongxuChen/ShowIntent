"""B diagnostic framing: prevent zoom pumping without changing B states."""
from scripts.render_b_preview import PreviewFraming,target_crop


def obj(tid='a',size=.12,y=.5):
    return dict(track_id=tid,bbox_norm=[.5-size/2,y-size/2,.5+size/2,y+size/2])


def test_larger_detail_crop_is_bounded():
    crop=target_crop([.4,.4,.5,.5],1.,5.)
    assert abs(crop[2]-crop[0]-.2)<1e-9
    assert all(0<=x<=1 for x in crop)


def test_box_size_and_track_relink_cannot_pump_episode_zoom():
    f=PreviewFraming();widths=[]
    for t in range(0,4001,50):
        tid='a' if t<2000 else 'linked'
        objects=[obj(tid,size=.12 if t%100==0 else .35,y=.5+t/20000)]
        crop=f.update(t,dict(state='SHOWING',target_id=tid),objects)
        widths.append(crop[2]-crop[0])
        assert all(0<=x<=1 for x in crop)
    assert all(b<=a+1e-9 for a,b in zip(widths,widths[1:]))
    assert max(widths[-10:])-min(widths[-10:])<1e-7


def test_short_exit_holds_then_monotonically_widens_and_never_changes_intent():
    f=PreviewFraming()
    for t in range(0,2001,50):f.update(t,dict(state='SHOWING',target_id='a'),[obj()])
    before=list(f.current)
    ending=dict(state='ENDING',target_id='a')
    for t in [2050,2100,2250,2300]:assert f.update(t,ending,[obj(size=.4,y=.8)])==before
    widths=[]
    for t in range(2350,4001,50):
        c=f.update(t,ending,[obj(size=.05,y=.9)]);widths.append(c[2]-c[0])
    assert all(b>=a for a,b in zip(widths,widths[1:]))
    assert widths[-1]>.99 and ending['state']=='ENDING'

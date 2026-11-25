"""Temporary B visualization, independent of the production C controller.

Only SHOWING requests a close-up. Preserve source audio and media timestamps;
save every applied crop so rendering cannot conceal intent decisions.
"""
import argparse
from fractions import Fraction
import hashlib
import json
import math
from pathlib import Path
from statistics import median

PREVIEW_POLICY='locked_scale_v2'


def target_crop(box, normalized_aspect, max_zoom, margin_factor=1.35):
    cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
    w=max((box[2]-box[0])*margin_factor,(box[3]-box[1])*margin_factor*normalized_aspect,1/max_zoom)
    w=min(w,1.,normalized_aspect);h=w/normalized_aspect
    left=min(max(cx-w/2,0),1-w);top=min(max(cy-h/2,0),1-h)
    return [left,top,left+w,top+h]


class PreviewFraming:
    """Diagnostic framing only: fix scale within an episode, smooth position.

    A brief exit holds the LAST crop, rather than chasing the lowering object
    or repeatedly recomputing its scale. B's state is always logged unchanged.
    """
    def __init__(self,max_zoom=5.,margin_factor=1.35,exit_hold_ms=300):
        self.max_zoom=max_zoom;self.margin_factor=margin_factor;self.exit_hold_ms=exit_hold_ms
        self.current=[0.,0.,1.,1.];self.last_ts=None;self.last_visible=None
        self.locked_width=None;self.history=[]

    def update(self,ts,intent,objects,aspect=1.):
        self.history=[(t,rows) for t,rows in self.history if ts-t<=500]
        self.history.append((ts,{o['track_id']:o['bbox_norm'] for o in objects}))
        obj=next((o for o in objects if o['track_id']==intent['target_id']),None)
        desired=[0.,0.,1.,1.]
        if intent['state']=='SHOWING' and obj:
            box=obj['bbox_norm'];cx,cy=(box[0]+box[2])/2,(box[1]+box[3])/2
            if self.locked_width is None:
                boxes=[rows[intent['target_id']] for _,rows in self.history if intent['target_id'] in rows]
                w=median(b[2]-b[0] for b in boxes);h=median(b[3]-b[1] for b in boxes)
                anchor=[cx-w/2,cy-h/2,cx+w/2,cy+h/2]
                initial=target_crop(anchor,aspect,self.max_zoom,self.margin_factor)
                self.locked_width=initial[2]-initial[0]
            w=self.locked_width;h=w/aspect
            left=min(max(cx-w/2,0),1-w);top=min(max(cy-h/2,0),1-h)
            desired=[left,top,left+w,top+h];self.last_visible=ts
        elif self.last_visible is not None and ts-self.last_visible<= (
                600 if intent['state']=='SHOWING' else self.exit_hold_ms):
            desired=list(self.current)
        else:
            self.locked_width=None
        dt=ts-self.last_ts if self.last_ts is not None else 0
        alpha=1-math.exp(-dt/180) if dt else 0
        self.current=[a+alpha*(b-a) for a,b in zip(self.current,desired)];self.last_ts=ts
        return list(self.current)


def main():
    parser=argparse.ArgumentParser(description='B-only diagnostic zoom preview; not final C rendering')
    parser.add_argument('--video',type=Path,required=True)
    parser.add_argument('--perception',type=Path,required=True)
    parser.add_argument('--intent',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    parser.add_argument('--max-zoom',type=float,default=5.)
    parser.add_argument('--margin-factor',type=float,default=1.35)
    parser.add_argument('--exit-hold-ms',type=int,default=300)
    parser.add_argument('--label',default='B preview / offline content')
    args=parser.parse_args()
    trace_path=args.out.with_suffix('.crops.jsonl');meta_path=args.out.with_suffix('.meta.json')
    if any(p.exists() for p in [args.out,trace_path,meta_path]):parser.error('output exists; choose a new filename')
    if not math.isfinite(args.max_zoom) or args.max_zoom<1:parser.error('invalid maximum zoom')
    if not math.isfinite(args.margin_factor) or args.margin_factor<1 or args.exit_hold_ms<0:
        parser.error('invalid margin or exit hold')
    import av
    import cv2
    from src.perception.run import apply_rotation,resolve_rotation
    read=lambda p:[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
    perception,intents=read(args.perception),read(args.intent)
    if not perception or [p['timestamp_ms'] for p in perception]!=[i['timestamp_ms'] for i in intents]:
        raise ValueError('perception/intent alignment mismatch')
    capture=cv2.VideoCapture(str(args.video))
    rotation,rotation_name=resolve_rotation(capture.get(cv2.CAP_PROP_ORIENTATION_META),'auto');capture.release()
    args.out.parent.mkdir(parents=True,exist_ok=True)
    framing=PreviewFraming(args.max_zoom,args.margin_factor,args.exit_hold_ms)
    count=0;crops=[]
    with av.open(str(args.video)) as source, av.open(str(args.out),'w') as destination:
        vin=source.streams.video[0];vin.thread_type='AUTO';vin.codec_context.thread_count=2
        vout=destination.add_stream('libx264',rate=vin.average_rate)
        # Phone sources can be variable-frame-rate. The encoder's default
        # average-FPS clock can round distinct source PTS onto one output tick,
        # causing duplicate DTS and a mux error. Keep the original precise clock.
        vout.time_base=vin.time_base
        vout.codec_context.time_base=vin.time_base
        vout.width=540;vout.height=960;vout.pix_fmt='yuv420p'
        vout.options={'preset':'veryfast','crf':'22'};vout.codec_context.thread_count=2
        audio=source.streams.audio[0] if source.streams.audio else None
        aout=destination.add_stream_from_template(audio) if audio else None
        for packet in source.demux([vin]+([audio] if audio else [])):
            if packet.stream==vin:
                for frame in packet.decode():
                    if count>=len(perception):raise ValueError('more decoded frames than B inputs')
                    p,it=perception[count],intents[count];ts=p['timestamp_ms']
                    actual=int(frame.pts*frame.time_base*1000)
                    if abs(actual-ts)>1:raise ValueError('video PTS differs from B media time')
                    image=apply_rotation(frame.to_ndarray(format='bgr24'),rotation)
                    height,width=image.shape[:2];aspect=(540/960)/(width/height)
                    obj=next((o for o in p['objects'] if o['track_id']==it['target_id']),None)
                    current=framing.update(ts,it,p['objects'],aspect)
                    x1,y1,x2,y2=current
                    canvas=image[int(y1*height):max(int(y1*height)+1,int(y2*height)),
                                 int(x1*width):max(int(x1*width)+1,int(x2*width))]
                    canvas=cv2.resize(canvas,(540,960),interpolation=cv2.INTER_AREA)
                    lines=[args.label,f'{ts/1000:.2f}s  {it["state"]}  zoom {1/(x2-x1):.1f}x',
                           f'Target: {it["target_id"] or "-"}  {obj["class_name"] if obj else ""}']
                    cv2.rectangle(canvas,(0,0),(540,86),(25,25,25),-1)
                    for j,line in enumerate(lines):cv2.putText(canvas,line,(10,23+j*25),cv2.FONT_HERSHEY_SIMPLEX,.52,(255,255,255),1,cv2.LINE_AA)
                    output=av.VideoFrame.from_ndarray(canvas,format='bgr24')
                    output.pts=frame.pts;output.time_base=frame.time_base
                    for encoded in vout.encode(output):destination.mux(encoded)
                    crops.append(dict(timestamp_ms=ts,state=it['state'],target_id=it['target_id'],crop_norm=list(current)))
                    count+=1
                    if count%150==0:print(f'{count}/{len(perception)} frames',flush=True)
            elif aout is not None and packet.dts is not None:
                packet.stream=aout;destination.mux(packet)
        for encoded in vout.encode():destination.mux(encoded)
    if count!=len(perception):raise ValueError('incomplete video decode')
    trace_path.write_text(''.join(json.dumps(p)+'\n' for p in crops))
    meta=dict(purpose='B diagnostic preview, not final C implementation',frames=count,
              video=str(args.video),perception=str(args.perception),intent=str(args.intent),
              output_size=[540,960],rotation=rotation_name,max_zoom=args.max_zoom,audio_preserved=audio is not None,
              preview_policy=PREVIEW_POLICY,margin_factor=args.margin_factor,exit_hold_ms=args.exit_hold_ms,
              input_sha256={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [args.perception,args.intent]})
    meta_path.write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
    print(f'Completed: {args.out}',flush=True)


if __name__=='__main__':main()

"""User-run timestamp extraction; decodes video but performs no model inference."""
import argparse
import json
from pathlib import Path

from src.perception.timing import timestamps_from_frames


def main():
    parser=argparse.ArgumentParser(description='从解码视频帧提取真实 PTS，保持帧号与媒体时间对应')
    parser.add_argument('--input',type=Path,required=True)
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists() or args.out.resolve()==args.input.resolve():parser.error('output exists or aliases input')
    import av
    with av.open(str(args.input)) as container:
        stream=container.streams.video[0]
        stream.thread_type='AUTO';stream.codec_context.thread_count=2
        values=timestamps_from_frames(container.decode(stream))
    record=dict(source_name=args.input.name,source_size_bytes=args.input.stat().st_size,
                timing_kind='decoded_video_pts',frame_count=len(values),timestamp_ms=values)
    args.out.parent.mkdir(parents=True,exist_ok=True)
    with args.out.open('x',encoding='utf-8') as output:output.write(json.dumps(record,ensure_ascii=False)+'\n')
    print(f'Extracted {len(values)} video presentation timestamps -> {args.out}')


if __name__=='__main__':main()

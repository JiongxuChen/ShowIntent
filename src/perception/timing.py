"""Validated video presentation times, independent of OpenCV position reporting."""
import json
from pathlib import Path


def validate_timestamps(values):
    if not isinstance(values,list) or not values:
        raise ValueError('timestamp map must contain a nonempty timestamp_ms list')
    previous=-1
    for value in values:
        if type(value) is not int or value<0 or value<=previous:
            raise ValueError('video timestamps must be non-negative, strictly increasing integer milliseconds')
        previous=value
    return values


def load_timestamps(path):
    data=json.loads(Path(path).read_text(encoding='utf-8'))
    values=validate_timestamps(data['timestamp_ms'])
    if data.get('frame_count')!=len(values):raise ValueError('timestamp map frame count mismatch')
    return values


def timestamps_from_frames(frames):
    values=[]
    for frame in frames:
        if frame.pts is None or frame.time_base is None:
            raise ValueError('decoded frame has no presentation timestamp')
        values.append(int(frame.pts*frame.time_base*1000))
    return validate_timestamps(values)

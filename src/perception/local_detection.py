"""Experimental local/full-frame fusion. Coordinates always stay full-frame."""
from .tracker import iou


def fuse_detections(global_objects, local_objects):
    result=list(local_objects)
    for obj in global_objects:
        # Local detections take precedence only where a same-class box overlaps.
        if not any(other['class_name']==obj['class_name'] and iou(tuple(obj['bbox_norm']),tuple(other['bbox_norm']))>.25 for other in local_objects):
            result.append(obj)
    return result

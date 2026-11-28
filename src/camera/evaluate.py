"""Measure the applied camera trajectory; match geometry only to B's selected ID."""
from collections import Counter
from math import hypot
from statistics import mean


def describe(values):
    if not values:
        return {"n": 0, "mean": None, "p95": None, "max": None}
    values = sorted(values)
    return {"n": len(values), "mean": round(mean(values), 5),
            "p95": round(values[round(.95*(len(values)-1))], 5), "max": round(values[-1], 5)}


def measure(rows):
    visible = retained = missing = showing = transitions = 0
    modes = Counter()
    durations = {"TRANSITION_IN": [], "TRANSITION_OUT": []}
    steps, zoom_steps, steady_motion, steady_jitter, responses = [], [], [], [], []
    previous_velocity = None
    start_ts = None
    waiting_since = None
    prev = None
    for r in rows:
        ts, mode, crop = r["timestamp_ms"], r["mode"], r["crop_norm"]
        if not (0 <= crop[0] < crop[2] <= 1 and 0 <= crop[1] < crop[3] <= 1):
            raise ValueError(f"out of bounds crop at {ts}")
        modes[mode] += 1
        if r["intent_state"] == "SHOWING":
            showing += 1
            box = r["target_bbox_norm"]
            if box is None:
                missing += 1
            else:
                visible += 1
                retained += int(crop[0] <= box[0]+1e-8 and crop[1] <= box[1]+1e-8
                                and crop[2] >= box[2]-1e-8 and crop[3] >= box[3]-1e-8)
        if prev is None or mode != prev["mode"]:
            transitions += int(prev is not None)
            if prev and prev["mode"] in durations:
                durations[prev["mode"]].append(ts-start_ts)
            start_ts = ts
        if r["reason"] == "ACQUIRE":
            waiting_since = ts
        elif r["intent_state"] != "SHOWING":
            waiting_since = None
        if prev is not None:
            before = prev["crop_norm"]
            step = max(abs(x-y) for x, y in zip(before, crop))
            steps.append(step)
            zoom_steps.append(abs(r["zoom"] - prev["zoom"]))
            if waiting_since is not None and step > 1e-6:
                responses.append(ts - waiting_since)
                waiting_since = None
            if mode == prev["mode"] == "CLOSEUP" and r["target_id"] == prev["target_id"]:
                dt = (ts-prev["timestamp_ms"])/1000
                dx, dy = ((crop[i]+crop[i+2]-before[i]-before[i+2])/2 for i in (0, 1))
                steady_motion.append(hypot(dx, dy)/dt)
                if previous_velocity is not None:
                    vx, vy = previous_velocity
                    steady_jitter.append(hypot(dx-vx*dt, dy-vy*dt))
                previous_velocity = (dx/dt, dy/dt)
            else:
                previous_velocity = None
        prev = r
    return dict(frames=len(rows), modes=dict(modes), showing=showing, visible=visible,
                retained=retained, missing=missing, mode_changes=transitions,
                max_zoom=max(r["zoom"] for r in rows),
                edge_step=describe(steps), zoom_step=describe(zoom_steps),
                steady_center_speed_per_s=describe(steady_motion),
                steady_center_jitter=describe(steady_jitter),
                transition_ms={k: describe(v) for k, v in durations.items()},
                request_to_motion_ms=describe(responses))

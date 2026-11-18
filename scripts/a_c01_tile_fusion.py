"""Diagnostic multi-crop fusion for the initial C01 tabletop sequence."""
import json
from pathlib import Path

import cv2
from ultralytics import YOLO

from src.perception.tracker import iou


def main():
    root = Path(__file__).resolve().parents[1]
    rows = [json.loads(line) for line in (root / "outputs/a_local_candidate/C01/raw.jsonl").read_text(encoding="utf-8").splitlines()]
    video = next((root.parent / "received_media").rglob("C01.mp4"))
    model = YOLO(str(root / "yolo11s.pt"))
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_ORIENTATION_AUTO, 0)
    angle = int(cap.get(cv2.CAP_PROP_ORIENTATION_META)) % 360
    rotations = {90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}
    output = root / "outputs/a_c01_tile_candidate"
    output.mkdir(parents=True, exist_ok=True)
    writer = None
    tracker = {}
    next_id = 1
    report = []
    selected = rows[:90]
    for row in selected:
        cap.set(cv2.CAP_PROP_POS_FRAMES, row["frame_id"])
        ok, frame = cap.read()
        if not ok:
            raise RuntimeError(f"Cannot decode C01 frame {row['frame_id']}")
        if angle in rotations:
            frame = cv2.rotate(frame, rotations[angle])
        h, w = frame.shape[:2]
        specs = [(0, int(.60*h), w, h), (0, int(.60*h), int(.68*w), h),
                 (int(.22*w), int(.60*h), int(.78*w), h), (int(.32*w), int(.60*h), w, h)]
        detections = []
        for x0, y0, x1, y1 in specs:
            pred = model(frame[y0:y1, x0:x1], imgsz=1280, conf=.20, verbose=False)[0]
            for box in pred.boxes:
                name = pred.names[int(box.cls[0])]
                if name != "apple":
                    continue
                ax1, ay1, ax2, ay2 = [float(v) for v in box.xyxy[0].tolist()]
                b = ((ax1+x0)/w, (ay1+y0)/h, (ax2+x0)/w, (ay2+y0)/h)
                # Suppress broad group boxes only when their center is covered by
                # at least two substantially smaller apple detections.
                area = (b[2]-b[0])*(b[3]-b[1])
                detections.append({"bbox":b,"score":float(box.conf[0]),"area":area,"source":"tile"})
        # Include cached full-frame detections as the reference proposals.
        for obj in row["objects"]:
            if obj["class_name"] == "apple":
                b=tuple(obj["bbox_norm"])
                detections.append({"bbox":b,"score":obj["score"],"area":(b[2]-b[0])*(b[3]-b[1]),"source":"global"})
        rejected = set()
        for i, big in enumerate(detections):
            x1,y1,x2,y2=big["bbox"]
            if big["area"] < .012:
                continue
            cx,cy=(x1+x2)/2,(y1+y2)/2
            contained=[]
            for j,small in enumerate(detections):
                if i==j or small["area"] >= big["area"]*.58:
                    continue
                a,b,c,d=small["bbox"]
                if a>=x1 and b>=y1 and c<=x2 and d<=y2 and small["area"]>0:
                    contained.append(j)
            if len(contained)>=2:
                rejected.add(i)
        kept=[]
        for i,d in enumerate(detections):
            if i in rejected:
                continue
            match=next((x for x in kept if iou(x["bbox"],d["bbox"])>.45),None)
            if match is None:
                kept.append(d)
            elif (d["source"] == "tile" and d["score"] > match["score"]) or (match["source"]=="tile" and d["source"]=="tile" and d["score"] > match["score"]):
                match.update(d)
        # If two retained same-class boxes overlap strongly, don't display both
        # as separate fruit instances. Prefer the better local proposal.
        deduped=[]
        for d in sorted(kept,key=lambda x:(x["source"]=="tile",x["score"]),reverse=True):
            if any(iou(x["bbox"],d["bbox"])>.28 for x in deduped):
                continue
            deduped.append(d)
        kept=deduped
        # Greedy frame association, only for readable review labels.
        for t in tracker.values():
            t["missed"]+=1
        used=set()
        for d in sorted(kept,key=lambda x:x["score"],reverse=True):
            x1,y1,x2,y2=d["bbox"]
            candidates=[]
            for tid,t in tracker.items():
                if tid in used or t["missed"]>15:
                    continue
                overlap=iou(t["bbox"],d["bbox"])
                if overlap>.15:
                    candidates.append((overlap,tid))
            if candidates:
                _,tid=max(candidates)
                tracker[tid].update(bbox=d["bbox"],missed=0)
            else:
                tid=f"obj_{next_id}"; next_id+=1
                tracker[tid]={"bbox":d["bbox"],"missed":0}
            d["track_id"]=tid; used.add(tid)
        canvas=frame.copy()
        for d in kept:
            x1,y1,x2,y2=[int(v*s) for v,s in zip(d["bbox"],(w,h,w,h))]
            cv2.rectangle(canvas,(x1,y1),(x2,y2),(50,230,70),3)
            cv2.rectangle(canvas,(x1,y1-28),(x1+95,y1),(0,0,0),-1)
            cv2.putText(canvas,d["track_id"],(x1+4,y1-7),cv2.FONT_HERSHEY_SIMPLEX,.65,(80,255,100),2)
        cv2.rectangle(canvas,(0,0),(w,38),(0,0,0),-1)
        cv2.putText(canvas,"C01 overlapping crop fusion (experimental)",(10,27),cv2.FONT_HERSHEY_SIMPLEX,.7,(255,255,255),2)
        if writer is None:
            writer=cv2.VideoWriter(str(output/"review_first3s.mp4"),cv2.VideoWriter_fourcc(*"mp4v"),30,(w,h))
        writer.write(canvas)
        report.append({"frame_id":row["frame_id"],"timestamp_ms":row["timestamp_ms"],"n_before_suppression":len(detections),"merged_candidates_rejected":len(rejected),"n_after_fusion":len(kept),"objects":[{"track_id":d["track_id"],"bbox_norm":list(d["bbox"]),"score":d["score"]} for d in kept]})
    cap.release()
    if writer:
        writer.release()
    (output/"diagnostics.json").write_text(json.dumps(report,indent=2),encoding="utf-8")
    print(output)


if __name__=="__main__":
    main()

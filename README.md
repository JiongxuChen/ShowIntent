# ShowIntent

## A / Visual Perception (P0)

The initial A module is under `src/perception`. It emits one `PerceptionPacket` JSON object per frame for B (presentation intent) and C (camera control). A detects candidate hands and objects, maintains short-term object IDs, and emits candidate hand-object relations; it does not decide whether an object is being presented.

Install the optional A dependencies:

```powershell
python -m pip install -r requirements-a.txt
```

Run the shared mock and validate its interface:

```powershell
python -m src.perception.make_mock --out fixtures\demo_object_show\perception.jsonl
python -m src.perception.validate_jsonl fixtures\demo_object_show\perception.jsonl
```

Run perception on a video or camera (`--input 0`):

```powershell
python -m src.perception.run --input path\to\clip.mp4 --out outputs\perception.jsonl --overlay-out outputs\perception_overlay.mp4 --metrics-out outputs\metrics.json --imgsz 1280 --rotate auto
```

The fixture and protocol are intentionally independent of any predeclared target object. The received evaluation videos carry a 90-degree rotation flag; `--rotate auto` applies it before detection and writes coordinates in the upright frame. The current evaluation report is in `docs/perception_video_report.md`. Original evaluation videos should remain local and should not be committed to the repository.

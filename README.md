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

## B / Presentation Intent (P0)

`src/intent` consumes A's PerceptionPacket JSONL and emits one `IntentPacket` per frame (`state` ∈ `NONE | CANDIDATE | SHOWING | ENDING`). It is rule-based and pure vision: no ASR, no learned model, and no predeclared target. A hand-object relation is only candidate evidence; SHOWING requires the object to be lifted above its table baseline and held stably for several consecutive frames, so ordinary carrying/tidying stays wide. Put-down or long target loss transitions through ENDING back to NONE, with hysteresis (different enter/exit thresholds) to avoid flicker.

```powershell
python -m src.intent.run --perception fixtures\demo_object_show\perception.jsonl --out outputs\intent.jsonl --mode showintent
python -m src.intent.run --perception outputs\perception_A02.jsonl --out outputs\intent_A02_baseline.jsonl --mode pickup-baseline
```

`--mode pickup-baseline` runs the naive pick-up trigger baseline (fires on any upward displacement) for false-trigger comparison; it emits the same IntentPacket schema. Thresholds are documented in `docs/intent_camera_interface.md`.

## C / Camera Controller (P0)

`src/camera` consumes PerceptionPacket + IntentPacket and emits one `ShotPacket` per frame (`mode` ∈ `WIDE | TRANSITION_IN | CLOSEUP | TRANSITION_OUT`). Crops are pure geometry (bbox + 30% margin, aspect-fit, clamped to [0,1]) smoothed with an EMA; brief target dropouts hold the last stable crop, longer ones ease back to WIDE. `--force-wide` / `set_manual_override(True)` is the human override hook. No future frames are used.

```powershell
python -m src.camera.run --perception outputs\perception_A02.jsonl --intent outputs\intent_A02.jsonl --out outputs\shot_A02.jsonl
python -m src.camera.run --video clip.mp4 --perception p.jsonl --intent i.jsonl --out outputs\shot.jsonl --overlay-out outputs\preview.mp4
```

The preview render applies the container rotation exactly once, matching A's coordinate frame. The full A→B→C contract is specified in `docs/intent_camera_interface.md`.

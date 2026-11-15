# ShowIntent P0 Demo Runbook

This is the shortest reproducible path for the final demonstration. It uses the existing A outputs and keeps the original videos outside Git.

## 1. Environment and regression check

```powershell
python -m pytest tests -q
```

Expected result: all tests pass.

## 2. Run B on one perception stream

```powershell
python -m src.intent.run `
  --perception "..\outputs\perception_A02.jsonl" `
  --out outputs\intent_A02.jsonl `
  --mode showintent
```

The intent output contains `NONE`, `CANDIDATE`, `SHOWING`, and `ENDING` packets. B never receives a predeclared target.

## 3. Run C and render a preview

```powershell
python -m src.camera.run `
  --video "..\received_media\A01-04&B01-03\A02.mp4" `
  --perception "..\outputs\perception_A02.jsonl" `
  --intent outputs\intent_A02.jsonl `
  --out outputs\shot_A02.jsonl `
  --overlay-out outputs\preview_A02.mp4
```

The source videos carry a 90-degree rotation flag. C must apply the same upright-frame convention as A; do not rotate A's normalized boxes a second time.

## 4. Recommended final clips

- `A02`: positive orange/tangerine presentation
- `B03`: hard negative; should remain wide
- `C02`: apple-to-tangerine target switch
- `D01`: complete mixed demo

For a human review, verify target selection, crop boundaries, smooth transitions, return to wide after ending, and no false close-up during B03.

## 5. Known limitations to state in the presentation

- Same-object `track_id` can fragment after occlusion or close-up changes.
- The current P0 pipeline runs as a reproducible replay pipeline; do not claim a fixed real-time FPS without measuring on the presentation machine.
- The object detector's `orange` class is the model vocabulary label; the dataset item is a tangerine (橘子).
- Ambiguous behavior is handled conservatively and is not a claim of reading private intent.


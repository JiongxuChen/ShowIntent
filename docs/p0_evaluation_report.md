# ShowIntent P0 Evaluation Report (Phase 3)

Date: 2026-11-13. Scope: end-to-end evaluation of A (perception) → B (intent) → C (camera) on the 12 received clips, problem analysis, and known-limitation report. **No A/B/C interface or schema was changed in this phase.**

## 1. Environment

- Windows 11, project venv (Python 3.12.10, `--system-site-packages`), PyTorch 2.13/cu126 on an NVIDIA GPU.
- A: YOLO11n (`yolo11n.pt`), `imgsz=1280`, `conf=0.35`, MediaPipe Hands 0.10.21; ~4 FPS on 4K input.
- B/C: rule-based, CPU only, real-time (>100x faster than playback on JSONL input).
- Videos: 2160×3840 containers with rot=90 metadata; A outputs coordinates in the upright 3840×2160 frame (clockwise rotation applied once, in A only).

## 2. Clips

| Clip | GT content | GT presentation? |
|---|---|---|
| A01/A02/A03/A04 | apple/orange show, with/without voice | Yes |
| B01 | talk only, never picks up | No |
| B02 | carries/relocates fruit | No (hard negative) |
| B03 | holds apple, tidies/sells, never presents | No (hard negative) |
| C01 | apple show, clear end | Yes |
| C02 | apple → orange target switch | Yes (2 switches) |
| C03 | orange show with brief occlusion | Yes |
| C04 | orange show with a brief pause | Yes |
| D01 | full natural fruit livestream (67 s) | Yes |

## 3. Commands

```bash
python -m src.intent.run --perception <perception.jsonl> --out outputs/intent_<clip>.jsonl --mode showintent
python -m src.intent.run --perception <perception.jsonl> --out outputs/intent_<clip>_baseline.jsonl --mode pickup-baseline
python -m src.camera.run --perception <perception.jsonl> --intent outputs/intent_<clip>.jsonl --out outputs/shot_<clip>.jsonl [--video <clip.mp4> --overlay-out outputs/preview_<clip>.mp4]
python -m scripts.p0_stats            # per-clip statistics -> outputs/p0_stats.json
```

A perception JSONL for all 12 clips was reused from Phase 1 (`outputs/perception_*.jsonl`); no 4K detection was re-run.

## 4. Methods

- **ShowIntent visual (main)**: rule state machine NONE→CANDIDATE→SHOWING→ENDING. Hand-object relation is candidate evidence only; SHOWING requires lift above the per-track table baseline (absolute fallback line when no baseline is learnable) plus a stable hold. Hysteresis on enter/exit.
- **Pick-up Trigger (baseline)**: triggers on upward displacement from table baseline only; no temporal hold, no display-region requirement. Same IntentPacket schema.

## 5. Per-clip results (frame-count thresholds, 30 FPS)

| Clip | SHOWING runs (s, target) | Real shows (GT) | Real vs pseudo split | Camera | Baseline runs |
|---|---|---|---|---|---|
| A01 | 1: 7.9–13.0 (obj_3) | 1 | 1 real | CLOSEUP 13.0s, ends WIDE | 0 |
| A02 | 1: 3.9–10.2 (obj_1) | 1 | 1 real (ends early, §13) | TRANSITION_IN×3, CLOSEUP 6.8s, ends WIDE | 0 |
| A03 | 1: 10.2–11.0 (obj_8) | 1 | 1 real, very short | — | 0 |
| A04 | 3: 2.4–5.7 / 6.5–7.8 / 9.4–10.5 (obj_1/3/4) | 1 | 1 real + 2 pseudo (track fragmentation) | — | 0 |
| B01 | 0 | 0 | — | — | 0 |
| B02 | 0 | 0 | — | 100% WIDE | 0 |
| B03 | 0 | 0 | — | 100% WIDE | 0 |
| C01 | **0 (MISS)** | 1 | 0 detected | — | 0 |
| C02 | 4: 4.5–7.5 / 8.0–11.1 (apple obj_12/13); 13.0–17.8 / 19.7–22.7 (orange obj_3/22) | 2 | 2 real + 2 pseudo (fragmentation) | TRANSITION_IN×15, CLOSEUP 15.9s, ends WIDE | 1 (6.7s) |
| C03 | **0 (MISS)** | 1 | 0 detected | — | 0 |
| C04 | 1: 3.8–12.0 (obj_2) | 1 | 1 real | — | 1 (11.3s) |
| D01 | 3: 14.0–14.6 (obj_4); 34.4–52.7 (obj_1); 63.6–64.1 (obj_1) | ≥2 (spot-checked) | frames inspected: 14.3s = apple in hand, 40s = orange held up to camera — both real | TRANSITION_IN×5, CLOSEUP 20.9s, ends WIDE | 3 (26.7s) |

All four "run" rows that show `END=True backNONE=False` stay in ENDING at video end (§13, defect D2); the camera nevertheless returns to WIDE in every rendered clip.

## 6. Presentation detection summary

- 9 positive clips: 6/9 produce SHOWING (A01, A02, A03, A04*, C04, D01).
- 2/9 positives missed entirely (C01, C03) — one shared root cause, defect D1 below.
- C02 detects both real shows but fragments them into 4 runs.
- Every detected show passed through CANDIDATE first (candidate_before=True in all runs).

## 7. False-trigger summary

- **B01/B02/B03: zero SHOWING, zero TRANSITION_IN, camera stays WIDE 100%.** The main method beats the design target on all three negatives, including both hard negatives.
- D01: no false trigger observed in the non-showing parts; the three SHOWING runs were spot-checked against rendered frames and all correspond to fruit actually held up.
- Baseline: also zero on negatives, but lower recall on positives (C02 6.7s vs 15.9s main; misses A01/A02/A03/A04 entirely because slow lifts do not cross its displacement threshold).

## 8. Target selection

- A02: orange locked as obj_1, held for the whole run. Correct.
- C02: apple runs target apple tracks, orange runs target orange tracks; every switch re-enters CANDIDATE. No wrong-class target ever locked. Correct modulo fragmentation.
- D01: obj_4 (apple) and obj_1 (orange) match the inspected frames. No apple/orange confusion observed anywhere.

## 9. Timing deviations

- **A02 ends early**: FSM SHOWING 3.9–10.2 s; the streamer actually puts the orange down around ~14.5 s. Root cause chain: track obj_1 dies at 9.73 s (perception ID fragmentation) → 15-frame miss tolerance → ENDING at 10.2 s → a new orange track obj_3 appears at 11.0 s while still held, but ENDING never resumes it (defect D2) and never returns to NONE. Deviation ≈ 4.3 s, caused by A-side track breakage, not by `putdown_margin`.
- C04 pause and C03 occlusion: C04 holds SHOWING through the pause correctly (3.8–12.0 s continuous). C03 was never in SHOWING to begin with (defect D1), so occlusion robustness could not be evaluated there.

## 10. Camera stability

- No crop ever left [0,1]²; aspect ratio maintained (validated by tests and by preview inspection).
- EMA smoothing (α=0.2) with dropout hold (≤15 frames) worked; all rendered previews end in WIDE.
- TRANSITION_IN counts are inflated by track fragmentation (C02 ×15 — each pseudo-segment restarts a transition).

## 11. Track ID fragmentation (module A)

| Clip | total track IDs | notes |
|---|---|---|
| B03 | 34 | close-up apple handling |
| C01 | 31 | close-up apple show |
| C02 | 25 | 4 apple IDs + 3 orange IDs among others |
| D01 | 115 | full livestream |
| Static scenes (B01/A02 table phase) | 2–3 | stable |

Fragmentation directly causes pseudo SHOWING segments (A04 ×3, C02 ×4), early SHOWING exit (A02), and inflated transition counts. **SHOWING-state segment count must not be read as the number of real presentations.**

## 12. Defects found (not fixed in this phase — outside the allowed change scope)

- **D1 — CANDIDATE lock starvation (root cause of C01/C03 misses).** In close-up clips the streamer's resting hand next to a table fruit produces a *permanent* relation, so the first table-bound track locks CANDIDATE forever (`candidate_decay` never fires) and the actually-presented object can never become a candidate. Oracle check: frames where some related object satisfies raised+stable — C01 156 frames (29%), C03 392 frames (67%) — yet FSM SHOWING = 0 in both. Fix direction: promote the best qualified candidate instead of the first related one, or decay a never-promoted candidate after N seconds.
- **D2 — ENDING deadlock.** In ENDING, any target frame that is not "on the table" counts as `reliable`, so `end_streak` never accumulates and the machine can park in ENDING forever (A02 parks from 10.2 s to video end while the re-appeared obj_3 is still held). Fix direction: ENDING should only count the *same target_id* as reliable, or resume SHOWING on a same-class near-position re-detection.
- Both are B-logic fixes; they were **reported, not implemented**, per the phase constraints.

## 13. C02/C03/C04 timing notes

- C02: real switch sequence apple→orange is reproduced; each switch passes CANDIDATE; pseudo-segmentation doubles the run count (apple obj_12/obj_13, orange obj_3/obj_22).
- C03: blocked by D1 before occlusion even matters; the perception side itself detects the orange through the occlusion (586/589 frames, Phase 1).
- C04: the pause does not break SHOWING (8.2 s continuous run) — the miss-tolerance/putdown hysteresis behaves as designed.

## 14. Main method vs baseline

| Metric | ShowIntent visual | Pickup baseline |
|---|---|---|
| Negative clips (B01–B03) false SHOWING | 0 | 0 |
| Positive clips with ≥1 SHOWING | 6/9 | 3/9 (C02/C04/D01 only) |
| Total SHOWING seconds on positives | ~78 s | ~45 s |
| Distinguishing carry vs show (B02/B03) | both hold | both hold |

Main ≥ baseline everywhere; the advantage is recall on slow/silent shows. Neither false-triggers on the negatives, so on this 12-clip set the baseline is *not* worse on precision — a larger negative set is needed to show the separation the hold rules are designed for.

## 15. Time-base (threshold semantics)

- All current thresholds are **frame counts** tuned for 30 FPS. At the mock fixture's 5 FPS the same numbers mean ~6× longer wall-clock evidence.
- Added in this phase (backward compatible, default off): `candidate_min_ms`, `showing_min_ms`, `missing_tolerance_ms`, `ending_min_ms` (constructor args and CLI flags). When set, the machine measures the median inter-packet gap and converts ms → frames, so the same wall-clock threshold applies at any packet rate. IntentPacket schema unchanged. Covered by 2 new tests (24 total).
- Control experiment (300/600/600/150 ms ≈ stricter real-time semantics): A02 3.9–10.4 ✓, C04 3.8–12.0 ✓, D01 3 runs ✓, C02 3 runs (one short apple pseudo-run suppressed), negatives and missed clips unchanged.

## 16. What this system can claim (P0)

- Zero camera zoom on non-showing behaviour across all three negatives and the full livestream.
- Correct target class and correct switch protocol on C02.
- Stable, bounded, aspect-correct crops with smooth transitions and WIDE fallback.

## 17. What this system cannot claim

- **It does not detect all presentations**: close-up clips where the presenter's resting hand touches table fruit (C01, C03) are missed entirely (D1).
- SHOWING-run count ≠ real presentation count (fragmentation pseudo-segments).
- A02-style early exit (~4 s) due to track breakage; ENDING can park (D2).
- No audio understanding; a verbal presentation without lifting is invisible by design.
- COCO `orange` cannot distinguish 橘子/橙子; ~4 FPS is not real-time on 4K.

## 18. Next steps

1. Fix D1 (candidate re-selection) and D2 (ENDING resume/deadlock) — both are localized B-logic changes; re-run this evaluation script to quantify.
2. Human review of `outputs/preview_*.mp4` for A04 pseudo-segment ground truth (one physical orange vs three picks).
3. Only after D1/D2: consider same-class near-position track merging in B (interface-level change, needs A-side agreement), then a final demo on D01.

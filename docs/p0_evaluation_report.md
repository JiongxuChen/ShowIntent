# ShowIntent P0 Evaluation Report (Phase 3, updated after D1/D2 fixes in Phase 4)

Date: 2026-11-13. Scope: end-to-end evaluation of A (perception) → B (intent) → C (camera) on the 12 received clips, problem analysis, defect fixes and known-limitation report. **No A/B/C interface or schema was changed.**

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

## 5. Per-clip results (after D1/D2 fixes, frame-count thresholds, 30 FPS)

| Clip | SHOWING runs (s, target) | Real shows (GT) | Real vs pseudo split | Camera | Baseline runs |
|---|---|---|---|---|---|
| A01 | 1: 6.5–13.0 (obj_3) | 1 | 1 real (starts 1.4 s earlier than before) | TRANSITION_IN×5, CLOSEUP 7.0s, ends WIDE | 0 |
| A02 | 2: 3.9–10.2 (obj_1); 14.1–14.4 (obj_3, re-detected after track break) | 1 | 1 real + 1 brief re-detect | TRANSITION_IN×4, CLOSEUP 7.7s, ends WIDE | 0 |
| A03 | 2: 3.7–4.6 (obj_4, frame-verified); 10.2–11.0 (obj_8) | 1 | 2 real (the early lift was missed before) | TRANSITION_IN×3, CLOSEUP 2.8s, ends WIDE | 0 |
| A04 | 3: 2.4–5.7 (obj_1); 6.5–7.8 (obj_3); 7.9–10.5 (obj_4) | 1 | 1 real + 2 pseudo (fragmentation) | TRANSITION_IN×13, CLOSEUP 8.4s, ends WIDE | 0 |
| B01 | 0 | 0 | — | 100% WIDE | 0 |
| B02 | 0 | 0 | — | 100% WIDE | 0 |
| B03 | 0 | 0 | — | 100% WIDE | 0 |
| C01 | 5: 3.4–5.1 / 5.2–6.3 / 6.3–8.1 / 8.2–8.7 / 12.0–12.7 (all apple tracks) | 1 | 1 real (frame-verified), fragmented into 5 pseudo-segments | TRANSITION_IN×17, CLOSEUP 7.1s, ends WIDE | 0 |
| C02 | 4: 4.5–7.5 / 8.0–11.1 (apple); 13.0–17.8 / 19.7–22.7 (orange) | 2 | 2 real + 2 pseudo | TRANSITION_IN×15, CLOSEUP 15.9s, ends WIDE | 1 (6.7s) |
| C03 | 2: 5.0–12.2 (obj_8, covers the occlusion); 15.4–16.3 (obj_10) | 1 | 1 real (frame-verified) + 1 brief re-detect | TRANSITION_IN×3, CLOSEUP 9.1s, ends WIDE | 0 |
| C04 | 1: 3.2–12.0 (obj_2) | 1 | 1 real | TRANSITION_IN×1, CLOSEUP 9.2s, ends WIDE | 1 (11.3s) |
| D01 | 4: 14.0–14.6 (obj_4); 14.7–24.3 (obj_33, frame-verified apple show); 34.4–52.7 (obj_1); 63.6–64.1 (obj_1) | ≥3 | spot-checked: all real | TRANSITION_IN×7, CLOSEUP 30.5s, ends WIDE | 3 (26.7s) |

Before the fixes, C01/C03 produced **0** SHOWING runs and A02 parked in ENDING until video end; see §12 for the before/after comparison.

## 6. Presentation detection summary

- 9 positive clips: **9/9 produce SHOWING** after the fixes (was 6/9).
- Every detected show passed through CANDIDATE first.
- Remaining inaccuracy is segmentation, not detection: close-up clips split one real show into several pseudo-segments (C01 ×5, C02 ×4, A04 ×3) because A's track IDs fragment.

## 7. False-trigger summary

- **B01/B02/B03: zero SHOWING, zero TRANSITION_IN, camera stays WIDE 100%** — unchanged by the fixes (regression-tested).
- D01: all four SHOWING runs spot-checked against rendered frames; all correspond to fruit actually held up.
- Baseline: also zero on negatives, but lower recall on positives (3/9 clips).

## 8. Target selection

- A02: orange locked as obj_1, held for the whole run. Correct.
- C02: apple runs target apple tracks, orange runs target orange tracks; every switch re-enters CANDIDATE. No wrong-class target ever locked. Correct modulo fragmentation.
- D01: obj_4 (apple) and obj_1 (orange) match the inspected frames. No apple/orange confusion observed anywhere.

## 9. Timing deviations

- **A02 early exit — resolved by D2.** Before: SHOWING 3.9–10.2 s then parked in ENDING until video end. After: the machine exits ENDING correctly; the re-appearing orange track obj_3 (still held, ~11.0 s) is re-linked / re-candidated and yields a second short SHOWING at 14.1–14.4 s, and the clip ends in NONE/CANDIDATE instead of ENDING. The residual gap versus the ~14.5 s put-down comes from A-side track breakage, which B now recovers from instead of deadlocking.
- C04 pause: SHOWING holds through the pause (3.2–12.0 s continuous).
- C03 occlusion: SHOWING holds through the occlusion (5.0–12.2 s continuous, frame-verified).

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

## 12. Defects D1/D2 — root cause and fix (Phase 4)

### D1 — CANDIDATE lock starvation (fixed)

**Root cause.** In close-up clips the streamer's resting hand next to a table fruit produces a permanent relation, so the first table-bound track entered CANDIDATE and, with `candidate_decay` never firing, occupied it forever; the actually-presented object could never become a candidate. Oracle: C01 156 frames (29%), C03 392 frames (67%) had a fully qualified target, yet FSM SHOWING was 0.

**Fix.** In CANDIDATE the machine now re-evaluates every frame. If the current candidate has been stuck below promotion for `candidate_switch_frames` (default 90 ≈ 3 s at 30 FPS) and another related object simultaneously satisfies raised + stable + relation streak ≥ `candidate_frames`, the candidate is replaced: `target_id` switches, the streak/age reset, and the packet carries `CANDIDATE_REPLACED`. The replacement emits CANDIDATE (never SHOWING) and must earn its own promotion streak, so a switch always re-passes the candidate stage. SHOWING keeps its target locked (unchanged).

### D2 — ENDING deadlock (fixed)

**Root cause.** In ENDING, any target frame that was not "on the table" counted as `reliable`, resetting `end_streak`; with the original track broken and a new track held in hand, the machine parked in ENDING forever (A02: 10.2 s → video end).

**Fix.** Three changes: (1) ENDING counts `_ending_frames` and force-returns to NONE after `ending_max_frames` (default 60 ≈ 2 s; `--ending-max-ms` available), so it can never outlive the clip. (2) The "not on table ⇒ reliable" reset was removed: while the original target is visible, only `related + raised` resumes SHOWING; anything else increments `end_streak` toward NONE. (3) When the original track is gone, within `relink_max_frames` (45 ≈ 1.5 s) a same-class, spatially close (`relink_max_dist` = 0.15), currently raised and hand-related track may resume SHOWING with `TARGET_RELINKED`; wrong-class or distant tracks never resume the timer.

### Before / after comparison

| Metric | before (ba466ec) | after (D1/D2 fixed) |
|---|---|---|
| Positive clips with ≥1 SHOWING | 6/9 | **9/9** |
| C01 / C03 SHOWING | 0 / 0 | 5 runs / 2 runs (frame-verified real) |
| A02 final state | ENDING (deadlock) | NONE / CANDIDATE |
| Negative clips (B01–B03) false SHOWING | 0 | 0 (unchanged) |
| C02 apple→orange protocol (re-CANDIDATE) | correct | correct (unchanged) |
| A03 detection | 1 short run | 2 runs (early lift now detected, frame-verified) |
| D01 real shows detected | 3 runs | 4 runs (14.7–24.3 s apple show added, frame-verified) |
| Tests | 24 passed | 32 passed (3 real-data regression tests included) |

New reason codes: `CANDIDATE_REPLACED`, `TARGET_RELINKED`. New parameters: `candidate_switch_frames=90`, `ending_max_frames=60` (+`ending_max_ms`), `relink_max_frames=45`, `relink_max_dist=0.15`. All defaults keep prior behaviour safe and no schema changed.

## 13. C02/C03/C04 timing notes

- C02: real switch sequence apple→orange is reproduced; each switch passes CANDIDATE; pseudo-segmentation doubles the run count (apple obj_12/obj_13, orange obj_3/obj_22).
- C03: after D1 the show is detected (5.0–12.2 s) and SHOWING holds through the occlusion; perception itself detects the orange through the occlusion (586/589 frames, Phase 1).
- C04: the pause does not break SHOWING (3.2–12.0 s continuous run) — the miss-tolerance/putdown hysteresis behaves as designed.

## 14. Main method vs baseline

| Metric | ShowIntent visual | Pickup baseline |
|---|---|---|
| Negative clips (B01–B03) false SHOWING | 0 | 0 |
| Positive clips with ≥1 SHOWING | **9/9** | 3/9 (C02/C04/D01 only) |
| Total SHOWING seconds on positives | ~102 s | ~45 s |

Main ≥ baseline everywhere; the advantage is recall on slow/silent/close-up shows. Neither false-triggers on the negatives.

## 15. Time-base (threshold semantics)

- All current thresholds are **frame counts** tuned for 30 FPS. At the mock fixture's 5 FPS the same numbers mean ~6× longer wall-clock evidence.
- Added in this phase (backward compatible, default off): `candidate_min_ms`, `showing_min_ms`, `missing_tolerance_ms`, `ending_min_ms` (constructor args and CLI flags). When set, the machine measures the median inter-packet gap and converts ms → frames, so the same wall-clock threshold applies at any packet rate. IntentPacket schema unchanged. Covered by 2 new tests (24 total).
- Control experiment (300/600/600/150 ms ≈ stricter real-time semantics): A02 3.9–10.4 ✓, C04 3.8–12.0 ✓, D01 3 runs ✓, C02 3 runs (one short apple pseudo-run suppressed), negatives and missed clips unchanged.

## 16. What this system can claim (P0)

- Detects at least one SHOWING on **all 9 positive clips**, including close-up (C01), occlusion (C03) and the full livestream (D01).
- Zero camera zoom on non-showing behaviour across all three negatives.
- Correct target class and correct switch protocol on C02; A02/C03 recover from track breakage instead of deadlocking.
- Stable, bounded, aspect-correct crops with smooth transitions and WIDE fallback.

## 17. What this system cannot claim

- SHOWING-run count ≠ real presentation count (fragmentation pseudo-segments; C01's single real show yields 5 runs).
- Residual timing noise: brief re-detect runs (A02 14.1–14.4 s, C03 15.4–16.3 s) and ±0.6 s start-time shifts.
- No audio understanding; a verbal presentation without lifting is invisible by design.
- COCO `orange` cannot distinguish 橘子/橙子; ~4 FPS is not real-time on 4K.

## 18. Next steps

1. Human review of `outputs/preview_*.mp4` (A04 pseudo-segment ground truth: one physical orange vs three picks).
2. Optional polish: merge adjacent same-class SHOWING runs separated by a relink (C01's 5 runs would collapse to 1–2) — interface-level change, needs A-side agreement on track identity.
3. Final demo on D01 with the current pipeline.

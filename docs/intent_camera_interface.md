# B / C 接口说明（P0）

本文档定义第二阶段 A → B → C 的数据契约与行为参数。PerceptionPacket schema 未做任何修改。

## 数据流

```
src/perception (A)  ──PerceptionPacket JSONL──►  src/intent (B)  ──IntentPacket JSONL──►
src/camera (C)  ──ShotPacket JSONL + 预览视频──► 展示
```

- 三个 JSONL 均为每帧一行，行数一致、顺序一致（按 frame_id 对齐）。
- 所有 bbox/crop 坐标基于**旋转矫正后的正立画面**（本数据集为容器 rot=90 → 顺时针旋转 90°，3840×2160）。
- B/C 都不处理视频旋转；只有 C 渲染预览时读取原始视频，需要按同一规则旋转一次。
- B 不接受任何"主播预先选择目标"的输入；A 输出什么就看什么。

## IntentPacket（B 输出，每帧一行）

```json
{"timestamp_ms": 2600, "state": "SHOWING", "target_id": "obj_7", "score": 0.82,
 "reason_codes": ["HAND_OBJECT_ASSOCIATED", "RAISED_IN_FRAME", "TEMPORAL_HOLD"]}
```

- `state` ∈ `NONE | CANDIDATE | SHOWING | ENDING`
- `target_id`：SHOWING/ENDING 时锁定；NONE 时为 null；CANDIDATE 时为候选对象
- `reason_codes`：`HAND_OBJECT_ASSOCIATED`（有手物关系）、`RAISED_IN_FRAME`（中心高于桌面基线或进入展示区）、`TEMPORAL_HOLD`（窗口内位移平稳）、`TARGET_LOST`、`BACK_ON_TABLE`、`INSUFFICIENT_EVIDENCE`

### 状态机（`src/intent/state_machine.py`，`--mode showintent`）

| 转移 | 条件（默认参数，30fps 语义） |
|---|---|
| NONE → CANDIDATE | 同一 track 连续 ≥3 帧有 `near_or_moving_together` 关系（score≥0.3） |
| CANDIDATE → SHOWING | 候选持续 ≥8 帧 **且** 目标中心高于桌面基线 0.12（基线未知时用绝对展示区 cy<0.72）**且** 最近 8 帧逐步位移 ≤0.05 |
| SHOWING → ENDING | 目标连续漏检 >15 帧，或目标回到桌面（cy ≥ 基线−0.05，连续 3 帧；基线未知时 cy≥0.75） |
| ENDING → NONE | 连续 3 帧没有可靠目标（或重新举起则直接回到 SHOWING） |
| CANDIDATE → NONE | 关系连续丢失 >8 帧 |

- 每个新目标必须重新经过 CANDIDATE（进入 CANDIDATE 的首帧只发 CANDIDATE，不允许同帧晋升）。
- SHOWING 期间锁定 target_id；短暂漏检不退出。
- 桌面基线：每 track 用"无手物关系帧"的中心 y 中位数估计（≥3 个样本）；不可得时用绝对阈值（举起线 0.72 / 桌面线 0.75，中间为缓冲带）。

### Baseline（`src/intent/baseline.py`，`--mode pickup-baseline`）

Pick-up Trigger：目标中心高于其桌面基线 0.10 即直接 SHOWING，无需关系持续、无需停留。用于对比误触发率。输出 schema 与主方法完全一致。

## ShotPacket（C 输出，每帧一行）

```json
{"timestamp_ms": 2633, "mode": "CLOSEUP", "target_id": "obj_7",
 "crop_norm": [0.28, 0.20, 0.78, 0.70], "zoom": 2.0, "manual_override": false}
```

- `mode` ∈ `WIDE | TRANSITION_IN | CLOSEUP | TRANSITION_OUT`
- `crop_norm`：[x1,y1,x2,y2]，按输出宽高比（默认 16:9）拟合、限制在 [0,1] 内
- `zoom` = 全画幅宽 / crop 宽

### 控制器（`src/camera/controller.py`）

| 规则 | 实现 |
|---|---|
| NONE/CANDIDATE → WIDE | crop 回归全画幅 |
| SHOWING → 特写 | crop = 目标 bbox 外扩 30%（每边，按 bbox 尺寸），16:9 拟合，clamp 到 [0,1] |
| 平滑 | 当前 crop 向目标 crop 做 EMA（α=0.2/帧），距离 ≤0.015 视为收敛 |
| 短暂漏检 | 目标缺失 ≤15 帧时冻结上一个稳定 crop（保持 CLOSEUP） |
| 超时漏检 / ENDING | crop 平滑回到全画幅（TRANSITION_OUT → WIDE） |
| 人工强制全景 | `CameraController.set_manual_override(True)` 或 CLI `--force-wide`，输出 `manual_override: true` |
| 无未来帧 | 逐帧顺序处理，仅用当前与历史数据 |

## CLI

```powershell
# B：mock
python -m src.intent.run --perception fixtures\demo_object_show\perception.jsonl --out outputs\intent.jsonl
# B：真实 A 输出 + baseline 对比
python -m src.intent.run --perception outputs\perception_A02.jsonl --out outputs\intent_A02.jsonl --mode showintent
python -m src.intent.run --perception outputs\perception_A02.jsonl --out outputs\intent_A02_baseline.jsonl --mode pickup-baseline
# C：仅 ShotPacket（无视频也可运行）
python -m src.camera.run --perception p.jsonl --intent i.jsonl --out outputs\shot.jsonl
# C：渲染裁剪预览（自动应用一次旋转）
python -m src.camera.run --video clip.mp4 --perception p.jsonl --intent i.jsonl --out outputs\shot.jsonl --overlay-out outputs\preview.mp4
```

## 已知限制（P0）

1. A 的 track_id 在手持特写/遮挡后会更换（无重识别）；B 的 SHOWING 因此可能在同一物理目标上出现多段（C02 实测 2 次展示 → 4 段 SHOWING）。段间都会重新经过 CANDIDATE，不会直接换镜头目标。
2. 手长时间搭在桌面旁时，桌上物体也会持续产生 relation，B 只用"举起"证据区分，不会因此特写。
3. B03 类"近距离整理/推销"若运动缓慢且位置偏上，仍可能误入 SHOWING——需在更多视频上量化误触发率。
4. 阈值均为帧数语义（默认 30fps），5fps 的 mock 会自动等比拉长时间跨度；如需按时长口径请调构造参数。

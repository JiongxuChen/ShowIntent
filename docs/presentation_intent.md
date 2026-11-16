# B：展示意图理解

B 负责判断讲解者是否正在展示物品、展示的是哪个实例，以及何时结束。输入是 A 的视觉事实，输出是 C 可以直接消费的展示状态。当前保留一套主实现 `src/intent/temporal.py`，实验基线单独位于 `baseline.py`。

## 运行方式

Python 3.10+ 即可运行 B，无需安装视觉模型。所有命令均在仓库根目录执行。

### 单段输入

```bash
python -m src.intent.run --perception fixtures/demo_object_show/perception.jsonl --out outputs/intent.jsonl --config configs/presentation_intent.json
```

实际使用时将 `--perception` 替换成 A 生成的文件：

```bash
python -m src.intent.run --perception data/perception_A01.jsonl --out outputs/intent_A01.jsonl --config configs/presentation_intent.json
```

`--config` 可省略，此时使用主实现的默认参数。显式给出的毫秒参数优先于配置文件，例如：

```bash
python -m src.intent.run --perception data/perception_A01.jsonl --out outputs/intent_A01.jsonl --config configs/presentation_intent.json --showing-min-ms 600
```

每个视频创建独立状态机，输入时间戳必须严格递增。不要将不同片段的日志直接拼接，也不要将输出路径设为感知输入本身。

### 批量输入与实验基线

当前素材命名为 `perception_A01.jsonl`、`perception_A02.jsonl` 等，可直接批量校验和回放：

```bash
python -m src.intent.replay_real --perception-dir data/perception --out outputs/intent_batch --clips A01 A02
```

省略 `--clips` 会处理当前样例集的全部编号。每段输出 `intent_showintent.jsonl`、`intent_pickup-baseline.jsonl` 和 `report.json`。主方法与基线读取相同感知数据；标签不进入推理过程。回放目录已有结果时需要指定新目录。

单独运行 Pick-up 对照：

```bash
python -m src.intent.run --perception data/perception_A01.jsonl --out outputs/intent_A01_pickup.jsonl --mode pickup-baseline
```

该基线只用于检验展示判断相对于简单拿起触发的价值，不作为正式镜头策略。它保留简单的帧数保持规则；比较时使用相同感知输入，不把它的保持时长视为跨采样率不变。

### 无真实视频的独立验证

```bash
python -m src.intent.mock_scenarios --out outputs/intent_mock --fps 30
python -m src.intent.diagnose --out outputs/intent_diagnostics --fps 5 15 30 --modes showintent --strict
python -m pip install -r requirements-b-dev.txt
python -m pytest -q
```

Mock 覆盖静物、手物接近、普通持物、搬运、展示、跨类别切换、短暂遮挡和长期丢失。`--strict` 在行为检查不满足时返回非零退出码。模拟场景只验证工程行为，不替代真实评价。

## 识别逻辑

手靠近物体只是候选证据。B 综合以下信息后决定是否展示：

1. **动作证据**：物体是否曾上移，或在已有手部上移动作背景下出现在呈现位置。
2. **呈现方式**：目标位置、运动趋势和相对桌面物体的尺度变化是否支持向观众展示。
3. **时序支持**：有效证据是否足够持续；短暂检测噪声可以容忍，长时间证据中断需要重新确认。
4. **实例连续性**：同类、短时消失且位置接近的 track 可以有限关联；存在相近竞争者时拒绝强行重关联。

手部左右标签不一定是唯一实例 ID，因此 B 内部维护短窗空间轨迹。框中心使用时间窗统计，减少检测抖动的影响。对于已具备动作和位置证据的目标，尺度增大可支持移动中的展示；尺度变化或高处静物本身不能独立触发。

确认举起与维持展示使用不同位置条件，避免讲解者稍微放低物品就退出。持续搬运、放下、长时间失去关联或目标跟丢会使证据减弱，最终退出展示。

### 四种状态

| 状态 | B 的含义 | 给 C 的行为约定 |
|---|---|---|
| NONE | 没有确认展示，目标为 null | 保持或恢复全景 |
| CANDIDATE | 有候选，但证据不足 | 不开启新特写 |
| SHOWING | 已确认展示 | 跟随指定 target_id |
| ENDING | 展示正在结束或可靠性不足 | 平滑退回全景 |

新目标要经过候选确认。展示期间不会因为另一个物体得分更高就随意切换；原 track 消失后的有限重关联会带 `TARGET_RELINKED`，可以在保持 SHOWING 时更新检测 ID。

### 主要配置

完整参数位于 `configs/presentation_intent.json`。以下为当前默认值，均是待继续验证的开发配置。

| 参数 | 默认值 | 用途 |
|---|---:|---|
| candidate_min_ms | 300 | 手物关系形成候选的时间 |
| showing_min_ms | 600 | 累计有效展示证据时长 |
| evidence_gap_ms | 250 | 短暂坏证据的容忍窗口 |
| missing_tolerance_ms | 600 | 展示目标失检保持 |
| ending_min_ms / ending_max_ms | 150 / 2000 | 结束等待和上限 |
| raise_margin / putdown_margin | 0.06 / 0.025 | 举起与回落的位置缓冲 |
| display_region_max_y / display_exit_y | 0.72 / 0.73 | 固定机位的呈现区域边界 |
| motion_window_ms | 300 | 运动统计窗口 |
| presentation_scale_ratio | 1.3 | 有其他证据时的相对尺度支持 |

时间按 `timestamp_ms` 比较，不按固定 FPS 或帧号推算。位置阈值使用原始正立输入的归一化坐标。换机位、桌面布局或拍摄方向后需要重新检查，不能直接认为参数通用。

## 交给 C 的内容

B 交付同一片段的 `intent.jsonl` 和对应配置；C 还需要 A 的原始 `perception.jsonl`。两份 JSONL 一一对应、行序相同、时间戳相同。B 不输出裁剪框、缩放倍数或最终视频。

```bash
python -m src.camera.run --perception fixtures/demo_object_show/perception.jsonl --intent outputs/intent.jsonl --out outputs/shot.jsonl
```

C 根据当前意图目标查找 `objects[].track_id`，使用该对象的 `bbox_norm` 构图。短暂找不到目标时有限保持上一画面；不要自行选择另一个物体。`score` 和 `reason_codes` 供诊断，不应在 C 中再次代替 B 判定展示。

内存接入可以沿用同一接口：

```python
from src.intent.run import build_engine
from src.camera.controller import CameraController

intent_engine = build_engine("showintent")
camera = CameraController()
# 每收到一个 A 的 packet：
# intent = intent_engine.update(packet)
# shot = camera.update(intent, packet)
```

开始新片段时重建这两个实例。详细的丢失、重关联、时间戳和人工覆盖约定见 [B/C 接口](intent_camera_interface.md)。

## 与 D 的后续对接

D 负责 ASR、对象词和展示性语言线索，提供带 `available_at_ms` 的 `SpeechEvent`。B 只使用当前已经可用的消息，保持最终 `state` 和 `target_id` 的决定权；D 不直接控制 C。

当前命令行尚未消费语音事件。融合属于下一步扩展，应作为可关闭的证据适配器接入：语音缺失或延迟不阻塞视觉链路，提到商品名不能独立触发。语言可以帮助处理视觉模糊情形，但仍需评价融合是否增加误触发。

## 当前结果与待解决问题

在已用于开发的 12 条素材上，9 条正例都在复核展示区间内出现正确类别触发，三条负例没有 SHOWING。抽帧复核的展示核心区间正确类别覆盖约 83.6%，但 A03、C01 分别只有约 33% 和 32%，仍有漏触发和保持中断。

这些是开发区间检查，不是独立测试准确率或事件 F1；同类物理实例也未完成逐段标注。现阶段保留不确定性，并为 D 的语音辅助留下清晰边界。评价口径和复现方式见 [evaluation.md](evaluation.md)。

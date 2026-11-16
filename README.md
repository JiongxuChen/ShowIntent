# ShowIntent

ShowIntent 根据讲解者自然的手物交互，判断是否正在展示某个物品，并为镜头模块提供展示状态和目标实例。无需预先指定商品名称、展示顺序或目标 ID。

```text
A：视觉感知 → PerceptionPacket → B：展示意图 → IntentPacket → C：取景与输出
                                      ↑ D：自然语音线索（可选）
```

## 当前模块

- **A**：检测手和候选物体，维护短期跟踪 ID，输出手物关系。
- **B**：综合举起、呈现位置、运动、尺度和时序证据，输出展示状态与目标。
- **C**：根据 B 的决定平滑进入、保持和退出特写，并提供人工覆盖。
- **D**：负责语音识别与自然语言线索。当前 B 可独立运行，语音融合尚未实现。

B 只有一套正式实现，位于 `src/intent/temporal.py`。`pickup-baseline` 是实验用的“拿起即触发”对照，不是另一个正式版本。

## 快速运行 B

需要 Python 3.10 或更高版本。B 的规则计算和 JSONL 回放仅使用标准库，不需要安装检测模型。以下命令均在仓库根目录执行。

先用仓库自带的模拟感知输入验证接口：

```bash
python -m src.perception.validate_jsonl fixtures/demo_object_show/perception.jsonl
python -m src.intent.run --perception fixtures/demo_object_show/perception.jsonl --out outputs/intent.jsonl --config configs/presentation_intent.json
```

得到的 `outputs/intent.jsonl` 每行对应一个输入包，保留输入时间戳。

使用 A 已生成的真实感知结果：

```bash
python -m src.intent.run --perception data/perception_A01.jsonl --out outputs/intent_A01.jsonl --config configs/presentation_intent.json
```

`data/perception_A01.jsonl` 是本地数据路径示例，需要替换成实际文件。原视频、模型权重和真实感知日志不随代码提供。

当前样例集的批量回放及基线对照：

```bash
python -m src.intent.replay_real --perception-dir data/perception --out outputs/intent_batch --clips A01 A02
python -m src.intent.run --perception data/perception_A01.jsonl --out outputs/intent_A01_pickup.jsonl --mode pickup-baseline
```

批量入口读取 `perception_<编号>.jsonl`，输出主方法、基线和输入质量报告；已有结果不会被静默覆盖。没有复核标签时，报告只作诊断，不计算准确率。

## 交给 C 运行

C 需要 B 的 `intent.jsonl` 和生成它的**同一份** `perception.jsonl`。先验证结构化链路，无需视频依赖：

```bash
python -m src.camera.run --perception fixtures/demo_object_show/perception.jsonl --intent outputs/intent.jsonl --out outputs/shot.jsonl
```

已有真实视频及对应 A/B 输出时，可生成预览：

```bash
python -m src.camera.run --video data/A01.mp4 --perception data/perception_A01.jsonl --intent outputs/intent_A01.jsonl --out outputs/shot_A01.jsonl --overlay-out outputs/preview_A01.mp4
```

视频渲染需要 A/C 的视觉依赖。C 按 `state` 控制镜头：`NONE/CANDIDATE` 不开启新特写，`SHOWING` 跟随 `target_id`，`ENDING` 平滑退回全景。目标短暂缺失时有限保持，不能自行换成其他物体。完整约定见 [B/C 接口说明](docs/intent_camera_interface.md)。

## 开发与验证

安装测试依赖并运行：

```bash
python -m pip install -r requirements-b-dev.txt
python -m pytest -q
```

生成独立 Mock、检查主方法在不同采样率下的行为：

```bash
python -m src.intent.mock_scenarios --out outputs/intent_mock --fps 30
python -m src.intent.diagnose --out outputs/intent_diagnostics --fps 5 15 30 --modes showintent --strict
```

提供本地真实感知数据后，可额外运行真实回归：

```bash
SHOWINTENT_PERCEPTION_DIR=data/perception python -m pytest -q
```

这条环境变量写法适用于 Bash；其他终端可先设置同名变量。未提供真实数据时，相应测试会明确跳过。

## 当前效果与边界

当前实现能在开发素材中区分多种展示、搬运和普通持物行为，并处理有限的漏检与跟踪 ID 变化。部分片段仍有确认偏晚、展示中断和同类实例歧义。开发区间的抽帧复核不是完整事件真值，不能据此声明通用准确率。

后续自然语音线索由 D 提供，B 保持最终状态与目标决定权。语音缺失时视觉路径仍可运行，语音提到物品名称也不应独立触发镜头。

- [B 的方法、配置与运行说明](docs/presentation_intent.md)
- [B→C 交接与消息契约](docs/intent_camera_interface.md)
- [开发评价口径](docs/evaluation.md)
- [提交 GitHub 前的文件清单](docs/publishing.md)

## 原有交付物

以下文件保留原仓库内容，记录早期实现和历史实验；当前 B 的方法与结果以本文链接的新说明为准。

- [A 的视觉感知报告](docs/perception_video_report.md)
- [原 P0 评价报告](docs/p0_evaluation_report.md)及[中文阅读版](docs/p0_evaluation_report.zh-CN.md)
- [原仓库逐项核对记录](docs/original_repository_inventory.md)

原有统计入口 `python -m scripts.p0_stats --help` 和早期原型也保留，未被当前正式入口调用。

# D：可选语音线索模块

`src.speech` 提供成员 D 的最终 ASR/字幕事件契约、中文/英文小词表，以及交给 B 的保守词汇证据。它不改变现有 A→B→C 文件格式，也不生成 `IntentPacket`、`target_id`、镜头参数或最终展示状态。

## 输入与时间约定

每条 `SpeechEvent` 是一行 JSON，可由真实 ASR 或可审计的字幕输入生成：

```json
{"start_ms":1000,"end_ms":1600,"available_at_ms":1900,"text":"给大家看这个苹果","mentioned_classes":["apple"],"cue_types":["PRESENTATION_LANGUAGE"],"is_final":true}
```

三个时间都以当前片段的媒体起点为准。只有 `available_at_ms <= PerceptionPacket.timestamp_ms` 且 `is_final=true` 的事件可以成为证据；这避免把未来语音倒灌到 B。`mentioned_classes` 是类别名而非 A 的 `track_id`。

## 给 B 的可选接口

```python
from src.speech.bridge import SpeechEvidenceBridge

speech = SpeechEvidenceBridge.from_jsonl("data/speech_events_A01.jsonl", enabled=True)
evidence = speech.evidence_for(perception_packet)
intent = visual_engine.update(perception_packet)  # B仍独立决定state和target_id
```

当同类目标可见多个、类别不在画面内、语音过期、尚未可用、否定、结束或换题时，接口返回拒绝原因而不是猜测实例。默认 `enabled=False`，因此没有语音文件时既有 B/C 行为不变。

运行词表位于 `config/speech_lexicon.json`，早期 `configs/speech_lexicon.json` 保留同内容镜像。`PRESENTATION_LANGUAGE` 只是一项词汇线索；`ENDING_LANGUAGE` 是结束表达的证据，不是强制把 B 切换到 `ENDING`。需要 Faster-Whisper/PyAV/音频读取依赖时安装 `requirements-d.txt`；核心契约和桥接模块只用标准库。

对本机已下载的 Faster-Whisper CTranslate2 模型，可运行：

```bash
python -m scripts.run_speech --audio data/A01.mp4 --model models/faster-whisper-base --out outputs/speech_events_A01.jsonl --mode causal --language zh
```

该入口转发到与本地一致的根目录 `run_speech.py`，支持 `offline/causal/agreement`，并检查旧输出与 sidecar 防止覆盖。根目录入口还支持 Mock 字幕和增量事件 journal；它允许显式重跑，因此要为不同实验选择独立输出路径。模型来源固定在 `config/asr_model.json`，可执行 `python scripts/fetch_asr_model.py` 下载并核对。

增量确认复用 `third_party/whisper_streaming/` 的上游 HypothesisBuffer；字幕解析复用 `third_party/srt/`，许可证和来源清单随代码提供。没有重新训练或自行重写 ASR/VAD。单模块开发集结果见 [人工参考报告](verified_speech_report.md)；词汇匹配和转写分数不证明 B 融合收益。

## 开发评测边界

本地录制、人工转写、模型权重、完整识别文本和运行输出均不进入 Git。已复核的开发样本用于诊断语音转写与词汇线索，不能单独证明视觉展示意图或 A+B+D 融合效果。B 可在后续独立分支中消费该接口并进行联合评测。

本轮仓库合并工作副本执行 `python -m pytest -q -rs`：325 项通过、6 项因缺少真实视觉感知文件而跳过，另有 391 项 subtest 通过。完整的人工校对网页仍保留在 D 的本地工作区；共享部分提供参考契约、评测器、来源校验和数值导出，无需为 B 运行安装校对网页的浏览器测试依赖。

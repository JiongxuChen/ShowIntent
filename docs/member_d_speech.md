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

词表位于 `configs/speech_lexicon.json`。`PRESENTATION_LANGUAGE` 只是一项词汇线索；`ENDING_LANGUAGE` 是结束表达的证据，不是强制把 B 切换到 `ENDING`。需要 Faster-Whisper/PyAV/音频读取依赖时安装 `requirements-d.txt`；核心契约和桥接模块只用标准库。

对本机已下载的 Faster-Whisper CTranslate2 模型，可运行：

```bash
python -m scripts.run_speech --audio data/A01.mp4 --model models/faster-whisper-base --out outputs/speech_events_A01.jsonl --mode causal --language zh
```

该入口只读取本地媒体和本地模型，创建新的事件、窗口和元数据文件，不会覆盖旧运行或上传音视频。

## 开发评测边界

本地录制、人工转写、模型权重、完整识别文本和运行输出均不进入 Git。已复核的开发样本用于诊断语音转写与词汇线索，不能单独证明视觉展示意图或 A+B+D 融合效果。B 可在后续独立分支中消费该接口并进行联合评测。

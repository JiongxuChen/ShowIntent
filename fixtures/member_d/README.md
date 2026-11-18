# D1 手写输入

`demo.srt` 是手写字幕，`manual_annotations.jsonl` 是相同内容附人工类别/语言标签的输入。两者都不是 ASR 输出或评价真值，也不要求主播照读。与规划中的杯子→苹果时间轴对应，只为独立验证 D 的消息契约，不选择目标实例。

SRT 转换时 `mentioned_classes=[]`、`cue_types=[]`；JSONL 里的标签只做人工透传。D3 才实现自动语言线索抽取。即便存在 `PRESENTATION_LANGUAGE`，B 仍须依据视觉决定是否展示。

后续 D3 已实现，显式加 `--extract-cues` 可对原 SRT 自动提取，不改动这里的 D1 期望样例。新增 `visual_candidates.jsonl` 是仅用于 D5/D6 证据适配器的简化候选包（省略未消费的 bbox/score 字段），不是 A 的真实检测或整个共享 schema 验收。4300 ms 单杯可辅助，4400 ms 双杯拒绝，8500 ms 否定拒绝，10600 ms 切换为 apple 类别。

`speech_events.jsonl` 从人工标注输入生成，模拟延迟为 200 ms；`speech_events.meta.json` 记录其模拟来源。`expected_from_srt.jsonl` 是 SRT 无语义标签版本。无需录音即可运行这两组输入。录音读块测试使用临时合成信号，不将其作为真人识别测试。

# 离线产物

- cache/<场景>/{perception,speech_raw,timestamps}：A/D输入与媒体时钟。
- results/<场景>/<视频>：正式三模式B输出；results/evaluation.json：四场景评价。
- benchmarks/<场景>/<模型>/evaluation.json：该场景的闭集模型对比分数。

场景均为fruit、books、cups、drinks。两组模型对比的B配置不同，详见docs/model_comparison.md；目录按场景组织不代表它们采用同一配置。

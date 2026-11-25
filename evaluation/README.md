# 评价数据

数据用途按场景区分：水果组用于开发；书籍、水杯、饮料三组用于评价。指标仍按四个场景分别报告。

fruit、books、cups、drinks四个目录分别保存对应场景的数据清单、展示标签、协议和身份映射。dataset_manifest.csv定义评价媒体时间与A缓存路径；annotations.csv定义连续SHOWING/NON_SHOWING区间；track_mapping.csv用于实例身份复核。未复核身份不能当作正确实例。

split_manifest.csv仅为评分器兼容元数据，不用于展示场景汇总。cups、drinks还含annotation_record.json，记录标签冻结哈希和标注状态。metadata保存名称映射、排除记录、复核记录与时间核验。

时间评价入口为python -m scripts.run_refined_b --out outputs/replay；规范和完整指标见docs/evaluation.md。

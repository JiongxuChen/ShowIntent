# 评价数据

水果用于开发；书籍、水杯、饮料用于评价。四个场景采用相同的目录结构与六种文件：

| 文件 | 用途 |
|---|---|
| dataset_manifest.csv | 视频编号、原片及感知路径、评价时间范围；各运行入口据此定位数据 |
| annotations.csv | 连续的SHOWING/NON_SHOWING区间、目标与标注复核状态 |
| annotation_record.json | 标注文件哈希、原视频哈希、逐视频复核状态及必要的标注依据 |
| protocol.json | 时间区间、事件匹配阈值和评分约定 |
| split_manifest.csv | 评价器读取的数据分组和参与者/会话元数据 |
| track_mapping.csv | 检测轨迹到真实物品的身份映射；尚未复核的身份不能计为正确 |

annotation_record.json采用统一字段，复核完成情况以逐视频review_status为准。dataset_manifest.csv是运行入口所需的数据索引，需与标注一起保留。

metadata保存跨场景的编号映射、用途、时间轴核验和复核资料。具体片段的判定依据集中在所属场景的annotation_record.json中，不另设单视频补丁文件。

离线重放：`python -m scripts.run_refined_b --out outputs/new_replay`。完整结果见[离线评价](../docs/offline.md)和[流式评价](../docs/stream.md)。

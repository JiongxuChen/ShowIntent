# 展示行为标注规范

`annotations.csv`、`track_mapping.csv` 和 `split_manifest.csv` 目前只有表头。`annotation_queue.csv` 是素材清单；`development_review.csv` 是待人工复核的抽帧初检区间，两者均不是正式事件真值。

## 标注步骤

1. 观看原视频及 A 的检测叠加结果，确认方向、时间轴和检测框正常，使用完整片段。
2. 独立于 B 的输出标注行为。向观众举起、呈现并停留或缓慢转动物品可记为展示；搬运、整理和普通持物分别检查。无法从画面确定的区间记为 UNCERTAIN，不根据语音中的商品名直接判为展示。
3. 采用 A 的 `timestamp_ms` 作为统一轴。先核对播放器时间与感知时间的起点偏移；不能默认两者相同，也不能直接用 `frame_id / 30` 推算时间。
4. 记录外显展示行为的开始与结束。先共同确认举起、确认停留和放下过渡的边界约定，再固定评价容忍窗口，不按 B 的状态倒推标签。
5. 为物理物品分配 `apple_1`、`orange_1` 等实例编号。A 的跟踪 ID 可能碎片化，在 `track_mapping.csv` 中分段记录映射；映射仅用于评价，不传给 B。
6. 一人标注后由另一人复核。争议保留 PENDING 或 UNCERTAIN，避免强行给出确定标签。

## 字段约定

- `annotations.csv`：label 使用 SHOWING / NON_SHOWING / UNCERTAIN，review_status 使用 PENDING / REVIEWED，confidence 使用 HIGH / MEDIUM / LOW。区间为 `[start_ms, end_ms)`，结束必须晚于开始。
- 负例也应标出确认过的非展示区间；没有 SHOWING 标签不等于整段已确认是负例。
- `track_mapping.csv`：多个跟踪 ID 可以在不同时间段对应同一物理实例。同类物品不能自动视为同一实例。
- `split_manifest.csv`：按参与者或录制会话划分数据，同源切片归入同一集合。未确认的信息留空。现有 12 条已用于开发，不作为未见测试集。

视频编号中的 A/B/C/D 表示场景组，不对应成员。无口播仍可存在视觉展示；当前标注以外显行为为准，后续语音融合另做对照。

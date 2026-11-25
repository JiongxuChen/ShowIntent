# A：增强视觉感知

A将原视频转换为逐帧视觉事实，供B判断展示意图。正式入口是`src/perception/enhanced.py`，使用全类别检测、手部局部复检、颜色外观特征与短期实例追踪。它不直接判断是否展示。

## 输入、输出与时间基准

输入为视频、原始解码PTS清单和本地检测权重。先用extract_video_timestamps抽取时间戳，感知按同序解码；视频解码帧数必须与PTS清单一致，完整运行不允许悄悄截短。

程序关闭OpenCV自动方向处理，读取旋转元数据后显式旋正一次。物体框和手框均相对于旋正后的画面，以[x1,y1,x2,y2]归一化至[0,1]。timestamp_ms采用原媒体毫秒，不由处理耗时或帧序号推算。

## 逐帧处理

1. **手部检测**：通过PerceptionModels调用MediaPipe，给左右手及同帧手实例建立唯一hand_id，输出手部包围框。手部分数不能解释为经校准的真实置信度。
2. **全帧物体检测**：默认YOLO11m、imgsz=1280、confidence=0.25，保留检测器支持的全部类别。无需提前为书、水杯或饮料新增白名单；类别范围仍受预训练检测器限制。
3. **手部区域复检**：以每只手和双手联合区域建立裁剪。宽高至少占对应整图方向的0.32，或扩至手框的2.8倍；重复区域IoU>0.7时去重，最多3个区域。每个区域用imgsz=960检测。
4. **映射与去重**：局部检测映射回整图；局部person不加入候选，距人工裁剪边缘不足3像素的截断框被拒绝。同类别框按分数排序，IoU>0.45时合并筛除重复。
5. **外观描述**：从框内部去掉12%边缘，缩放为48×48，计算12×4的色相/饱和度直方图与8档亮度分布，权重0.7/0.3。颜色占比超过0.22时取最多两种颜色。这不是分割或材质识别，背景仍可能影响结果。
6. **实例关联**：AppearanceTracker结合类别、位置、重叠、运动预测与外观距离，输出item_编号。不为暂时失检目标虚构当前框。
7. **手物关系**：计算手框中心到物体框的最短归一化距离，距离<0.09时建立关系，score=1-distance/0.12。person不作为手持物品关联对象。

## 追踪细节

轨迹默认保留1500ms。仅同类别候选可关联，运动预测时间最大0.3秒；中心距离>0.18或外观距离>0.5时拒绝。重叠<0.05且距离>0.08时，还要求外观距离不超过0.35。

符合条件的关联按`2×距离+0.25×(1-IoU)+0.7×外观距离`排序，逐一分配，确保检测与旧ID一对一。新目标分配新ID。速度按媒体时间更新并限制到每方向[-1,1]。这些约束用于短时关联，不是可靠的长期重识别。

## PerceptionPacket接口

每行JSON包含：

| 字段 | 内容 |
|---|---|
| timestamp_ms / frame_id | 媒体时间 / 解码序号 |
| objects | class_name、track_id、bbox_norm、score、appearance、attributes.colors |
| hands | hand_id、bbox_norm、score |
| relations | hand_id、object_id、type、score |

关系名near_or_moving_together在当前适配器中由空间邻近产生，没有验证共同运动，也不表示抓取、举起或展示。B需要进一步累积动作证据。

## 独立运行

```bash
python -m scripts.extract_video_timestamps --input Example_Data2/E01.mp4 --out outputs/E01_pts.json
python -m src.perception.enhanced --input Example_Data2/E01.mp4 --timestamp-map outputs/E01_pts.json --yolo-model models/yolo11m.pt --out outputs/E01_perception.jsonl --metrics-out outputs/E01_perception_metrics.json
```

需要自行提供视频和模型，并安装README中的完整依赖；输出路径不能已存在。完整A/D处理也可使用scripts.run_complementary，命令见README。

现有输入位于outputs/cache/<场景>/perception；配套时间戳位于timestamps。运行指标记录帧数、总耗时、适配器处理FPS、裁剪推理次数、旋转方式和尺寸；这个FPS包含检测、手部、追踪及写盘，不是裸检测器速度。

## 已知限制

同类密集、堆叠书籍、局部遮挡、封面图案及短暂误分类可能造成漏框、误框或ID变化。颜色只是弱属性。局部复检不能保证恢复每个物体，框更多也不等于更准确。后端可选RT-DETR，模型对比见[model_comparison](model_comparison.md)。

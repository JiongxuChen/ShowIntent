# A 模块候选配置与验收状态

## 已确认的开发反馈

用户人工观察：A02 方向正确、手框及橘子框贴合；B03/C01/C02 原配置存在漏框、同类合框和 ID 断裂。C03 原结果同一橘子 obj_8→obj_10，未观察到串物。

small 检测对照：B03 漏框明显减少、合框几乎消失；C01 改善但残留一次合框。此结论来自开发素材人工比较，不能当作独立测试集准确率。

relink 追踪对照：C03 主要为 obj_1，中间短暂 obj_6 后恢复 obj_1，遮挡后仍为 obj_1。用户因标签重叠未能确认是否串物。已将标签改到画外固定列表，最终复核尚未完成。

## 可选运行方式

从仓库目录运行，输出到新目录避免覆盖基线：

```powershell
python -m src.perception.run --input "..\received_media\C01-04&D01\C03.mp4" --yolo-model yolo11s.pt --imgsz 1280 --confidence 0.35 --tracker relink --out outputs\candidate\perception_C03.jsonl --metrics-out outputs\candidate\metrics_C03.json --overlay-out outputs\candidate\overlay_C03.mp4
```

默认仍是 nano + simple；候选需要显式指定。模型权重不提交。PerceptionPacket 字段不变。

## 追踪参数的限制

relink 的 max_missed=8 按实际处理帧计，30FPS 全帧约0.27秒；对照脚本在5FPS抽样下设8次更新，约1.6秒。因此当前抽样对照收益不能直接外推到正式30FPS运行。正式采用前须在全帧短片复核并按媒体时间校准保留窗口，不能仅凭 ID 总数减少验收。

不输出虚构的漏检框；只对再次检测到的同类、近位置、尺度相近目标尝试重关联，歧义时允许新 ID。仍然不保证同类交叉或长遮挡重识别。

## 2026-11-17 复核与修复

用户确认更新后的 C03 手持阶段及遮挡前后保持 obj_1，放回桌面变为 obj_8。此反馈只适用于5FPS开发对照，未构成完整帧率验收，也未确认所有多实例串 ID 情况。

缓存诊断：5.6秒 obj_1 与额外 obj_6 同时存在，不能简单计为 ID 切换；12.4秒局部露出框面积骤减导致关联被拒绝。候选增加“短缺失后局部框与旧框充分重叠”的关联规则，并保留完整尺寸参考以避免下一帧尺寸恢复时再次断裂。16秒放下位移超过距离门槛，仍允许新 ID，避免放宽后误接桌上同类。

正式 CLI 新增 --track-keep-ms（默认1600，仅 relink 生效），按源 FPS 转为保留帧数并记录到 metrics；这是完整逐帧回放的近似时间口径。可变帧率和丢帧实时流仍需按真实时间戳实现，不能声称已解决。默认 simple 行为不变。

## 下一步人工复核

打开仓库 outputs/a_tracker_compare 下的 C03_compare.mp4、C01_compare.mp4、B03_compare.mp4。右侧为候选，ID 位于下方固定列表。确认原 ID 没有转给另一颗水果，重点是 C03 遮挡期间与 C01/B03 拿起放回时。定位小编号是逐帧标记，不是永久 ID。

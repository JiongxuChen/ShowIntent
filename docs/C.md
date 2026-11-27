# 交付C：正式镜头模块接入指南

B负责展示时机和目标选择，C负责构图、放大、平滑和异常回退。现有render_b_preview只作为效果参考，正式C可以独立实现。

## 交付文件

全部23段正式B输出在`outputs/results/<场景>/<编号>/showintent.jsonl`。场景目录为fruit、books、cups、drinks。请使用showintent，不要误用visual-only或audio-only。

配套A文件在`outputs/cache/<场景>/perception/perception_<编号>.jsonl`。配置是configs/presentation_intent.json和configs/semantic_priority_fusion.json。标注和原视频路径见evaluation/<场景>/dataset_manifest.csv以及evaluation/cups、drinks_v1内的manifest；编号映射见evaluation/metadata/video_name_mapping.csv。原视频另行提供，仓库不包含视频。

例如E01这一组是：

- 视频：Example_Data2/E01.mp4。
- A：outputs/cache/books/perception/perception_E01.jsonl。
- B：outputs/results/books/E01/showintent.jsonl。

## 对齐与取景规则

逐行检查A、B的timestamp_ms相等、严格递增且行数相同。时间戳为媒体毫秒，不保证从零开始；渲染时按媒体时间匹配，不能把每行当作固定帧率。画面按旋转元数据保持正立，只旋转一次。

SHOWING时用target_id查同包objects中的track_id，再读取bbox_norm=[x1,y1,x2,y2]。它是正立画面的归一化物体框，C自行计算符合输出宽高比、边界和最大倍率约束的裁剪框。禁止改选同类别或最高分物体。

NONE维持全景；CANDIDATE不发起新特写；ENDING平滑退场。短暂找不到指定目标时有限保持上次构图，超过C设置的超时则回全景并记录原因。ID变化需要平滑衔接，不能直接跳切。人工强制全景优先。

## 开发起点

仓库现有src/camera为基础镜头实现，scripts/render_b_preview.py提供带原音轨的参考渲染。先用E01验证联结，再覆盖同类多物体、遮挡、放回、切换和纯负例。

```bash
python -m src.camera.run --perception outputs/cache/books/perception/perception_E01.jsonl --intent outputs/results/books/E01/showintent.jsonl --out outputs/E01_shots.jsonl
```

结构化镜头输出可在没有视频时检查；视觉验收需要另行放好原视频。参考预览命令见README。正式C应输出镜头日志，至少记录媒体时间、目标、裁剪框、倍率、状态及回退原因。

## 验收清单

1. 23段均正确对齐，不混用感知运行的ID。
2. 七个纯负例B01/B02/B03/E04/F03/F04/G04保持全景。
3. 展示时裁剪不越界，目标可见，倍率有上限，展示与放回时平滑过渡。
4. 检查E02/E03的多书籍切换、F01的连续换杯、G01的换物；记录已知B漏检，不用C重新判断来掩盖。
5. 检查音画同步、方向、完整时长和帧率；评价C时另报目标保留率、裁剪抖动、倍率变化、过渡时长和人工观看结果。

当前B覆盖率只评价展示时间，不代表实例身份或框准确。C的构图评价需要独立完成。更详细字段见[intent_camera_interface](C.md)。


## 输出字段

IntentPacket含timestamp_ms、state、target_id、score、reason_codes；timestamp_ms为媒体毫秒，score为启发式强度。NONE时target_id为空，其他状态可携带候选或旧目标。C只在SHOWING且指定目标可用时发起特写，不依赖固定reason_codes列表。

## 简易预览的实际实现

PreviewFraming在一段SHOWING开始时根据近期约500ms的目标框历史取尺寸中值，锁定该段的裁剪尺度，并平滑更新位置。默认最大放大5倍、目标边距系数1.35；框的宽高及输出比例共同限制裁剪大小。这样可减轻放回过程中的忽大忽小，但不代表正式C已完成。

退出时默认保留最后取景300ms再平滑恢复全景，不持续追随正在放下的物品。程序按原媒体时间渲染并保留原音轨，同时记录实际裁剪，便于区分B状态与C构图的差异。

## 正式C可自行选择的参数

正式C应明确配置输出比例、最大倍率、位置与尺度平滑速度、边界约束、目标缺失容忍、进入/退出时长、人工覆盖和日志格式。不要把B的候选门槛复制进C重新作展示判断。

当目标ID变化时，先检查新框是否与现有镜头连续；必要时过渡或回全景，避免跳到另一实例。B在SHOWING时偶发目标不可见，C应有限保持后回退，不能用旧框无限放大。

## 测试素材

小型接口样例位于tests/fixtures/demo_object_show，供自动化测试验证A/B/C字段、行序与入口联通，不作为真实视频准确率证据。完整联调使用outputs/cache与outputs/results的对应23段，原视频另行提供。

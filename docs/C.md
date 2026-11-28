# Member C · Camera & Rendering

C 接收 A 的候选框与 B 的展示状态，输出可播放的取景视频、逐帧 ShotPacket 和测量结果。团队的正式离线输入为 `outputs/cache/<场景>/perception/perception_<编号>.jsonl` 与 `outputs/results/<场景>/<编号>/showintent.jsonl`。书籍 E01–E04、水杯 F01–F04 和饮料 G01–G04 与水果共 23 段。原视频路径见各场景的 `evaluation/<场景>/dataset_manifest.csv`，旧文件名映射见 `evaluation/metadata/video_name_mapping.csv`。早期的 `scripts/render_b_preview.py` 仍可复现 B 的诊断预览。

## 从哪里开始

以下命令在仓库根目录执行。普通渲染只需 `requirements-c.txt`；本次已在 Conda 环境 `showintent-c` 安装。输出使用新文件名，避免覆盖以前的导出。

```powershell
conda activate showintent-c
python -m scripts.camera_mock --out outputs/my_mock
python -m scripts.run_camera_collection --out outputs/my_camera --render
python -m scripts.verify_camera_collection --root outputs/my_camera
```

第一条 Python 命令用团队的合成杯子和苹果轨迹、视觉 B 和正式 C 写出可播放 MP4。第二条遍历 manifest，保存每个片段的 `shots.jsonl` 和有原片时的 `camera.mp4`。第三条复查完整导出集的帧数、时间戳、方向、音频包、时长和镜头指标。指定 `--clips C02 F01` 可运行子集；省略 `--render` 可仅检查 23 条结构化轨迹。只有数据包时，默认输入输出宽高比相同。

在新环境安装可执行 `conda create -n showintent-c python=3.11 pip`，激活后执行 `python -m pip install -r requirements-c.txt pytest`。重跑测试使用 `python -X utf8 -m pytest -q tests/test_camera_controller.py tests/test_camera_rendering.py`。

## 单段渲染与演示

```powershell
python -m src.camera.run --video Example_Data1/C02.mp4 --perception outputs/cache/fruit/perception/perception_C02.jsonl --intent outputs/results/fruit/C02/showintent.jsonl --out outputs/my_C02/shots.jsonl --video-out outputs/my_C02/camera.mp4
```

默认输出干净画面。用 `--compare` 导出左侧固定全景和右侧 C 结果的同步分屏。用 `--hud` 叠加镜头模式、ID、倍率及目标框。 `--out-height 1920 --out-width 1080` 可将竖屏 4K 原片导出为竖屏 1080p。默认长边至多 1280 像素并保留正立原片比例。若显式指定不同宽高比，全景采用源画面中央能容纳的最大同形裁剪，`zoom` 相对这个裁剪计量，极宽物体可能无法完整放入。配套输出的 `crop_norm` 始终指向原始正立画面。单段命令只给 `--perception --intent --out` 时可直接输出镜头日志。

## 实时摄像头与控制

实时模式复用团队的 `EnhancedStreamPerception` 与 `SparseVisualIntentStateMachine`，本地运行 YOLO11n、MediaPipe Hands 和纯视觉 B；异步检测每次只接收一张最新可用帧。预览持续对当前帧应用同一个 C 控制器和渲染函数。可分别测量画面帧率、已收到的检测频率、采样到检测结果被消费的延迟。摄像头模式默认只保存结构化日志与指标。

```powershell
python -m pip install "ultralytics>=8.3,<9" "mediapipe==0.10.21" "opencv-contrib-python>=4.8,<4.12"
python -m scripts.download_closed_set_models --models yolo11n
python -m src.camera.live --source 0 --out outputs/my_live/shots.jsonl
```

在预览窗口中，**W** 立即强制全景，**P** 暂停或继续自动构图，**R** 恢复自动控制，**V** 切换原图与结果的分屏，**Q / Esc** 退出。强制全景优先于暂停。用 `--source 视频路径` 将本地视频作为按播放时钟输入的摄像头替身。若 4K 解码跟不上，可先用仓库的 `python -m src.streaming.prepare_stream_proxy --video Example_Data1/C02.mp4 --out outputs/my_C02_proxy.mp4 --height 960 --fps 15`，该工具另需 `imageio-ffmpeg`。用 `--headless --seconds 6` 做有时限的硬件冒烟测试。实时指标记录在日志同目录的 `*.metrics.json`。最新检测超过保留时限时回全景。

## 输入、画面和时序

- A/B 两个 JSONL 必须同一运行、行数相同、媒体毫秒相等且严格递增。原片方向只旋转一次。对每个视频 PTS 使用已到达的最近一对包，首包前为全景；稀疏输入也会导出完整源帧序列。正式批量验收另将两个包的时间与仓库的逐帧 PTS 地图核对。
- `SHOWING` 时只按 B 的 `target_id` 查同包对象。目标框四周留 30% 边距，附近已关联的手可在每轴额外贡献最多 0.35 个物体边长的构图范围。每次确认新 ID 时重新定尺度，持续跟随时只按需要放宽，缩小的框不会反复拉近镜头。
- 默认最大 2.5 倍，相同宽高比下请求宽度已达全幅 96% 时使用全景。平滑系数按 30 FPS 时的 0.20 换算为实际媒体时间，每边最多移动每秒 1.6 个归一化单位。边界修正在同一速度预算内保留可见目标。新目标也从现有构图过渡。
- `NONE` 与 `CANDIDATE` 均以全景为终点。`ENDING` 先固定最后构图 200 ms，再平滑退场；缺失已确认目标最多保持到上次观测后 500 ms。可用 `--margin --max-zoom --miss-hold-ms --exit-hold-ms --force-wide` 调整。手动暂停固定当前构图，强制全景立即生效。
- 在源文件的时间基上编码 H.264，逐包复用原音轨，使用精细 MP4 媒体时钟保留手机视频的非零起点和可变帧间隔。固定全景、候选、失检或切换都保留真实时长。

## 日志与评分

基本字段为 `timestamp_ms / mode / target_id / crop_norm / zoom / manual_override`。日志另含 `intent_state / intent_target_id`（B 原始请求）、`reason`（如 ACQUIRE、TRACK、TARGET_MISSING_HOLD、TARGET_UNAVAILABLE、FULL_FRAME_FIT、FORCED_WIDE）、`target_bbox_norm`、`packet_timestamp_ms` 与 A 的 `frame_id`；渲染时还记录 `source_frame_id`。模式包括 WIDE、TRANSITION_IN、CLOSEUP、TRANSITION_OUT。

几何目标保留率的分母为 B 请求 SHOWING 且 A 中指定 ID 有有效框的帧。分子为整个可见框都在实际归一化裁剪内的帧。缺框单列。稳定性报告相邻帧最大边位移、倍率差、稳定跟随中心移动速度、中心位置相对前两帧恒速外推的偏差，以及过渡次数和时长。位置误差单位为源宽高归一化坐标。每个已结束的过渡进入统计；片尾尚未完成的过渡可在逐帧日志查看。纯负例以实际输出是否全程 WIDE 单独检查。物理实例准确率、上游动作识别延迟和观众打分需要各自的标注或实验。

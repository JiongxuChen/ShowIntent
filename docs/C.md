# Member C · 自动取景与视频导出

C 根据 A 提供的物体框和 B 确认的展示目标自动取景。离线导出直接读取仓库里的 A/B 结果，在 CPU 上即可生成带原声的 MP4。

## 安装与原片

首次使用，在 PowerShell 中克隆仓库并安装离线依赖。已有仓库的组员可在本地 `main` 上运行 `git pull --ff-only origin main`。下文命令均在仓库根目录运行。

```powershell
git clone https://github.com/JiongxuChen/ShowIntent.git
cd ShowIntent
conda create -n showintent-c python=3.11 pip -y
conda activate showintent-c
python -m pip install -r requirements-c.txt
```

从组内共享取得要渲染的原片，按下表放置。例如 C02 的路径是 `Example_Data1/C02.mp4`。早期文件名见[编号映射](../evaluation/metadata/video_name_mapping.csv)。

| 仓库内目录 | MP4 编号 |
|---|---|
| `Example_Data1/` 水果 | A01–A03、B01–B03、C01–C04、D01 |
| `Example_Data2/` 书籍 | E01–E04，E04 可使用组内提供的 720×1280 副本 |
| `Example_Data3/` 水杯 | F01–F04 |
| `Example_Data4/` 饮料 | G01–G04 |

没有原片也可运行 `python -m scripts.camera_mock --out outputs/camera_mock`，打开 `outputs/camera_mock/camera.mp4` 查看合成双目标演示。

## 导出成片

高清特写直接从原片裁剪，保持正立宽高比，默认最长边为 1920 像素，受原片尺寸限制。构图框每侧留 13% 边距，最大放大 3.0×。镜头推近时逐渐增强亮度边缘（Luma Sharpen），限制平坦区域的噪点与强边缘的光晕，成片以 H.264 CRF 17 编码。拍摄时已经失焦或有运动拖影的文字，清晰度仍受原片限制。

```powershell
python -m scripts.run_camera_collection --out outputs/camera --render
python -m scripts.verify_camera_collection --root outputs/camera
```

成片在 `outputs/camera/<编号>/camera.mp4`，同目录的 `shots.jsonl` 记录逐帧镜头。备齐原片后，核验应输出 `Verified 23 movies and 23 shot traces`。`outputs/camera/` 下的 `metrics.json` 记录渲染与取景指标，`media_audit.json` 和 `shot_metrics.json` 分别记录音画与逐片构图核验结果。

只做 C02 和 D01 的展示时，可单独运行 `python -m scripts.run_camera_collection --out outputs/selected --render --clips C02 D01`。缺少原片的片段只生成日志，终端显示 `packets only`。全量核验需要全部 23 份镜头日志。重复导出时请使用新的输出目录。

## 制作 PPT 素材

备齐原片后，在仓库根目录运行一次即可生成全部 23 段完整的同步分屏视频。每段左侧是固定全景，右侧是与成片相同的高清自动取景，保留原声。此命令直接读取原片和 A/B 缓存，可独立运行。

```powershell
python -m scripts.run_camera_collection --out outputs/slides --compare
```

终端显示 `Exported 23/23 split-screen videos` 后，可打开 `outputs/slides/<编号>/compare.mp4` 挑选画面。若只备了部分原片，终端会列出实际导出数量。

下面以 C02 第 8 秒的苹果截图和第 5–19 秒的有声短片为例。更换素材时，只需改视频编号、截图时间、短片起点和时长。时间单位均为秒。

```powershell
python -m pip install "imageio-ffmpeg>=0.5,<0.7"
$ffmpeg = python -c "import imageio_ffmpeg; print(imageio_ffmpeg.get_ffmpeg_exe())"

$clip = "C02"
$shotSec = 8
$startSec = 5
$durationSec = 14

$folder = "outputs/slides/$clip"
& $ffmpeg -v error -n -i "$folder/compare.mp4" -ss $shotSec -frames:v 1 "$folder/frame_${shotSec}s.png"
& $ffmpeg -v error -n -i "$folder/compare.mp4" -ss $startSec -t $durationSec -c:v libx264 -crf 17 -pix_fmt yuv420p -c:a aac -movflags +faststart "$folder/clip_${startSec}s_${durationSec}s.mp4"
```

C02 第 17 秒可截到橙子特写。要展示普通持物时镜头保持全景，可选 D01 第 63 秒。

## 取景与日志

- A/B 包按同一媒体时间对齐。C 在 `SHOWING` 时跟随 B 指定的 `target_id`，将目标和邻近的关联手部纳入构图框。
- `NONE` 和 `CANDIDATE` 以全景为终点。目标短时失检最多保持 500 ms，`ENDING` 保持最后构图 200 ms 后平滑退出。
- `shots.jsonl` 记录 `timestamp_ms`、镜头模式 `mode`、目标 ID、源画面归一化裁剪框 `crop_norm`、倍率 `zoom` 和决策原因 `reason`。目标保留率按 B 请求展示且 A 有对应可见框的帧计算，缺框另计。

## 实时摄像头（可选）

```powershell
python -m pip install "ultralytics>=8.3,<9" "mediapipe==0.10.21" "opencv-contrib-python>=4.8,<4.12"
python -m scripts.download_closed_set_models --models yolo11n
python -m src.camera.live --source 0 --out outputs/live/shots.jsonl
```

实时预览使用相同的亮度锐化，预览单幅长边上限为 1280 像素。在预览窗口按 **W** 强制全景，**P** 暂停或继续构图，**R** 恢复自动控制，**V** 切换分屏，**Q / Esc** 退出。实时模式使用摄像头上的视觉检测与意图状态，日志和运行指标写入 `outputs/live/`。


## 延时流式左右分屏导出

延时分屏的左侧按输入时钟播放原视频，右侧按流式输出时刻播放自动取景画面，通常相差约6秒。两侧左上角显示各自的源视频时间，声音跟随右侧。开始时右侧等待缓冲；左侧播放结束后保持最后一帧，右侧继续播放剩余内容。

### 运行

安装requirements-c.txt，并按上表放置原视频。在仓库根目录导出全部23段：

```bash
python -m scripts.export_delayed_split --out outputs/delayed_split
```

只处理指定视频：

```bash
python -m scripts.export_delayed_split --clips C02 D01 --out outputs/delayed_selected
```

输出目录须不存在。延时分屏使用流式代理视频，每侧默认高 720 像素，右侧自动取景沿用亮度锐化，成片以 H.264 CRF 17 编码。`--height` 可调整高度，`--max-seconds 24` 可导出前 24 秒播放时间的短测。

### 输入和输出

仓库的outputs/stream提供感知、语音和运行时序，render_inputs.json登记源文件哈希、代理尺寸和帧率。程序使用outputs/stream_media中的代理视频；缺少时校验自备原视频并自动生成。视频代理和成片不提交GitHub，不需要其他项目目录。

重建代理需要与流式记录一致的源文件。E04的720×1280渲染副本可用于离线导出，不能直接替代哈希不同的流式源文件。

每段输出位于outputs/delayed_split/<编号>/：

| 文件 | 内容 |
|---|---|
| compare.mp4 | 延时分屏成片及右侧对应的原声 |
| split.jsonl | 每帧的播放时间、两侧源时间和镜头状态 |
| export.json | 输入来源、缓冲、画面尺寸及导出参数 |

### 判断与取景

默认current-b按保存的感知与语音可用时序回放B，再交给C取景；--intent-source recorded使用实测保存的B状态。此过程不重跑检测或ASR，是已记录输入上的受控回放，不代表重新测得了端到端实时吞吐。

流式取景允许1000毫秒观测间断，与B的有效期对齐。同一目标持续展示时，短时缺框保留取景和缩放尺度，检测恢复后继续跟踪；持续超时才退回全景。NONE和CANDIDATE进入全景过渡，ENDING保持200毫秒后退出。

跟踪最大边缘速度为每秒1.6个归一化画面长度，退出速度为每秒0.8，平滑因子0.20。离线取景采用500毫秒失检保持；两种模式共用C的构图逻辑。长时间缺框和B的展示分段仍可能产生镜头过渡。

右侧按记录的实际输出时刻呈现，声音按配置缓冲延后，因此帧迟滞和导出帧率量化可能带来小幅音画偏差。模型预热、实测运行与C重渲染耗时分别记录和理解。镜头回放核验见outputs/stream/camera_validation.json，整体评价见[流式结果](stream.md)。

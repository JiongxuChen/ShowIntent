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

```powershell
python -m scripts.run_camera_collection --out outputs/camera --render
python -m scripts.verify_camera_collection --root outputs/camera
```

成片在 `outputs/camera/<编号>/camera.mp4`，同目录的 `shots.jsonl` 记录逐帧镜头。备齐原片后，核验应输出 `Verified 23 movies and 23 shot traces`。`outputs/camera/` 下的 `metrics.json` 记录渲染与取景指标，`media_audit.json` 和 `shot_metrics.json` 分别记录音画与逐片构图核验结果。

只做 C02 和 D01 的展示时，可单独运行 `python -m scripts.run_camera_collection --out outputs/selected --render --clips C02 D01`。缺少原片的片段只生成日志，终端显示 `packets only`。全量核验需要全部 23 份镜头日志。重复导出时请使用新的输出目录。

## 制作 PPT 素材

备齐原片后，在仓库根目录运行一次即可生成全部 23 段完整的同步分屏视频。每段左侧是固定全景，右侧是自动取景，保留原声。此命令直接读取原片和 A/B 缓存，可独立运行。

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
& $ffmpeg -v error -n -i "$folder/compare.mp4" -ss $startSec -t $durationSec -c:v libx264 -pix_fmt yuv420p -c:a aac -movflags +faststart "$folder/clip_${startSec}s_${durationSec}s.mp4"
```

C02 第 17 秒可截到橙子特写。要展示普通持物时镜头保持全景，可选 D01 第 63 秒。

## 取景与日志

- A/B 包按同一媒体时间对齐。C 在 `SHOWING` 时跟随 B 指定的 `target_id`，将目标和邻近的关联手部纳入构图框，每侧按相应边长留 **13%**，最大放大 **3.0 倍**。默认成片保持原片正立宽高比，长边至多 1280 像素。
- `NONE` 和 `CANDIDATE` 以全景为终点。目标短时失检最多保持 500 ms，`ENDING` 保持最后构图 200 ms 后平滑退出。
- `shots.jsonl` 记录 `timestamp_ms`、镜头模式 `mode`、目标 ID、源画面归一化裁剪框 `crop_norm`、倍率 `zoom` 和决策原因 `reason`。目标保留率按 B 请求展示且 A 有对应可见框的帧计算，缺框另计。

## 实时摄像头（可选）

```powershell
python -m pip install "ultralytics>=8.3,<9" "mediapipe==0.10.21" "opencv-contrib-python>=4.8,<4.12"
python -m scripts.download_closed_set_models --models yolo11n
python -m src.camera.live --source 0 --out outputs/live/shots.jsonl
```

在预览窗口按 **W** 强制全景，**P** 暂停或继续构图，**R** 恢复自动控制，**V** 切换分屏，**Q / Esc** 退出。实时模式使用摄像头上的视觉检测与意图状态，日志和运行指标写入 `outputs/live/`。

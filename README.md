# ShowIntent

ShowIntent根据手物交互、物体轨迹和语音证据判断何时展示物品，向C镜头模块输出展示状态与目标框。提供当前离线方式和6秒缓冲的延时流式原型。

仅水果场景用于开发，书籍、水杯、饮料用于评价。共23段视频、31个展示事件：水果11段，书籍E01–E04、水杯F01–F04、饮料G01–G04；视频与权重暂不发布，标注、预测和评价已提供。

## 安装

使用Python 3.10或3.11，先安装适合本机的PyTorch，再执行：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-open-vocabulary.txt
```

无需模型的B缓存重放只使用标准库。GPU推理还需要与本机CUDA和CTranslate2兼容的运行库。

## 离线：当前B与三模态

```bash
python -m scripts.run_refined_b --out outputs/new_replay
```

从`outputs/cache/<场景>`读取默认YOLO11m感知、原始large-v3语音及时间轴，重放展示意图状态机，生成23段纯视觉、纯语音和视听结果。先验证E04可加`--clips E04`。输出目录必须是新目录。

正式结果在`outputs/results`。五模型对比在`outputs/benchmarks`：YOLO11n、YOLO11m、YOLO11l、RT-DETR-L、场景先验Grounding DINO Tiny。DINO获知候选物品范围，属于额外场景信息，不能宣称等信息模型排名。

## 五模型重新推理

自行准备Example_Data1～4及对应当前编号视频，下载权重：

```bash
python -m scripts.download_closed_set_models
python -m scripts.download_grounding_dino
python -m scripts.download_joint_models --models speech
python -m scripts.run_closed_set_benchmark --out outputs/new_benchmark
```

该入口对五模型重新做增强视觉检测，再用相同已发布ASR和当前B完成三模态评价；仅跑书籍负例可加`--clips E04`。为控制仓库大小，仅默认YOLO11m的感知缓存随仓库发布，其余模型的完整评分及预测事件区间均在benchmark报告中。重新检测需要原视频和权重。

若要重做ASR：

```bash
python -m scripts.run_speech --audio Example_Data2/E04.mp4 \
  --model models/faster-whisper-large-v3 --out outputs/new_speech_E04.jsonl \
  --mode offline --window-ms 6000 --context-ms 1500 --beam-size 5 \
  --device cuda --compute-type float32
```

重新识别语音属于新实验；五模型对比默认复用相同语音，避免混淆变量。

## 延时流式

准备原视频、`models/yolo11m.pt`和`models/faster-whisper-base`：

```bash
python -m scripts.run_stream_collection --suite all \
  --out outputs/new_stream
```

默认增强视觉1280、检测10Hz、6秒缓冲、base CPU int8/beam1、4秒音频块、adapted B。可加`--clips A02 E03 E04`做小集合验证。流式内部调度、进程和代理视频工具集中在`src/streaming`。

单视频也可执行：

```bash
python -m src.streaming.prepare_stream_proxy \
  --video Example_Data4/G04.mp4 --out outputs/input_G04.mp4
python -m scripts.run_delayed_stream \
  --video outputs/input_G04.mp4 --out outputs/new_G04 --device cuda
```

本机必须有足够吞吐才能维持固定播放缓冲；模型预热时间另计。不能只凭输出文件存在宣称实时。23段已完成流式的实际播放状态、语音完成时间及评价保存在`outputs/stream`。纯语音输出只有时间意图，没有物体框。

## 文档和结果

- [A视觉](docs/A.md)、[B当前方法及流式差异](docs/B.md)、[C接口与呈现](docs/C.md)、[D语音](docs/D.md)
- [评价协议与流式69条逐视频指标](docs/stream.md)
- [离线五模型345条逐视频模态指标](docs/offline.md)
- [目录结构](docs/project_structure.md)、[发布操作](docs/publishing.md)

默认YOLO11m视听事件F1为0.935、覆盖率88.10%；场景先验DINO为0.889、95.22%；延时流式YOLO11m为0.935、83.73%。覆盖率高也可能将多次展示合并，必须同时查看事件计数、误触发时间和边界误差。语音增益有限，详见正式评价，不宣称已经实现显著全场景互补。

测试：`python -m pytest -q`。原始视频、模型权重、运行环境及重复试验不随仓库提交。

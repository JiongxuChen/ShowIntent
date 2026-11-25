# ShowIntent

数据用途按场景区分：水果组用于开发；书籍、水杯、饮料三组用于评价。指标仍按四个场景分别报告。

ShowIntent根据讲解者的手部动作、物体轨迹和语音线索，判断何时展示哪个物品，并生成平滑放大的离线预览。

- A：全类别物体检测、手部感知、交互关系和追踪。
- B：视觉为主、语音辅助的展示意图判断，当前使用`semantic-priority-rest`。
- 简易C：根据B的目标生成保持、缩放和退出预览，供观察触发效果；不是最终镜头模块。
- D：Whisper语音转写与轻量展示语言分析，不依赖语言大模型。

## 数据与结果

| 目录 | 场景 | 视频编号 |
|---|---|---|
| Example_Data1 | 水果 | A01–A03、B01–B03、C01–C04、D01，共11段；排除A04 |
| Example_Data2 | 书籍 | E01–E03 |
| Example_Data3 | 水杯 | F01–F04 |
| Example_Data4 | 饮料 | G01–G04 |

视频和模型权重不随仓库分发。原名对应关系见[evaluation/metadata/video_name_mapping.csv](evaluation/metadata/video_name_mapping.csv)。仓库包含22段的标注、缓存感知、语音事件、逐帧B结果及评价，可直接重放B。

[逐视频指标](docs/evaluation.md)列出纯视觉、纯语音、视听三种模式。全量覆盖率分别为85.73%、54.89%、88.10%；视听事件F1为0.9508。覆盖率仅表示展示时间覆盖，不表示目标身份或放大框正确率。水杯、饮料标签仍待复核。

## 快速复现B（无需视频、模型或GPU）

在仓库根目录使用Python 3.10或3.11：

```bash
python -m scripts.run_refined_b --profile semantic-priority-rest --out outputs/replay
```

输出目录必须是新目录。三个模式分别输出为`visual-only.jsonl`、`audio-only.jsonl`、`showintent.jsonl`，另有`evaluation.json`。B重放只需标准库；测试可安装`requirements-b-dev.txt`后运行`python -m pytest -q`。

## 从原视频运行A与D

完整流程需要自行准备视频及模型。建议新建Python 3.10或3.11虚拟环境，按PyTorch官方说明安装适合设备的PyTorch，再安装：

```bash
python -m pip install -r requirements.txt
python -m scripts.run_complementary --input-dir Example_Data2 --out outputs/new_books --vision-model models/yolo11m.pt --speech-model models/faster-whisper-large-v3
```

视觉权重使用YOLO11m；ASR目录需为可供faster-whisper加载的本地large-v3模型。该入口生成感知、语音及原融合流程结果；其内置B属于较早基线，当前v3重放以上节入口为准。迁入当前数据集时需先在manifest登记新缓存路径，再运行v3，不能把旧入口的B评分当作当前v3。

## 生成简易C预览

本地放好E01.mp4后：

```bash
python -m scripts.render_b_preview --video Example_Data2/E01.mp4 --perception outputs/cache/books/perception/perception_E01.jsonl --intent outputs/results/books/E01/showintent.jsonl --out outputs/E01_preview.mp4
```

预览会保留音轨并记录裁剪信息。模型检测类别有限，仍可能出现漏检、书籍误识别、目标绑定失败或确认偏晚。

## 四种视觉模型对比

项目包含YOLO11n、YOLO11m、YOLO11l和RT-DETR-L四种闭集检测器的对比，覆盖水果、书籍、水杯、饮料四个场景。实现、配置和16份评价JSON均已包含，权重另行下载。

- [四场景结果表、实验说明与运行命令](docs/model_comparison.md)
- 结果：`outputs/benchmarks/<场景>/<模型>/evaluation.json`
- 运行：`scripts/run_closed_set_benchmark.py`
- 配置：`configs/closed_set_benchmark.json`

## 文档与目录

完整目录及每个文件用途见[项目结构与逐文件说明](docs/project_structure.md)。

- [A：检测、手部与追踪](docs/A.md)
- [B：展示意图与语言融合](docs/B.md)
- [C：简易预览与正式交付](docs/C.md)
- [D：语音转写与语言线索](docs/D.md)
- [评价规范与22段逐视频指标](docs/evaluation.md)
- [闭集模型对比](docs/model_comparison.md)
- [上传GitHub](docs/publishing.md)

src为实现，scripts保留10个功能入口；evaluation保存标注与清单；outputs/cache保存A/D输入，outputs/results保存正式结果，outputs/benchmarks保存模型对比评价。视频和模型不随仓库分发。

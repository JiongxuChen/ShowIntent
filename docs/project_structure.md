# 项目结构与逐文件说明

本文对应仓库中的实际文件。先读数据流和目录职责，再按表查具体文件。路径均相对仓库根目录；`.git/`为Git内部数据，不属于项目实现。Python字节码缓存、权重、原视频和运行环境不在本清单中。

## 从哪里开始

| 任务 | 入口 / 说明 |
|---|---|
| 了解项目与安装 | [README](../README.md) |
| 复现22段正式B结果 | `python -m scripts.run_refined_b --out outputs/replay` |
| 理解算法 | [A](A.md)、[B](B.md)、[C](C.md)、[D](D.md) |
| 查逐视频指标 | [评价与结果](evaluation.md) |
| 正式C接入 | [C交付与验收](C.md)，使用同份A感知与B的showintent.jsonl |
| 查模型差异 | [模型对比](model_comparison.md) |
| 上传代码 | [发布操作](publishing.md) |

## 目录总览

```text
ShowIntent/
├── README.md、requirements*.txt、pytest.ini、release_manifest.json
├── src/
│   ├── perception/    A：检测、手部、外观与追踪
│   ├── intent/        B：时序、实例绑定与视听展示判断
│   ├── speech/        D：ASR、语言规则和证据接口
│   ├── camera/        C：基础镜头控制与渲染
│   └── evaluation/    共享场景评分工具
├── scripts/           可执行运行、下载、预览与评价入口
├── configs/           视觉、语音、融合与模型对比参数
├── evaluation/
│   ├── fruit/、books/、cups/、drinks/
│   └── metadata/      名称、排除、复核及时间记录
├── outputs/
│   ├── cache/<场景>/{perception,speech_raw,timestamps}/
│   ├── results/<场景>/<视频>/
│   └── benchmarks/<场景>/<模型>/
├── tests/
│   └── fixtures/      自动化联通测试的小型输入
└── docs/              模块、评价、模型对比、发布与结构说明
```

## 数据流与文件关系

原视频 → 媒体PTS → A感知；原视频音轨 → D语音事件；A感知＋D语言证据 → B三模式序列；A感知＋B视听输出＋原视频 → 简易C预览。标注只在预测完成后由评分器读取，不参与触发推理。

`outputs/cache`是可复用输入，`outputs/results`是正式预测及评分，`outputs/benchmarks`是模型对比。它们不是三份相同数据；删除缓存会使无需模型的B重放失去输入。重放到新输出目录，不覆盖随仓库提供的正式结果。

每个A包与B包使用相同timestamp_ms。target_id仅对应同一次A运行的track_id；纯语音序列没有定位框。标签按原媒体时间定义，不能把包序号当固定帧率。

场景映射：fruit为水果11段，books为E01～E03，cups为F01～F04，drinks为G01～G04；A04排除。四个闭集模型报告分别保存在四个场景目录；不同场景报告的B配置差异见模型对比说明。

## 逐文件清单

以下清单对每个项目文件给出用途。重复的22段缓存和预测仍逐个列出，以便直接定位；同类后缀的字段解释分别见A、B、D及评价文档。

### 根目录

| 文件 | 作用 |
|---|---|
| [.gitattributes](../.gitattributes) | 定义CSV的行尾与差异检查规则。 |
| [.gitignore](../.gitignore) | 排除视频、权重、环境与临时输出，仅允许精选缓存和结果进入版本管理。 |
| [README.md](../README.md) | 项目总览、安装要求、离线重放、原视频处理与简易预览命令。 |
| [ShowIntent_Four_Person_Roadmap_v2_1.md](../ShowIntent_Four_Person_Roadmap_v2_1.md) | A/B/C/D职责与阶段工作规划；具体运行以README及模块文档为准。 |
| [ShowIntent_Project_Description_v2_1.md](../ShowIntent_Project_Description_v2_1.md) | 项目任务、目标能力与范围说明。 |
| [pytest.ini](../pytest.ini) | pytest测试发现范围与目录排除规则。 |
| [release_manifest.json](../release_manifest.json) | 发布文件实际SHA-256及验证摘要；用于核对文件是否变化，不参与推理。 |
| [requirements-a.txt](../requirements-a.txt) | 单独运行视觉模块所需依赖。 |
| [requirements-b-dev.txt](../requirements-b-dev.txt) | 测试工具依赖；缓存B重放自身仅需标准库。 |
| [requirements-joint.txt](../requirements-joint.txt) | 视听联合运行的补充依赖约束。 |
| [requirements.txt](../requirements.txt) | 完整视觉、语音与渲染运行依赖；PyTorch按设备单独准备。 |

### 运行脚本

| 文件 | 作用 |
|---|---|
| [scripts/__init__.py](../scripts/__init__.py) | Python包标记，提供该模块的导入边界；没有独立运行命令。 |
| [scripts/download_closed_set_models.py](../scripts/download_closed_set_models.py) | 下载模型对比所需官方检测权重并记录校验信息。 |
| [scripts/download_joint_models.py](../scripts/download_joint_models.py) | 下载联合流程所需检测和ASR权重，支持文件校验和断点恢复。 |
| [scripts/enrich_speech.py](../scripts/enrich_speech.py) | 为已保存语音事件附加轻量语言解释与耗时记录。 |
| [scripts/evaluate_formal.py](../scripts/evaluate_formal.py) | 对指定场景的已有预测单独评分，支持时间与实例映射检查；不是推理入口。 |
| [scripts/extract_video_timestamps.py](../scripts/extract_video_timestamps.py) | 解码原视频，提取逐帧媒体PTS，供A逐帧对齐。 |
| [scripts/render_b_preview.py](../scripts/render_b_preview.py) | 简易C预览：目标框转裁剪、段内锁定尺度、平滑位置、退场保持和音轨封装。 |
| [scripts/run_closed_set_benchmark.py](../scripts/run_closed_set_benchmark.py) | 固定后处理、B及ASR条件，替换闭集检测器并输出时间指标与处理统计。 |
| [scripts/run_complementary.py](../scripts/run_complementary.py) | 原视频A/D处理与融合诊断入口；包含阶段计划、输入哈希、检查点与恢复机制。其内置B诊断输出与正式缓存重放应按README区分。 |
| [scripts/run_refined_b.py](../scripts/run_refined_b.py) | 正式离线B入口：读取四场景缓存，重放纯视觉/纯语音/视听并统一评分。 |
| [scripts/run_speech.py](../scripts/run_speech.py) | 独立ASR入口，将音频转写为SpeechEvent及窗口统计。 |

### 源代码：A

| 文件 | 作用 |
|---|---|
| [src/perception/__init__.py](../src/perception/__init__.py) | Python包标记，提供该模块的导入边界；没有独立运行命令。 |
| [src/perception/appearance_tracker.py](../src/perception/appearance_tracker.py) | 按类别、位置、重叠、运动与外观距离进行短期实例关联，只输出当前检测框。 |
| [src/perception/closed_set.py](../src/perception/closed_set.py) | YOLO与RT-DETR统一预测适配层，服务模型对比。 |
| [src/perception/downsample.py](../src/perception/downsample.py) | 视频缩放、方向处理与降采样工具；不改变默认B逻辑。 |
| [src/perception/enhanced.py](../src/perception/enhanced.py) | 正式增强视觉适配器：全帧检测、手部局部复检、去重、颜色外观、实例追踪和手物关系。 |
| [src/perception/make_mock.py](../src/perception/make_mock.py) | 生成接口验证所需小型感知样例，不作为真实视频评分。 |
| [src/perception/relations.py](../src/perception/relations.py) | 基础手物关系计算工具。 |
| [src/perception/run.py](../src/perception/run.py) | 基础视觉入口及模型/旋转/坐标工具；enhanced复用其中PerceptionModels等组件。 |
| [src/perception/timing.py](../src/perception/timing.py) | 读取、校验媒体时间戳，防止感知时间错位。 |
| [src/perception/tracker.py](../src/perception/tracker.py) | 基础几何追踪、IoU和距离工具；外观追踪也复用几何函数。 |
| [src/perception/types.py](../src/perception/types.py) | 视觉数据字段与包结构定义。 |
| [src/perception/validate_jsonl.py](../src/perception/validate_jsonl.py) | 检查感知JSONL结构、取值和时间序列。 |

### 源代码：B

| 文件 | 作用 |
|---|---|
| [src/intent/__init__.py](../src/intent/__init__.py) | Python包标记，提供该模块的导入边界；没有独立运行命令。 |
| [src/intent/analyze_joint_run.py](../src/intent/analyze_joint_run.py) | 分析已有联合运行日志与输入完整性，不执行模型推理。 |
| [src/intent/audio_timeline.py](../src/intent/audio_timeline.py) | 纯语音时间基线：依据语言动作开启/延长/结束展示，无物体定位。 |
| [src/intent/baseline.py](../src/intent/baseline.py) | 仅按抬起触发的对照方法，不是正式融合方法。 |
| [src/intent/complementarity.py](../src/intent/complementarity.py) | 比较不同模态预测及差异区间；额外触发不能自动等同增益。 |
| [src/intent/diagnose.py](../src/intent/diagnose.py) | 运行合成场景并检查状态包、展示区间与边界行为。 |
| [src/intent/evaluate_review.py](../src/intent/evaluate_review.py) | 辅助区间评分工具，不是当前22段正式评分入口。 |
| [src/intent/formal_evaluation.py](../src/intent/formal_evaluation.py) | 标签校验、时间IoU、一对一事件匹配、时间及实例评分核心。 |
| [src/intent/fusion.py](../src/intent/fusion.py) | 视听候选证据、权重与确认条件；分数是规则强度。 |
| [src/intent/grounding.py](../src/intent/grounding.py) | 将语句绑定到语句时刻附近的真实实例，处理缺失、指代范围和放回失效。 |
| [src/intent/mock_scenarios.py](../src/intent/mock_scenarios.py) | 确定性动作/语音场景生成器，用于边界与时序测试。 |
| [src/intent/persistent.py](../src/intent/persistent.py) | 持续展示、有限交互记忆、呈现过程约束与语言绑定基础层。 |
| [src/intent/replay_real.py](../src/intent/replay_real.py) | 感知包加载校验及多模式辅助重放；服务诊断入口和检测器对比。 |
| [src/intent/rest_aware.py](../src/intent/rest_aware.py) | 正式B顶层实现：静置区域约束的语言放回判断及组合输出策略。 |
| [src/intent/run.py](../src/intent/run.py) | 通用单份感知JSONL的B接口入口，供接口测试；正式22段运行用run_refined_b。 |
| [src/intent/semantic_priority.py](../src/intent/semantic_priority.py) | 语音提前确认及基础视觉保障；由正式B内部复用。 |
| [src/intent/state_machine.py](../src/intent/state_machine.py) | 基础状态机和包结构，供兼容性与行为测试；正式方法基于temporal组合。 |
| [src/intent/temporal.py](../src/intent/temporal.py) | 共享时序状态机及视觉轨迹证据，维护NONE/CANDIDATE/SHOWING/ENDING转换。 |
| [src/intent/types.py](../src/intent/types.py) | 意图包与几何、桌面基准等共享数据结构。 |

### 源代码：D

| 文件 | 作用 |
|---|---|
| [src/speech/__init__.py](../src/speech/__init__.py) | Python包标记，提供该模块的导入边界；没有独立运行命令。 |
| [src/speech/asr.py](../src/speech/asr.py) | faster-whisper音频分窗、过去上下文、词时间去重及识别耗时统计。 |
| [src/speech/bridge.py](../src/speech/bridge.py) | 加载语音事件，将可用语言证据对齐到感知时间。 |
| [src/speech/cue_adapter.py](../src/speech/cue_adapter.py) | 将词汇线索适配为可按时间消费的语音证据，不自行选择实例。 |
| [src/speech/events.py](../src/speech/events.py) | SpeechEvent字段、时间校验、最终事件与可用性约定。 |
| [src/speech/intent_language.py](../src/speech/intent_language.py) | 展示、介绍、描述、继续、停止、切换等动作及指代范围规则。 |
| [src/speech/language.py](../src/speech/language.py) | 基础中英词汇、类别及线索抽取工具。 |
| [src/speech/presentation_language.py](../src/speech/presentation_language.py) | 正式离线重放采用的上下文展示语言解释。 |
| [src/speech/semantics.py](../src/speech/semantics.py) | 校验语言结构并转为标准线索，防止非法语义字段进入融合。 |

### 源代码：C

| 文件 | 作用 |
|---|---|
| [src/camera/__init__.py](../src/camera/__init__.py) | Python包标记，提供该模块的导入边界；没有独立运行命令。 |
| [src/camera/controller.py](../src/camera/controller.py) | 基础C控制器：根据意图和目标状态产生镜头控制包。 |
| [src/camera/render.py](../src/camera/render.py) | 基础C画面渲染支持；独立于B判断。 |
| [src/camera/run.py](../src/camera/run.py) | A/B序列联结与ShotPacket输出入口，也支持视频渲染参数。 |

### 共享评价工具

| 文件 | 作用 |
|---|---|
| [src/evaluation/__init__.py](../src/evaluation/__init__.py) | Python包标记，提供该模块的导入边界；没有独立运行命令。 |
| [src/evaluation/collection.py](../src/evaluation/collection.py) | 场景标签冻结校验、时钟一致性检查、三模式评分汇总与场景评分辅助函数。 |

### 源代码包标记

| 文件 | 作用 |
|---|---|
| [src/__init__.py](../src/__init__.py) | Python包标记，提供该模块的导入边界；没有独立运行命令。 |

### 配置

| 文件 | 作用 |
|---|---|
| [configs/audio_priority_experiment.json](../configs/audio_priority_experiment.json) | 语音优先策略辅助参数记录；正式入口默认读取semantic_priority_fusion。 |
| [configs/closed_set_benchmark.json](../configs/closed_set_benchmark.json) | 模型列表、权重路径、检测尺寸、置信度及模型对比控制参数。 |
| [configs/evidence_fusion.json](../configs/evidence_fusion.json) | 通用语音证据接口与辅助重放的基础配置。 |
| [configs/presentation_intent.json](../configs/presentation_intent.json) | 视觉候选、确认、缺失、结束、轨迹与动作判定参数。 |
| [configs/semantic_priority_fusion.json](../configs/semantic_priority_fusion.json) | 正式离线B融合权重、支持门槛和视觉/联合确认时间。 |
| [configs/speech_lexicon.json](../configs/speech_lexicon.json) | 物品类别称呼、同义词和展示/结束等词汇规则。 |
| [configs/stable_fusion.json](../configs/stable_fusion.json) | 基础融合层与部分对比流程的参数。 |

### 文档

| 文件 | 作用 |
|---|---|
| [docs/A.md](../docs/A.md) | 增强视觉算法、默认参数、追踪、安全边界与独立运行说明。 |
| [docs/B.md](../docs/B.md) | 正式展示意图逻辑、视觉/语音绑定、确认与退出及配置说明。 |
| [docs/C.md](../docs/C.md) | 简易预览实现及交给正式C的文件、接口、取景约束和验收步骤。 |
| [docs/D.md](../docs/D.md) | ASR分窗、SpeechEvent、语言解释、时间可用性与错误边界。 |
| [docs/evaluation.md](../docs/evaluation.md) | 评价规范、指标定义、22段三模式逐视频和场景汇总。 |
| [docs/model_comparison.md](../docs/model_comparison.md) | 四个闭集检测器在四场景中的结果、配置范围与运行方式。 |
| [docs/project_structure.md](../docs/project_structure.md) | 本文件：目录导航、逐文件用途和数据流说明。 |
| [docs/publishing.md](../docs/publishing.md) | 直接提交并推送main的详细步骤、文件检查及验证记录。 |

### 评价数据

| 文件 | 作用 |
|---|---|
| [evaluation/README.md](../evaluation/README.md) | 四场景评价文件及metadata的阅读指引。 |
| [evaluation/books/annotations.csv](../evaluation/books/annotations.csv) | 该场景完整SHOWING/NON_SHOWING区间、事件、目标与复核状态。 |
| [evaluation/books/dataset_manifest.csv](../evaluation/books/dataset_manifest.csv) | 该场景视频、感知缓存路径及有效评价起止时间。 |
| [evaluation/books/protocol.json](../evaluation/books/protocol.json) | 该场景时间区间、IoU与身份评分等固定评价约定。 |
| [evaluation/books/split_manifest.csv](../evaluation/books/split_manifest.csv) | 评分器兼容分组和录制元数据；面向读者的汇总按场景呈现。 |
| [evaluation/books/track_mapping.csv](../evaluation/books/track_mapping.csv) | track_id到物理实例的时间映射，待复核项不算已确认身份。 |
| [evaluation/cups/annotation_record.json](../evaluation/cups/annotation_record.json) | 标签冻结哈希、原视频校验与标注状态；缓存重放不强制读取原视频。 |
| [evaluation/cups/annotations.csv](../evaluation/cups/annotations.csv) | 该场景完整SHOWING/NON_SHOWING区间、事件、目标与复核状态。 |
| [evaluation/cups/dataset_manifest.csv](../evaluation/cups/dataset_manifest.csv) | 该场景视频、感知缓存路径及有效评价起止时间。 |
| [evaluation/cups/protocol.json](../evaluation/cups/protocol.json) | 该场景时间区间、IoU与身份评分等固定评价约定。 |
| [evaluation/cups/split_manifest.csv](../evaluation/cups/split_manifest.csv) | 评分器兼容分组和录制元数据；面向读者的汇总按场景呈现。 |
| [evaluation/cups/track_mapping.csv](../evaluation/cups/track_mapping.csv) | track_id到物理实例的时间映射，待复核项不算已确认身份。 |
| [evaluation/drinks/annotation_record.json](../evaluation/drinks/annotation_record.json) | 标签冻结哈希、原视频校验与标注状态；缓存重放不强制读取原视频。 |
| [evaluation/drinks/annotations.csv](../evaluation/drinks/annotations.csv) | 该场景完整SHOWING/NON_SHOWING区间、事件、目标与复核状态。 |
| [evaluation/drinks/dataset_manifest.csv](../evaluation/drinks/dataset_manifest.csv) | 该场景视频、感知缓存路径及有效评价起止时间。 |
| [evaluation/drinks/protocol.json](../evaluation/drinks/protocol.json) | 该场景时间区间、IoU与身份评分等固定评价约定。 |
| [evaluation/drinks/split_manifest.csv](../evaluation/drinks/split_manifest.csv) | 评分器兼容分组和录制元数据；面向读者的汇总按场景呈现。 |
| [evaluation/drinks/track_mapping.csv](../evaluation/drinks/track_mapping.csv) | track_id到物理实例的时间映射，待复核项不算已确认身份。 |
| [evaluation/fruit/annotations.csv](../evaluation/fruit/annotations.csv) | 该场景完整SHOWING/NON_SHOWING区间、事件、目标与复核状态。 |
| [evaluation/fruit/dataset_manifest.csv](../evaluation/fruit/dataset_manifest.csv) | 该场景视频、感知缓存路径及有效评价起止时间。 |
| [evaluation/fruit/protocol.json](../evaluation/fruit/protocol.json) | 该场景时间区间、IoU与身份评分等固定评价约定。 |
| [evaluation/fruit/split_manifest.csv](../evaluation/fruit/split_manifest.csv) | 评分器兼容分组和录制元数据；面向读者的汇总按场景呈现。 |
| [evaluation/fruit/track_mapping.csv](../evaluation/fruit/track_mapping.csv) | track_id到物理实例的时间映射，待复核项不算已确认身份。 |
| [evaluation/metadata/annotation_review_record.json](../evaluation/metadata/annotation_review_record.json) | 标注复核记录与文件哈希。 |
| [evaluation/metadata/exclusions.json](../evaluation/metadata/exclusions.json) | A04排除理由与范围记录。 |
| [evaluation/metadata/media_timebase_audit.json](../evaluation/metadata/media_timebase_audit.json) | 媒体时间、方向和标注/感知对齐核验记录。 |
| [evaluation/metadata/video_name_mapping.csv](../evaluation/metadata/video_name_mapping.csv) | 原视频编号与E/F/G统一编号的对应表。 |

| [evaluation/metadata/reporting_roles.json](../evaluation/metadata/reporting_roles.json) | 项目负责人确认的场景报告用途：水果用于开发，其余三组用于评价；不作为统计独立性声明。 |

### 测试与小型样例

| 文件 | 作用 |
|---|---|
| [tests/fixtures/demo_object_show/README.md](../tests/fixtures/demo_object_show/README.md) | 小型联通样例的字段与使用说明。 |
| [tests/fixtures/demo_object_show/perception.jsonl](../tests/fixtures/demo_object_show/perception.jsonl) | 固定感知输入，供命令行接口测试，不依赖检测器或真实视频。 |
| [tests/fixtures/demo_object_show/speech_events.jsonl](../tests/fixtures/demo_object_show/speech_events.jsonl) | 与样例感知配对的固定语音事件。 |
| [tests/test_asr_context.py](../tests/test_asr_context.py) | 验证音频上下文与重叠词去重。 |
| [tests/test_audio_timeline.py](../tests/test_audio_timeline.py) | 验证纯语音时间基线。 |
| [tests/test_b_preview.py](../tests/test_b_preview.py) | 验证裁剪尺度、位置与退出保持。 |
| [tests/test_camera_controller.py](../tests/test_camera_controller.py) | 验证镜头状态与回退。 |
| [tests/test_closed_set_benchmark.py](../tests/test_closed_set_benchmark.py) | 验证模型对比输入和场景评分。 |
| [tests/test_complementary_intent.py](../tests/test_complementary_intent.py) | 验证语言解释与实例互补规则。 |
| [tests/test_complementary_pipeline.py](../tests/test_complementary_pipeline.py) | 验证完整流程检查点、恢复和产物联通。 |
| [tests/test_cups_evaluation.py](../tests/test_cups_evaluation.py) | 验证场景标签与时间校验。 |
| [tests/test_download_resume.py](../tests/test_download_resume.py) | 验证下载中断恢复和校验。 |
| [tests/test_enhanced_perception.py](../tests/test_enhanced_perception.py) | 验证增强视觉外观和手部裁剪。 |
| [tests/test_evidence_fusion.py](../tests/test_evidence_fusion.py) | 验证语音融合和A/B/C入口联通。 |
| [tests/test_formal_evaluation.py](../tests/test_formal_evaluation.py) | 验证标签、事件及身份评分。 |
| [tests/test_general_categories.py](../tests/test_general_categories.py) | 验证非预设类别的兼容处理。 |
| [tests/test_intent_language.py](../tests/test_intent_language.py) | 验证语言动作与指代范围。 |
| [tests/test_intent_regression.py](../tests/test_intent_regression.py) | 验证意图行为一致性。 |
| [tests/test_intent_state_machine.py](../tests/test_intent_state_machine.py) | 验证基础状态转换。 |
| [tests/test_joint_audit.py](../tests/test_joint_audit.py) | 验证联合结果审计。 |
| [tests/test_joint_log_regression.py](../tests/test_joint_log_regression.py) | 验证已有日志的状态核查。 |
| [tests/test_member_d_speech.py](../tests/test_member_d_speech.py) | 验证D事件与线索接口。 |
| [tests/test_perception_core.py](../tests/test_perception_core.py) | 验证视觉坐标、旋转与基本包。 |
| [tests/test_persistent_intent.py](../tests/test_persistent_intent.py) | 验证持续展示、交互记忆与退出。 |
| [tests/test_pickup_baseline.py](../tests/test_pickup_baseline.py) | 验证抬起对照方法。 |
| [tests/test_presentation_intent.py](../tests/test_presentation_intent.py) | 验证公开字段与时序约定。 |
| [tests/test_real_data_replay.py](../tests/test_real_data_replay.py) | 验证实际感知格式的重放。 |
| [tests/test_rest_aware.py](../tests/test_rest_aware.py) | 验证静置区域与语言放回失效。 |
| [tests/test_speech_observations.py](../tests/test_speech_observations.py) | 验证语音观察字段和时序。 |
| [tests/test_temporal_intent.py](../tests/test_temporal_intent.py) | 验证共享时序状态机。 |
| [tests/test_trigger_precision.py](../tests/test_trigger_precision.py) | 验证展示/搬动/负例的触发边界。 |
| [tests/test_visual_guard.py](../tests/test_visual_guard.py) | 验证基础视觉保障与语音提前确认。 |
| [tests/test_visual_policy.py](../tests/test_visual_policy.py) | 验证视觉输出策略。 |

### 缓存输入

| 文件 | 作用 |
|---|---|
| [outputs/cache/books/perception/perception_E01.jsonl](../outputs/cache/books/perception/perception_E01.jsonl) | books场景E01的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/books/perception/perception_E02.jsonl](../outputs/cache/books/perception/perception_E02.jsonl) | books场景E02的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/books/perception/perception_E03.jsonl](../outputs/cache/books/perception/perception_E03.jsonl) | books场景E03的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/books/speech_raw/speech_events_E01.jsonl](../outputs/cache/books/speech_raw/speech_events_E01.jsonl) | books场景E01的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/books/speech_raw/speech_events_E02.jsonl](../outputs/cache/books/speech_raw/speech_events_E02.jsonl) | books场景E02的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/books/speech_raw/speech_events_E03.jsonl](../outputs/cache/books/speech_raw/speech_events_E03.jsonl) | books场景E03的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/books/timestamps/E01.json](../outputs/cache/books/timestamps/E01.json) | books场景E01的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/books/timestamps/E02.json](../outputs/cache/books/timestamps/E02.json) | books场景E02的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/books/timestamps/E03.json](../outputs/cache/books/timestamps/E03.json) | books场景E03的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/cups/perception/perception_F01.jsonl](../outputs/cache/cups/perception/perception_F01.jsonl) | cups场景F01的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/cups/perception/perception_F02.jsonl](../outputs/cache/cups/perception/perception_F02.jsonl) | cups场景F02的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/cups/perception/perception_F03.jsonl](../outputs/cache/cups/perception/perception_F03.jsonl) | cups场景F03的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/cups/perception/perception_F04.jsonl](../outputs/cache/cups/perception/perception_F04.jsonl) | cups场景F04的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/cups/speech_raw/speech_events_F01.jsonl](../outputs/cache/cups/speech_raw/speech_events_F01.jsonl) | cups场景F01的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/cups/speech_raw/speech_events_F02.jsonl](../outputs/cache/cups/speech_raw/speech_events_F02.jsonl) | cups场景F02的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/cups/speech_raw/speech_events_F03.jsonl](../outputs/cache/cups/speech_raw/speech_events_F03.jsonl) | cups场景F03的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/cups/speech_raw/speech_events_F04.jsonl](../outputs/cache/cups/speech_raw/speech_events_F04.jsonl) | cups场景F04的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/cups/timestamps/F01.json](../outputs/cache/cups/timestamps/F01.json) | cups场景F01的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/cups/timestamps/F02.json](../outputs/cache/cups/timestamps/F02.json) | cups场景F02的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/cups/timestamps/F03.json](../outputs/cache/cups/timestamps/F03.json) | cups场景F03的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/cups/timestamps/F04.json](../outputs/cache/cups/timestamps/F04.json) | cups场景F04的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/drinks/perception/perception_G01.jsonl](../outputs/cache/drinks/perception/perception_G01.jsonl) | drinks场景G01的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/drinks/perception/perception_G02.jsonl](../outputs/cache/drinks/perception/perception_G02.jsonl) | drinks场景G02的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/drinks/perception/perception_G03.jsonl](../outputs/cache/drinks/perception/perception_G03.jsonl) | drinks场景G03的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/drinks/perception/perception_G04.jsonl](../outputs/cache/drinks/perception/perception_G04.jsonl) | drinks场景G04的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/drinks/speech_raw/speech_events_G01.jsonl](../outputs/cache/drinks/speech_raw/speech_events_G01.jsonl) | drinks场景G01的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/drinks/speech_raw/speech_events_G02.jsonl](../outputs/cache/drinks/speech_raw/speech_events_G02.jsonl) | drinks场景G02的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/drinks/speech_raw/speech_events_G03.jsonl](../outputs/cache/drinks/speech_raw/speech_events_G03.jsonl) | drinks场景G03的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/drinks/speech_raw/speech_events_G04.jsonl](../outputs/cache/drinks/speech_raw/speech_events_G04.jsonl) | drinks场景G04的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/drinks/timestamps/G01.json](../outputs/cache/drinks/timestamps/G01.json) | drinks场景G01的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/drinks/timestamps/G02.json](../outputs/cache/drinks/timestamps/G02.json) | drinks场景G02的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/drinks/timestamps/G03.json](../outputs/cache/drinks/timestamps/G03.json) | drinks场景G03的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/drinks/timestamps/G04.json](../outputs/cache/drinks/timestamps/G04.json) | drinks场景G04的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/fruit/perception/perception_A01.jsonl](../outputs/cache/fruit/perception/perception_A01.jsonl) | fruit场景A01的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/fruit/perception/perception_A02.jsonl](../outputs/cache/fruit/perception/perception_A02.jsonl) | fruit场景A02的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/fruit/perception/perception_A03.jsonl](../outputs/cache/fruit/perception/perception_A03.jsonl) | fruit场景A03的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/fruit/perception/perception_B01.jsonl](../outputs/cache/fruit/perception/perception_B01.jsonl) | fruit场景B01的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/fruit/perception/perception_B02.jsonl](../outputs/cache/fruit/perception/perception_B02.jsonl) | fruit场景B02的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/fruit/perception/perception_B03.jsonl](../outputs/cache/fruit/perception/perception_B03.jsonl) | fruit场景B03的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/fruit/perception/perception_C01.jsonl](../outputs/cache/fruit/perception/perception_C01.jsonl) | fruit场景C01的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/fruit/perception/perception_C02.jsonl](../outputs/cache/fruit/perception/perception_C02.jsonl) | fruit场景C02的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/fruit/perception/perception_C03.jsonl](../outputs/cache/fruit/perception/perception_C03.jsonl) | fruit场景C03的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/fruit/perception/perception_C04.jsonl](../outputs/cache/fruit/perception/perception_C04.jsonl) | fruit场景C04的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/fruit/perception/perception_D01.jsonl](../outputs/cache/fruit/perception/perception_D01.jsonl) | fruit场景D01的逐帧A感知，含物体、手、关系、外观与ID。 |
| [outputs/cache/fruit/speech_raw/speech_events_A01.jsonl](../outputs/cache/fruit/speech_raw/speech_events_A01.jsonl) | fruit场景A01的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/fruit/speech_raw/speech_events_A02.jsonl](../outputs/cache/fruit/speech_raw/speech_events_A02.jsonl) | fruit场景A02的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/fruit/speech_raw/speech_events_A03.jsonl](../outputs/cache/fruit/speech_raw/speech_events_A03.jsonl) | fruit场景A03的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/fruit/speech_raw/speech_events_B01.jsonl](../outputs/cache/fruit/speech_raw/speech_events_B01.jsonl) | fruit场景B01的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/fruit/speech_raw/speech_events_B02.jsonl](../outputs/cache/fruit/speech_raw/speech_events_B02.jsonl) | fruit场景B02的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/fruit/speech_raw/speech_events_B03.jsonl](../outputs/cache/fruit/speech_raw/speech_events_B03.jsonl) | fruit场景B03的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/fruit/speech_raw/speech_events_C01.jsonl](../outputs/cache/fruit/speech_raw/speech_events_C01.jsonl) | fruit场景C01的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/fruit/speech_raw/speech_events_C02.jsonl](../outputs/cache/fruit/speech_raw/speech_events_C02.jsonl) | fruit场景C02的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/fruit/speech_raw/speech_events_C03.jsonl](../outputs/cache/fruit/speech_raw/speech_events_C03.jsonl) | fruit场景C03的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/fruit/speech_raw/speech_events_C04.jsonl](../outputs/cache/fruit/speech_raw/speech_events_C04.jsonl) | fruit场景C04的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/fruit/speech_raw/speech_events_D01.jsonl](../outputs/cache/fruit/speech_raw/speech_events_D01.jsonl) | fruit场景D01的原始ASR事件，供语言重放和错误复核。 |
| [outputs/cache/fruit/timestamps/A01.json](../outputs/cache/fruit/timestamps/A01.json) | fruit场景A01的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/fruit/timestamps/A02.json](../outputs/cache/fruit/timestamps/A02.json) | fruit场景A02的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/fruit/timestamps/A03.json](../outputs/cache/fruit/timestamps/A03.json) | fruit场景A03的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/fruit/timestamps/B01.json](../outputs/cache/fruit/timestamps/B01.json) | fruit场景B01的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/fruit/timestamps/B02.json](../outputs/cache/fruit/timestamps/B02.json) | fruit场景B02的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/fruit/timestamps/B03.json](../outputs/cache/fruit/timestamps/B03.json) | fruit场景B03的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/fruit/timestamps/C01.json](../outputs/cache/fruit/timestamps/C01.json) | fruit场景C01的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/fruit/timestamps/C02.json](../outputs/cache/fruit/timestamps/C02.json) | fruit场景C02的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/fruit/timestamps/C03.json](../outputs/cache/fruit/timestamps/C03.json) | fruit场景C03的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/fruit/timestamps/C04.json](../outputs/cache/fruit/timestamps/C04.json) | fruit场景C04的原媒体PTS清单，用于逐帧对齐。 |
| [outputs/cache/fruit/timestamps/D01.json](../outputs/cache/fruit/timestamps/D01.json) | fruit场景D01的原媒体PTS清单，用于逐帧对齐。 |

### 正式结果

| 文件 | 作用 |
|---|---|
| [outputs/results/books/E01/audio-only.jsonl](../outputs/results/books/E01/audio-only.jsonl) | books场景E01：纯语音时间状态，无物体框。 |
| [outputs/results/books/E01/showintent.jsonl](../outputs/results/books/E01/showintent.jsonl) | books场景E01：视听融合，正式交给C的意图序列。 |
| [outputs/results/books/E01/visual-only.jsonl](../outputs/results/books/E01/visual-only.jsonl) | books场景E01：纯视觉意图序列，用于消融对照。 |
| [outputs/results/books/E02/audio-only.jsonl](../outputs/results/books/E02/audio-only.jsonl) | books场景E02：纯语音时间状态，无物体框。 |
| [outputs/results/books/E02/showintent.jsonl](../outputs/results/books/E02/showintent.jsonl) | books场景E02：视听融合，正式交给C的意图序列。 |
| [outputs/results/books/E02/visual-only.jsonl](../outputs/results/books/E02/visual-only.jsonl) | books场景E02：纯视觉意图序列，用于消融对照。 |
| [outputs/results/books/E03/audio-only.jsonl](../outputs/results/books/E03/audio-only.jsonl) | books场景E03：纯语音时间状态，无物体框。 |
| [outputs/results/books/E03/showintent.jsonl](../outputs/results/books/E03/showintent.jsonl) | books场景E03：视听融合，正式交给C的意图序列。 |
| [outputs/results/books/E03/visual-only.jsonl](../outputs/results/books/E03/visual-only.jsonl) | books场景E03：纯视觉意图序列，用于消融对照。 |
| [outputs/results/cups/F01/audio-only.jsonl](../outputs/results/cups/F01/audio-only.jsonl) | cups场景F01：纯语音时间状态，无物体框。 |
| [outputs/results/cups/F01/showintent.jsonl](../outputs/results/cups/F01/showintent.jsonl) | cups场景F01：视听融合，正式交给C的意图序列。 |
| [outputs/results/cups/F01/visual-only.jsonl](../outputs/results/cups/F01/visual-only.jsonl) | cups场景F01：纯视觉意图序列，用于消融对照。 |
| [outputs/results/cups/F02/audio-only.jsonl](../outputs/results/cups/F02/audio-only.jsonl) | cups场景F02：纯语音时间状态，无物体框。 |
| [outputs/results/cups/F02/showintent.jsonl](../outputs/results/cups/F02/showintent.jsonl) | cups场景F02：视听融合，正式交给C的意图序列。 |
| [outputs/results/cups/F02/visual-only.jsonl](../outputs/results/cups/F02/visual-only.jsonl) | cups场景F02：纯视觉意图序列，用于消融对照。 |
| [outputs/results/cups/F03/audio-only.jsonl](../outputs/results/cups/F03/audio-only.jsonl) | cups场景F03：纯语音时间状态，无物体框。 |
| [outputs/results/cups/F03/showintent.jsonl](../outputs/results/cups/F03/showintent.jsonl) | cups场景F03：视听融合，正式交给C的意图序列。 |
| [outputs/results/cups/F03/visual-only.jsonl](../outputs/results/cups/F03/visual-only.jsonl) | cups场景F03：纯视觉意图序列，用于消融对照。 |
| [outputs/results/cups/F04/audio-only.jsonl](../outputs/results/cups/F04/audio-only.jsonl) | cups场景F04：纯语音时间状态，无物体框。 |
| [outputs/results/cups/F04/showintent.jsonl](../outputs/results/cups/F04/showintent.jsonl) | cups场景F04：视听融合，正式交给C的意图序列。 |
| [outputs/results/cups/F04/visual-only.jsonl](../outputs/results/cups/F04/visual-only.jsonl) | cups场景F04：纯视觉意图序列，用于消融对照。 |
| [outputs/results/drinks/G01/audio-only.jsonl](../outputs/results/drinks/G01/audio-only.jsonl) | drinks场景G01：纯语音时间状态，无物体框。 |
| [outputs/results/drinks/G01/showintent.jsonl](../outputs/results/drinks/G01/showintent.jsonl) | drinks场景G01：视听融合，正式交给C的意图序列。 |
| [outputs/results/drinks/G01/visual-only.jsonl](../outputs/results/drinks/G01/visual-only.jsonl) | drinks场景G01：纯视觉意图序列，用于消融对照。 |
| [outputs/results/drinks/G02/audio-only.jsonl](../outputs/results/drinks/G02/audio-only.jsonl) | drinks场景G02：纯语音时间状态，无物体框。 |
| [outputs/results/drinks/G02/showintent.jsonl](../outputs/results/drinks/G02/showintent.jsonl) | drinks场景G02：视听融合，正式交给C的意图序列。 |
| [outputs/results/drinks/G02/visual-only.jsonl](../outputs/results/drinks/G02/visual-only.jsonl) | drinks场景G02：纯视觉意图序列，用于消融对照。 |
| [outputs/results/drinks/G03/audio-only.jsonl](../outputs/results/drinks/G03/audio-only.jsonl) | drinks场景G03：纯语音时间状态，无物体框。 |
| [outputs/results/drinks/G03/showintent.jsonl](../outputs/results/drinks/G03/showintent.jsonl) | drinks场景G03：视听融合，正式交给C的意图序列。 |
| [outputs/results/drinks/G03/visual-only.jsonl](../outputs/results/drinks/G03/visual-only.jsonl) | drinks场景G03：纯视觉意图序列，用于消融对照。 |
| [outputs/results/drinks/G04/audio-only.jsonl](../outputs/results/drinks/G04/audio-only.jsonl) | drinks场景G04：纯语音时间状态，无物体框。 |
| [outputs/results/drinks/G04/showintent.jsonl](../outputs/results/drinks/G04/showintent.jsonl) | drinks场景G04：视听融合，正式交给C的意图序列。 |
| [outputs/results/drinks/G04/visual-only.jsonl](../outputs/results/drinks/G04/visual-only.jsonl) | drinks场景G04：纯视觉意图序列，用于消融对照。 |
| [outputs/results/evaluation.json](../outputs/results/evaluation.json) | 四场景正式三模式评价、逐视频分数及输入/配置来源记录。 |
| [outputs/results/fruit/A01/audio-only.jsonl](../outputs/results/fruit/A01/audio-only.jsonl) | fruit场景A01：纯语音时间状态，无物体框。 |
| [outputs/results/fruit/A01/showintent.jsonl](../outputs/results/fruit/A01/showintent.jsonl) | fruit场景A01：视听融合，正式交给C的意图序列。 |
| [outputs/results/fruit/A01/visual-only.jsonl](../outputs/results/fruit/A01/visual-only.jsonl) | fruit场景A01：纯视觉意图序列，用于消融对照。 |
| [outputs/results/fruit/A02/audio-only.jsonl](../outputs/results/fruit/A02/audio-only.jsonl) | fruit场景A02：纯语音时间状态，无物体框。 |
| [outputs/results/fruit/A02/showintent.jsonl](../outputs/results/fruit/A02/showintent.jsonl) | fruit场景A02：视听融合，正式交给C的意图序列。 |
| [outputs/results/fruit/A02/visual-only.jsonl](../outputs/results/fruit/A02/visual-only.jsonl) | fruit场景A02：纯视觉意图序列，用于消融对照。 |
| [outputs/results/fruit/A03/audio-only.jsonl](../outputs/results/fruit/A03/audio-only.jsonl) | fruit场景A03：纯语音时间状态，无物体框。 |
| [outputs/results/fruit/A03/showintent.jsonl](../outputs/results/fruit/A03/showintent.jsonl) | fruit场景A03：视听融合，正式交给C的意图序列。 |
| [outputs/results/fruit/A03/visual-only.jsonl](../outputs/results/fruit/A03/visual-only.jsonl) | fruit场景A03：纯视觉意图序列，用于消融对照。 |
| [outputs/results/fruit/B01/audio-only.jsonl](../outputs/results/fruit/B01/audio-only.jsonl) | fruit场景B01：纯语音时间状态，无物体框。 |
| [outputs/results/fruit/B01/showintent.jsonl](../outputs/results/fruit/B01/showintent.jsonl) | fruit场景B01：视听融合，正式交给C的意图序列。 |
| [outputs/results/fruit/B01/visual-only.jsonl](../outputs/results/fruit/B01/visual-only.jsonl) | fruit场景B01：纯视觉意图序列，用于消融对照。 |
| [outputs/results/fruit/B02/audio-only.jsonl](../outputs/results/fruit/B02/audio-only.jsonl) | fruit场景B02：纯语音时间状态，无物体框。 |
| [outputs/results/fruit/B02/showintent.jsonl](../outputs/results/fruit/B02/showintent.jsonl) | fruit场景B02：视听融合，正式交给C的意图序列。 |
| [outputs/results/fruit/B02/visual-only.jsonl](../outputs/results/fruit/B02/visual-only.jsonl) | fruit场景B02：纯视觉意图序列，用于消融对照。 |
| [outputs/results/fruit/B03/audio-only.jsonl](../outputs/results/fruit/B03/audio-only.jsonl) | fruit场景B03：纯语音时间状态，无物体框。 |
| [outputs/results/fruit/B03/showintent.jsonl](../outputs/results/fruit/B03/showintent.jsonl) | fruit场景B03：视听融合，正式交给C的意图序列。 |
| [outputs/results/fruit/B03/visual-only.jsonl](../outputs/results/fruit/B03/visual-only.jsonl) | fruit场景B03：纯视觉意图序列，用于消融对照。 |
| [outputs/results/fruit/C01/audio-only.jsonl](../outputs/results/fruit/C01/audio-only.jsonl) | fruit场景C01：纯语音时间状态，无物体框。 |
| [outputs/results/fruit/C01/showintent.jsonl](../outputs/results/fruit/C01/showintent.jsonl) | fruit场景C01：视听融合，正式交给C的意图序列。 |
| [outputs/results/fruit/C01/visual-only.jsonl](../outputs/results/fruit/C01/visual-only.jsonl) | fruit场景C01：纯视觉意图序列，用于消融对照。 |
| [outputs/results/fruit/C02/audio-only.jsonl](../outputs/results/fruit/C02/audio-only.jsonl) | fruit场景C02：纯语音时间状态，无物体框。 |
| [outputs/results/fruit/C02/showintent.jsonl](../outputs/results/fruit/C02/showintent.jsonl) | fruit场景C02：视听融合，正式交给C的意图序列。 |
| [outputs/results/fruit/C02/visual-only.jsonl](../outputs/results/fruit/C02/visual-only.jsonl) | fruit场景C02：纯视觉意图序列，用于消融对照。 |
| [outputs/results/fruit/C03/audio-only.jsonl](../outputs/results/fruit/C03/audio-only.jsonl) | fruit场景C03：纯语音时间状态，无物体框。 |
| [outputs/results/fruit/C03/showintent.jsonl](../outputs/results/fruit/C03/showintent.jsonl) | fruit场景C03：视听融合，正式交给C的意图序列。 |
| [outputs/results/fruit/C03/visual-only.jsonl](../outputs/results/fruit/C03/visual-only.jsonl) | fruit场景C03：纯视觉意图序列，用于消融对照。 |
| [outputs/results/fruit/C04/audio-only.jsonl](../outputs/results/fruit/C04/audio-only.jsonl) | fruit场景C04：纯语音时间状态，无物体框。 |
| [outputs/results/fruit/C04/showintent.jsonl](../outputs/results/fruit/C04/showintent.jsonl) | fruit场景C04：视听融合，正式交给C的意图序列。 |
| [outputs/results/fruit/C04/visual-only.jsonl](../outputs/results/fruit/C04/visual-only.jsonl) | fruit场景C04：纯视觉意图序列，用于消融对照。 |
| [outputs/results/fruit/D01/audio-only.jsonl](../outputs/results/fruit/D01/audio-only.jsonl) | fruit场景D01：纯语音时间状态，无物体框。 |
| [outputs/results/fruit/D01/showintent.jsonl](../outputs/results/fruit/D01/showintent.jsonl) | fruit场景D01：视听融合，正式交给C的意图序列。 |
| [outputs/results/fruit/D01/visual-only.jsonl](../outputs/results/fruit/D01/visual-only.jsonl) | fruit场景D01：纯视觉意图序列，用于消融对照。 |

### 闭集模型对比

| 文件 | 作用 |
|---|---|
| [outputs/benchmarks/books/rtdetr-l/evaluation.json](../outputs/benchmarks/books/rtdetr-l/evaluation.json) | books场景、rtdetr-l检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/books/yolo11l/evaluation.json](../outputs/benchmarks/books/yolo11l/evaluation.json) | books场景、yolo11l检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/books/yolo11m/evaluation.json](../outputs/benchmarks/books/yolo11m/evaluation.json) | books场景、yolo11m检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/books/yolo11n/evaluation.json](../outputs/benchmarks/books/yolo11n/evaluation.json) | books场景、yolo11n检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/cups/rtdetr-l/evaluation.json](../outputs/benchmarks/cups/rtdetr-l/evaluation.json) | cups场景、rtdetr-l检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/cups/yolo11l/evaluation.json](../outputs/benchmarks/cups/yolo11l/evaluation.json) | cups场景、yolo11l检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/cups/yolo11m/evaluation.json](../outputs/benchmarks/cups/yolo11m/evaluation.json) | cups场景、yolo11m检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/cups/yolo11n/evaluation.json](../outputs/benchmarks/cups/yolo11n/evaluation.json) | cups场景、yolo11n检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/drinks/rtdetr-l/evaluation.json](../outputs/benchmarks/drinks/rtdetr-l/evaluation.json) | drinks场景、rtdetr-l检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/drinks/yolo11l/evaluation.json](../outputs/benchmarks/drinks/yolo11l/evaluation.json) | drinks场景、yolo11l检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/drinks/yolo11m/evaluation.json](../outputs/benchmarks/drinks/yolo11m/evaluation.json) | drinks场景、yolo11m检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/drinks/yolo11n/evaluation.json](../outputs/benchmarks/drinks/yolo11n/evaluation.json) | drinks场景、yolo11n检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/fruit/rtdetr-l/evaluation.json](../outputs/benchmarks/fruit/rtdetr-l/evaluation.json) | fruit场景、rtdetr-l检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/fruit/yolo11l/evaluation.json](../outputs/benchmarks/fruit/yolo11l/evaluation.json) | fruit场景、yolo11l检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/fruit/yolo11m/evaluation.json](../outputs/benchmarks/fruit/yolo11m/evaluation.json) | fruit场景、yolo11m检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |
| [outputs/benchmarks/fruit/yolo11n/evaluation.json](../outputs/benchmarks/fruit/yolo11n/evaluation.json) | fruit场景、yolo11n检测器的逐视频分数、场景汇总和处理统计；B配置以文件内记录及模型对比说明为准。 |

### 产物导航

| 文件 | 作用 |
|---|---|
| [outputs/README.md](../outputs/README.md) | 缓存、正式结果、模型对比三类产物的定位说明。 |

## 修改和扩展时的依赖

- 修改A：涉及检测、外观或追踪时，要重新生成感知；原track_id映射不应自动复用于新运行。
- 修改B：优先复用cache重放，保持标签和评价口径一致；C使用新生成的showintent与其配套A。
- 修改D：区分重新转写和仅重新解释语音；前者需模型，后者可用原始ASR缓存。
- 修改C：保持B意图不变，记录镜头、目标、裁剪和回退原因，独立评价构图。
- 新增数据：在对应场景中维护manifest、连续标签、复核与时间记录，再运行；不要只添加空结果参与评分。
- 修改发布文件后更新release_manifest.json中的实际SHA-256；运行验证临时输出不需要提交。

`src/intent`中的基础层、对照与诊断模块并不都是正式入口。正式方法由run_refined_b调用rest_aware及其共享依赖；基础类和小型样例仍用于自动化测试与模型对比，不能因名字不在README命令中就直接删除。

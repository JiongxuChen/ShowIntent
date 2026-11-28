# 项目结构

沿用原仓库布局，不引入工作目录的试验树。

```text
src/           perception、intent、speech、camera、evaluation、streaming
scripts/       运行、下载、预览入口
configs/       当前运行配置
evaluation/    fruit、books、cups、drinks、metadata
outputs/       cache、results、benchmarks、stream
tests/         自动化测试与小型fixtures
docs/          A、B、C、D、evaluation、model_comparison、project_structure、publishing
```

`cache`只保存默认YOLO11m感知、离线语音和时间轴，供无需视频的B重放；`results`保存默认三模态预测；`benchmarks`只保存五模型完整评分及预测事件区间，不保存其余模型大体积逐帧缓存；`stream`保存实际流式播放状态、语音完成时间和评价。

书籍标注统一保存在`evaluation/books`。流式进程、调度和评价工具在`src/streaming`，避免将内部工具铺在scripts下。原视频和模型不发布。详细命令见根目录README，指标见评价与模型对比文档。

延时分屏入口为scripts/export_delayed_split.py，依赖outputs/stream中的发布记录。outputs/stream_media由程序在本地准备并复用，不提交视频文件。

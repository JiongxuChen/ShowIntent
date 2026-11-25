# D：语音与语言线索

使用faster-whisper从视频转写语音，并输出带时间戳的SpeechEvent。当前离线流程使用large-v3缓存，语言阶段使用轻量规则，提取展示、描述、继续、停止、换物及物体属性线索。B负责把语言绑定到A的真实候选。

安装及完整运行见[README](../README.md)。原始语音事件位于outputs下各场景的speech_raw目录。

## 转写与语言事件

scripts/run_speech.py使用faster-whisper转写音频，离线输入采用large-v3。src/speech/asr.py负责音频分窗和上下文，事件含start_ms、end_ms、available_at_ms、text、is_final等字段。scripts/enrich_speech.py将轻量语言解释附加到事件。

语言层提取展示、介绍、描述、继续、停止和切换动作，处理当前/其他/过去指代以及类别和颜色属性。它不自行生成视觉对象；B通过语句附近的真实轨迹完成绑定。语音错误或指代含糊时，保留视觉确认路径。

## 缓存与三模式评价

outputs/cache/<场景>/speech_raw为原始语音事件。B重放按话语结束时间使用语言证据，这是离线内容时间评价，不代表真实ASR延迟。纯语音基线只输出时间展示状态，默认保持8秒，无定位框，不能直接交给C裁剪。

ASR可能误转物品名、遗漏描述或切分句子；时间覆盖高也可能来自保持过长，应结合误触发和事件召回解释。

## 音频分窗与上下文

完整离线入口以6000ms窗口转写，并携带1500ms过去音频上下文，beam_size=5、中文、温度0、VAD开启。上下文用于连接前后词语；重叠窗口按词的时间中点归属当前窗口，避免上一窗口命令重复生效。音频按统一采样率解码，事件时间加上媒体偏移。

ASR同时记录文本内容时间与计算后的available_at_ms。离线B采用内容时间进行对齐；真实识别耗时仍可从原始语音与窗口记录分析，两者不可混作端到端延迟。

## SpeechEvent字段

| 字段 | 含义 |
|---|---|
| start_ms / end_ms | 语句对应媒体时间 |
| available_at_ms | 该结果可用时间 |
| text / is_final | 文本及是否为最终结果 |
| mentioned_classes / cue_types | 提及类别与线索 |
| semantics | 可选语言动作、指代范围、属性等结构 |

B只消费最终且符合时间条件的证据。词语被识别出来不等于能绑定具体物体，D不负责按最高检测分数替B选目标。

## 语言解释的边界

ContextualPresentationLanguage结合句子及短期语境解释动作，区分当前、先前、其他物品；SpeechEvidenceBridge将可用事件对齐到感知时间；UtteranceGrounder属于B，负责真实实例绑定。三者职责分开，便于定位转写错误、动作解释错误和目标绑定错误。

单独转写可运行`python -m scripts.run_speech --help`查看本地模型、窗口、设备和输出参数。需要完整模型时按README准备；仅重放B则使用缓存，不加载ASR模型。

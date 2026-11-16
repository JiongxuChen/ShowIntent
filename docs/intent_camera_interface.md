# B→C 交接说明

B 决定“是否展示、展示哪个实例、何时结束”；C 决定“如何取景、如何平滑切换、何时因自身异常回到全景”。人工强制全景或暂停自动导播的优先级高于自动判断。

## 输入和输出

C 需要同一来源的两个逐包序列：

- A 的 PerceptionPacket：timestamp_ms、frame_id、objects、hands、relations。
- B 的 IntentPacket：timestamp_ms、state、target_id、score、reason_codes。

B 每接收一个 A 包就输出一个意图包。`IntentPacket` 不带 frame_id，文件模式通过行序及 timestamp_ms 对齐；接入方应检查两者都一致，不能只检查行数。

当前 C 的文件入口只检查行数，时间戳核对仍属于接入验收项；这次整理未修改 C 的实现。

```json
{"timestamp_ms":2600,"state":"SHOWING","target_id":"obj_7","score":0.82,"reason_codes":["HAND_OBJECT_ASSOCIATED","TEMPORAL_HOLD"]}
```

| 字段 | 约定 |
|---|---|
| timestamp_ms | 相对当前片段的媒体毫秒；严格递增，不是机器时钟 |
| state | NONE / CANDIDATE / SHOWING / ENDING |
| target_id | A 的 track_id；NONE 时为 null，其他状态可保留候选或旧目标 |
| score | 启发式证据强度，范围 0–1，不是校准的意图概率 |
| reason_codes | 可扩展的诊断字符串；C 不依赖固定列表决策 |

objects[].bbox_norm 是正立输入画面的 [x1,y1,x2,y2]，归一化到 [0,1]，不是最终输出裁剪框。JSONL 不携带图像。C 渲染原视频时应与 A 使用同一方向且仅应用一次必要旋转。

## C 的处理约定

| B 状态/情况 | C 应做什么 |
|---|---|
| NONE | 平滑恢复或维持全景 |
| CANDIDATE | 不触发新特写；已有退场动画可以继续 |
| SHOWING 且目标可见 | 根据指定实例的框计算和平滑裁剪 |
| SHOWING 且短暂失检 | 有限保持上次取景，不猜另一个物体 |
| 长期找不到目标 | 独立回退全景并记录异常 |
| ENDING | 进入平滑退场，不把旧目标当新展示 |
| TARGET_RELINKED | 当前 ID 已由 B 关联更新；用新 ID 的框平滑续接 |
| 人工覆盖 | 强制全景/暂停优先，不等待 B 同意 |

一次展示保持同一物理实例的假设，但检测 ID 可能因受控重关联更新。C 不应把所有 ID 变化当作新商品，也不能自己按类别或最高分挑目标。B 的短时关联也不是可靠的长期重识别保证。

## 联调命令

先运行 B：

```bash
python -m src.intent.run --perception fixtures/demo_object_show/perception.jsonl --out outputs/intent.jsonl --config configs/presentation_intent.json
```

验证 C 的结构化输出：

```bash
python -m src.camera.run --perception fixtures/demo_object_show/perception.jsonl --intent outputs/intent.jsonl --out outputs/shot.jsonl
```

使用本地真实文件渲染：

```bash
python -m src.camera.run --video data/A01.mp4 --perception data/perception_A01.jsonl --intent outputs/intent_A01.jsonl --out outputs/shot_A01.jsonl --overlay-out outputs/preview_A01.mp4
```

渲染需要安装 A/C 所需依赖；B 单独计算及无视频的 ShotPacket 验证无需这些依赖。输出比例、边界、最大放大、动画速度属于 C 的实现范围，JSONL 联通本身不代表构图或观看效果已验收。

## 交接验收

交给 C 的每段材料应包含 A 的感知文件、B 的意图文件、生成该结果的配置，以及需要渲染时的对应原视频。注明是否完整片段、画面方向和媒体时间起点。不要混用不同感知运行生成的 track_id。

检查静物与搬运保持全景、展示只放大指定实例、跨类别重新确认、短遮挡不乱切、放下退场和人工接管。B 的同类歧义、目标缺失及 C 的越界/性能问题要分别记录。

当前 B 使用媒体时间；C 现有部分平滑/保持参数仍按帧计数。不同采样率联调时需分别检查两者的时间行为。D 后续的语音证据仍先交给 B，C 的包接口不因此改变。

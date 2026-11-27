# A：视觉感知

视觉模块将视频帧转为带媒体时间戳的物体、手部、轨迹及交互关系，为B提供可连续追踪的候选物品。模型并不直接决定是否放大，展示状态由B结合时间证据判断。

## 帧处理流程

1. 解码视频并读取真实媒体PTS，统一旋转方向，保留原画面坐标系。
2. 对全帧运行物体检测；长边设为1280，保持宽高比。
3. 检测手部位置，并围绕手部生成局部区域，以长边960再次检测，补充被手遮挡或全帧下较小的物体。
4. 合并全帧和局部结果，将局部坐标映射回原画面，去除重复与不可靠边界框。
5. 计算物体外观、手物接近关系及运动信息，以外观与几何信息维持轨迹ID。
6. 输出逐帧JSONL；每个包携带timestamp_ms、frame_id、objects、hands、relations。

## 五种检测配置

闭集方法为YOLO11n、YOLO11m、YOLO11l和RT-DETR-L，共享后续手部增强、关系计算和追踪流程。默认采用YOLO11m。检测器输出全部可识别类别，不为书籍等类别单独设置允许列表。

Grounding DINO Tiny采用场景候选词：水果apple/orange、书籍book、水杯cup/bottle、饮料bottle，各场景保留person。同场景正负例使用相同提示，不提供目标位置、展示次数或时间。阈值为0.25，权重及revision见configs/grounding_dino_scenario_prior.json。它获得额外类别信息，因此五种方法是系统配置比较，不是等信息模型排名。

## 与B的接口

物体含track_id、class_name、bbox_norm和score；手部含hand_id及归一化框；relations连接手与物体。track_id只在同一次感知运行内有效，不能跨模型直接比较。person用于场景上下文，不作为B展示目标。所有状态以timestamp_ms对齐，不能用包序号假设固定帧率。

实现入口为src/perception/enhanced.py，开放词汇适配器为src/perception/open_vocabulary.py。模型对比运行及详细指标见[离线评价](offline.md)与根目录README。

## 局限

类别识别正确不代表展示意图正确。遮挡、相似外观、物体放回桌面和重新拿起可能造成轨迹断裂或合并；DINO较高时间覆盖率也不保证事件分段正确。当前评价主要测展示时间，不等价于检测mAP或实例身份准确率。

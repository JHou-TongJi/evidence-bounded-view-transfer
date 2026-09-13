# 第三方组件、数据与许可说明

本目录仅包含本项目自有代码快照和配置，不复制第三方数据、模型权重或外部代码仓库。复现者应从官方渠道分别取得组件，并以对应仓库或模型卡中发布的**当前许可证文本**为准。

| 组件 | 本项目用途 | 官方来源 | 许可与使用边界 |
| --- | --- | --- | --- |
| NVIDIA PhysicalAI-Autonomous-Vehicles-NCore | 源多相机图像、LiDAR、标定、时间戳、车辆位姿和场景标签 | [Hugging Face 数据集页](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore) | 受 NVIDIA 数据访问协议与数据集页面条款约束。数据为受控输入，不随本提交分发；不得擅自公开原始图像、点云、标定、track 或可逆数据副本。 |
| NVIDIA InstantNuRec | 上游静态 Gaussian 场景导出、NCore 读取流程参考 | [官方 GitHub](https://github.com/NVIDIA/instant-nurec) | 代码、权重和相关组件的许可可能不同。应按仓库 `LICENSE`、模型卡及 NVIDIA 条款获取和使用；本提交不携带其副本或权重。 |
| 2D Gaussian Splatting | 表面约束的静态背景训练与目标相机渲染接口 | [官方 GitHub](https://github.com/hbb1/2d-gaussian-splatting) | 官方仓库采用 Gaussian-Splatting License；该许可对研究、评估和商用用途有单独约束。复现前须阅读仓库根目录许可并获得适当授权。 |
| gsplat | PLY Gaussian 的基础 rasterization 与部分渲染工具 | [官方 GitHub](https://github.com/nerfstudio-project/gsplat) | 依照其仓库 LICENSE 与安装包元数据使用。它作为 Python/CUDA 依赖安装，不在本提交中重新分发。 |
| NVIDIA SegFormer B5 ADE20K | 从本地权重推理天空语义，构建 direction-domain temporal sky | [模型页](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) | 权重、配置和 Transformer 代码分别遵守模型卡、NVIDIA 条款及 Hugging Face / Transformers 的适用许可。仅从已获授权的本地副本加载。 |
| Depth Anything V2 | 静态场景道路表面初始化的低权重单目深度先验 | [官方 GitHub](https://github.com/DepthAnything/Depth-Anything-V2) | 代码和各权重的许可以官方仓库和模型发布页面为准。其输出只作辅助约束，不替代 NCore LiDAR。 |
| FFmpeg / OpenH264 | 将逐帧 PNG 编码为 H.264、`yuv420p` MP4 | [FFmpeg](https://ffmpeg.org/)；[OpenH264](https://github.com/cisco/openh264) | 编码器的可用性、编译选项与专利/再分发义务取决于本地 FFmpeg 构建。复现者应使用符合其发布环境要求的编码器和许可证。 |

本项目引用的论文、模型和开源实现用于工程实验与可复现描述，不代表获得其数据、权重、商标或商业部署授权。若最终用途超出赛事评审、研究或内部验证范围，应单独完成数据合规、软件许可、模型许可和知识产权审查。

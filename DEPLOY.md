# 工程代码与复现说明

本目录提供最终 L4 7V 演示的冻结代码、目标相机 rig、可移植的重渲染脚本与环境变量模板。它支持两种复现范围：

1. **最终结果重渲染**：在已具备本场景静态 2DGS checkpoint、temporal sky 和 actor registry 的前提下，重新生成 7×299 张 1920×1080 RGB、七路 MP4 与 7V mosaic；
2. **从授权 NCore 数据重建资产**：从 NCore clip、InstantNuRec 导出、天空构建、2DGS 训练到 actor registry 的完整工程流程。该流程需要授权数据、外部仓库及较长 GPU 训练，不随提交包直接分发。

`configs/7fd4_neolix_x3_size_7v.json` 是目标 L4 7V 工程 rig；其中 `ncore_path` 是占位符，运行脚本会以 `NCORE_SEQUENCE_JSON` 自动写入副本。`` 包含 192 个冻结的代码与配置文件；训练数据、模型权重、2DGS checkpoint、天空资产、actor registry 和完整 PNG 序列均不包含在提交包内。

## 外部项目、数据与许可

| 组件 | 用途 | 获取方式 |
| --- | --- | --- |
| [NVIDIA InstantNuRec](https://github.com/NVIDIA/instant-nurec) | 从 NCore V4 驾驶日志导出初始 Gaussian 场景和上游资产 | 遵循官方安装说明及其代码/模型许可 |
| [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) | 训练与渲染表面约束的静态 2DGS 背景 | `git clone --recursive` 官方仓库；遵循其 Gaussian-Splatting License |
| [NVIDIA PhysicalAI-Autonomous-Vehicles-NCore](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore) | 多相机、LiDAR、标定、时间戳与场景描述输入 | 申请并遵守数据集访问协议；不得随提交重新分发 |
| [NVIDIA SegFormer B5 ADE20K](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) | 构建世界方向 temporal sky 时的天空语义分割 | 本地下载权重，遵守模型卡及上游许可 |

本项目提交的结果和代码不附带上述外部仓库、数据或权重。使用者应在开始前确认数据访问权限、模型许可与赛事规则。

## 参考环境

以下组合是本提交完成 GPU 渲染时使用的参考环境；不同显卡可以采用兼容版本，但应重新运行测试和小规模 smoke。

| 类别 | 已验证版本/要求 |
| --- | --- |
| 操作系统与 Python | Linux；Python 3.11 |
| PyTorch / CUDA | PyTorch 2.7.0 + CUDA 12.8；NVIDIA GPU 计算能力应与 PyTorch CUDA 扩展兼容 |
| 核心渲染库 | `gsplat 1.5.3`、`numpy 1.26.4`、`Pillow 12.2.0`、`plyfile >= 1.0` |
| NCore 读取 | `nvidia-ncore 18.7.0`、`universal-pathlib >= 0.2` |
| 天空/深度可选依赖 | `transformers 4.57.1`、`huggingface-hub 0.36.2`、`safetensors >= 0.4.3`、`scipy >= 1.10` |
| 视频编码 | FFmpeg 8.1.2；H.264 编码器（本提交使用 `libopenh264`）和 `yuv420p` 像素格式 |

建议先按 InstantNuRec 与 2DGS 官方仓库完成 CUDA/PyTorch 环境安装，再安装本提交中的渲染器：

```bash
cd 03_工程代码/source_snapshot
python -m pip install -e '.[ncore,sky,depth]'
```

首次使用 `gsplat`、2DGS 子模块或其他 CUDA 扩展时可能触发 JIT 编译。可通过 `TORCH_CUDA_ARCH_LIST` 指定显卡计算能力；RTX 40 系列的参考值为 `8.9`。

## 资产接口

复制 `env.example` 为任意 shell 配置文件并填写路径。路径由使用者自行决定，脚本不假设服务器目录结构。

| 变量 | 必需内容 | 最低检查 |
| --- | --- | --- |
| `NCORE_SEQUENCE_JSON` | 授权 NCore sequence JSON | 文件存在，且相对 component store 路径可访问 |
| `TWO_DGS_ROOT` | 官方 2DGS checkout 根目录 | 包含 `scene/gaussian_model.py` |
| `STATIC_DATASET` | 本场景 20-tile 训练代理数据集 | 包含 `ncore_2dgs_manifest.json` |
| `STATIC_MODEL` | 训练完成的静态 2DGS 模型目录 | 包含 `point_cloud/iteration_12000/point_cloud.ply` |
| `SKY_ASSET` | temporal sky JSON 索引 | 文件存在且引用的 slot 资产可访问 |
| `ACTOR_REGISTRY` | 通过来源时间留出验收的刚性 actor registry | `chunk0-rigid-incumbent-v1.json` 或等价 schema 文件 |
| `OUTPUT_ROOT` | 新建输出根目录 | 具有写权限；不得覆盖既有正式结果 |

本提交最终资产标识如下：静态 checkpoint 为 `chunk0-clean-static-road-nearfield-fullclip-20tile-12000-v1`，actor registry 为 `chunk0-rigid-incumbent-v1.json`，时序长度为 299 帧，帧率为 30 FPS。

## 最终 1080p 结果重渲染

```bash
cd 03_工程代码
cp env.example env.local
# 编辑 env.local，填写上表中的变量。
source env.local
bash run_render_1080p_7v.sh
```

脚本执行顺序为：

1. 将目标 rig 中的 NCore 输入占位符替换为 `NCORE_SEQUENCE_JSON`，生成本次运行的 resolved rig；
2. 调用 `two_dgs_target_render_cli`，以静态 2DGS、目标 rig 和 temporal sky 直接光栅化 7×299 张 1920×1080 静态 RGB；
3. 调用 `two_dgs_actor_target_render_cli`，读取同一静态 RGB 和冻结 registry，以最大观察角 25°、距离比 0.60–1.60、100 ms fade 选择经过来源留出验收的 actor expert；无可靠证据的帧保持静态背景；
4. 使用 FFmpeg 将七路 RGB 编码为 30 FPS、H.264、`yuv420p` MP4，并生成 3×3 布局的 7V mosaic。

该流程的输出目录会包含 `static/`、`actors/`、`media/` 和本次 resolved rig。脚本拒绝覆盖非空 `static/` 或 `actors/` 目录，以避免误覆盖已有实验结果。

最终 1080p 图像由目标 pinhole 相机参数直接光栅化，并非简单放大旧版 480×270 PNG。静态 2DGS 的训练代理分辨率仍为 480×270，因此输出分辨率提高主要改善采样密度与边缘呈现；动态对象完整性和可恢复几何范围仍受原始观测与训练资产限制。

## 从 NCore 完整重建资产

完整重建应按以下阶段执行，并在每个阶段保存 manifest、命令、软件版本和输入哈希：

1. **上游静态场景初始化**：使用 InstantNuRec 对授权 NCore clip 执行导出，保留 PLY、输入 clip 标识和模型版本；
2. **动态实例与天空资产**：以 NCore 标定、FTheta/rolling-shutter 位姿、动态实例 mask 与 SegFormer 构建 temporal sky；
3. **静态 2DGS 训练代理**：运行 `two_dgs_dataset_cli` 生成 20-tile、480×270 的 pinhole 代理、动态排除 mask 和多帧 LiDAR 道路监督；
4. **静态背景训练与选择**：使用官方 2DGS 训练 12,000 iteration，前 3,000 iteration 后冻结 densification；在独立道路 LiDAR holdout 上选择 checkpoint；
5. **actor expert 训练与筛选**：仅将来源时间留出满足 RGB、alpha IoU 和面积门槛的刚性 actor 写入 registry；不满足门槛的动态目标在目标视角回退为静态背景；
6. **目标 L4 7V 重渲染与验收**：运行本目录脚本，检查每路 299 帧完整性、视频元数据、actor 外 RGB 保留性及 `checksums.sha256`。

`examples/` 保留了部分历史实验脚本，适合了解数据集构建和训练超参数；最终交付以本 README、`env.example` 与 `run_render_1080p_7v.sh` 为准。由于 checkpoint、sky、actor registry 和训练数据均不随提交分发，完整重建需要重新获得授权数据并执行上述训练阶段，不能由提交包单独完成。

## 验证

渲染完成后，在 `03_工程代码` 目录执行：

```bash
sha256sum -c ../05_评价结果/checksums.sha256
```

该命令验证**已包含在提交包中的文件**。重新渲染出的新目录应独立进行文件数、视频帧数、分辨率、帧率和 manifest 检查，不应覆盖提交包内的基准媒体。

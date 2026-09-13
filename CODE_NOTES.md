# 冻结工程代码快照

本目录是最终提交使用的渲染、相机读取、天空合成、2DGS 数据接口和动态 actor 筛选代码快照。它配合本仓库根目录的 [`DEPLOY.md`](DEPLOY.md)、`env.example` 与 `run_render_1080p_7v.sh` 使用。DEPLOY.md 是最终 1080p L4 7V 演示的**权威复现入口**；本文件说明代码边界、安装方式和接口语义。

本快照不含 NCore 数据、模型权重、InstantNuRec/2DGS 外部代码、训练 proxy、2DGS checkpoint、天空资产、actor registry 或完整 PNG 结果。因此它可复现工程和重渲染过程，但完整训练复现仍需要取得数据与模型授权，并重新执行相应的 GPU 训练。

## 代码范围

### 生产使用的静态背景与目标视角链路

最终演示使用以下链路：

```text
NCore 标定、时间戳和车辆位姿
      + 静态 2DGS checkpoint + temporal sky
      + 目标 L4 7V rig
      -> 目标 pinhole 相机直接光栅化 1920×1080 静态 RGB
      + 来源时间留出验收通过的 actor registry
      -> 保守的动态 actor 合成 RGB
      -> 单路 MP4 与 7V mosaic
```

`nurec_gs_renderer.two_dgs_target_render_cli` 负责静态 2DGS 的目标相机渲染，支持 temporal sky。`nurec_gs_renderer.two_dgs_actor_target_render_cli` 只从已经验收的 registry 选取动态 actor；不满足视角、距离或可见性条件时，该区域保持静态背景。最终脚本采用最大观察角 25°、目标/来源距离比 0.60–1.60 和 100 ms 淡入淡出，避免把缺少可靠来源证据的 actor 强行投影到新视角。

### 通用 Gaussian 渲染链路

`nurec_gs_renderer.cli` 可渲染 InstantNuRec 或 Graphdeco 格式的 3D Gaussian PLY，并输出 RGB、几何 alpha 和 expected-depth。它支持：

- 无畸变 pinhole 目标相机；
- 直接读取 NCore `FThetaCameraModelParameters` 的源相机；
- NCore rolling-shutter 首/尾曝光位姿和扫描方向；
- 多相机 rig 与 NCore 车辆轨迹；
- 可选世界方向 cubemap / temporal sky；
- 可选、可逆的 Gaussian DC 色彩 sidecar。

变换名称遵循 NCore 的 `T_source_target` 语义。对目标相机安装外参，代码计算：

```text
T_camera_world(t) = T_rig_world(t) @ T_camera_rig
```

其中 `T_camera_rig` 把 camera 坐标中的点变换到 rig 坐标；随后转换为 `gsplat` 所需的 OpenCV world-to-camera 表达。相机 OpenCV 轴为 `+x` 向右、`+y` 向下、`+z` 向前。目标 L4 rig 使用 global-shutter 中点位姿，源 NCore FTheta 相机则保留逐像素 rolling-shutter 射线。

### 天空与动态实验代码

`sky_cli` 从 7 路 NCore 观测构建方向域 cubemap 或 temporal sky 索引。天空仅填充 `1 - geometry_alpha`，输出的 alpha 与 expected-depth 始终保持几何语义。`dynamic_*`、`raw_dynamic_*`、`two_dgs_actor_*`、`emernerf_*` 等模块保留用于动态层归因、训练数据生成和消融实验；它们不是最终静态 2DGS 演示的前置运行步骤。

## 目录结构

| 位置 | 内容 |
| --- | --- |
| `src/nurec_gs_renderer/` | Python 包及各 CLI 实现 |
| `configs/` | 相机、评测和历史实验配置；最终 target rig 位于上一级 `configs/` |
| `examples/` | 已冻结的历史 2DGS 训练/诊断脚本，含特定实验超参数 |
| `pyproject.toml` | 包依赖、可选依赖组与 console scripts |
| `third_party.md` | 第三方数据、代码、模型与许可边界 |

历史配置和 `examples/` 用于追溯实验，而不是可直接复制的最终命令；其中的资产标识必须替换为本地持有的等价资产。最终重渲染请使用上一级的 `env.example` 和 `run_render_1080p_7v.sh`，它们只依赖环境变量，不绑定特定机器目录。

## 环境安装

建议先安装与本地 GPU/CUDA 兼容的 PyTorch，再按 [NVIDIA InstantNuRec](https://github.com/NVIDIA/instant-nurec) 和 [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) 的官方说明准备外部项目。之后，在本目录中安装本包：

```bash
python -m pip install -e '.[ncore,sky,depth]'
```

依赖组含义如下：

| 组 | 用途 |
| --- | --- |
| 无可选组 | PLY、`torch`、`gsplat`、pinhole 渲染 |
| `ncore` | 读取 NCore V4 sequence、FTheta 标定与位姿图 |
| `sky` | 本地 SegFormer 权重推理和 cubemap / temporal sky 构建 |
| `depth` | Depth Anything 等深度先验的可选数据构建功能 |
| `dev` | pytest 测试工具 |

参考组合为 Linux、Python 3.11、PyTorch 2.7.0 + CUDA 12.8、`gsplat 1.5.3`、`nvidia-ncore 18.7.0`、`transformers 4.57.1` 与 `huggingface-hub 0.36.2`。首次调用 `gsplat` 或 2DGS CUDA 扩展可能触发 JIT 编译；应设置与本机 GPU 匹配的 `TORCH_CUDA_ARCH_LIST`。例如 Ada 架构 GPU 的值为 `8.9`。

## 最小健康检查

完成安装后可先检查 CLI 是否可导入：

```bash
PYTHONPATH=src python -m nurec_gs_renderer.cli --help
PYTHONPATH=src python -m nurec_gs_renderer.two_dgs_target_render_cli --help
PYTHONPATH=src python -m nurec_gs_renderer.two_dgs_actor_target_render_cli --help
```

若快照同时包含测试目录，可运行：

```bash
PYTHONPATH=src pytest -q
```

真实 GPU smoke 至少应验证一张目标相机 RGB、alpha、expected-depth 的尺寸与无 NaN/异常全黑输出；完整 7V 重渲染的帧数应为 7×299，单路视频应为 1920×1080、30 FPS、299 帧。

## 输入与输出语义

静态 2DGS 训练代理采用从源 NCore FTheta/rolling-shutter 图像精确反投影得到的 pinhole tile。最终 L4 输出并非把 480×270 训练 PNG 放大：`two_dgs_target_render_cli` 以目标 1920×1080 pinhole 内外参直接光栅化 checkpoint。训练代理的观测密度仍限制可恢复的高频几何与动态目标细节。

动态 actor 的来源 mask、实例 track、LiDAR 和跨视角观测只用作保守筛选与验收。原始动态对象若没有可验证的表面、可见性或来源时间留出支持，不会被强制合成；这能减少新视角下的错误重影，但也意味着部分车辆和行人以静态背景或弱动态形式呈现。

## 许可与数据边界

使用本快照前，请阅读 [`third_party.md`](third_party.md) 并确认 NVIDIA NCore 数据访问协议、InstantNuRec、2DGS 代码及各本地模型权重的许可均允许预期用途。本提交不得附带或公开分发原始图像、LiDAR、标定、track、预训练权重或外部项目副本。

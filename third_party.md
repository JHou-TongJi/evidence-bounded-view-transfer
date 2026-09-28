# Third-Party Components, Data and Licences

This repository contains only the project's own code snapshot and configuration.
It copies no third-party data, model weights or external code repositories.
Anyone reproducing the work should obtain each component from its official
channel, and treat the **current licence text** published in the corresponding
repository or model card as authoritative.

| Component | Use in this project | Official source | Licence and boundary of use |
| --- | --- | --- | --- |
| NVIDIA PhysicalAI-Autonomous-Vehicles-NCore | Source multi-camera images, LiDAR, calibration, timestamps, vehicle poses and scene labels | [Hugging Face dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore) | Governed by the NVIDIA data access agreement and the dataset page terms. The data is a controlled input and is not distributed with this submission; source images, point clouds, calibration, tracks or reversible copies must not be published without authorisation. |
| NVIDIA InstantNuRec | Upstream static Gaussian scene export; reference for the NCore reading pipeline | [Official GitHub](https://github.com/NVIDIA/instant-nurec) | The licences for code, weights and related components may differ. Obtain and use them according to the repository `LICENSE`, the model card and NVIDIA's terms; this submission carries neither a copy nor any weights. |
| 2D Gaussian Splatting | Surface-constrained static background training and the target camera rendering interface | [Official GitHub](https://github.com/hbb1/2d-gaussian-splatting) | The official repository uses the Gaussian-Splatting License, which places separate constraints on research, evaluation and commercial use. Read the licence at the repository root and obtain appropriate authorisation before reproducing. |
| gsplat | Base rasterisation for PLY Gaussians and some rendering utilities | [Official GitHub](https://github.com/nerfstudio-project/gsplat) | Used according to its repository LICENSE and package metadata. It is installed as a Python/CUDA dependency and is not redistributed here. |
| NVIDIA SegFormer B5 ADE20K | Sky semantic inference from local weights, for building the direction-domain temporal sky | [Model page](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) | Weights, configuration and the Transformers code are each subject to the model card, NVIDIA's terms, and the applicable Hugging Face / Transformers licences. Loaded only from an authorised local copy. |
| Depth Anything V2 | Low-weight monocular depth prior for static road-surface initialisation | [Official GitHub](https://github.com/DepthAnything/Depth-Anything-V2) | The licences for the code and for each set of weights are governed by the official repository and model release pages. Its output serves only as an auxiliary constraint and does not replace NCore LiDAR. |
| FFmpeg / OpenH264 | Encoding per-frame PNGs into H.264 `yuv420p` MP4 | [FFmpeg](https://ffmpeg.org/); [OpenH264](https://github.com/cisco/openh264) | Encoder availability, build options and patent/redistribution obligations depend on the local FFmpeg build. Use an encoder and licence appropriate to your release environment. |

The papers, models and open-source implementations cited by this project are used
for engineering experiments and reproducible description. Citing them does not
constitute authorisation for their data, weights, trademarks or commercial
deployment. If the end use goes beyond competition review, research or internal
validation, a separate data-compliance, software-licence, model-licence and
intellectual-property review should be completed.

# Data Compliance Statement

## 1. Data licence and preconditions for use

The source road data for this project comes from NVIDIA's
[PhysicalAI-Autonomous-Vehicles-NCore](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore)
dataset. That dataset is not unconditionally downloadable: before accessing any
file, a user must sign in to Hugging Face and accept the **NVIDIA Autonomous
Vehicle Dataset License Agreement**.

The dataset card marks the licence as:

```text
nvidia-av-dataset
```

Before using the data, an appropriately authorised individual or institution must
confirm that the agreement has been accepted and is being observed. This document
does not substitute for the original licence text; in case of conflict, the
current version of the NVIDIA dataset page and licence agreement governs.

## 2. Key licence constraints

Under the NVIDIA Autonomous Vehicle Dataset License Agreement as published on the
dataset card, this project observes the following in particular:

1. Downloading, use, modification and reproduction of the dataset are limited to
   the autonomous-driving research and development purposes the agreement
   explicitly permits.
2. The dataset page is publicly visible, but the data files are governed by an
   access agreement; they must not be treated as freely redistributable open data.
3. The dataset, in whole or in part, must not be sold, rented, sub-licensed,
   transferred, hosted, or supplied to third parties.
4. No attempt may be made to identify, de-anonymise or profile individuals,
   licence plates or other identifiable subjects in the data.
5. No face recognition, biometric processing, identity recognition, emotion
   recognition, social scoring, or individual tracking based on pedestrian
   behaviour may be performed.
6. The data must not be used for unlawful surveillance, law-enforcement purposes,
   or any other use contrary to applicable law.
7. The storage locations of the source data and its copies must be recorded, and
   licence termination, data-update requirements and deletion notices must be
   acted on as the agreement requires.
8. Whether source data, partial derivatives, performance benchmarks or
   competitive analyses may be disclosed externally is governed by the
   confidentiality and distribution clauses of the licence together with the
   scope of authorisation granted by the competition organiser.

## 3. Data isolation and minimal-submission measures

To reduce the risk of leaking source road-collection data, annotations and model
assets, this project follows the principle of *process inside the authorised
environment, export the minimum necessary, ship a package from which the source
data cannot be reconstructed*.

| Data category | In the final package | Isolation and handling |
|---|---|---|
| Source multi-camera images, LiDAR, calibration, trajectories and scene labels | No | Read and processed only inside a local compute environment holding dataset authorisation; never copied into submission material and never publicly distributed. |
| NCore data indices, sequence descriptions and intermediate data formats | No | Used only for local loading, time synchronisation, camera-pose resolution and the training/rendering pipeline; not provided as submission content. |
| Pre-trained model weights | No | Used only for local inference or training; the final submission carries no model parameters, and each model's own licence terms are observed separately. |
| Reconstruction models, intermediate checkpoints and Gaussian scene assets | No | Kept as local experimental assets for reproducing the experimental pipeline; the submission retains only the necessary configuration notes, version information and result metrics. |
| Full per-frame RGB image sequences | No | The complete PNG frame set is not packaged, so no large-scale derived image corpus can be bulk-extracted and redistributed. |
| Final demonstration video and representative frames | Yes | Only a compressed demonstration video and a small number of representative frames, retained to illustrate the seven-view reconstruction for the target L4 vehicle. Not released as general training data or as a public dataset. |
| Configuration files, code snapshot, evaluation metrics and experiment records | Yes | Non-sensitive engineering information required for reproduction. Contains no source images, point clouds, annotations, model weights, or anything from which the source data could be directly reconstructed. |

The final package is addressed to competition review and technical presentation.
It contains only the necessary engineering documentation, reproducible experiment
configuration, evaluation results, representative images and demonstration video.
Source collection data, sensor calibration and labels, full frame sequences, model
weights and external project code all remain managed inside the authorised
environment.

## 4. Disclosure boundary for derived video and representative frames

The seven-view MP4, the synchronised mosaic and the representative frames in the
final submission are derived visual results generated from NCore observations.
Their disclosure is limited to competition review, project acceptance, or an
internal R&D environment covered by the licence.

Without the NVIDIA data licence and the explicit permission of the competition
organiser, the following must not be done:

- publicly uploading the final MP4, representative frames, full rendered PNGs, or
  any reversible high-quality version of them to an open website;
- republishing the derived imagery as a standalone dataset;
- circulating it as source data or as real collected data from the target vehicle;
- extracting, circulating or analysing licence plates, faces, pedestrian identity
  or other identifiable information from the imagery;
- using the video for personnel tracking, behavioural profiling, law enforcement
  or surveillance.

Where competition rules require public presentation, the presentation channel,
resolution and access permissions approved by the organiser take precedence, and
the presentation material must retain the data-source and use-boundary notes.

> **Note on this repository.** The published repository carries only 640×360
> demonstration clips and a single still whose per-tile resolution is lower still.
> Full-resolution output and representative frames are withheld under §4 above.

## 5. Licences of third-party code, models and public material

Besides NCore, this project uses third-party open-source code, pre-trained models
and public product material. These do not constitute a new road-collection
dataset, but each carries its own licence and conditions of use. Models are loaded
and code is run only inside a locally controlled environment; model weights,
external repositories and source data are never copied into the final package.

| Category | Name and version / source | Use in this project | Licence or terms | Key restrictions and submission handling |
|---|---|---|---|---|
| Scene-representation code | [2D Gaussian Splatting](https://github.com/hbb1/2d-gaussian-splatting) | Training the static surface-aligned Gaussian scene and rasterising it directly into the target seven-view L4 rig | The repository's `LICENSE.md` is the **Gaussian-Splatting License**, granted by Inria and MPII | The licence covers research and evaluation use by academic and industrial research users and explicitly restricts commercial use. Distributing its code or derived code requires shipping the full licence and retaining copyright, patent, trademark and attribution notices. The code snapshot in the final package retains the upstream licence and third-party notices; this project distributes none of its pre-trained assets. |
| 2DGS dependencies | `diff-surfel-rasterization`, `simple-knn` and other Git submodules | CUDA rasterisation, neighbourhood queries and training acceleration | Each submodule carries its own licence | The main 2DGS repository's licence must not be assumed to cover every submodule. If submodule source is distributed, each repository's LICENSE/NOTICE must be retained individually, and the commit and licence listed in the third-party statement. |
| Upstream reconstruction code | [NVIDIA InstantNuRec](https://github.com/NVIDIA/instant-nurec) | Early static/dynamic Gaussian asset generation, 3DGS control experiments and upstream compatibility checks | **Apache License 2.0** | May be used, modified and redistributed; distributing modified source requires retaining copyright, licence and NOTICE, and clearly marking modified files. Apache-2.0 grants no NVIDIA trademark rights. The final static background is produced by 2DGS; InstantNuRec serves only as an upstream/control component. |
| Upstream reconstruction weights | [NVIDIA InstantNuRec model card](https://huggingface.co/nvidia/instant-nurec) | Local inference for early Gaussian assets and control experiments | **NVIDIA Open Model License Agreement** | The code licence and the weight licence differ. The model card is marked `nvidia-open-model-license` and states that the model system is governed by the NVIDIA Open Model License Agreement. Weights do not enter the submission package; the model card and licence text must be re-checked before any re-download, deployment, redistribution or commercial use. |
| Sky / semantic segmentation model | [NVIDIA SegFormer B5 ADE20K 640×640](https://huggingface.co/nvidia/segformer-b5-finetuned-ade-640-640) | Identifying sky regions and supporting construction of the world-direction temporal sky cubemap; in some experiments, filtering vehicle semantic regions | The Hugging Face model card is marked `license: other` and points to the [NVIDIA SegFormer License](https://github.com/NVlabs/SegFormer/blob/master/LICENSE) | The upstream licence is the **NVIDIA Source Code License for SegFormer**. It permits copying, modification and distribution but restricts these to non-commercial research or evaluation; redistribution must retain the complete licence, copyright and attribution notices, and derivative works must carry the same non-commercial restriction. Weights are loaded locally only and are not distributed with the final package. |
| Road depth-prior model | [Depth Anything V2 repository](https://github.com/DepthAnything/Depth-Anything-V2) | Supplying a low-weight depth prior for static road regions after LiDAR scale calibration | The code repository is **Apache License 2.0** | Apache-2.0 permits use, modification and redistribution, but requires retaining the licence, copyright and NOTICE and marking modifications; it grants no trademark rights. The project uses the depth output only as an auxiliary constraint and never treats it as raw LiDAR or as depth ground truth. |
| Monocular depth weights | [Depth Anything V2 Metric Outdoor Small (Transformers)](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Outdoor-Small-hf) | The outdoor metric-depth small model used in the current experiments | The upstream repository states that the **Depth-Anything-V2-Small model is Apache-2.0**; the Base/Large/Giant models are **CC-BY-NC-4.0** | The Small variant is the one in use. This specific Hugging Face model card does not carry its own `license` field, so the project relies on the official repository's Apache-2.0 statement for the Small weights, and recommends re-checking that checkpoint's model card before any formal release. The Small licence must not be conflated with those of Base/Large/Giant. Weights do not enter the submission package. |
| Target-vehicle public material | [Neolix product technology page](https://www.neolix.cn/productTechnology) | Reference for the size class, payload capacity, operating scenarios and multi-sensor topology of low-speed autonomous delivery vehicles | Public web material; not an open-source-software or open-data licence | Only public product-level facts are cited, to form an engineering assumption about the target vehicle. The page grants no right to redistribute software, imagery, trademarks, product designs or OEM calibration data. Submission material must not use Neolix trademarks, logos or product images, or imply official cooperation, certification or OEM calibration authorisation. |
| Camera specification material | [Leopard Imaging LI-IMX390-GMSL2 datasheet](https://leopardimaging.com/wp-content/uploads/2023/12/LI-IMX390-GMSL2-xxxH_Datasheet_V1.4.pdf) | Reference for the resolution and field-of-view capability range of automotive wide-angle cameras | Public vendor datasheet | Used only as a capability-range reference. Drawings, trademarks, images or the full contents of the datasheet must not be redistributed as project assets. The target camera intrinsics and extrinsics remain an engineering-defined virtual calibration, not this camera's actual mounted calibration. |

## 6. Privacy and de-identification requirements

This project does not perform:

- face recognition, person identification or biometric extraction;
- licence-plate recognition or unique-vehicle-identifier tracking;
- personal profiling from pedestrian pose, trajectory or behaviour;
- multi-frame association to infer the identity of a specific person or vehicle;
- inference, classification or annotation of sensitive attributes.

When inspecting video by hand, extracting representative frames, writing up
failure cases and preparing public material, information that might identify a
person or a vehicle must not be highlighted, enlarged or re-annotated. If source
data or derived results are found to carry a risk of insufficient anonymisation,
circulation must stop, the data provider must be notified, and the matter handled
according to the applicable licence and institutional requirements.

## 7. Veracity and safety boundary of the final result

This project outputs a virtual view sequence generated from the source vehicle's
multi-modal observations and a target L4 rig definition. It is suitable for
cross-vehicle view reconstruction, scene visualisation and method validation. It
must not be described as:

- real collected sensor data from the target L4 vehicle;
- a basis for autonomous-driving safety certification, road compliance, or
  collision-liability determination;
- ground truth for dynamic object trajectories, pedestrian behaviour or vehicle
  state;
- a data source usable directly for identity, behaviour or sensitive-attribute
  analysis.

Any subsequent use must remain within the source data licence, the licences of
third-party components, the competition rules, applicable data-protection law,
and internal institutional approval.

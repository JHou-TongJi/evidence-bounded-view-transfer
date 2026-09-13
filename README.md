# Evidence-Bounded Cross-Vehicle View Transfer with Fail-Closed Abstention

Re-rendering a calibrated passenger-car recording into the geometry of a
seven-camera L4 delivery-robot rig, under the constraint that **no image of the
target rig exists** against which the result could be scored.

This repository holds the renderer, the rig configuration, the evaluation
evidence, and the build system for the manuscript. It deliberately does **not**
hold the source dataset, the trained models, or the full rendered output — see
[What is not here](#what-is-not-here-and-why).

---

## What the method does

A passenger car and a delivery robot differ in camera centres, mounting height,
field of view, lens model and shutter model — all at once. That makes this
*rig substitution*, not novel-view synthesis, and it removes the held-out
reference image that view synthesis is normally scored against.

The approach is to render only what the source recording supports, and to
refuse rather than extrapolate:

- **Static geometry** — surface-aligned 2D Gaussian splatting, supervised by
  road LiDAR, rendered directly into the target camera.
- **Sky** — a world-direction temporal cubemap, queried only where static alpha
  is incomplete.
- **Dynamic actors** — a frozen registry of rigid actor experts. An expert is
  admitted only if it reproduces its own source camera on a withheld holdout,
  *and* the target-side view-angle, distance-ratio and temporal gates all pass.
  Otherwise the renderer emits nothing and records which gate refused it.

**The fail-closed guarantee covers the actor layer only.** The static and sky
layers are unconditional reconstructions that may extrapolate, and they carry
97.3% of delivered pixels. The manuscript states this scope explicitly; so does
this README, because it is the single easiest thing to overread.

---

## Repository layout

| Path | Contents |
|---|---|
| `src/nurec_gs_renderer/` | Renderer, camera models, sky compositing, 2DGS data interfaces, actor registry and gates (146 modules) |
| `configs/` | Rig configuration and per-experiment configs. `7fd4_neolix_x3_size_7v.json` is the target rig |
| `evidence/` | Every measured quantity the manuscript reports, as JSON + CSV, with SHA-256 checksums |
| `paper_build/` | Deterministic manuscript build: content modules, figure generators, and the audit that checks the text against `evidence/` |
| `docs/` | Technical report, data-source and sensor notes, data-compliance statement, measured-results report, failure log |
| `demo/` | Low-resolution demonstration videos (see licence note below) |
| `DEPLOY.md` | Reproduction entry point — environment, asset interfaces, re-render and verification |

---

## Install

Python 3.10+ with CUDA for the rendering path.

```bash
pip install -r requirements.txt
pip install -e .
```

The 2D Gaussian Splatting backend and its CUDA rasteriser are **not vendored
here**. Obtain them from their own repositories under their own licences; see
[`third_party.md`](third_party.md), which lists every external component, its
source, and its licence boundary.

## Deploy and reproduce

[`DEPLOY.md`](DEPLOY.md) is the authoritative entry point. In outline:

```bash
cp env.example env.local        # fill in dataset and output roots
./run_render_1080p_7v.sh        # re-render the 1080p seven-view delivery
```

Rendering requires NCore access (below) and a GPU. Everything downstream of the
render — the evidence JSONs, every number in the manuscript, and every figure —
rebuilds from this repository alone:

```bash
cd paper_build
python build_access_paper.py    # build the manuscript
python verify_claims.py         # 106 deterministic checks against evidence/
```

`verify_claims.py` re-extracts the built text on every run and fails on any
mismatch between what the manuscript says and what `evidence/` contains. It is
the check to run after any edit.

---

## What is not here, and why

The source recording is the **NVIDIA PhysicalAI Autonomous Vehicles NCore**
dataset, used under the NVIDIA Autonomous Vehicle Dataset License Agreement.
Those terms govern the rendered output as much as the input: a redistribution
of the target-rig stream is a redistribution of a derivative work, not the
creation of an unencumbered one.

Accordingly this repository does **not** contain:

- source images, LiDAR, calibration, trajectories or scene labels;
- trained model weights or Gaussian scene assets;
- the full per-frame rendered PNG sequences;
- the full-resolution delivered videos.

To reproduce the render you must obtain NCore access yourself from the
[dataset page](https://huggingface.co/datasets/nvidia/PhysicalAI-Autonomous-Vehicles-NCore)
and accept its licence.

### The demonstration videos

`demo/` holds two 640×360 clips, downscaled from the delivered 1920×1080
output. They are included to show coverage and cross-camera synchronisation.
They are **not** a dataset release: do not redistribute them as one, do not
mine them for licence plates, faces or pedestrian identity, and do not present
them as a recording from any commercial vehicle. The full-resolution output and
the representative frames are withheld under the terms above.

The source clip is a public-road recording and contains identifiable third
parties. The renderer does not remove them — it withholds only what it cannot
support geometrically, which is a different criterion from privacy. Every frame
carries a provenance record naming its source timestamp, so an erasure request
against a source interval resolves to an exact set of derived frames.

---

## Licensing

The code and configuration in this repository are licensed under
[Apache-2.0](LICENSE).

That licence covers **this repository's own contents only**. It does not and
cannot relicense:

- the NCore dataset or anything derived from it, including `demo/`;
- the 2D Gaussian Splatting code and its submodules, which carry the
  Gaussian-Splatting License (Inria / MPII) restricting use to research and
  evaluation;
- any other third-party component in [`third_party.md`](third_party.md).

---

## Status and honest limits

Results come from **one 9.97 s clip** re-rendered into **one engineering proxy
rig**, with a registry of rigid actors and no deformable objects. The rendered
rig is a 2.69 × 1.08 × 1.82 m proxy, not the larger deployment platform the
work is aimed at; the coverage figures characterise the proxy. No head-to-head
superiority claim is made, because no shared target-view benchmark exists.

The decisive missing experiment is stated in the manuscript: hold out one
physical source camera entirely, render its real poses, and score against its
real images. That would convert the current coverage numbers into a genuine
risk–coverage trade-off. It has not been run.

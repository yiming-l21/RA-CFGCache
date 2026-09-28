<h1 align="center">RA-CFGCache</h1>

<p align="center">
  <strong>Risk-Aligned Caching for Classifier-Free Guidance</strong><br>
  Training-free diffusion acceleration under CFG via guided-risk composition and propagation-aware rescaling.
</p>

<p align="center">
  <a href="#citation">
    <img src="https://img.shields.io/badge/Paper-arXiv-b31b1b.svg" alt="Paper">
  </a>
  <a href="#license">
    <img src="https://img.shields.io/badge/License-GPL--3.0-blue.svg" alt="License">
  </a>
  <a href="#quick-start">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  </a>
  <a href="#quick-start">
    <img src="https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg" alt="PyTorch">
  </a>
</p>

---

## Overview

**RA-CFGCache** is a training-free caching framework for diffusion models under **classifier-free guidance (CFG)**.

Existing caching methods are mostly designed for standard sampling, where reuse is controlled by single-prediction or branch-wise criteria. Under CFG, however, the quantity that actually drives denoising is the **guided prediction**, not either branch alone. This creates two mismatches:

- **Branch–guided mismatch**: branch-wise reuse criteria may not reflect the true error of the guided prediction.
- **Local–final mismatch**: local guided error at one timestep does not directly determine the final output deviation, because errors propagate differently across timesteps.

RA-CFGCache addresses these two issues with:

- **CFG-aware Guided-Risk Composition**: composes branch-wise proxy risks into a guided risk aligned with the guided prediction.
- **Propagation-Aware Rescaling**: rescales the local guided risk using timestep-dependent propagation gain to better reflect final deviation.
- **Threshold-based Scheduling**: triggers refresh when the accumulated risk exceeds a threshold.

In our paper, RA-CFGCache is evaluated on **FLUX.1-dev**, **Wan2.1-T2V-1.3B**, and **CogVideoX-2B**, and achieves a stronger efficiency–fidelity trade-off than existing training-free caching baselines.

---
## Motivation
<p align="center">
  <img src="docs/figures/mismatches.png" alt="Two Mismatches" width="98%">
</p>
Classifier-free guidance (CFG) improves conditional generation quality, but it also changes what a caching rule should care about. Existing training-free caching methods are often defined on single-prediction or branch-wise changes. Under CFG, however, the sampler is driven by the guided prediction rather than either branch alone, so small branch-wise errors do not necessarily imply a small perturbation on the actual guided update. In addition, diffusion is iterative: even when the local guided error is similar, its effect on the final sample can vary significantly depending on when the perturbation is introduced.

These two observations motivate RA-CFGCache:

- Branch–guided mismatch: branch-local reuse criteria are not aligned with the quantity that actually enters the sampler.

- Local–final mismatch: local guided error alone does not determine the final output deviation because downstream propagation is timestep-dependent.

These mismatches suggest that effective cache control under CFG should be aligned with both the guided prediction and its propagated final impact. RA-CFGCache addresses this by composing branch-wise proxy risks into a guided risk, then rescaling that risk with a timestep-dependent propagation gain before making refresh decisions.

---


## Method at a Glance

<p align="center">
  <img src="docs/figures/overview.png" alt="RA-CFGCache overview" width="85%">
</p>

RA-CFGCache reformulates cache control under CFG as **guided-risk control**:

1. **Compose** branch-wise proxy risks into a guided risk that matches the CFG error geometry.
2. **Rescale** the guided risk with a timestep-dependent propagation gain.
3. **Schedule** refresh/reuse decisions using a threshold on the accumulated risk.

---

## Calibration Branch

For standard inference and reproduction of the main results, use the [`main`](https://github.com/yiming-l21/RA-CFGCache/tree/main) branch. This branch contains the offline calibration pipeline used to estimate the branch-alignment statistics and propagation-aware rescaling curves required by RA-CFGCache.

## Quick Start

### 1. Create environment

```bash
cd /path/to/RA-CFGCache
bash scripts/create_env.sh
```

### 2. Configure model weights

Model weights are not included in this repository. Download them from their official model pages and review the applicable license before use:

| Model | Official weights | Weight license | Configure with |
|---|---|---|---|
| FLUX.1-dev | [black-forest-labs/FLUX.1-dev](https://huggingface.co/black-forest-labs/FLUX.1-dev) | FLUX.1-dev Non-Commercial License | `FLUX_MODEL_DIR` or `--model_dir` |
| Wan2.1-T2V-1.3B | [Wan-AI/Wan2.1-T2V-1.3B-Diffusers](https://huggingface.co/Wan-AI/Wan2.1-T2V-1.3B-Diffusers) | Apache-2.0 | `WAN_MODEL_PATH` or `--model_path` |
| CogVideoX-2B | [THUDM/CogVideoX-2b](https://huggingface.co/THUDM/CogVideoX-2b) | Apache-2.0 | `COGVIDEOX_MODEL_PATH` or `--model_path` |

For example:

```bash
export FLUX_MODEL_DIR=/path/to/FLUX.1-dev
export WAN_MODEL_PATH=/path/to/Wan2.1-T2V-1.3B-Diffusers
export COGVIDEOX_MODEL_PATH=/path/to/CogVideoX-2b
```

The FLUX directory must use the layout expected by the reference FLUX implementation, including `flux1-dev.safetensors`, `ae.safetensors`, and the text-encoder/tokenizer subdirectories. The Wan and CogVideoX paths should point to local Diffusers-format model directories.

### 3. Run rho calibration

```bash
bash RUN/run_calibration_flux.sh --calibrate_rho --gpus 0 --num_gpus 1
bash RUN/run_calibration_wan.sh --calibrate_rho --gpus 0 --num_gpus 1
bash RUN/run_calibration_cogvideox.sh --calibrate_rho --gpus 0 --num_gpus 1
```

To collect ground-truth propagation statistics instead, replace `--calibrate_rho` with `--calibrate_gt --interval 3`.

## Supported Calibration Backends

| Model | Task | Calibration launcher |
|---|---|---|
| FLUX.1-dev | Text-to-Image | `RUN/run_calibration_flux.sh` |
| Wan2.1-T2V-1.3B | Text-to-Video | `RUN/run_calibration_wan.sh` |
| CogVideoX-2B | Text-to-Video | `RUN/run_calibration_cogvideox.sh` |

## Released Rho Tables

Configuration-specific rho tables used by the released experiments are provided under `calibration/`:

- `calibration/flux_rho_all_cfg.npz`
- `calibration/wan_rho_cfg5.npz`
- `calibration/cogvideox_rho_cfg6.npz`

These statistics contain numerical calibration data only. Because calibration is tied to the model and inference configuration, validate or regenerate the table when changing the scheduler, resolution, number of steps, CFG scale, or prompt distribution.

## Offline Calibration Workflow

The complete workflow contains two stages:

1. offline calibration on the `calibration` branch;
2. inference on the `main` branch.

The calibration stage estimates two types of artifacts:

- `rho`: branch-alignment statistics used by the CFG-aware risk estimator;
- `gt`: ground-truth propagation statistics used to fit the propagation-aware rescaling curve.

### 1. Run Offline Calibration

Before running calibration, make sure the calibration configuration is aligned with the inference configuration on the `main` branch. In particular, check:

- model path;
- prompt file;
- resolution;
- number of frames;
- number of denoising steps;
- CFG / guidance scale;
- scheduler and scheduler-specific parameters;
- random seed;
- prompt limit;

The calibration launch scripts are located under `RUN/`.

For FLUX:

```bash
bash RUN/run_calibration_flux.sh
```

For Wan2.1:

```bash
bash RUN/run_calibration_wan.sh
```

For CogVideoX:

```bash
bash RUN/run_calibration_cogvideox.sh
```

Each backend needs two calibration runs: one for `rho` and one for `gt`.

### 1.1 Calibrate rho

Run the corresponding launcher with `--calibrate_rho`:

```bash
bash RUN/run_calibration_<backend>.sh --calibrate_rho
```

The raw rho calibration results will be saved under:

```text
calibration_rho/
```

After the raw calibration run finishes, update the paths in the post-processing script and run:

```bash
bash calibration_rho/calibration_rho.sh
```

This produces the final rho table, usually saved as an `.npz` file.

### 1.2 Calibrate ground-truth propagation curve

For the ground-truth curve, calibrate every 3 denoising steps with:

```bash
bash RUN/run_calibration_<backend>.sh --calibrate_gt --interval 3
```

The raw GT calibration results will be saved under:

```text
calibration_gt/
```

After the raw GT calibration run finishes, update the paths in the post-processing script and run:

```bash
bash calibration_gt/calibration_gt.sh
```

This produces a fitted propagation curve figure. The figure also reports the fitted propagation parameters:

```text
a, b, alpha
```

These values should be copied into the RA-CFGCache configuration used during inference.

### 2. Use the Calibration Artifacts

After calibration, switch back to the main branch:

```bash
git switch main
```

Follow the inference instructions in the main-branch README and point `RHO_PROXY_TABLES_PATH` to the `.npz` file produced by rho calibration:

```bash
RHO_PROXY_TABLES_PATH="/path/to/calibration_rho/<backend>/rho_table.npz"
```

The propagation-aware rescaling parameters should be updated in `cache_init.py` according to the fitted GT curve:

```python
prop_a = ...
prop_b = ...
prop_alpha = ...
```

The cache threshold should also be configured in `cache_init.py` according to the desired speed-quality trade-off.

### Notes

- Custom calibration is only required when the model, resolution, scheduler, number of steps, CFG scale, prompt set, or other key settings are changed.
- The calibration configuration and inference configuration must be aligned. Otherwise, the calibrated rho table and propagation parameters may not match the actual inference setting.
- `rho` calibration produces the rho table used by RA-CFGCache.
- `gt` calibration produces the fitted propagation curve and the parameters `a`, `b`, and `alpha`.
- The GT calibration interval is set to `3` by default, meaning one calibration point is collected every 3 denoising steps.
- Currently, the cache threshold and propagation-aware parameters are configured in `cache_init.py`. They should be kept consistent with the fitted calibration results.

## Results

### Main Results

We evaluate RA-CFGCache on both text-to-image and text-to-video generation. All metrics are computed against the full-step vanilla baseline under the same prompt set and random seed.

| Model | Task / Setting | Method | Speedup ↑ | Latency (s) ↓ | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---|---:|---:|---:|---:|---:|
| FLUX.1-dev | Text-to-Image, 1024×1024, CFG=3.5 | RA-CFGCache-Slow (τ=0.18) | 3.19× | 6.58 | **0.1137** | **0.8680** | **24.02** |
| FLUX.1-dev | Text-to-Image, 1024×1024, CFG=3.5 | RA-CFGCache-Fast (τ=0.3) | **3.95×** | **5.32** | 0.1484 | 0.8340 | 22.75 |
| Wan2.1-1.3B | Text-to-Video, 81 frames, 480P, CFG=5.0 | RA-CFGCache-Slow (τ=0.4) | 2.56× | 36.48 | **0.0993** | **0.8778** | **28.03** |
| Wan2.1-1.3B | Text-to-Video, 81 frames, 480P, CFG=5.0 | RA-CFGCache-Fast (τ=0.6) | **2.99×** | **31.25** | 0.1409 | 0.8386 | 26.12 |
| CogVideoX-2B | Text-to-Video, 49 frames, 480P, CFG=6.0 | RA-CFGCache-Slow (τ=0.6) | 2.39× | 16.96 | **0.0968** | **0.8746** | **26.93** |
| CogVideoX-2B | Text-to-Video, 49 frames, 480P, CFG=6.0 | RA-CFGCache-Fast (τ=1.4) | **3.13×** | **12.91** | 0.2052 | 0.7614 | 21.89 |

### Trade-off Curve

<p align="center">
  <img src="docs/figures/tradeoff.png" alt="FLUX trade-off curve" width="85%">
</p>

### Qualitative Examples

<p align="center">
  <img src="docs/figures/qualitative_main.png" alt="Qualitative examples" width="95%">
</p>


---

## Limitations

RA-CFGCache is effective in practice, but several limitations remain:

- The propagation gain is a first-order approximation of downstream error propagation.
- The scheduler is an online threshold-based controller rather than a globally optimal sequential policy.
- Offline calibration is configuration-specific; transferring to a different model, scheduler, resolution, step count, CFG scale, or prompt distribution may require validation or recalibration.


---

## Acknowledgement

This repository is initialized from [HiCache](https://github.com/fenglang918/HiCache). We sincerely thank the HiCache authors for releasing their codebase, which provides an important foundation for this project.

We also thank the authors of prior training-free diffusion acceleration and caching methods, including TeaCache, MagCache, DiCache, FasterCache, and TaylorSeer, for their inspiring works and open-source contributions.

## Citation

The arXiv link and BibTeX entry will be added here when the preprint is available.

## License

RA-CFGCache is released under the [GNU General Public License v3.0](LICENSE).

Third-party code, data, and model weights remain subject to their respective licenses and attribution requirements. Notices and license copies shipped with this repository are kept under `resources/third_party/`. Model weights are not distributed in this repository; consult each official model page before downloading or using them.

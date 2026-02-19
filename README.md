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
    <img src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" alt="License">
  </a>
  <a href="#installation">
    <img src="https://img.shields.io/badge/Python-3.10+-blue.svg" alt="Python">
  </a>
  <a href="#installation">
    <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg" alt="PyTorch">
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

In our paper, RA-CFGCache is evaluated on **FLUX.1-dev**, **Qwen-Image**, and **CogVideoX-2B**, and achieves a stronger efficiency–fidelity trade-off than existing training-free caching baselines.

---
## Motivation
<p align="center">
  <img src="docs/figures/branch_vs_guided_error.png" alt="Branch-wise errors vs guided error" width="48%">
  <img src="docs/figures/local_vs_final_error.png" alt="Local guided error vs final deviation" width="48%">
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

## Quick Start

### 1. Create environment

```bash
conda create -n racfgcache python=3.10 -y
conda activate racfgcache
cd /path/to/RA-CFGCache
bash scripts/create_env.sh
```

### 2. Run a minimal example

#### FLUX.1-dev

```bash
bash scripts/demo_flux.sh
```

#### Qwen-Image

```bash
bash scripts/demo_qwen_image.sh
```

#### CogVideoX-2B

```bash
bash scripts/demo_cogvideox.sh
```

---

## Installation

### Requirements

- Python 3.10+
- PyTorch 2.1+
- CUDA 12.1+ (recommended)
- `diffusers`, `transformers`, `accelerate`, `safetensors`

### Basic install

```bash
git clone https://github.com/<your-org>/RA-CFGCache.git
cd RA-CFGCache
pip install -r requirements.txt
pip install -e .
```

### Optional backend dependencies

Some backends may require additional packages or model-specific runtime dependencies.
Please check the corresponding script/config before running:

- `scripts/demo_flux.sh`
- `scripts/demo_qwen_image.sh`
- `scripts/demo_cogvideox.sh`

---

## Supported Models

| Model | Task | Status | Example Script | Notes |
|---|---|---:|---|---|
| FLUX.1-dev | Text-to-Image | ✅ | `scripts/demo_flux.sh` | Main T2I backend |
| Qwen-Image | Text-to-Image | ✅ | `scripts/demo_qwen_image.sh` | Main T2I backend |
| CogVideoX-2B | Text-to-Video | ✅ | `scripts/demo_cogvideox.sh` | Main T2V backend |

---

## Reproduce Main Results

This repository is organized to make it easy to reproduce the main paper results.

### A. FLUX.1-dev

```bash
# Step 1: offline calibration
bash scripts/calibrate_flux.sh

# Step 2: run slow mode
bash scripts/eval_flux_slow.sh

# Step 3: run fast mode
bash scripts/eval_flux_fast.sh
```

### B. Qwen-Image

```bash
bash scripts/calibrate_qwen.sh
bash scripts/eval_qwen_slow.sh
bash scripts/eval_qwen_fast.sh
```

### C. CogVideoX-2B

```bash
bash scripts/calibrate_cogvideox.sh
bash scripts/eval_cogvideox_slow.sh
bash scripts/eval_cogvideox_fast.sh
```

### Outputs

By default, results are saved to:

```text
outputs/
├── images/
├── videos/
├── metrics/
└── logs/
```

---

## Calibration

RA-CFGCache uses **offline calibration** to estimate:

1. **Pairwise alignment statistics** for guided-risk composition
2. **Propagation coefficients** for propagation-aware rescaling

### What calibration produces

```text
artifacts/calibration/
├── <model_name>/
│   ├── rho_table.pt
│   ├── propagation_gain.pt
│   ├── propagation_fit.json
│   └── metadata.json
```

### Example

```bash
python tools/calibrate.py \
  --model flux \
  --prompt_file data/calibration/prompts.txt \
  --output_dir artifacts/calibration/flux
```

### Notes

- Calibration is performed once per model/setup.
- Calibrated artifacts can be reused for all subsequent runs under the same setting.
- We recommend storing calibration outputs under `artifacts/calibration/` for reproducibility.

---

## Usage

### Run on a single prompt

```bash
python run.py \
  --model flux \
  --prompt "a red sports car driving on a rainy city street" \
  --mode fast
```

### Run on a prompt file

```bash
python run.py \
  --model qwen_image \
  --prompt_file data/prompts/drawbench_200.txt \
  --mode slow
```

### Switch between fast and slow modes

```bash
python run.py --model flux --mode slow
python run.py --model flux --mode fast
```

### Specify a custom threshold

```bash
python run.py \
  --model flux \
  --tau 0.12
```

### Select base proxy family

```bash
python run.py \
  --model flux \
  --base_proxy tea

python run.py \
  --model flux \
  --base_proxy mag

python run.py \
  --model flux \
  --base_proxy dicache
```

### Save intermediate traces

```bash
python run.py \
  --model flux \
  --save_traces \
  --trace_dir artifacts/traces/flux
```

---

## Results

### Main Results

| Model | Mode | Speedup ↑ | LPIPS ↓ | SSIM ↑ | PSNR ↑ |
|---|---|---:|---:|---:|---:|
| FLUX.1-dev | Slow | 3.19× | 0.1294 | 0.8529 | 23.25 |
| FLUX.1-dev | Fast | 3.43× | 0.1453 | 0.8364 | 22.57 |
| Qwen-Image | Slow | 2.18× | 0.0511 | 0.9338 | 28.64 |
| Qwen-Image | Fast | 2.49× | 0.0878 | 0.8922 | 26.39 |
| CogVideoX-2B | Slow | 2.40× | 0.1045 | 0.8610 | 25.85 |
| CogVideoX-2B | Fast | 3.04× | 0.2098 | 0.7546 | 21.64 |

### Trade-off Curve

<p align="center">
  <img src="docs/assets/flux_tradeoff.png" alt="FLUX trade-off curve" width="85%">
</p>

### Qualitative Examples

<p align="center">
  <img src="docs/assets/qualitative_main.png" alt="Qualitative examples" width="95%">
</p>

---

## Checkpoints / Model Paths

Please prepare the corresponding model checkpoints before running experiments.

### Example structure

```text
checkpoints/
├── FLUX.1-dev/
├── Qwen-Image/
└── CogVideoX-2B/
```

You can either:

- download checkpoints manually and place them under `checkpoints/`, or
- specify paths through config files / command-line arguments.

### Example

```bash
python run.py \
  --model flux \
  --ckpt_path checkpoints/FLUX.1-dev
```

---

## Repository Structure

```text
RA-CFGCache/
├── backends/                # FLUX / Qwen-Image / CogVideoX wrappers
├── racfgcache/              # core logic: composition, rescaling, scheduler
├── calibration/             # calibration code
├── configs/                 # model and experiment configs
├── scripts/                 # runnable scripts for demo and reproduction
├── tools/                   # utility scripts
├── data/                    # prompt files and metadata
├── artifacts/               # calibration artifacts and traces
├── outputs/                 # generated results and evaluation outputs
├── docs/                    # figures and documentation assets
└── README.md
```

---

## Limitations

RA-CFGCache is effective in practice, but several limitations remain:

- The propagation gain is a first-order approximation of downstream error propagation.
- The scheduler is an online threshold-based controller rather than a globally optimal sequential policy.
- The framework is most naturally compatible with proxy families that have explicit cumulative reuse semantics.

---

## TODO

- [ ] Release full training-free evaluation scripts
- [ ] Add more proxy backends
- [ ] Add support for more diffusion / flow models
- [ ] Release a lightweight benchmark suite for CFG caching
- [ ] Add multi-GPU inference examples

---

## License

This project is released under the Apache 2.0 License. See [LICENSE](LICENSE) for details.

---

## Citation

If you find this repository useful, please consider citing our work:

```bibtex
@article{ra_cfgcache_2026,
  title={RA-CFGCache: From Branch-Level Heuristics to Guided-Risk Control under Classifier-Free Guidance},
  author={Anonymous Authors},
  journal={arXiv preprint arXiv:xxxx.xxxxx},
  year={2026}
}
```

---

## Acknowledgement

We thank the authors of prior training-free diffusion acceleration and caching methods for releasing their code and inspiring this project.

---

## Contact

For questions or collaborations, please open an issue or contact the maintainers.


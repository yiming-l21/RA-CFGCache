#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Offline GT calibration from fixed CFG perturbation traces.

This script reads dumped tensors, aggregates prompt results, draws GT curves,
and fits:

    y(t) = a * ((T - t) / T) ** alpha + b

Main modification:
    By default, the first 5 GT curve points are masked as NaN before averaging
    and fitting, to remove extreme early-step GT values.

Expected layout:

    calibration_gt/
    └── {model_name}/
        └── cfg{scale}_traces/
            ├── gt_plan.json                         # optional, used to infer T
            ├── prompt_000000/
            │   ├── baseline/
            │   │   └── final_latents.pt
            │   ├── perturb_step_001/
            │   │   ├── final_latents.pt
            │   │   ├── guided_old.pt
            │   │   ├── guided_ref.pt
            │   │   ├── meta.json
            │   │   └── run_meta.json
            │   └── ...
            └── ...

For each prompt and perturbation step t:

    latent_error(t) = D(perturb_step_t/final_latents.pt,
                        baseline/final_latents.pt)

    guided_error(t) = D(perturb_step_t/guided_old.pt,
                        perturb_step_t/guided_ref.pt)

    gt(t) = latent_error(t) / guided_error(t)

Outputs under:

    {out_root}/{model_name}/
        online_gt_all_cfg.npz
        online_gt_all_cfg.csv
        online_gt_all_cfg_summary.json
        online_gt_fit_params.json
        online_gt_curve.png
        gt_cfg{scale}.npz              # if --also_save_per_cfg
        gt_cfg{scale}.csv              # if --also_save_per_cfg
        gt_cfg{scale}_curve.png        # if --also_save_per_cfg

Example:

    python calibration_gt/calibrate_gt_from_traces.py \
        --root calibration_gt \
        --models flux \
        --cfg_scales 3.5 \
        --out_root calibration \
        --device cpu \
        --mask_head_gt_steps 5 \
        --mask_head_gt_mode position
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm


# -----------------------------------------------------------------------------
# Fixed layout helpers
# -----------------------------------------------------------------------------


def parse_cfg_scale_from_dirname(name: str) -> float | None:
    """Parse fixed cfg directory names like cfg3.5_traces."""
    m = re.match(r"^cfg([0-9]+(?:\.[0-9]+)?)_traces$", name)
    if m is None:
        return None
    return float(m.group(1))


def cfg_tag(scale: float) -> str:
    """3.5 -> cfg3.5_traces; 4.0 -> cfg4.0_traces."""
    return f"cfg{float(scale):.1f}_traces"


def discover_models(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def discover_cfg_scales(model_dir: Path) -> list[float]:
    scales: list[float] = []
    for p in model_dir.iterdir():
        if not p.is_dir():
            continue
        s = parse_cfg_scale_from_dirname(p.name)
        if s is not None:
            scales.append(s)
    return sorted(set(scales))


def perturb_dir(prompt_dir: Path, step: int) -> Path:
    return prompt_dir / f"perturb_step_{step:03d}"


def candidate_step_ids(prompt_dir: Path) -> list[int]:
    ids: list[int] = []
    for p in prompt_dir.glob("perturb_step_*"):
        if not p.is_dir():
            continue
        m = re.match(r"^perturb_step_(\d+)$", p.name)
        if m:
            ids.append(int(m.group(1)))
    return sorted(set(ids))


def has_complete_step(prompt_dir: Path, step: int) -> bool:
    d = perturb_dir(prompt_dir, step)
    return (
        (d / "final_latents.pt").is_file()
        and (d / "guided_old.pt").is_file()
        and (d / "guided_ref.pt").is_file()
    )


def step_ids_in_prompt_dir(prompt_dir: Path) -> list[int]:
    return [s for s in candidate_step_ids(prompt_dir) if has_complete_step(prompt_dir, s)]


def baseline_file(prompt_dir: Path) -> Path:
    return prompt_dir / "baseline" / "final_latents.pt"


# -----------------------------------------------------------------------------
# Tensor and distance helpers
# -----------------------------------------------------------------------------


def load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    x = torch.load(path, map_location=device)
    if not torch.is_tensor(x):
        raise TypeError(f"File does not contain a torch.Tensor: {path}")
    return x.detach().to(device=device, dtype=torch.float32).flatten()


def tensor_distance(x: torch.Tensor, y: torch.Tensor, metric: str, eps: float) -> float:
    if x.numel() != y.numel():
        raise ValueError(f"Tensor size mismatch after flatten: {x.numel()} vs {y.numel()}")

    diff = x - y
    if metric == "rel_l1":
        return float((torch.sum(torch.abs(diff)) / (torch.sum(torch.abs(y)) + eps)).item())
    if metric == "rel_l2":
        return float((torch.linalg.vector_norm(diff) / (torch.linalg.vector_norm(y) + eps)).item())
    if metric == "l1":
        return float(torch.mean(torch.abs(diff)).item())
    if metric == "l2":
        return float(torch.linalg.vector_norm(diff).item())
    if metric == "mse":
        return float(torch.mean(diff * diff).item())

    raise ValueError(f"Unsupported metric: {metric}")


# -----------------------------------------------------------------------------
# GT masking
# -----------------------------------------------------------------------------


def mask_head_gt_outliers(
    gt_stack: np.ndarray,
    steps: list[int],
    head_steps: int = 5,
    mode: str = "position",
) -> tuple[np.ndarray, np.ndarray]:
    """
    Mask extreme GT values in the first few denoising steps.

    Args:
        gt_stack:
            Shape [P, S], per-prompt GT curves.
        steps:
            Perturbation step ids, length S.
        head_steps:
            Number of early GT points to mask.
        mode:
            "position": mask the first head_steps columns in the curve.
                        Recommended when perturb steps are sparse, e.g. 1, 4, 7...
            "step_id":  mask columns whose actual step id < head_steps.

    Returns:
        gt_stack_clean:
            Copy of gt_stack with selected early GT entries set to NaN.
        masked_cols:
            Boolean array of shape [S], indicating masked columns.
    """
    gt_stack_clean = gt_stack.copy()
    step_arr = np.asarray(steps, dtype=np.int64)

    if head_steps <= 0:
        return gt_stack_clean, np.zeros((len(steps),), dtype=bool)

    if mode == "position":
        masked_cols = np.zeros((len(steps),), dtype=bool)
        masked_cols[: min(head_steps, len(steps))] = True
    elif mode == "step_id":
        masked_cols = step_arr < int(head_steps)
    else:
        raise ValueError(f"Unsupported head GT mask mode: {mode}")

    gt_stack_clean[:, masked_cols] = np.nan
    return gt_stack_clean, masked_cols


# -----------------------------------------------------------------------------
# GT computation
# -----------------------------------------------------------------------------


def compute_prompt_gt_curve(
    prompt_dir: Path,
    steps: list[int],
    device: torch.device,
    metric: str,
    eps: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute one prompt's GT curve.

    Returns:
        gt:           [T]
        latent_errs:  [T]
        guided_errs:  [T]
    """
    base_file = baseline_file(prompt_dir)
    if not base_file.is_file():
        raise FileNotFoundError(f"Missing baseline final latents: {base_file}")

    baseline = load_tensor(base_file, device)

    gt = np.full((len(steps),), np.nan, dtype=np.float32)
    latent_errs = np.full((len(steps),), np.nan, dtype=np.float32)
    guided_errs = np.full((len(steps),), np.nan, dtype=np.float32)

    for i, step in enumerate(steps):
        d = perturb_dir(prompt_dir, step)
        final_file = d / "final_latents.pt"
        guided_old_file = d / "guided_old.pt"
        guided_ref_file = d / "guided_ref.pt"

        if not final_file.is_file() or not guided_old_file.is_file() or not guided_ref_file.is_file():
            raise FileNotFoundError(f"Missing fixed GT tensors under: {d}")

        final_latents = load_tensor(final_file, device)
        guided_old = load_tensor(guided_old_file, device)
        guided_ref = load_tensor(guided_ref_file, device)

        latent_err = tensor_distance(final_latents, baseline, metric=metric, eps=eps)
        guided_err = tensor_distance(guided_old, guided_ref, metric=metric, eps=eps)

        latent_errs[i] = latent_err
        guided_errs[i] = guided_err
        if math.isfinite(guided_err) and abs(guided_err) > eps:
            gt[i] = latent_err / guided_err

    return gt, latent_errs, guided_errs


def calibrate_one_cfg(
    cfg_trace_dir: Path,
    device: torch.device,
    max_prompts: int | None,
    prompt_start: int,
    step_mode: str,
    metric: str,
    eps: float,
    mask_head_gt_steps: int,
    mask_head_gt_mode: str,
) -> tuple[list[int], dict[str, np.ndarray], dict[str, Any], dict[str, np.ndarray]]:
    if not cfg_trace_dir.is_dir():
        raise FileNotFoundError(f"Trace dir not found: {cfg_trace_dir}")

    prompt_dirs = sorted(p for p in cfg_trace_dir.glob("prompt_*") if p.is_dir())
    if prompt_start > 0:
        prompt_dirs = prompt_dirs[prompt_start:]
    if max_prompts is not None and max_prompts > 0:
        prompt_dirs = prompt_dirs[:max_prompts]
    if not prompt_dirs:
        raise RuntimeError(f"No prompt directories found in: {cfg_trace_dir}")

    prompt_steps: dict[Path, list[int]] = {}
    valid_prompt_dirs: list[Path] = []
    skipped_layout: list[str] = []

    for p in prompt_dirs:
        if not baseline_file(p).is_file():
            print(f"[WARN] Skip prompt without baseline/final_latents.pt: {p}")
            skipped_layout.append(str(p))
            continue

        steps = step_ids_in_prompt_dir(p)
        if len(steps) == 0:
            print(f"[WARN] Skip prompt without complete perturb_step_XXX dumps: {p}")
            skipped_layout.append(str(p))
            continue

        prompt_steps[p] = steps
        valid_prompt_dirs.append(p)

    if not valid_prompt_dirs:
        first = prompt_dirs[0]
        print("\n========== First prompt layout ==========")
        print(f"prompt_dir: {first}")
        print(f"baseline exists: {baseline_file(first).is_file()} -> {baseline_file(first)}")
        print(f"candidate steps: {candidate_step_ids(first)}")
        for step in candidate_step_ids(first)[:5]:
            d = perturb_dir(first, step)
            print(f"  {d.name}: {[x.name for x in sorted(d.glob('*'))]}")
        print("========================================\n")
        raise RuntimeError(f"No valid prompt directories in {cfg_trace_dir}")

    if step_mode == "intersection":
        common = set(prompt_steps[valid_prompt_dirs[0]])
        for p in valid_prompt_dirs[1:]:
            common &= set(prompt_steps[p])
        steps = sorted(common)
    elif step_mode == "first":
        steps = prompt_steps[valid_prompt_dirs[0]]
    else:
        raise ValueError(f"Unsupported step_mode: {step_mode}")

    if len(steps) == 0:
        raise RuntimeError(f"No common perturb steps found in {cfg_trace_dir}. Try --step_mode first.")

    gt_list: list[np.ndarray] = []
    latent_list: list[np.ndarray] = []
    guided_list: list[np.ndarray] = []
    used_prompts: list[str] = []
    skipped_prompts: list[str] = []

    for prompt_dir in tqdm(valid_prompt_dirs, desc=f"Calibrating {cfg_trace_dir.name}"):
        if not set(steps).issubset(set(prompt_steps[prompt_dir])):
            skipped_prompts.append(str(prompt_dir))
            continue

        try:
            gt, latent_errs, guided_errs = compute_prompt_gt_curve(
                prompt_dir=prompt_dir,
                steps=steps,
                device=device,
                metric=metric,
                eps=eps,
            )
            gt_list.append(gt)
            latent_list.append(latent_errs)
            guided_list.append(guided_errs)
            used_prompts.append(str(prompt_dir))
        except Exception as e:
            print(f"[WARN] Failed on {prompt_dir}: {e!r}")
            skipped_prompts.append(str(prompt_dir))

    if not gt_list:
        raise RuntimeError(f"No prompt was successfully calibrated in: {cfg_trace_dir}")

    gt_stack_raw = np.stack(gt_list, axis=0).astype(np.float32)      # [P, S]
    latent_stack = np.stack(latent_list, axis=0).astype(np.float32)  # [P, S]
    guided_stack = np.stack(guided_list, axis=0).astype(np.float32)  # [P, S]

    gt_stack, masked_head_cols = mask_head_gt_outliers(
        gt_stack=gt_stack_raw,
        steps=steps,
        head_steps=mask_head_gt_steps,
        mode=mask_head_gt_mode,
    )

    gt_count = np.sum(np.isfinite(gt_stack), axis=0).astype(np.float32)
    latent_count = np.sum(np.isfinite(latent_stack), axis=0).astype(np.float32)
    guided_count = np.sum(np.isfinite(guided_stack), axis=0).astype(np.float32)

    gt_mean = np.nanmean(gt_stack, axis=0).astype(np.float32)
    gt_std = np.nanstd(gt_stack, axis=0).astype(np.float32)
    latent_mean = np.nanmean(latent_stack, axis=0).astype(np.float32)
    latent_std = np.nanstd(latent_stack, axis=0).astype(np.float32)
    guided_mean = np.nanmean(guided_stack, axis=0).astype(np.float32)
    guided_std = np.nanstd(guided_stack, axis=0).astype(np.float32)

    ratio_of_means = np.full_like(gt_mean, np.nan, dtype=np.float32)
    valid = np.isfinite(guided_mean) & (np.abs(guided_mean) > eps)
    ratio_of_means[valid] = latent_mean[valid] / guided_mean[valid]

    curves = {
        "gt_mean": gt_mean,
        "gt_std": gt_std,
        "gt_count": gt_count,
        "latent_error_mean": latent_mean,
        "latent_error_std": latent_std,
        "latent_error_count": latent_count,
        "guided_error_mean": guided_mean,
        "guided_error_std": guided_std,
        "guided_error_count": guided_count,
        "gt_ratio_of_means": ratio_of_means,
    }

    raw = {
        "gt_stack": gt_stack,
        "gt_stack_raw": gt_stack_raw,
        "latent_error_stack": latent_stack,
        "guided_error_stack": guided_stack,
        "masked_head_cols": masked_head_cols.astype(np.bool_),
    }

    summary = {
        "trace_dir": str(cfg_trace_dir),
        "num_prompt_dirs_found": len(prompt_dirs),
        "num_valid_prompt_dirs": len(valid_prompt_dirs),
        "num_prompts_used": len(used_prompts),
        "num_prompts_skipped": len(skipped_prompts),
        "steps": steps,
        "step_mode": step_mode,
        "metric": metric,
        "eps": eps,
        "mask_head_gt_steps": int(mask_head_gt_steps),
        "mask_head_gt_mode": str(mask_head_gt_mode),
        "masked_head_gt_step_ids": [int(s) for s, m in zip(steps, masked_head_cols) if bool(m)],
        "fixed_layout": {
            "baseline": "baseline/final_latents.pt",
            "perturb_dir": "perturb_step_%03d",
            "final_latents": "final_latents.pt",
            "guided_old": "guided_old.pt",
            "guided_ref": "guided_ref.pt",
        },
        "used_prompts_first_10": used_prompts[:10],
        "skipped_prompts_first_10": skipped_prompts[:10],
        "layout_skipped_first_10": skipped_layout[:10],
    }

    return steps, curves, summary, raw


# -----------------------------------------------------------------------------
# Fit y = a * ((T - t) / T) ** alpha + b by MSE
# -----------------------------------------------------------------------------


def _as_number(x: Any) -> float | None:
    if isinstance(x, (int, float)):
        return float(x)
    return None


def _search_key_recursive(obj: Any, keys: set[str]) -> float | None:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in keys:
                val = _as_number(v)
                if val is not None:
                    return val
        for v in obj.values():
            found = _search_key_recursive(v, keys)
            if found is not None:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = _search_key_recursive(v, keys)
            if found is not None:
                return found
    return None


def infer_total_steps(cfg_trace_dir: Path, steps: list[int], explicit_total_steps: int | None) -> int:
    if explicit_total_steps is not None and explicit_total_steps > 0:
        return int(explicit_total_steps)

    plan_path = cfg_trace_dir / "gt_plan.json"
    if plan_path.is_file():
        try:
            with open(plan_path, "r", encoding="utf-8") as f:
                payload = json.load(f)
            val = _search_key_recursive(
                payload,
                keys={
                    "num_steps",
                    "num_denoise_steps",
                    "denoise_steps",
                    "total_steps",
                    "n_steps",
                    "T",
                },
            )
            if val is not None and val > max(steps):
                return int(round(val))
        except Exception as e:
            print(f"[WARN] Failed to parse gt_plan.json for total steps: {e!r}")

    return int(max(steps) + 1)


def solve_ab_for_alpha(x: np.ndarray, y: np.ndarray, alpha: float) -> tuple[float, float, float]:
    z = np.power(x, alpha)
    A = np.stack([z, np.ones_like(z)], axis=1)
    sol, _, _, _ = np.linalg.lstsq(A, y, rcond=None)
    a = float(sol[0])
    b = float(sol[1])
    pred = a * z + b
    mse = float(np.mean((pred - y) ** 2))
    return a, b, mse


def fit_power_curve(
    steps: np.ndarray,
    y: np.ndarray,
    total_steps: int,
    alpha_min: float,
    alpha_max: float,
    alpha_grid_size: int,
) -> dict[str, Any]:
    mask = np.isfinite(steps) & np.isfinite(y)
    t = steps[mask].astype(np.float64)
    yy = y[mask].astype(np.float64)

    if yy.size < 3:
        raise RuntimeError("Need at least 3 valid GT points to fit a, b, alpha.")

    T = float(total_steps)
    x = (T - t) / T
    x = np.clip(x, 1e-12, None)

    alpha_min = max(float(alpha_min), 1e-8)
    alpha_max = max(float(alpha_max), alpha_min * 1.0001)
    alpha_grid = np.exp(np.linspace(np.log(alpha_min), np.log(alpha_max), int(alpha_grid_size)))

    best = None
    mses = []
    for alpha in alpha_grid:
        a, b, mse = solve_ab_for_alpha(x, yy, float(alpha))
        mses.append(mse)
        if best is None or mse < best["mse"]:
            best = {"a": a, "b": b, "alpha": float(alpha), "mse": mse}

    assert best is not None

    log_grid = np.log(alpha_grid)
    best_idx = int(np.argmin(mses))
    lo_idx = max(0, best_idx - 2)
    hi_idx = min(len(alpha_grid) - 1, best_idx + 2)
    lo = float(log_grid[lo_idx])
    hi = float(log_grid[hi_idx])

    def eval_log_alpha(log_alpha: float) -> dict[str, float]:
        alpha = float(np.exp(log_alpha))
        a, b, mse = solve_ab_for_alpha(x, yy, alpha)
        return {"a": a, "b": b, "alpha": alpha, "mse": mse}

    for _ in range(80):
        m1 = lo + (hi - lo) / 3.0
        m2 = hi - (hi - lo) / 3.0
        f1 = eval_log_alpha(m1)
        f2 = eval_log_alpha(m2)
        if f1["mse"] <= f2["mse"]:
            hi = m2
            if f1["mse"] < best["mse"]:
                best = f1
        else:
            lo = m1
            if f2["mse"] < best["mse"]:
                best = f2

    all_x = (T - steps.astype(np.float64)) / T
    all_x = np.clip(all_x, 1e-12, None)
    fit = best["a"] * np.power(all_x, best["alpha"]) + best["b"]

    ss_res = float(np.sum((best["a"] * np.power(x, best["alpha"]) + best["b"] - yy) ** 2))
    ss_tot = float(np.sum((yy - np.mean(yy)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "a": float(best["a"]),
        "b": float(best["b"]),
        "alpha": float(best["alpha"]),
        "mse": float(best["mse"]),
        "r2": float(r2),
        "total_steps_T": int(total_steps),
        "fit_curve": fit.astype(np.float32),
        "num_points": int(yy.size),
        "alpha_min": float(alpha_min),
        "alpha_max": float(alpha_max),
        "alpha_grid_size": int(alpha_grid_size),
        "formula": "gt(t) = a * ((T - t) / T) ** alpha + b",
    }


# -----------------------------------------------------------------------------
# Saving and plotting
# -----------------------------------------------------------------------------


def save_csv(
    csv_path: Path,
    cfg_scales: list[float],
    steps: list[int],
    curves_by_cfg: list[dict[str, np.ndarray]],
    fits_by_cfg: list[dict[str, Any]],
) -> None:
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cfg_scale",
            "step",
            "count",
            "gt_mean",
            "gt_std",
            "gt_fit",
            "latent_error_mean",
            "latent_error_std",
            "guided_error_mean",
            "guided_error_std",
            "gt_ratio_of_means",
        ])
        for cfg_scale, curves, fit in zip(cfg_scales, curves_by_cfg, fits_by_cfg):
            fit_curve = fit["fit_curve"]
            for i, step in enumerate(steps):
                writer.writerow([
                    float(cfg_scale),
                    int(step),
                    float(curves["gt_count"][i]),
                    float(curves["gt_mean"][i]) if np.isfinite(curves["gt_mean"][i]) else "nan",
                    float(curves["gt_std"][i]) if np.isfinite(curves["gt_std"][i]) else "nan",
                    float(fit_curve[i]) if np.isfinite(fit_curve[i]) else "nan",
                    float(curves["latent_error_mean"][i]) if np.isfinite(curves["latent_error_mean"][i]) else "nan",
                    float(curves["latent_error_std"][i]) if np.isfinite(curves["latent_error_std"][i]) else "nan",
                    float(curves["guided_error_mean"][i]) if np.isfinite(curves["guided_error_mean"][i]) else "nan",
                    float(curves["guided_error_std"][i]) if np.isfinite(curves["guided_error_std"][i]) else "nan",
                    float(curves["gt_ratio_of_means"][i]) if np.isfinite(curves["gt_ratio_of_means"][i]) else "nan",
                ])


def save_per_prompt_csv(csv_path: Path, cfg_scale: float, steps: list[int], raw: dict[str, np.ndarray]) -> None:
    gt_stack = raw["gt_stack"]
    gt_stack_raw = raw["gt_stack_raw"]
    latent_stack = raw["latent_error_stack"]
    guided_stack = raw["guided_error_stack"]

    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cfg_scale",
            "prompt_index",
            "step",
            "gt_clean",
            "gt_raw",
            "latent_error",
            "guided_error",
        ])
        for p_idx in range(gt_stack.shape[0]):
            for s_idx, step in enumerate(steps):
                writer.writerow([
                    float(cfg_scale),
                    int(p_idx),
                    int(step),
                    float(gt_stack[p_idx, s_idx]) if np.isfinite(gt_stack[p_idx, s_idx]) else "nan",
                    float(gt_stack_raw[p_idx, s_idx]) if np.isfinite(gt_stack_raw[p_idx, s_idx]) else "nan",
                    float(latent_stack[p_idx, s_idx]) if np.isfinite(latent_stack[p_idx, s_idx]) else "nan",
                    float(guided_stack[p_idx, s_idx]) if np.isfinite(guided_stack[p_idx, s_idx]) else "nan",
                ])


def plot_gt_curve(
    png_path: Path,
    cfg_scales: list[float],
    steps: list[int],
    curves_by_cfg: list[dict[str, np.ndarray]],
    fits_by_cfg: list[dict[str, Any]],
    title: str,
) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception as e:
        print(f"[WARN] matplotlib unavailable; skip plotting {png_path}: {e!r}")
        return

    x = np.array(steps, dtype=np.int64)

    plt.figure(figsize=(8.5, 5.2))
    for cfg_scale, curves, fit in zip(cfg_scales, curves_by_cfg, fits_by_cfg):
        y = curves["gt_mean"]
        y_std = curves["gt_std"]
        y_fit = fit["fit_curve"]

        label_data = f"CFG {cfg_scale:g} mean GT"
        label_fit = (
            f"CFG {cfg_scale:g} fit: "
            f"a={fit['a']:.4g}, b={fit['b']:.4g}, "
            f"alpha={fit['alpha']:.4g}, mse={fit['mse']:.3g}"
        )

        valid = np.isfinite(y)
        if valid.any():
            plt.plot(x[valid], y[valid], marker="o", linewidth=1.8, label=label_data)

        valid_std = valid & np.isfinite(y_std)
        if valid_std.any():
            plt.fill_between(
                x[valid_std],
                y[valid_std] - y_std[valid_std],
                y[valid_std] + y_std[valid_std],
                alpha=0.15,
            )

        valid_fit = np.isfinite(y_fit)
        if valid_fit.any():
            plt.plot(x[valid_fit], y_fit[valid_fit], linestyle="--", linewidth=2.0, label=label_fit)

    plt.xlabel("Perturb step")
    plt.ylabel("GT")
    plt.title(title)
    plt.grid(True, linestyle="--", alpha=0.35)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(png_path, dpi=220)
    plt.close()
    print(f"[INFO] Saved curve plot: {png_path}")


def save_outputs(
    model_out_dir: Path,
    cfg_scales: list[float],
    steps: list[int],
    curves_by_cfg: list[dict[str, np.ndarray]],
    fits_by_cfg: list[dict[str, Any]],
    summaries: list[dict[str, Any]],
    root: Path,
    model_name: str,
    metric: str,
    eps: float,
) -> None:
    model_out_dir.mkdir(parents=True, exist_ok=True)

    out_npz = model_out_dir / "online_gt_all_cfg.npz"
    out_csv = model_out_dir / "online_gt_all_cfg.csv"
    out_json = model_out_dir / "online_gt_all_cfg_summary.json"
    out_fit_json = model_out_dir / "online_gt_fit_params.json"
    out_png = model_out_dir / "online_gt_curve.png"

    np.savez_compressed(
        out_npz,
        cfg_scales=np.array(cfg_scales, dtype=np.float32),
        steps=np.array(steps, dtype=np.int64),
        gt_curve_all=np.stack([c["gt_mean"] for c in curves_by_cfg], axis=0).astype(np.float32),
        gt_curve_std_all=np.stack([c["gt_std"] for c in curves_by_cfg], axis=0).astype(np.float32),
        gt_curve_count_all=np.stack([c["gt_count"] for c in curves_by_cfg], axis=0).astype(np.float32),
        latent_error_curve_all=np.stack([c["latent_error_mean"] for c in curves_by_cfg], axis=0).astype(np.float32),
        latent_error_std_all=np.stack([c["latent_error_std"] for c in curves_by_cfg], axis=0).astype(np.float32),
        guided_error_curve_all=np.stack([c["guided_error_mean"] for c in curves_by_cfg], axis=0).astype(np.float32),
        guided_error_std_all=np.stack([c["guided_error_std"] for c in curves_by_cfg], axis=0).astype(np.float32),
        gt_ratio_of_means_all=np.stack([c["gt_ratio_of_means"] for c in curves_by_cfg], axis=0).astype(np.float32),
        fit_curve_all=np.stack([f["fit_curve"] for f in fits_by_cfg], axis=0).astype(np.float32),
        fit_params_all=np.array(
            [[f["a"], f["b"], f["alpha"], f["mse"], f["r2"], f["total_steps_T"]] for f in fits_by_cfg],
            dtype=np.float32,
        ),
    )

    save_csv(out_csv, cfg_scales, steps, curves_by_cfg, fits_by_cfg)

    fit_payload = {
        "model_name": model_name,
        "formula": "gt(t) = a * ((T - t) / T) ** alpha + b",
        "metric": metric,
        "fits": [
            {
                "cfg_scale": float(cfg),
                "a": float(fit["a"]),
                "b": float(fit["b"]),
                "alpha": float(fit["alpha"]),
                "mse": float(fit["mse"]),
                "r2": float(fit["r2"]),
                "total_steps_T": int(fit["total_steps_T"]),
                "num_points": int(fit["num_points"]),
                "alpha_min": float(fit["alpha_min"]),
                "alpha_max": float(fit["alpha_max"]),
                "alpha_grid_size": int(fit["alpha_grid_size"]),
            }
            for cfg, fit in zip(cfg_scales, fits_by_cfg)
        ],
    }
    with open(out_fit_json, "w", encoding="utf-8") as f:
        json.dump(fit_payload, f, indent=2, ensure_ascii=False)

    payload = {
        "model_name": model_name,
        "root": str(root),
        "out_npz": str(out_npz),
        "out_csv": str(out_csv),
        "out_fit_json": str(out_fit_json),
        "out_png": str(out_png),
        "cfg_scales": cfg_scales,
        "steps": steps,
        "num_cfg_scales": len(cfg_scales),
        "metric": metric,
        "eps": eps,
        "summaries": summaries,
    }
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    plot_gt_curve(
        png_path=out_png,
        cfg_scales=cfg_scales,
        steps=steps,
        curves_by_cfg=curves_by_cfg,
        fits_by_cfg=fits_by_cfg,
        title=f"GT Curve and Power Fit ({model_name})",
    )

    print(f"[INFO] Saved merged GT curve: {out_npz}")
    print(f"[INFO] Saved merged CSV:      {out_csv}")
    print(f"[INFO] Saved summary:         {out_json}")
    print(f"[INFO] Saved fit params:      {out_fit_json}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description="Clean offline GT calibration from fixed CFG perturbation traces.")

    parser.add_argument("--root", type=str, default="calibration_gt", help="Root containing {model}/cfg{scale}_traces.")
    parser.add_argument("--models", nargs="+", default=["all"], help="Model names under root. Use 'all' to auto-discover.")
    parser.add_argument("--cfg_scales", nargs="+", default=["auto"], help="CFG scales. Use 'auto' to discover all.")
    parser.add_argument("--out_root", type=str, default="calibration", help="Output root directory.")
    parser.add_argument("--max_prompts", type=int, default=None, help="Max prompts per cfg. Default: all.")
    parser.add_argument("--prompt_start", type=int, default=0, help="Prompt directory start offset after sorting.")
    parser.add_argument("--step_mode", type=str, default="intersection", choices=["intersection", "first"], help="How to choose step ids.")
    parser.add_argument("--device", type=str, default="cpu", help="cpu or cuda. CPU is safer for large dumps.")
    parser.add_argument("--metric", type=str, default="rel_l1", choices=["rel_l1", "rel_l2", "l1", "l2", "mse"], help="Distance metric.")
    parser.add_argument("--eps", type=float, default=1e-12, help="Numerical epsilon.")
    parser.add_argument("--also_save_per_cfg", action="store_true", help="Also save one npz/csv/png/per-prompt csv per cfg scale.")

    parser.add_argument("--total_steps", type=int, default=None, help="T in a*((T-t)/T)^alpha+b. If omitted, read gt_plan.json or use max(step)+1.")
    parser.add_argument("--fit_alpha_min", type=float, default=0.01, help="Minimum alpha for grid search.")
    parser.add_argument("--fit_alpha_max", type=float, default=10.0, help="Maximum alpha for grid search.")
    parser.add_argument("--fit_alpha_grid_size", type=int, default=2000, help="Number of log-grid alpha candidates before local refinement.")

    parser.add_argument(
        "--mask_head_gt_steps",
        type=int,
        default=0,
        help="Mask GT values in the first N curve positions before averaging/fitting. Use 0 to disable.",
    )
    parser.add_argument(
        "--mask_head_gt_mode",
        type=str,
        default="position",
        choices=["position", "step_id"],
        help="'position' masks the first N curve points; 'step_id' masks actual step ids < N.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve()
    device = torch.device(args.device)

    if not root.is_dir():
        raise FileNotFoundError(f"Root dir not found: {root}")

    models = discover_models(root) if args.models == ["all"] else args.models
    if not models:
        raise RuntimeError(f"No model directories found under: {root}")

    print("========================================")
    print("Clean GT calibration from fixed traces")
    print(f"root:       {root}")
    print(f"models:     {models}")
    print(f"cfg_scales: {args.cfg_scales}")
    print(f"out_root:   {out_root}")
    print(f"device:     {device}")
    print(f"metric:     {args.metric}")
    print(f"step_mode:  {args.step_mode}")
    print(f"mask_head_gt_steps: {args.mask_head_gt_steps}")
    print(f"mask_head_gt_mode:  {args.mask_head_gt_mode}")
    print("layout:     baseline/final_latents.pt + perturb_step_%03d/{final_latents,guided_old,guided_ref}.pt")
    print("fit:        gt(t) = a * ((T - t) / T) ** alpha + b")
    print("========================================")

    for model_name in models:
        model_dir = root / model_name
        if not model_dir.is_dir():
            print(f"[WARN] Skip missing model dir: {model_dir}")
            continue

        cfg_scales = discover_cfg_scales(model_dir) if args.cfg_scales == ["auto"] else [float(x) for x in args.cfg_scales]
        if not cfg_scales:
            print(f"[WARN] No cfg trace dirs found for model: {model_name}")
            continue

        print(f"\n========== Model: {model_name} ==========")
        print(f"cfg_scales: {cfg_scales}")

        all_steps: list[int] | None = None
        used_cfg_scales: list[float] = []
        curves_by_cfg: list[dict[str, np.ndarray]] = []
        fits_by_cfg: list[dict[str, Any]] = []
        summaries: list[dict[str, Any]] = []

        for scale in cfg_scales:
            trace_dir = model_dir / cfg_tag(scale)
            if not trace_dir.is_dir():
                print(f"[WARN] Missing trace dir, skip: {trace_dir}")
                continue

            steps, curves, summary, raw = calibrate_one_cfg(
                cfg_trace_dir=trace_dir,
                device=device,
                max_prompts=args.max_prompts,
                prompt_start=args.prompt_start,
                step_mode=args.step_mode,
                metric=args.metric,
                eps=args.eps,
                mask_head_gt_steps=args.mask_head_gt_steps,
                mask_head_gt_mode=args.mask_head_gt_mode,
            )

            if all_steps is None:
                all_steps = steps
            elif steps != all_steps:
                raise RuntimeError(
                    f"Step ids mismatch for model={model_name}, cfg={scale}. "
                    f"Expected {all_steps}, got {steps}. Run cfg scales separately or use consistent traces."
                )

            total_steps = infer_total_steps(trace_dir, steps, args.total_steps)
            fit = fit_power_curve(
                steps=np.array(steps, dtype=np.float64),
                y=curves["gt_mean"],
                total_steps=total_steps,
                alpha_min=args.fit_alpha_min,
                alpha_max=args.fit_alpha_max,
                alpha_grid_size=args.fit_alpha_grid_size,
            )

            print(
                f"[FIT] model={model_name} cfg={scale:g}: "
                f"T={fit['total_steps_T']}, "
                f"a={fit['a']:.8g}, b={fit['b']:.8g}, "
                f"alpha={fit['alpha']:.8g}, mse={fit['mse']:.8g}, r2={fit['r2']:.6g}, "
                f"masked_head_steps={summary['masked_head_gt_step_ids']}"
            )

            used_cfg_scales.append(float(scale))
            curves_by_cfg.append(curves)
            fits_by_cfg.append(fit)
            summaries.append(
                {
                    "cfg_scale": float(scale),
                    "fit": {k: v for k, v in fit.items() if k != "fit_curve"},
                    **summary,
                }
            )

            if args.also_save_per_cfg:
                per_cfg_dir = out_root / model_name
                per_cfg_dir.mkdir(parents=True, exist_ok=True)
                per_npz = per_cfg_dir / f"gt_cfg{float(scale):g}.npz"
                per_csv = per_cfg_dir / f"gt_cfg{float(scale):g}.csv"
                per_prompt_csv = per_cfg_dir / f"gt_cfg{float(scale):g}_per_prompt.csv"
                per_png = per_cfg_dir / f"gt_cfg{float(scale):g}_curve.png"

                np.savez_compressed(
                    per_npz,
                    cfg_scale=np.array([float(scale)], dtype=np.float32),
                    steps=np.array(steps, dtype=np.int64),
                    gt_curve=curves["gt_mean"].astype(np.float32),
                    gt_curve_std=curves["gt_std"].astype(np.float32),
                    gt_curve_count=curves["gt_count"].astype(np.float32),
                    latent_error_curve=curves["latent_error_mean"].astype(np.float32),
                    latent_error_std=curves["latent_error_std"].astype(np.float32),
                    guided_error_curve=curves["guided_error_mean"].astype(np.float32),
                    guided_error_std=curves["guided_error_std"].astype(np.float32),
                    gt_ratio_of_means=curves["gt_ratio_of_means"].astype(np.float32),
                    fit_curve=fit["fit_curve"].astype(np.float32),
                    fit_params=np.array(
                        [fit["a"], fit["b"], fit["alpha"], fit["mse"], fit["r2"], fit["total_steps_T"]],
                        dtype=np.float32,
                    ),
                    gt_stack=raw["gt_stack"].astype(np.float32),
                    gt_stack_raw=raw["gt_stack_raw"].astype(np.float32),
                    masked_head_cols=raw["masked_head_cols"].astype(np.bool_),
                )
                save_csv(per_csv, [float(scale)], steps, [curves], [fit])
                save_per_prompt_csv(per_prompt_csv, float(scale), steps, raw)
                plot_gt_curve(
                    png_path=per_png,
                    cfg_scales=[float(scale)],
                    steps=steps,
                    curves_by_cfg=[curves],
                    fits_by_cfg=[fit],
                    title=f"GT Curve and Power Fit ({model_name}, CFG {float(scale):g})",
                )
                print(f"[INFO] Saved per-cfg GT curve: {per_npz}")
                print(f"[INFO] Saved per-cfg CSV:      {per_csv}")
                print(f"[INFO] Saved per-prompt CSV:   {per_prompt_csv}")

        if not curves_by_cfg:
            print(f"[WARN] No cfg scale successfully calibrated for model: {model_name}")
            continue

        assert all_steps is not None
        save_outputs(
            model_out_dir=out_root / model_name,
            cfg_scales=used_cfg_scales,
            steps=all_steps,
            curves_by_cfg=curves_by_cfg,
            fits_by_cfg=fits_by_cfg,
            summaries=summaries,
            root=root,
            model_name=model_name,
            metric=args.metric,
            eps=args.eps,
        )


if __name__ == "__main__":
    main()
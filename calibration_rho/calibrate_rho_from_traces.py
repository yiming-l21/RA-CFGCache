#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Generic offline rho calibration from saved CFG traces + adjacent-step rho plotting.

Expected trace layout:

    calibration_rho/
    └── {model_name}/
        └── cfg{scale}_traces/
            ├── prompt_000000/
            │   ├── step_0_xt.pt
            │   ├── step_0_cond.pt
            │   ├── step_0_uncond.pt
            │   ├── step_1_xt.pt
            │   ├── step_1_cond.pt
            │   ├── step_1_uncond.pt
            │   └── ...
            ├── prompt_000001/
            └── ...

For each prompt and each pair of steps (a, t), this script computes:

    r_c(s) = cond(s)   - xt(s)
    r_u(s) = uncond(s) - xt(s)

    rho(a, t) = cosine(
        r_u(a) - r_u(t),
        r_c(a) - r_c(t)
    )

Then it averages rho(a, t) over prompts.

Main outputs:

    calibration/{model_name}/online_rho_all_cfg.npz
    calibration/{model_name}/online_rho_all_cfg_summary.json

The merged npz contains:

    cfg_scales
    steps
    rho_table_at_all      # [N_cfg, T, T]
    rho_table_std_all     # [N_cfg, T, T]
    rho_table_count_all   # [N_cfg, T, T]

Additional plot outputs by default:

    calibration/{model_name}/adjacent_rho_curves/
        adjacent_rho_cfg{scale}.png
        adjacent_rho_cfg{scale}.csv
        adjacent_rho_all_cfg.png
        adjacent_rho_plot_summary.json

Adjacent-step rho curve means:

    rho_adjacent[t] = rho_table_at[t, t + 1]

Optional comparison with another rho npz:

    --compare_npz /export/home/liuyiming54/CFGCache/calibration/flux_rho_all_cfg.npz

Example:

    python calibration/calibrate_rho_from_traces_with_adjacent_plot.py \
        --root calibration_rho \
        --models flux \
        --cfg_scales 3.5 \
        --out_root calibration

    python calibration/calibrate_rho_from_traces_with_adjacent_plot.py \
        --root calibration_rho \
        --models flux \
        --cfg_scales auto \
        --compare_npz /export/home/liuyiming54/CFGCache/calibration/flux_rho_all_cfg.npz
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
from pathlib import Path
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

# Headless backend for servers without display.
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# -----------------------------
# Basic helpers
# -----------------------------

def parse_cfg_scale_from_dirname(name: str) -> float | None:
    """
    Parse cfg scale from directory name like:
        cfg3.5_traces
        cfg4_traces
        cfg7.50_traces
    """
    m = re.match(r"^cfg([0-9]+(?:\.[0-9]+)?)_traces$", name)
    if m is None:
        return None
    return float(m.group(1))


def cfg_tag(scale: float) -> str:
    """
    Historical default tag.

    Note:
        Some traces may be named cfg4_traces while others are cfg4.0_traces.
        The main loop uses find_cfg_trace_dir() below to try both forms.
    """
    return f"cfg{float(scale):.1f}_traces"


def compact_float(x: float) -> str:
    """3.0 -> 3, 3.5 -> 3.5"""
    return f"{float(x):g}"


def cfg_file_tag(scale: float) -> str:
    """Safe cfg string for filenames."""
    return compact_float(float(scale)).replace(".", "p")


def find_cfg_trace_dir(model_dir: Path, scale: float) -> Path:
    """
    Try both cfg3.5_traces and cfg3_traces / cfg4_traces variants.
    This keeps compatibility with your current script while avoiding cfg4.0/cfg4 mismatch.
    """
    candidates = [
        model_dir / f"cfg{float(scale):.1f}_traces",
        model_dir / f"cfg{compact_float(scale)}_traces",
    ]
    seen = set()
    unique_candidates = []
    for p in candidates:
        if str(p) not in seen:
            unique_candidates.append(p)
            seen.add(str(p))

    for p in unique_candidates:
        if p.is_dir():
            return p

    # Return historical default path for clearer warning downstream.
    return unique_candidates[0]


def step_ids_in_prompt_dir(prompt_dir: Path) -> list[int]:
    """
    Return step ids that have all three files:
        step_i_xt.pt
        step_i_cond.pt
        step_i_uncond.pt
    """
    xt_steps = set()
    cond_steps = set()
    uncond_steps = set()

    for p in prompt_dir.glob("step_*_xt.pt"):
        m = re.match(r"step_(\d+)_xt\.pt$", p.name)
        if m:
            xt_steps.add(int(m.group(1)))

    for p in prompt_dir.glob("step_*_cond.pt"):
        m = re.match(r"step_(\d+)_cond\.pt$", p.name)
        if m:
            cond_steps.add(int(m.group(1)))

    for p in prompt_dir.glob("step_*_uncond.pt"):
        m = re.match(r"step_(\d+)_uncond\.pt$", p.name)
        if m:
            uncond_steps.add(int(m.group(1)))

    return sorted(xt_steps & cond_steps & uncond_steps)


def load_tensor(path: Path, device: torch.device) -> torch.Tensor:
    """
    Load tensor safely and flatten to 1D float32 tensor.
    """
    x = torch.load(path, map_location=device)
    if not torch.is_tensor(x):
        raise TypeError(f"File does not contain a torch.Tensor: {path}")

    return x.detach().to(device=device, dtype=torch.float32).flatten()


def cosine_1d(x: torch.Tensor, y: torch.Tensor, eps: float = 1e-12) -> float:
    """
    Cosine similarity between two flattened tensors.
    If either vector is too small, return 1.0.
    """
    nx = torch.linalg.vector_norm(x)
    ny = torch.linalg.vector_norm(y)

    if nx.item() < eps or ny.item() < eps:
        return 1.0

    val = torch.dot(x, y) / (nx * ny + eps)
    val = torch.clamp(val, -1.0, 1.0)
    return float(val.item())


# -----------------------------
# Rho calibration
# -----------------------------

def compute_prompt_rho_table(
    prompt_dir: Path,
    steps: list[int],
    device: torch.device,
) -> np.ndarray:
    """
    Compute rho table for one prompt.

    Returns:
        rho_table: [T, T], where rows are anchor step a, columns are target step t.
    """
    cond_residuals: dict[int, torch.Tensor] = {}
    uncond_residuals: dict[int, torch.Tensor] = {}

    for step in steps:
        xt = load_tensor(prompt_dir / f"step_{step}_xt.pt", device)
        cond = load_tensor(prompt_dir / f"step_{step}_cond.pt", device)
        uncond = load_tensor(prompt_dir / f"step_{step}_uncond.pt", device)

        cond_residuals[step] = cond - xt
        uncond_residuals[step] = uncond - xt

    T = len(steps)
    rho = np.ones((T, T), dtype=np.float32)

    for i, step_a in enumerate(steps):
        ru_a = uncond_residuals[step_a]
        rc_a = cond_residuals[step_a]

        for j, step_t in enumerate(steps):
            if i == j:
                rho[i, j] = 1.0
                continue

            ru_t = uncond_residuals[step_t]
            rc_t = cond_residuals[step_t]

            du = ru_a - ru_t
            dc = rc_a - rc_t

            rho[i, j] = cosine_1d(du, dc)

    return rho


def calibrate_one_cfg(
    cfg_trace_dir: Path,
    device: torch.device,
    max_prompts: int | None = None,
    prompt_start: int = 0,
    step_mode: str = "intersection",
) -> tuple[list[int], np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Calibrate one cfg scale directory.

    Args:
        cfg_trace_dir:
            e.g. calibration_rho/flux/cfg3.5_traces
        step_mode:
            intersection: use steps common to all selected prompts
            first: use steps from first selected prompt and skip prompts with missing files

    Returns:
        steps
        rho_mean [T, T]
        rho_std [T, T]
        rho_count [T, T]
        summary dict
    """
    if not cfg_trace_dir.is_dir():
        raise FileNotFoundError(f"Trace dir not found: {cfg_trace_dir}")

    prompt_dirs = sorted(
        p for p in cfg_trace_dir.glob("prompt_*")
        if p.is_dir()
    )

    if prompt_start > 0:
        prompt_dirs = prompt_dirs[prompt_start:]

    if max_prompts is not None and max_prompts > 0:
        prompt_dirs = prompt_dirs[:max_prompts]

    if not prompt_dirs:
        raise RuntimeError(f"No prompt directories found in: {cfg_trace_dir}")

    prompt_steps = {}
    valid_prompt_dirs = []

    for p in prompt_dirs:
        steps = step_ids_in_prompt_dir(p)
        if len(steps) == 0:
            print(f"[WARN] Skip prompt with no complete steps: {p}")
            continue
        prompt_steps[p] = steps
        valid_prompt_dirs.append(p)

    if not valid_prompt_dirs:
        raise RuntimeError(f"No valid prompt directories found in: {cfg_trace_dir}")

    if step_mode == "intersection":
        common_steps = set(prompt_steps[valid_prompt_dirs[0]])
        for p in valid_prompt_dirs[1:]:
            common_steps &= set(prompt_steps[p])
        steps = sorted(common_steps)
    elif step_mode == "first":
        steps = prompt_steps[valid_prompt_dirs[0]]
    else:
        raise ValueError(f"Unsupported step_mode: {step_mode}")

    if len(steps) == 0:
        raise RuntimeError(
            f"No common complete steps found in {cfg_trace_dir}. "
            f"Try --step_mode first."
        )

    tables = []
    used_prompts = []
    skipped_prompts = []

    for prompt_dir in tqdm(valid_prompt_dirs, desc=f"Calibrating {cfg_trace_dir.name}"):
        have = set(prompt_steps[prompt_dir])
        need = set(steps)

        if not need.issubset(have):
            skipped_prompts.append(str(prompt_dir))
            continue

        try:
            table = compute_prompt_rho_table(
                prompt_dir=prompt_dir,
                steps=steps,
                device=device,
            )
            tables.append(table)
            used_prompts.append(str(prompt_dir))
        except Exception as e:
            print(f"[WARN] Failed on {prompt_dir}: {e!r}")
            skipped_prompts.append(str(prompt_dir))

    if not tables:
        raise RuntimeError(f"No prompt was successfully calibrated in: {cfg_trace_dir}")

    stack = np.stack(tables, axis=0).astype(np.float32)  # [P, T, T]

    rho_mean = stack.mean(axis=0)
    rho_std = stack.std(axis=0)
    rho_count = np.full_like(rho_mean, fill_value=len(tables), dtype=np.float32)

    summary = {
        "trace_dir": str(cfg_trace_dir),
        "num_prompt_dirs_found": len(prompt_dirs),
        "num_valid_prompt_dirs": len(valid_prompt_dirs),
        "num_prompts_used": len(used_prompts),
        "num_prompts_skipped": len(skipped_prompts),
        "steps": steps,
        "step_mode": step_mode,
        "used_prompts_first_10": used_prompts[:10],
        "skipped_prompts_first_10": skipped_prompts[:10],
    }

    return steps, rho_mean, rho_std, rho_count, summary


def discover_models(root: Path) -> list[str]:
    return sorted(p.name for p in root.iterdir() if p.is_dir())


def discover_cfg_scales(model_dir: Path) -> list[float]:
    scales = []
    for p in model_dir.iterdir():
        if not p.is_dir():
            continue
        s = parse_cfg_scale_from_dirname(p.name)
        if s is not None:
            scales.append(s)
    return sorted(scales)


# -----------------------------
# Adjacent rho plotting
# -----------------------------

def adjacent_rho_from_table(
    rho_table: np.ndarray,
    rho_std: np.ndarray | None = None,
    rho_count: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None]:
    """
    Extract adjacent-step rho curve from upper diagonal:

        rho_adjacent[i] = rho_table[i, i + 1]

    Returns:
        adjacent_mean, adjacent_std, adjacent_count
    """
    rho_table = np.asarray(rho_table, dtype=np.float64)
    if rho_table.ndim != 2 or rho_table.shape[0] != rho_table.shape[1]:
        raise ValueError(f"rho_table must be a square 2D matrix, got shape={rho_table.shape}")

    adj = np.diag(rho_table, k=1).astype(np.float64)

    adj_std = None
    if rho_std is not None:
        rho_std = np.asarray(rho_std, dtype=np.float64)
        adj_std = np.diag(rho_std, k=1).astype(np.float64)

    adj_count = None
    if rho_count is not None:
        rho_count = np.asarray(rho_count, dtype=np.float64)
        adj_count = np.diag(rho_count, k=1).astype(np.float64)

    return adj, adj_std, adj_count


def save_adjacent_rho_csv(
    out_csv: Path,
    steps: list[int],
    adjacent_mean: np.ndarray,
    adjacent_std: np.ndarray | None = None,
    adjacent_count: np.ndarray | None = None,
):
    """Save adjacent rho curve to csv without depending on pandas."""
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    step_a = list(steps[:-1])
    step_t = list(steps[1:])
    n = len(adjacent_mean)

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("index,step_a,step_t,rho_adjacent")
        if adjacent_std is not None:
            f.write(",rho_adjacent_std")
        if adjacent_count is not None:
            f.write(",rho_adjacent_count")
        f.write("\n")

        for i in range(n):
            a = step_a[i] if i < len(step_a) else i
            t = step_t[i] if i < len(step_t) else i + 1
            f.write(f"{i},{a},{t},{float(adjacent_mean[i])}")
            if adjacent_std is not None:
                f.write(f",{float(adjacent_std[i])}")
            if adjacent_count is not None:
                f.write(f",{float(adjacent_count[i])}")
            f.write("\n")


def plot_adjacent_rho_one_cfg(
    *,
    out_png: Path,
    model_name: str,
    cfg_scale: float,
    steps: list[int],
    adjacent_mean: np.ndarray,
    adjacent_std: np.ndarray | None = None,
    show_std_band: bool = True,
    dpi: int = 300,
):
    """Plot adjacent rho curve for one cfg scale."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    x = np.asarray(steps[:-1], dtype=np.int64)
    y = np.asarray(adjacent_mean, dtype=np.float64)

    if len(x) != len(y):
        x = np.arange(len(y))

    plt.figure(figsize=(8.5, 5.0))
    plt.plot(x, y, marker="o", linewidth=2.0, markersize=4.0, label=f"CFG={compact_float(cfg_scale)}")

    if show_std_band and adjacent_std is not None and len(adjacent_std) == len(y):
        std = np.asarray(adjacent_std, dtype=np.float64)
        lower = y - std
        upper = y + std
        plt.fill_between(x, lower, upper, alpha=0.18, label="±1 std over prompts")

    plt.xlabel("Anchor step a")
    plt.ylabel(r"Adjacent $\rho(a, a+1)$")
    plt.title(f"{model_name}: adjacent-step rho curve, CFG={compact_float(cfg_scale)}")
    plt.grid(True, alpha=0.3)
    plt.ylim(0,1)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def plot_adjacent_rho_all_cfg(
    *,
    out_png: Path,
    model_name: str,
    cfg_scales: list[float],
    steps: list[int],
    adjacent_curves: list[np.ndarray],
    dpi: int = 300,
):
    """Plot all cfg adjacent rho curves into one figure."""
    out_png.parent.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(9.5, 5.6))

    for cfg_scale, curve in zip(cfg_scales, adjacent_curves):
        y = np.asarray(curve, dtype=np.float64)
        x = np.asarray(steps[:-1], dtype=np.int64)
        if len(x) != len(y):
            x = np.arange(len(y))
        plt.plot(x, y, marker="o", linewidth=2.0, markersize=3.5, label=f"CFG={compact_float(cfg_scale)}")

    plt.xlabel("Anchor step a")
    plt.ylabel(r"Adjacent $\rho(a, a+1)$")
    plt.title(f"{model_name}: adjacent-step rho curves across CFG scales")
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.05, 1.05)
    plt.legend(ncol=2)
    plt.tight_layout()
    plt.savefig(out_png, dpi=dpi)
    plt.close()


def write_adjacent_plot_outputs(
    *,
    model_name: str,
    out_dir: Path,
    cfg_scales: list[float],
    steps: list[int],
    rho_tables: list[np.ndarray],
    rho_stds: list[np.ndarray] | None = None,
    rho_counts: list[np.ndarray] | None = None,
    show_std_band: bool = True,
    dpi: int = 300,
) -> dict[str, Any]:
    """Save per-cfg adjacent rho png/csv and all-cfg overview."""
    out_dir.mkdir(parents=True, exist_ok=True)

    adjacent_curves = []
    outputs = {
        "plot_dir": str(out_dir),
        "per_cfg": [],
        "all_cfg_png": None,
    }

    for idx, cfg_scale in enumerate(cfg_scales):
        rho_std = rho_stds[idx] if rho_stds is not None else None
        rho_count = rho_counts[idx] if rho_counts is not None else None

        adj, adj_std, adj_count = adjacent_rho_from_table(
            rho_table=rho_tables[idx],
            rho_std=rho_std,
            rho_count=rho_count,
        )
        adjacent_curves.append(adj)

        tag = cfg_file_tag(cfg_scale)
        out_png = out_dir / f"adjacent_rho_cfg{tag}.png"
        out_csv = out_dir / f"adjacent_rho_cfg{tag}.csv"

        save_adjacent_rho_csv(
            out_csv=out_csv,
            steps=steps,
            adjacent_mean=adj,
            adjacent_std=adj_std,
            adjacent_count=adj_count,
        )
        plot_adjacent_rho_one_cfg(
            out_png=out_png,
            model_name=model_name,
            cfg_scale=cfg_scale,
            steps=steps,
            adjacent_mean=adj,
            adjacent_std=adj_std,
            show_std_band=show_std_band,
            dpi=dpi,
        )

        outputs["per_cfg"].append({
            "cfg_scale": float(cfg_scale),
            "png": str(out_png),
            "csv": str(out_csv),
            "num_adjacent_pairs": int(len(adj)),
            "mean": float(np.nanmean(adj)),
            "min": float(np.nanmin(adj)),
            "max": float(np.nanmax(adj)),
        })

        print(f"[INFO] Saved adjacent rho csv: {out_csv}")
        print(f"[INFO] Saved adjacent rho png: {out_png}")

    all_png = out_dir / "adjacent_rho_all_cfg.png"
    plot_adjacent_rho_all_cfg(
        out_png=all_png,
        model_name=model_name,
        cfg_scales=cfg_scales,
        steps=steps,
        adjacent_curves=adjacent_curves,
        dpi=dpi,
    )
    outputs["all_cfg_png"] = str(all_png)
    print(f"[INFO] Saved all-cfg adjacent rho png: {all_png}")

    summary_json = out_dir / "adjacent_rho_plot_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved adjacent rho plot summary: {summary_json}")

    return outputs


# -----------------------------
# Optional comparison npz plotting
# -----------------------------

def find_first_existing_key(data: np.lib.npyio.NpzFile, candidates: list[str]) -> str | None:
    for k in candidates:
        if k in data.files:
            return k
    return None


def load_rho_npz_for_compare(npz_path: Path) -> tuple[list[float], list[int], list[np.ndarray]]:
    """
    Load common rho npz formats for optional comparison.

    Supported merged keys include:
        cfg_scales + steps + rho_table_at_all
        cfg_scales + steps + rho_table_all
        cfg_scales + steps + rho_all
        cfg_scales + steps + rho

    Also supports a single 2D rho table with rho_table_at / rho.
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"compare npz not found: {npz_path}")

    data = np.load(npz_path, allow_pickle=True)
    try:
        cfg_key = find_first_existing_key(data, ["cfg_scales", "cfg_scale", "true_cfg_scales", "scales", "cfgs"])
        steps_key = find_first_existing_key(data, ["steps", "step_ids", "timesteps"])
        rho_key = find_first_existing_key(
            data,
            [
                "rho_table_at_all",
                "rho_table_all",
                "rho_tables",
                "rho_mean_all",
                "rho_all",
                "rho",
                "rho_table_at",
                "rho_table",
            ],
        )

        if rho_key is None:
            raise KeyError(f"No rho table key found in {npz_path}. keys={list(data.files)}")

        rho = np.asarray(data[rho_key], dtype=np.float32)

        if cfg_key is not None:
            cfg_arr = np.asarray(data[cfg_key]).reshape(-1)
            cfg_scales = [float(x) for x in cfg_arr.tolist()]
        else:
            cfg_scales = []

        if steps_key is not None:
            steps = [int(x) for x in np.asarray(data[steps_key]).reshape(-1).tolist()]
        else:
            if rho.ndim == 2:
                steps = list(range(rho.shape[0]))
            elif rho.ndim == 3:
                steps = list(range(rho.shape[-1]))
            else:
                raise ValueError(f"Cannot infer steps from rho shape={rho.shape}")

        tables: list[np.ndarray] = []

        if rho.ndim == 2:
            if not cfg_scales:
                cfg_scales = [0.0]
            tables = [rho]
        elif rho.ndim == 3:
            # [C, T, T]
            if rho.shape[-1] == rho.shape[-2]:
                if not cfg_scales:
                    cfg_scales = [float(i) for i in range(rho.shape[0])]
                if len(cfg_scales) != rho.shape[0]:
                    raise ValueError(
                        f"cfg_scales length {len(cfg_scales)} does not match rho first dim {rho.shape[0]}"
                    )
                tables = [rho[i] for i in range(rho.shape[0])]
            # [T, T, C]
            elif rho.shape[0] == rho.shape[1]:
                if not cfg_scales:
                    cfg_scales = [float(i) for i in range(rho.shape[-1])]
                if len(cfg_scales) != rho.shape[-1]:
                    raise ValueError(
                        f"cfg_scales length {len(cfg_scales)} does not match rho last dim {rho.shape[-1]}"
                    )
                tables = [rho[:, :, i] for i in range(rho.shape[-1])]
            else:
                raise ValueError(f"Unsupported 3D rho shape={rho.shape}")
        else:
            raise ValueError(f"Unsupported rho ndim={rho.ndim}, shape={rho.shape}")

        return cfg_scales, steps, tables
    finally:
        data.close()


def find_matching_cfg_index(cfg_scales: list[float], target: float, tol: float = 1e-5) -> int | None:
    for i, cfg in enumerate(cfg_scales):
        if abs(float(cfg) - float(target)) <= tol:
            return i
    return None


def plot_compare_adjacent_rho(
    *,
    out_dir: Path,
    model_name: str,
    online_cfg_scales: list[float],
    online_steps: list[int],
    online_rho_tables: list[np.ndarray],
    compare_npz: Path,
    compare_label: str = "compare",
    dpi: int = 300,
) -> dict[str, Any]:
    """Overlay online adjacent rho curves with another npz file."""
    out_dir.mkdir(parents=True, exist_ok=True)

    cmp_cfg_scales, cmp_steps, cmp_tables = load_rho_npz_for_compare(compare_npz)

    outputs = {
        "compare_npz": str(compare_npz),
        "compare_label": compare_label,
        "per_cfg": [],
        "all_cfg_png": None,
    }

    common_items = []
    for i, cfg in enumerate(online_cfg_scales):
        j = find_matching_cfg_index(cmp_cfg_scales, cfg)
        if j is not None:
            common_items.append((i, j, float(cfg)))

    if not common_items:
        print(f"[WARN] No common cfg scales between online result and compare npz: {compare_npz}")
        print(f"       online cfg:  {online_cfg_scales}")
        print(f"       compare cfg: {cmp_cfg_scales}")
        return outputs

    plt.figure(figsize=(10.0, 5.8))

    for i, j, cfg in common_items:
        online_adj, _, _ = adjacent_rho_from_table(online_rho_tables[i])
        compare_adj, _, _ = adjacent_rho_from_table(cmp_tables[j])

        min_len = min(len(online_adj), len(compare_adj))
        online_adj = online_adj[:min_len]
        compare_adj = compare_adj[:min_len]
        diff = online_adj - compare_adj

        x = np.asarray(online_steps[:-1], dtype=np.int64)[:min_len]
        if len(x) != min_len:
            x = np.arange(min_len)

        # Per-cfg comparison png
        tag = cfg_file_tag(cfg)
        per_png = out_dir / f"adjacent_rho_compare_cfg{tag}.png"
        per_csv = out_dir / f"adjacent_rho_compare_cfg{tag}.csv"

        plt_one = plt.figure(figsize=(8.8, 5.0))
        ax = plt_one.add_subplot(111)
        ax.plot(x, online_adj, marker="o", linewidth=2.0, markersize=4.0, label="online")
        ax.plot(x, compare_adj, marker="s", linewidth=2.0, markersize=4.0, label=compare_label)
        ax.set_xlabel("Anchor step a")
        ax.set_ylabel(r"Adjacent $\rho(a, a+1)$")
        ax.set_title(f"{model_name}: online vs {compare_label}, CFG={compact_float(cfg)}")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0,1)
        ax.legend()
        plt_one.tight_layout()
        plt_one.savefig(per_png, dpi=dpi)
        plt.close(plt_one)

        with open(per_csv, "w", encoding="utf-8") as f:
            f.write("index,step_a,online_rho,compare_rho,diff_online_minus_compare,abs_diff\n")
            for k in range(min_len):
                f.write(
                    f"{k},{int(x[k])},{float(online_adj[k])},{float(compare_adj[k])},"
                    f"{float(diff[k])},{float(abs(diff[k]))}\n"
                )

        corr = float(np.corrcoef(online_adj, compare_adj)[0, 1]) if min_len > 1 else float("nan")
        outputs["per_cfg"].append({
            "cfg_scale": cfg,
            "png": str(per_png),
            "csv": str(per_csv),
            "mean_abs_diff": float(np.nanmean(np.abs(diff))),
            "max_abs_diff": float(np.nanmax(np.abs(diff))),
            "rmse": float(np.sqrt(np.nanmean(diff ** 2))),
            "corr": corr,
        })

        print(f"[INFO] Saved compare adjacent rho png: {per_png}")
        print(f"[INFO] Saved compare adjacent rho csv: {per_csv}")
        print(
            f"[INFO] Compare CFG={compact_float(cfg)}: "
            f"mean_abs_diff={np.nanmean(np.abs(diff)):.6f}, "
            f"max_abs_diff={np.nanmax(np.abs(diff)):.6f}, corr={corr:.6f}"
        )

        # Add to all-cfg overview, online solid, compare dashed.
        plt.plot(x, online_adj, linewidth=2.0, label=f"online CFG={compact_float(cfg)}")
        plt.plot(x, compare_adj, linewidth=2.0, linestyle="--", label=f"{compare_label} CFG={compact_float(cfg)}")

    all_png = out_dir / "adjacent_rho_compare_all_cfg.png"
    plt.xlabel("Anchor step a")
    plt.ylabel(r"Adjacent $\rho(a, a+1)$")
    plt.title(f"{model_name}: adjacent-step rho comparison")
    plt.grid(True, alpha=0.3)
    plt.ylim(-1.05, 1.05)
    plt.legend(ncol=2, fontsize=8)
    plt.tight_layout()
    plt.savefig(all_png, dpi=dpi)
    plt.close()

    outputs["all_cfg_png"] = str(all_png)
    print(f"[INFO] Saved all-cfg compare adjacent rho png: {all_png}")

    summary_json = out_dir / "adjacent_rho_compare_summary.json"
    with open(summary_json, "w", encoding="utf-8") as f:
        json.dump(outputs, f, indent=2, ensure_ascii=False)
    print(f"[INFO] Saved adjacent rho compare summary: {summary_json}")

    return outputs


# -----------------------------
# Main
# -----------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Generic rho calibration from CFG traces, with adjacent rho curve plotting."
    )

    parser.add_argument(
        "--root",
        type=str,
        default="calibration_rho",
        help="Root directory containing {model}/cfg{scale}_traces.",
    )
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Model names under root, e.g. flux qwen. Use 'all' to auto-discover.",
    )
    parser.add_argument(
        "--cfg_scales",
        nargs="+",
        default=["auto"],
        help="CFG scales, e.g. 1.5 3.5 5.0. Use 'auto' to discover all.",
    )
    parser.add_argument(
        "--out_root",
        type=str,
        default="calibration",
        help="Output root directory.",
    )
    parser.add_argument(
        "--max_prompts",
        type=int,
        default=None,
        help="Max prompts to use per cfg scale. Default: use all.",
    )
    parser.add_argument(
        "--prompt_start",
        type=int,
        default=0,
        help="Prompt directory start offset after sorting.",
    )
    parser.add_argument(
        "--step_mode",
        type=str,
        default="intersection",
        choices=["intersection", "first"],
        help="How to choose step ids.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="cpu or cuda. CPU is safer for large traces.",
    )
    parser.add_argument(
        "--also_save_per_cfg",
        action="store_true",
        help="Also save one npz per cfg scale.",
    )

    # New plot arguments.
    parser.add_argument(
        "--plot_adjacent_rho",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to plot adjacent rho curves after calibration. Default: True.",
    )
    parser.add_argument(
        "--plot_std_band",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Whether to draw ±1 std band for per-cfg adjacent rho curves. Default: True.",
    )
    parser.add_argument(
        "--plot_dpi",
        type=int,
        default=300,
        help="DPI for saved figures. Default: 300.",
    )
    parser.add_argument(
        "--plot_dir",
        type=str,
        default=None,
        help="Optional plot directory. Default: {out_root}/{model}/adjacent_rho_curves.",
    )

    # Optional comparison with another rho npz.
    parser.add_argument(
        "--compare_npz",
        type=str,
        default=None,
        help="Optional rho npz to overlay with the newly calibrated online rho curves.",
    )
    parser.add_argument(
        "--compare_label",
        type=str,
        default="offline",
        help="Legend label for --compare_npz. Default: offline.",
    )

    args = parser.parse_args()

    root = Path(args.root).resolve()
    out_root = Path(args.out_root).resolve()
    device = torch.device(args.device)

    if not root.is_dir():
        raise FileNotFoundError(f"Root dir not found: {root}")

    if args.models == ["all"]:
        models = discover_models(root)
    else:
        models = args.models

    if not models:
        raise RuntimeError(f"No model directories found under: {root}")

    print("========================================")
    print("Generic rho calibration")
    print(f"root:       {root}")
    print(f"models:     {models}")
    print(f"cfg_scales: {args.cfg_scales}")
    print(f"out_root:   {out_root}")
    print(f"device:     {device}")
    print(f"plot_adjacent_rho: {args.plot_adjacent_rho}")
    print(f"compare_npz: {args.compare_npz}")
    print("========================================")

    for model_name in models:
        model_dir = root / model_name
        if not model_dir.is_dir():
            print(f"[WARN] Skip missing model dir: {model_dir}")
            continue

        if args.cfg_scales == ["auto"]:
            cfg_scales = discover_cfg_scales(model_dir)
        else:
            cfg_scales = [float(x) for x in args.cfg_scales]

        if not cfg_scales:
            print(f"[WARN] No cfg trace dirs found for model: {model_name}")
            continue

        print(f"\n========== Model: {model_name} ==========")
        print(f"cfg_scales: {cfg_scales}")

        all_steps = None
        rho_tables = []
        rho_stds = []
        rho_counts = []
        summaries = []

        for scale in cfg_scales:
            trace_dir = find_cfg_trace_dir(model_dir, scale)

            if not trace_dir.is_dir():
                print(f"[WARN] Missing trace dir, skip: {trace_dir}")
                continue

            steps, rho_mean, rho_std, rho_count, summary = calibrate_one_cfg(
                cfg_trace_dir=trace_dir,
                device=device,
                max_prompts=args.max_prompts,
                prompt_start=args.prompt_start,
                step_mode=args.step_mode,
            )

            if all_steps is None:
                all_steps = steps
            else:
                if steps != all_steps:
                    raise RuntimeError(
                        f"Step ids mismatch for model={model_name}, cfg={scale}. "
                        f"Expected {all_steps}, got {steps}. "
                        f"Use consistent num_steps traces or run cfg scales separately."
                    )

            rho_tables.append(rho_mean)
            rho_stds.append(rho_std)
            rho_counts.append(rho_count)
            summaries.append({
                "cfg_scale": float(scale),
                **summary,
            })

            if args.also_save_per_cfg:
                per_cfg_out_dir = out_root / model_name
                per_cfg_out_dir.mkdir(parents=True, exist_ok=True)

                per_cfg_path = per_cfg_out_dir / f"rho_cfg{compact_float(scale)}.npz"
                np.savez_compressed(
                    per_cfg_path,
                    cfg_scale=np.array([float(scale)], dtype=np.float32),
                    steps=np.array(steps, dtype=np.int64),
                    rho_table_at=rho_mean.astype(np.float32),
                    rho_table_std=rho_std.astype(np.float32),
                    rho_table_count=rho_count.astype(np.float32),
                )
                print(f"[INFO] Saved per-cfg rho table: {per_cfg_path}")

        if not rho_tables:
            print(f"[WARN] No cfg scale successfully calibrated for model: {model_name}")
            continue

        model_out_dir = out_root / model_name
        model_out_dir.mkdir(parents=True, exist_ok=True)

        out_npz = model_out_dir / "online_rho_all_cfg.npz"
        out_json = model_out_dir / "online_rho_all_cfg_summary.json"

        used_cfg_scales = [s["cfg_scale"] for s in summaries]

        np.savez_compressed(
            out_npz,
            cfg_scales=np.array(used_cfg_scales, dtype=np.float32),
            steps=np.array(all_steps, dtype=np.int64),
            rho_table_at_all=np.stack(rho_tables, axis=0).astype(np.float32),
            rho_table_std_all=np.stack(rho_stds, axis=0).astype(np.float32),
            rho_table_count_all=np.stack(rho_counts, axis=0).astype(np.float32),
        )

        payload = {
            "model_name": model_name,
            "root": str(root),
            "out_npz": str(out_npz),
            "cfg_scales": used_cfg_scales,
            "steps": all_steps,
            "num_cfg_scales": len(used_cfg_scales),
            "summaries": summaries,
        }

        with open(out_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Saved merged rho table: {out_npz}")
        print(f"[INFO] Saved summary:          {out_json}")

        if args.plot_adjacent_rho:
            if args.plot_dir is not None:
                # If multiple models are processed, avoid overwriting by nesting model name.
                base_plot_dir = Path(args.plot_dir).resolve()
                plot_dir = base_plot_dir if len(models) == 1 else base_plot_dir / model_name
            else:
                plot_dir = model_out_dir / "adjacent_rho_curves"

            write_adjacent_plot_outputs(
                model_name=model_name,
                out_dir=plot_dir,
                cfg_scales=used_cfg_scales,
                steps=all_steps,
                rho_tables=rho_tables,
                rho_stds=rho_stds,
                rho_counts=rho_counts,
                show_std_band=args.plot_std_band,
                dpi=args.plot_dpi,
            )

            if args.compare_npz is not None:
                compare_dir = plot_dir / "compare"
                plot_compare_adjacent_rho(
                    out_dir=compare_dir,
                    model_name=model_name,
                    online_cfg_scales=used_cfg_scales,
                    online_steps=all_steps,
                    online_rho_tables=rho_tables,
                    compare_npz=Path(args.compare_npz).resolve(),
                    compare_label=args.compare_label,
                    dpi=args.plot_dpi,
                )


if __name__ == "__main__":
    main()

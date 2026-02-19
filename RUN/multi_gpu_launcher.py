#!/usr/bin/env python3

"""Launch multiple sample.sh runs across GPUs.

This launcher supports multiple backends, but all backends are launched through
scripts/sample.sh.

Supported backends:
    - flux
    - wan
    - cogvideox

Responsibilities:
    1. Split prompt file into GPU shards.
    2. Launch one sample.sh process per GPU.
    3. Pass backend/mode/prompt_file/output_dir/start_index to sample.sh.
    4. Merge all shard outputs into one final directory.

Important:
    --mode is only an output-stage label, such as "single", "cfg", or "calib".
    It is not a cache mode and does not enable any cache logic by itself.
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, List


PROJECT_ROOT = Path(__file__).resolve().parent.parent
SAMPLE_SH = PROJECT_ROOT / "scripts" / "sample.sh"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "resources" / "prompts" / "prompt.txt"

SUPPORTED_BACKENDS = {"flux", "wan", "cogvideox"}

# Common numbered outputs.
# FLUX usually: img_000000.png / img_000000.jpg
# Wan / CogVideoX usually: video_000000.mp4, often under videos/
NUMBERED_OUTPUT_PATTERN = re.compile(
    r"(?:img|image|video)_(\d+)\.(?:png|jpe?g|webp|mp4|gif)$",
    re.IGNORECASE,
)


@dataclass
class ShardJob:
    gpu: str
    prompts: List[str]
    start_index: int
    prompt_file: Path
    output_dir: Path
    process: subprocess.Popen | None = None
    full_output_dir: Path | None = None


def normalize_backend(name: str) -> str:
    """
    Normalize user-facing backend names.

    We accept wan / wan2.1 / wan2_1 / wan2.2 / wan2_2 for convenience,
    but always forward "wan" to scripts/sample.sh.
    """
    name = name.strip().lower().replace("-", "_")

    aliases = {
        "flux": "flux",
        "wan": "wan",
        "wan2": "wan",
        "wan21": "wan",
        "wan2_1": "wan",
        "wan2.1": "wan",
        "wan22": "wan",
        "wan2_2": "wan",
        "wan2.2": "wan",
        "cogvideo": "cogvideox",
        "cogvideo_x": "cogvideox",
        "cogvideox": "cogvideox",
    }

    if name not in aliases:
        raise ValueError(
            f"Unsupported backend: {name}. "
            f"Supported: {', '.join(sorted(SUPPORTED_BACKENDS))}"
        )

    backend = aliases[name]
    if backend not in SUPPORTED_BACKENDS:
        raise ValueError(f"Unsupported normalized backend: {backend}")

    return backend


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch sample.sh on multiple GPUs for flux/wan/cogvideox"
    )

    parser.add_argument(
        "--backend",
        default="flux",
        choices=[
            "flux",
            "wan",
            "wan2",
            "wan21",
            "wan2_1",
            "wan2.1",
            "wan22",
            "wan2_2",
            "wan2.2",
            "cogvideox",
        ],
        help="Backend to launch through scripts/sample.sh.",
    )

    parser.add_argument(
        "--mode",
        default="single",
        help="Output stage label only, e.g. single, cfg, calib_rho. Not a cache mode.",
    )

    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help="Path to active prompt list.",
    )

    parser.add_argument(
        "--full-prompt-file",
        help="Path to master prompt list for manifest. Defaults to --prompt-file.",
    )

    parser.add_argument(
        "--gpus",
        help="Comma-separated GPU ids, e.g. 0,1,3.",
    )

    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Number of GPUs to use starting from 0 when --gpus is not provided.",
    )

    parser.add_argument(
        "--base-output-dir",
        default=None,
        help=(
            "Base output directory. "
            "Defaults to results/<backend> if not specified."
        ),
    )

    parser.add_argument(
        "--run-name",
        help="Optional name appended to output directories. Defaults to timestamp.",
    )

    parser.add_argument(
        "--keep-temp",
        action="store_true",
        help="Keep generated prompt shards under RUN/tmp_multi_gpu_*.",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print commands without executing them.",
    )

    parser.add_argument(
        "--report-path",
        help="Optional JSON file path to record the final merged output directory.",
    )

    parser.add_argument(
        "--start-offset",
        type=int,
        default=0,
        help="Global start index offset for numbering/resume.",
    )

    parser.add_argument(
        "sample_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to sample.sh after --.",
    )

    args = parser.parse_args()
    args.backend = normalize_backend(args.backend)
    return args


def detect_gpus(gpu_arg: str | None, num_gpus: int | None) -> List[str]:
    if gpu_arg:
        return [gpu.strip() for gpu in gpu_arg.split(",") if gpu.strip()]

    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        tokens = [token.strip() for token in env.split(",") if token.strip()]
        if tokens:
            if num_gpus is not None:
                return tokens[:num_gpus]
            return tokens

    if num_gpus is not None:
        return [str(i) for i in range(num_gpus)]

    try:
        import torch

        count = torch.cuda.device_count()
    except Exception:
        count = 0

    if count > 0:
        return [str(i) for i in range(count)]

    return ["0"]


def read_prompts(path: Path) -> List[str]:
    with path.open("r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]

    if not prompts:
        raise ValueError(f"Prompt file '{path}' is empty")

    return prompts


def split_prompts(prompts: List[str], max_shards: int) -> List[tuple[int, List[str]]]:
    if max_shards <= 0:
        raise ValueError("Number of shards must be positive")

    total = len(prompts)
    if total == 0:
        raise ValueError("Prompt list is empty")

    shards = min(max_shards, total)
    base = total // shards
    remainder = total % shards

    result: List[tuple[int, List[str]]] = []
    start = 0

    for shard_idx in range(shards):
        size = base + (1 if shard_idx < remainder else 0)
        end = start + size
        chunk = prompts[start:end]

        if chunk:
            result.append((start, chunk))

        start = end

    return result


def _flag_aliases(flag: str) -> set[str]:
    if flag.startswith("--"):
        core = flag[2:]
        return {"--" + core, "--" + core.replace("_", "-")}
    return {flag}


def sanitize_sample_args(sample_args: Iterable[str]) -> List[str]:
    args = list(sample_args)

    if args and args[0] == "--":
        args = args[1:]

    # Launcher manages these arguments.
    managed_flags = {
        "--backend",
        "--mode",
        "--prompt_file",
        "--prompt-file",
        "--output_dir",
        "--output-dir",
        "--start_index",
        "--start-index",
    }

    for token in args:
        for flag in managed_flags:
            aliases = _flag_aliases(flag)
            for alias in aliases:
                if token == alias or token.startswith(alias + "="):
                    raise ValueError(
                        f"Launcher manages '{alias}'; remove it from forwarded arguments: {args}"
                    )

    return args


def extract_arg_value(args: List[str], flag: str) -> str | None:
    aliases = _flag_aliases(flag)
    i = 0

    while i < len(args):
        tok = args[i]

        if tok in aliases:
            if i + 1 >= len(args):
                raise ValueError(f"{tok} requires a value")
            return args[i + 1]

        for alias in aliases:
            if tok.startswith(alias + "="):
                return tok.split("=", 1)[1]

        i += 1

    return None


def replace_arg_value(args: List[str], flag: str, new_value: str) -> List[str]:
    aliases = _flag_aliases(flag)
    out: List[str] = []
    i = 0
    replaced = False

    while i < len(args):
        tok = args[i]

        if tok in aliases:
            if i + 1 >= len(args):
                raise ValueError(f"{tok} requires a value")
            out.extend([tok, new_value])
            i += 2
            replaced = True
            continue

        matched = False
        for alias in aliases:
            if tok.startswith(alias + "="):
                out.append(alias + "=" + new_value)
                i += 1
                replaced = True
                matched = True
                break

        if matched:
            continue

        out.append(tok)
        i += 1

    if not replaced:
        out.extend([flag, new_value])

    return out


def build_output_dir(stage_root: Path, gpu: str) -> Path:
    return stage_root / f"gpu{gpu}"


def write_prompt_shard(prompts: List[str], directory: Path, name: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    safe_name = str(name).replace("/", "_").replace(",", "_")
    shard_path = directory / f"prompts_{safe_name}.txt"

    with shard_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(prompts))
        f.write("\n")

    return shard_path


def launch_process(
    *,
    backend: str,
    gpu: str,
    prompt_file: Path,
    mode: str,
    output_dir: Path,
    start_index: int,
    extra_args: List[str],
    dry_run: bool,
) -> subprocess.Popen | None:
    """
    All backends go through scripts/sample.sh.
    """
    cmd = [
        "bash",
        str(SAMPLE_SH),
        "--backend",
        backend,
        "--mode",
        mode,
        "--prompt_file",
        str(prompt_file),
        "--output_dir",
        str(output_dir),
        "--start_index",
        str(start_index),
    ] + extra_args

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("PYTHON_BIN", sys.executable)

    print("[LAUNCH] GPU", gpu, "->", " ".join(cmd))

    if dry_run:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, cwd=PROJECT_ROOT, env=env)


def find_full_output_dir(job_output_dir: Path) -> Path:
    """
    sample.sh is expected to write .full_output_dir.

    If it does not exist, we fallback to job_output_dir itself. This makes the
    launcher usable even if a backend has not implemented the marker yet.
    """
    marker = job_output_dir / ".full_output_dir"

    if marker.exists():
        full_output_dir = Path(marker.read_text(encoding="utf-8").strip()).resolve()

        if not full_output_dir.exists():
            raise FileNotFoundError(
                f"Expected output directory '{full_output_dir}' from marker not found"
            )

        return full_output_dir

    print(f"[WARN] Missing .full_output_dir marker in {job_output_dir}; using shard output dir itself.")
    return job_output_dir.resolve()


def iter_files_recursive(root: Path) -> List[Path]:
    files: List[Path] = []

    if not root.exists():
        return files

    for path in sorted(root.rglob("*")):
        if path.is_file():
            files.append(path)

    return files


def numbered_index_from_path(path: Path) -> int | None:
    match = NUMBERED_OUTPUT_PATTERN.fullmatch(path.name)
    if not match:
        return None
    return int(match.group(1))


def copy_tree_merge(src_root: Path, dst_root: Path):
    """
    Merge src_root into dst_root recursively.

    If a destination file already exists, keep both by suffixing the filename.
    """
    for src in iter_files_recursive(src_root):
        rel = src.relative_to(src_root)
        dst = dst_root / rel
        dst.parent.mkdir(parents=True, exist_ok=True)

        if not dst.exists():
            shutil.copy2(src, dst)
            continue

        stem = dst.stem
        suffix = dst.suffix
        parent = dst.parent

        k = 1
        while True:
            candidate = parent / f"{stem}_dup{k}{suffix}"
            if not candidate.exists():
                shutil.copy2(src, candidate)
                break
            k += 1


def aggregate_outputs(
    jobs: List[ShardJob],
    aggregated_root: Path,
    master_prompts: List[str],
) -> Path | None:
    if not jobs:
        return None

    if aggregated_root.exists():
        shutil.rmtree(aggregated_root)

    aggregated_root.mkdir(parents=True, exist_ok=True)

    prompt_path = aggregated_root / "prompts.txt"
    prompt_path.write_text("\n".join(master_prompts) + "\n", encoding="utf-8")

    manifest: List[dict] = []

    for job in sorted(jobs, key=lambda j: j.start_index):
        full_output_dir = find_full_output_dir(job.output_dir)
        job.full_output_dir = full_output_dir

        # Preserve the backend/sample.sh internal directory structure.
        try:
            rel_subpath = full_output_dir.relative_to(job.output_dir)
        except ValueError:
            rel_subpath = Path(full_output_dir.name)

        if str(rel_subpath) in (".", ""):
            dest_root = aggregated_root
        else:
            dest_root = aggregated_root / rel_subpath

        dest_root.mkdir(parents=True, exist_ok=True)

        # Copy everything, including videos/, metadata/, eval_frames/, jsonl, etc.
        copy_tree_merge(full_output_dir, dest_root)

        # Build a lightweight manifest for numbered image/video outputs.
        for src_path in iter_files_recursive(full_output_dir):
            index = numbered_index_from_path(src_path)
            if index is None:
                continue

            rel = src_path.relative_to(full_output_dir)
            dest_path = dest_root / rel
            prompt_text = master_prompts[index] if index < len(master_prompts) else None

            manifest.append(
                {
                    "index": index,
                    "prompt": prompt_text,
                    "source_gpu": job.gpu,
                    "source_dir": str(full_output_dir),
                    "source_file": str(src_path),
                    "dest_file": str(dest_path),
                    "relative_path": str(rel),
                }
            )

    manifest.sort(key=lambda entry: (entry["index"], entry["dest_file"]))

    manifest_path = aggregated_root / "merged_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    return aggregated_root


def move_aggregated_to_target(
    *,
    aggregated_root: Path,
    target_mode_dir: Path,
    timestamp_label: str,
) -> Path:
    """
    Move merged outputs from temporary aggregated_root to final target_mode_dir.
    """
    target_mode_dir.mkdir(parents=True, exist_ok=True)

    for child in sorted(aggregated_root.iterdir()):
        dest = target_mode_dir / child.name

        if dest.exists():
            dest = target_mode_dir / f"{child.name}_{timestamp_label}"

        shutil.move(str(child), str(dest))

    return target_mode_dir


def write_report(
    *,
    report_path: str | None,
    success: bool,
    final_output_path: Path | None,
    timestamp_label: str,
    target_mode_dir: Path,
    backend: str,
    mode: str,
):
    if not report_path:
        return

    report_payload = {
        "success": success,
        "backend": backend,
        "mode": mode,
        "final_output_path": str(final_output_path) if final_output_path else None,
        "timestamp_label": timestamp_label,
        "target_mode_dir": str(target_mode_dir),
    }

    report_file = Path(report_path)
    report_file.parent.mkdir(parents=True, exist_ok=True)
    report_file.write_text(
        json.dumps(report_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def main() -> int:
    args = parse_args()

    if not SAMPLE_SH.exists():
        raise FileNotFoundError(f"Missing sample.sh: {SAMPLE_SH}")

    gpu_list = detect_gpus(args.gpus, args.num_gpus)
    if not gpu_list:
        raise RuntimeError("No GPU devices available")

    prompt_path = Path(args.prompt_file)
    full_prompt_path = Path(args.full_prompt_file) if args.full_prompt_file else prompt_path

    master_prompts = read_prompts(full_prompt_path)
    prompt_list = read_prompts(prompt_path)

    start_offset = max(int(args.start_offset or 0), 0)

    if start_offset >= len(master_prompts):
        raise ValueError(
            f"Start offset {start_offset} exceeds available master prompts ({len(master_prompts)})."
        )

    if args.full_prompt_file:
        # prompt_file may already be a filtered temporary prompt list.
        active_prompts = prompt_list
    else:
        if start_offset >= len(prompt_list):
            raise ValueError("Start offset exceeds prompt file length.")
        active_prompts = prompt_list[start_offset:]

    if not active_prompts:
        raise ValueError("No prompts remain after applying prompt filtering.")

    shard_specs = split_prompts(active_prompts, len(gpu_list))

    if len(shard_specs) < len(gpu_list):
        unused_gpus = gpu_list[len(shard_specs):]
        if unused_gpus:
            print(f"[INFO] GPUs {', '.join(unused_gpus)} idle; no prompts assigned")

    extra_args = sanitize_sample_args(args.sample_args)

    # -----------------------------
    # Shard negative_prompt_file if provided.
    # This is needed for true-CFG / per-prompt negatives.
    # -----------------------------
    neg_file = extract_arg_value(extra_args, "--negative_prompt_file")
    neg_prompts_active: List[str] | None = None

    if neg_file:
        neg_path = Path(neg_file)

        if not neg_path.is_absolute():
            neg_path = (PROJECT_ROOT / neg_path).resolve()

        if not neg_path.exists():
            raise FileNotFoundError(f"negative_prompt_file not found: {neg_path}")

        neg_prompts_active = read_prompts(neg_path)

        if len(neg_prompts_active) != len(active_prompts):
            raise ValueError(
                f"negative_prompt_file lines ({len(neg_prompts_active)}) != active prompts ({len(active_prompts)}). "
                "They must align 1:1 after preprocessing."
            )

    base_output_dir = (
        Path(args.base_output_dir).resolve()
        if args.base_output_dir
        else (PROJECT_ROOT / "results" / args.backend).resolve()
    )
    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    mode_label = args.mode.lower()

    stage_root = base_output_dir / ".multi_gpu_tmp" / f"{mode_label}_{timestamp_label}"
    stage_root.mkdir(parents=True, exist_ok=True)

    final_mode_dir_name = mode_label if not args.run_name else f"{mode_label}_{args.run_name}"
    target_mode_dir = base_output_dir / final_mode_dir_name

    aggregated_root = stage_root / "merged"

    tmp_root = Path(
        tempfile.mkdtemp(
            prefix="tmp_multi_gpu_",
            dir=PROJECT_ROOT / "RUN",
        )
    )

    exit_code = 0
    jobs: List[ShardJob] = []
    final_output_path: Path | None = None
    success = False

    try:
        print("========== Multi-GPU Launcher ==========")
        print(f"backend:           {args.backend}")
        print(f"mode:              {mode_label}")
        print(f"gpus:              {','.join(gpu_list)}")
        print(f"num_shards:        {len(shard_specs)}")
        print(f"active_prompts:    {len(active_prompts)}")
        print(f"start_offset:      {start_offset}")
        print(f"base_output_dir:   {base_output_dir}")
        print(f"run_name:          {args.run_name}")
        print(f"dry_run:           {args.dry_run}")
        print("========================================")

        for gpu, (start_index, shard_prompts) in zip(gpu_list, shard_specs):
            shard_file = write_prompt_shard(shard_prompts, tmp_root, f"gpu{gpu}")
            shard_output = build_output_dir(stage_root, gpu)
            global_start = start_offset + start_index

            job = ShardJob(
                gpu=gpu,
                prompts=shard_prompts,
                start_index=global_start,
                prompt_file=shard_file,
                output_dir=shard_output,
            )

            shard_extra_args = list(extra_args)

            if neg_prompts_active is not None:
                shard_negs = neg_prompts_active[start_index : start_index + len(shard_prompts)]

                if len(shard_negs) != len(shard_prompts):
                    raise ValueError("Negative shard length mismatch.")

                shard_neg_file = write_prompt_shard(shard_negs, tmp_root, f"gpu{gpu}_neg")
                shard_extra_args = replace_arg_value(
                    shard_extra_args,
                    "--negative_prompt_file",
                    str(shard_neg_file),
                )

            job.process = launch_process(
                backend=args.backend,
                gpu=gpu,
                prompt_file=shard_file,
                mode=mode_label,
                output_dir=shard_output,
                start_index=global_start,
                extra_args=shard_extra_args,
                dry_run=args.dry_run,
            )

            jobs.append(job)

        if args.dry_run:
            print("[DRY-RUN] No commands executed.")
            success = True
            return 0

        for job in jobs:
            if job.process is None:
                continue

            ret = job.process.wait()
            if ret != 0:
                print(
                    f"[ERROR] Process on GPU {job.gpu} exited with code {ret}",
                    file=sys.stderr,
                )
                exit_code = ret if exit_code == 0 else exit_code

        if exit_code != 0:
            return exit_code

        merged_dir = aggregate_outputs(
            jobs=jobs,
            aggregated_root=aggregated_root,
            master_prompts=master_prompts,
        )

        if merged_dir is not None:
            final_output_path = move_aggregated_to_target(
                aggregated_root=merged_dir,
                target_mode_dir=target_mode_dir,
                timestamp_label=timestamp_label,
            )
            print(f"[MERGE] Aggregated outputs merged into: {final_output_path}")
        else:
            print("[MERGE] No aggregated outputs were produced.")

        success = final_output_path is not None

        # Cleanup shard output dirs.
        for job in jobs:
            try:
                if job.output_dir.exists():
                    shutil.rmtree(job.output_dir, ignore_errors=True)
                    print(f"[CLEANUP] Removed shard directory: {job.output_dir}")
            except Exception as cleanup_err:
                print(
                    f"[WARN] Failed to remove shard directory {job.output_dir}: {cleanup_err}"
                )

        return 0

    finally:
        write_report(
            report_path=args.report_path,
            success=success,
            final_output_path=final_output_path,
            timestamp_label=timestamp_label,
            target_mode_dir=target_mode_dir,
            backend=args.backend,
            mode=mode_label,
        )

        if not args.keep_temp and tmp_root.exists():
            shutil.rmtree(tmp_root, ignore_errors=True)

        if exit_code == 0:
            if aggregated_root.exists():
                shutil.rmtree(aggregated_root, ignore_errors=True)
            if stage_root.exists():
                shutil.rmtree(stage_root, ignore_errors=True)


if __name__ == "__main__":
    try:
        sys.exit(main())
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        sys.exit(1)
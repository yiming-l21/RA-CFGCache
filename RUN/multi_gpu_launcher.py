#!/usr/bin/env python3

"""Launch multiple sample runs across available GPUs.

The script splits the prompt list into shards, one per GPU, and spawns a
separate process for each shard with an isolated prompt file and output
directory.

Supported backends:
  - flux
  - wan
  - chipmunk
  - cogvideox

Important:
  - flux / chipmunk still go through scripts/sample.sh.
  - wan / cogvideox directly launch their sample.py
    because video sample.py uses --taylor_method and does not support
    generic sample.sh-injected args like --cache_mode or --true_cfg_scale.

Video merged output layout:
  output_dir/
    videos/video_XXXXXX.mp4
    metadata/video_XXXXXX.json
    eval_frames/video_XXXXXX/*.png
    manifest.jsonl
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
RUN_BACKEND_SH = PROJECT_ROOT / "scripts" / "run_backend.sh"
DEFAULT_PROMPT_FILE = PROJECT_ROOT / "resources" / "prompts" / "prompt.txt"
CHIPMUNK_EXAMPLE_DIR = PROJECT_ROOT / "models" / "chipmunk" / "examples" / "flux"


@dataclass
class ShardJob:
    gpu: str
    prompts: List[str]
    start_index: int
    prompt_file: Path
    output_dir: Path
    process: subprocess.Popen | None = None
    full_output_dir: Path | None = None
    placeholder_image: Path | None = None


IMG_PATTERN = re.compile(r"img_(\d+)\.(?:jpe?g|png)$", re.IGNORECASE)
VIDEO_PATTERN = re.compile(r"video_(\d+)\.(?:mp4|webm|gif|mov)$", re.IGNORECASE)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch sample.sh or backend runners on multiple GPUs in parallel"
    )
    parser.add_argument(
        "--backend",
        default="flux",
        choices=["flux", "wan", "chipmunk", "cogvideox"],
        help="Backend to run for each shard.",
    )
    parser.add_argument("--mode", required=True, help="Cache mode to pass to sample.sh/sample.py")
    parser.add_argument(
        "--prompt-file",
        default=str(DEFAULT_PROMPT_FILE),
        help="Path to the active prompt list.",
    )
    parser.add_argument(
        "--full-prompt-file",
        help="Path to the master prompt list for global index / manifest. Defaults to --prompt-file.",
    )
    parser.add_argument(
        "--gpus",
        help="Comma-separated GPU ids to use, e.g. 0,1,3. If omitted, tries CUDA_VISIBLE_DEVICES or torch.cuda.device_count().",
    )
    parser.add_argument(
        "--num-gpus",
        type=int,
        help="Number of GPUs to use starting from 0 when --gpus is not provided.",
    )
    parser.add_argument(
        "--base-output-dir",
        default=str(PROJECT_ROOT / "results"),
        help="Base directory for outputs.",
    )
    parser.add_argument(
        "--run-name",
        help="Optional name appended to output directory.",
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
        "--chipmunk-param-tag",
        help="Relative output subdirectory for chipmunk backend runs.",
    )
    parser.add_argument(
        "sample_args",
        nargs=argparse.REMAINDER,
        help="Additional arguments forwarded to sample.sh or video sample.py. Prefix with --.",
    )
    return parser.parse_args()

def find_module_file(module_name: str) -> bool:
    rel_path = module_name.replace(".", "/") + ".py"
    return (PROJECT_ROOT / rel_path).is_file()


def first_existing_module(*module_names: str) -> str | None:
    for module_name in module_names:
        if find_module_file(module_name):
            return module_name
    return None


def resolve_video_sample_module(backend: str) -> str:
    if backend == "cogvideox":
        module_name = first_existing_module(
            "models.cogvideox.src.sample",
            "models.cogvideox.sample",
            "models.cogvideo_x.src.sample",
            "models.cogvideo_x.sample",
        )
        if module_name is None:
            raise FileNotFoundError(
                "未找到 CogVideoX Python module: "
                "models.cogvideox.src.sample / models.cogvideox.sample / "
                "models.cogvideo_x.src.sample / models.cogvideo_x.sample"
            )
        return module_name

    if backend == "wan":
        module_name = first_existing_module(
            "models.wan.src.sample",
            "models.wan.sample",
        )
        if module_name is None:
            raise FileNotFoundError(
                "未找到 Wan Python module: models.wan.src.sample 或 models.wan.sample"
            )
        return module_name

    raise ValueError(f"Unsupported video backend: {backend}")

def detect_gpus(gpu_arg: str | None, num_gpus: int | None) -> List[str]:
    if gpu_arg:
        return [gpu.strip() for gpu in gpu_arg.split(",") if gpu.strip()]

    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        tokens = [token.strip() for token in env.split(",") if token.strip()]
        if tokens:
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


def sanitize_sample_args(sample_args: Iterable[str]) -> List[str]:
    args = list(sample_args)

    if args and args[0] == "--":
        args = args[1:]

    managed_flags = {
        "--mode",
        "--prompt_file",
        "--prompt-file",
        "--prompt",
        "--output_dir",
        "--output-dir",
        "--start_index",
        "--start-index",
        "--taylor_method",
        "--taylor-method",
    }

    for token in args:
        for flag in managed_flags:
            if token == flag or token.startswith(flag + "="):
                raise ValueError(
                    f"Launcher manages '{flag}'; remove it from forwarded arguments: {args}"
                )

    return args


def _flag_aliases(flag: str) -> set[str]:
    if flag.startswith("--"):
        core = flag[2:]
        return {"--" + core, "--" + core.replace("_", "-")}
    return {flag}


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


def remove_args(args: List[str], flags: set[str]) -> List[str]:
    cleaned: List[str] = []
    i = 0

    while i < len(args):
        tok = args[i]

        if tok in flags:
            i += 2
            continue

        if any(tok.startswith(flag + "=") for flag in flags):
            i += 1
            continue

        cleaned.append(tok)
        i += 1

    return cleaned


def strip_video_unsupported_args(extra_args: List[str]) -> List[str]:
    """
    Wan / CogVideoX sample.py do not support:
      --cache_mode / --cache-mode
      --true_cfg_scale / --true-cfg-scale

    Video sample.py uses --taylor_method instead of --cache_mode.
    CFG strength is controlled by --guidance_scale.
    """
    unsupported_flags = {
        "--cache_mode",
        "--cache-mode",
        "--true_cfg_scale",
        "--true-cfg-scale",
    }
    return remove_args(extra_args, unsupported_flags)


def build_output_dir(stage_root: Path, gpu: str) -> Path:
    return stage_root / f"gpu{gpu}"


def write_prompt_shard(prompts: List[str], directory: Path, gpu: str) -> Path:
    directory.mkdir(parents=True, exist_ok=True)

    safe_gpu = gpu.replace("/", "_").replace(",", "_")
    shard_path = directory / f"prompts_gpu{safe_gpu}.txt"

    with shard_path.open("w", encoding="utf-8") as f:
        f.write("\n".join(prompts))
        f.write("\n")

    return shard_path


def normalize_chipmunk_args(extra_args: List[str]) -> List[str]:
    normalized: List[str] = []
    i = 0

    while i < len(extra_args):
        token = extra_args[i]

        if token in {"--chipmunk_config", "--chipmunk-config"}:
            if i + 1 >= len(extra_args):
                raise ValueError(f"{token} requires a path argument")

            path_token = extra_args[i + 1]
            resolved = Path(path_token)

            if not resolved.is_absolute():
                resolved = (PROJECT_ROOT / resolved).resolve()

            normalized.append("--chipmunk-config")
            normalized.append(str(resolved))
            i += 2
            continue

        if token.startswith("--chipmunk_config=") or token.startswith("--chipmunk-config="):
            _, path_token = token.split("=", 1)
            resolved = Path(path_token)

            if not resolved.is_absolute():
                resolved = (PROJECT_ROOT / resolved).resolve()

            normalized.append("--chipmunk-config=" + str(resolved))
            i += 1
            continue

        normalized.append(token)
        i += 1

    return normalized


def setup_chipmunk_output(
    job: ShardJob,
    mode_label: str,
    param_tag: str,
    dry_run: bool,
) -> tuple[Path, Path | None]:
    if not param_tag:
        raise ValueError("chipmunk backend requires --chipmunk-param-tag")

    final_dir = job.output_dir / mode_label / param_tag
    placeholder: Path | None = None

    if not dry_run:
        final_dir.mkdir(parents=True, exist_ok=True)
        marker_path = job.output_dir / ".full_output_dir"
        marker_path.write_text(str(final_dir.resolve()), encoding="utf-8")

        if job.start_index > 0:
            placeholder = final_dir / f"img_{job.start_index - 1}.jpg"
            placeholder.touch(exist_ok=True)

    return final_dir, placeholder


def launch_process(
    gpu: str,
    prompt_file: Path,
    mode: str,
    output_dir: Path,
    start_index: int,
    extra_args: List[str],
    dry_run: bool,
    backend: str,
    prompts: List[str] | None = None,
) -> subprocess.Popen | None:
    run_cwd = PROJECT_ROOT

    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = gpu
    env.setdefault("PYTHON_BIN", sys.executable)

    if backend in {"cogvideox", "wan"}:
        video_args = strip_video_unsupported_args(extra_args)
        video_module = resolve_video_sample_module(backend)

        cmd = [
            env.get("PYTHON_BIN", sys.executable),
            "-m",
            video_module,
            "--taylor_method",
            mode,
            "--prompt_file",
            str(prompt_file),
            "--output_dir",
            str(output_dir),
            "--start_index",
            str(start_index),
        ] + video_args

    else:
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

    if backend == "chipmunk":
        chipmunk_src = str(CHIPMUNK_EXAMPLE_DIR / "src")
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = chipmunk_src if not existing else chipmunk_src + os.pathsep + existing

    print("[LAUNCH] GPU", gpu, "->", " ".join(cmd), flush=True)

    if dry_run:
        return None

    output_dir.mkdir(parents=True, exist_ok=True)
    return subprocess.Popen(cmd, cwd=run_cwd, env=env)


def collect_numbered_files(directory: Path, pattern: re.Pattern) -> List[tuple[int, Path]]:
    items: List[tuple[int, Path]] = []

    if not directory.exists() or not directory.is_dir():
        return items

    for path in sorted(directory.iterdir()):
        if not path.is_file():
            continue

        match = pattern.fullmatch(path.name)
        if match:
            items.append((int(match.group(1)), path))

    items.sort(key=lambda item: item[0])
    return items


def _copy_file_no_overwrite(src_path: Path, dest_path: Path):
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    if dest_path.exists():
        raise FileExistsError(f"Duplicate output file detected: {dest_path}")

    shutil.copy2(src_path, dest_path)


def _resolve_full_output_dir(job: ShardJob) -> tuple[Path, bool]:
    marker = job.output_dir / ".full_output_dir"

    if marker.exists():
        full_output_dir = Path(marker.read_text(encoding="utf-8").strip()).resolve()
        return full_output_dir, True

    return job.output_dir.resolve(), False


def _copy_video_outputs(
    *,
    full_output_dir: Path,
    dest_dir: Path,
    job: ShardJob,
    prompts: List[str],
    manifest: List[dict],
    manifest_jsonl_lines: List[str],
) -> bool:
    videos_src_dir = full_output_dir / "videos"
    metadata_src_dir = full_output_dir / "metadata"
    eval_frames_src_dir = full_output_dir / "eval_frames"

    if not videos_src_dir.is_dir():
        return False

    videos_dest_dir = dest_dir / "videos"
    metadata_dest_dir = dest_dir / "metadata"
    eval_frames_dest_dir = dest_dir / "eval_frames"

    found_any = False

    for index, src_video_path in collect_numbered_files(videos_src_dir, VIDEO_PATTERN):
        found_any = True
        video_stem = src_video_path.stem

        dest_video_path = videos_dest_dir / src_video_path.name
        _copy_file_no_overwrite(src_video_path, dest_video_path)

        src_metadata_path = metadata_src_dir / f"{video_stem}.json"
        dest_metadata_path = metadata_dest_dir / f"{video_stem}.json"
        metadata_rel = None

        if src_metadata_path.exists():
            _copy_file_no_overwrite(src_metadata_path, dest_metadata_path)
            metadata_rel = str(dest_metadata_path.relative_to(dest_dir))

        src_eval_frame_dir = eval_frames_src_dir / video_stem
        dest_eval_frame_dir = eval_frames_dest_dir / video_stem
        eval_frame_rel = None

        if src_eval_frame_dir.is_dir():
            if dest_eval_frame_dir.exists():
                raise FileExistsError(
                    f"Duplicate eval frame directory detected: {dest_eval_frame_dir}"
                )

            dest_eval_frame_dir.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(src_eval_frame_dir, dest_eval_frame_dir)
            eval_frame_rel = str(dest_eval_frame_dir.relative_to(dest_dir))

        prompt_text = prompts[index] if index < len(prompts) else None

        entry = {
            "index": index,
            "prompt_index": index,
            "prompt": prompt_text,
            "source_gpu": job.gpu,
            "source_dir": str(full_output_dir),
            "source_video": str(src_video_path),
            "dest_video": str(dest_video_path),
            "video_path": str(dest_video_path.relative_to(dest_dir)),
            "metadata_path": metadata_rel,
            "eval_frame_dir": eval_frame_rel,
        }
        manifest.append(entry)

        manifest_jsonl_lines.append(
            json.dumps(
                {
                    "video_id": video_stem,
                    "prompt_index": index,
                    "prompt": prompt_text,
                    "video_path": str(dest_video_path.relative_to(dest_dir)),
                    "metadata_path": metadata_rel,
                    "eval_frame_dir": eval_frame_rel,
                },
                ensure_ascii=False,
            )
        )

    return found_any


def aggregate_outputs(
    jobs: List[ShardJob],
    aggregated_root: Path,
    prompts: List[str],
    backend: str = "flux",
) -> Path | None:
    if not jobs:
        return None

    if aggregated_root.exists():
        shutil.rmtree(aggregated_root)

    aggregated_root.mkdir(parents=True, exist_ok=True)

    prompt_path = aggregated_root / "prompts.txt"
    prompt_path.write_text("\n".join(prompts) + "\n", encoding="utf-8")

    manifest: List[dict] = []
    manifest_jsonl_lines: List[str] = []
    rel_subpaths: set[Path] = set()

    for job in sorted(jobs, key=lambda j: j.start_index):
        full_output_dir, has_marker = _resolve_full_output_dir(job)

        if not full_output_dir.exists():
            raise FileNotFoundError(f"Expected output directory '{full_output_dir}' not found")

        job.full_output_dir = full_output_dir

        if has_marker:
            try:
                rel_subpath = full_output_dir.relative_to(job.output_dir)
            except ValueError:
                rel_subpath = Path(full_output_dir.name)
        else:
            rel_subpath = Path(".")

        rel_subpaths.add(rel_subpath)

        dest_dir = aggregated_root / rel_subpath
        dest_dir.mkdir(parents=True, exist_ok=True)

        # Flux / Qwen image-style outputs.
        for index, src_path in collect_numbered_files(full_output_dir, IMG_PATTERN):
            dest_path = dest_dir / src_path.name
            _copy_file_no_overwrite(src_path, dest_path)

            prompt_text = prompts[index] if index < len(prompts) else None
            manifest.append(
                {
                    "index": index,
                    "prompt": prompt_text,
                    "source_gpu": job.gpu,
                    "source_dir": str(full_output_dir),
                    "source_image": str(src_path),
                    "dest_image": str(dest_path),
                }
            )

        # CogVideoX video-style outputs.
        copied_video_outputs = _copy_video_outputs(
            full_output_dir=full_output_dir,
            dest_dir=dest_dir,
            job=job,
            prompts=prompts,
            manifest=manifest,
            manifest_jsonl_lines=manifest_jsonl_lines,
        )

        # Logs.
        logs_dir = full_output_dir / "logs"
        if logs_dir.is_dir():
            dest_logs_dir = dest_dir / "logs" / f"gpu{job.gpu}"
            shutil.copytree(logs_dir, dest_logs_dir, dirs_exist_ok=True)

        # Auxiliary files.
        skip_names = {"logs"}

        if copied_video_outputs:
            skip_names.update({"videos", "metadata", "eval_frames", "manifest.jsonl"})

        for item in full_output_dir.iterdir():
            if item.name in skip_names:
                continue

            if item.is_file() and IMG_PATTERN.fullmatch(item.name):
                continue

            if item.is_file() and VIDEO_PATTERN.fullmatch(item.name):
                continue

            dest_item = dest_dir / item.name

            if dest_item.exists():
                continue

            if item.is_dir():
                shutil.copytree(item, dest_item)
            else:
                shutil.copy2(item, dest_item)

    manifest.sort(key=lambda entry: entry.get("index", entry.get("prompt_index", 0)))

    manifest_path = aggregated_root / "merged_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    if manifest_jsonl_lines:
        jsonl_path = aggregated_root / "manifest.jsonl"
        jsonl_path.write_text("\n".join(manifest_jsonl_lines) + "\n", encoding="utf-8")

    if len(rel_subpaths) == 1:
        only_subpath = next(iter(rel_subpaths))
        return aggregated_root / only_subpath

    return aggregated_root


def move_aggregated_outputs(
    *,
    aggregated_root: Path,
    final_dir: Path | None,
    target_mode_dir: Path,
    mode_label: str,
    timestamp_label: str,
) -> Path | None:
    if final_dir is None:
        return None

    source_mode_dir = aggregated_root / mode_label

    if source_mode_dir.exists():
        target_mode_dir.mkdir(parents=True, exist_ok=True)

        for child in sorted(source_mode_dir.iterdir()):
            dest = target_mode_dir / child.name
            if dest.exists():
                dest = target_mode_dir / f"{child.name}_{timestamp_label}"
            shutil.move(str(child), str(dest))

        for meta_name in ("prompts.txt", "merged_manifest.json", "manifest.jsonl"):
            meta_src = aggregated_root / meta_name
            if not meta_src.exists():
                continue

            if meta_name == "manifest.jsonl":
                dest_meta = target_mode_dir / meta_name
                if dest_meta.exists():
                    dest_meta = target_mode_dir / f"{timestamp_label}_{meta_name}"
            else:
                dest_meta = target_mode_dir / f"{timestamp_label}_{meta_name}"

            shutil.move(str(meta_src), str(dest_meta))

        try:
            rel_path = Path(final_dir).relative_to(source_mode_dir)
            return target_mode_dir / rel_path
        except ValueError:
            return target_mode_dir

    target_mode_dir.mkdir(parents=True, exist_ok=True)

    for child in sorted(aggregated_root.iterdir()):
        if child.name in {"prompts.txt", "merged_manifest.json"}:
            continue

        if child.name == "manifest.jsonl":
            dest = target_mode_dir / child.name
            if dest.exists():
                dest = target_mode_dir / f"{timestamp_label}_{child.name}"
        else:
            dest = target_mode_dir / child.name
            if dest.exists():
                dest = target_mode_dir / f"{child.name}_{timestamp_label}"

        shutil.move(str(child), str(dest))

    for meta_name in ("prompts.txt", "merged_manifest.json"):
        meta_src = aggregated_root / meta_name
        if meta_src.exists():
            stamped = target_mode_dir / f"{timestamp_label}_{meta_name}"
            shutil.move(str(meta_src), str(stamped))

    return target_mode_dir


def main() -> int:
    args = parse_args()

    if args.backend == "chipmunk" and not args.chipmunk_param_tag:
        raise ValueError("Chipmunk backend requires --chipmunk-param-tag to organize outputs.")

    gpu_list = detect_gpus(args.gpus, args.num_gpus)
    if not gpu_list:
        raise RuntimeError("No GPU devices available")

    prompt_path = Path(args.prompt_file)
    if not prompt_path.is_absolute():
        prompt_path = (PROJECT_ROOT / prompt_path).resolve()

    full_prompt_path = Path(args.full_prompt_file) if args.full_prompt_file else prompt_path
    if not full_prompt_path.is_absolute():
        full_prompt_path = (PROJECT_ROOT / full_prompt_path).resolve()

    master_prompts = read_prompts(full_prompt_path)
    prompt_list = read_prompts(prompt_path)

    start_offset = max(int(args.start_offset or 0), 0)

    if start_offset >= len(master_prompts):
        raise ValueError(
            f"Start offset {start_offset} exceeds available prompts ({len(master_prompts)})."
        )

    if args.full_prompt_file:
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
            print(f"[INFO] GPUs {', '.join(unused_gpus)} idle (no prompts assigned)")

    extra_args = sanitize_sample_args(args.sample_args)

    if args.backend == "chipmunk":
        extra_args = normalize_chipmunk_args(extra_args)

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

    base_output_dir = Path(args.base_output_dir).resolve()
    default_root = (PROJECT_ROOT / "results").resolve()

    if base_output_dir == default_root:
        backend_root = args.backend.replace("-", "_")
        base_output_dir = default_root / backend_root

    base_output_dir.mkdir(parents=True, exist_ok=True)

    timestamp_label = datetime.now().strftime("%Y%m%d_%H%M%S")
    backend_tag = args.backend.replace("-", "_")
    mode_label = args.mode.lower()

    stage_label = mode_label if args.backend in {"flux", "chipmunk"} else f"{backend_tag}_{mode_label}"
    stage_root = base_output_dir / ".multi_gpu_tmp" / f"{stage_label}_{timestamp_label}"
    stage_root.mkdir(parents=True, exist_ok=True)

    final_mode_dir_name = stage_label if not args.run_name else f"{stage_label}_{args.run_name}"
    aggregated_root = stage_root / "merged"
    target_mode_dir = base_output_dir / final_mode_dir_name

    tmp_root = Path(tempfile.mkdtemp(prefix="tmp_multi_gpu_", dir=PROJECT_ROOT / "RUN"))

    exit_code = 0
    jobs: List[ShardJob] = []
    final_output_path: Path | None = None
    success = False

    try:
        for gpu, (start_index, shard_prompts) in zip(gpu_list, shard_specs):
            shard_file = write_prompt_shard(shard_prompts, tmp_root, gpu)
            shard_output = build_output_dir(stage_root, gpu)
            global_start = start_offset + start_index

            job = ShardJob(
                gpu=gpu,
                prompts=shard_prompts,
                start_index=global_start,
                prompt_file=shard_file,
                output_dir=shard_output,
            )

            backend_output_dir = shard_output

            if args.backend == "chipmunk":
                full_output_dir, placeholder = setup_chipmunk_output(
                    job,
                    mode_label=mode_label,
                    param_tag=args.chipmunk_param_tag,
                    dry_run=args.dry_run,
                )
                job.full_output_dir = full_output_dir
                job.placeholder_image = placeholder
                backend_output_dir = full_output_dir

            shard_extra_args = extra_args

            if neg_prompts_active is not None:
                shard_negs = neg_prompts_active[start_index : start_index + len(shard_prompts)]

                if len(shard_negs) != len(shard_prompts):
                    raise ValueError("Negative shard length mismatch.")

                shard_neg_file = write_prompt_shard(shard_negs, tmp_root, f"{gpu}_neg")
                shard_extra_args = replace_arg_value(
                    shard_extra_args,
                    "--negative_prompt_file",
                    str(shard_neg_file),
                )

            job.process = launch_process(
                gpu=gpu,
                prompt_file=shard_file,
                mode=args.mode,
                output_dir=backend_output_dir,
                start_index=global_start,
                extra_args=shard_extra_args,
                dry_run=args.dry_run,
                backend=args.backend,
                prompts=shard_prompts,
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

        if args.backend == "chipmunk":
            for job in jobs:
                placeholder = job.placeholder_image

                if placeholder and placeholder.exists():
                    try:
                        placeholder.unlink(missing_ok=True)
                    except Exception as cleanup_err:
                        print(f"[WARN] Failed to remove placeholder {placeholder}: {cleanup_err}")

                    job.placeholder_image = None

        if exit_code != 0:
            return exit_code

        final_dir = aggregate_outputs(
            jobs,
            aggregated_root,
            master_prompts,
            backend=args.backend,
        )

        if final_dir is not None:
            final_output_path = move_aggregated_outputs(
                aggregated_root=aggregated_root,
                final_dir=final_dir,
                target_mode_dir=target_mode_dir,
                mode_label=mode_label,
                timestamp_label=timestamp_label,
            )
            print(f"[MERGE] Aggregated outputs merged into: {final_output_path}")
        else:
            print("[MERGE] No aggregated outputs were produced.")

        success = final_output_path is not None

        for job in jobs:
            try:
                if job.output_dir.exists():
                    shutil.rmtree(job.output_dir, ignore_errors=True)
                    print(f"[CLEANUP] Removed shard directory: {job.output_dir}")
            except Exception as cleanup_err:
                print(f"[WARN] Failed to remove shard directory {job.output_dir}: {cleanup_err}")

        return 0

    finally:
        if args.report_path:
            report_payload = {
                "success": success,
                "final_output_path": str(final_output_path) if final_output_path else None,
                "timestamp_label": timestamp_label,
                "target_mode_dir": str(target_mode_dir),
            }

            report_file = Path(args.report_path)
            report_file.parent.mkdir(parents=True, exist_ok=True)
            report_file.write_text(
                json.dumps(report_payload, ensure_ascii=False, indent=2),
                encoding="utf-8",
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
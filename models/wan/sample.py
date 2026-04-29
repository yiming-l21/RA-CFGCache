import os
import json
import time
import inspect
import contextlib
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .pipeline_wan import WanPipeline
from .cache_functions import cache_init

try:
    from diffusers import AutoencoderKLWan
except Exception:
    AutoencoderKLWan = None

try:
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
except Exception:
    UniPCMultistepScheduler = None

from .util import load_wan_pipeline_fast


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "wan2.1"

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)

CACHE_METHODS = [
    "original",
    "CFGCache",
    "TeaCache",
    "MagCache",
    "DiCache",
    "Taylor",
    "HiCache",
    "GroupedTaylor",
]


@dataclass
class SamplingOptions:
    prompts: list[str]
    width: int
    height: int
    num_frames: int
    fps: int
    num_steps: int
    guidance_scale: float
    guidance_scale_2: float | None
    seed: int | None
    num_videos_per_prompt: int
    batch_size: int
    model_name: str
    output_dir: str
    model_path: str
    add_sampling_metadata: bool
    negative_prompt: str | None
    negative_prompts: list[str] | None
    start_index: int
    limit: int | None
    cpu_offload: bool
    max_sequence_length: int
    output_type: str
    use_fast_loader: bool
    scheduler: str
    flow_shift: float | None
    load_vae_fp32: bool

    interval: int
    max_order: int
    first_enhance: int
    taylor_method: str
    hicache_scale: float
    rel_l1_thresh: float
    proxy_tables_path: str | None

    save_eval_frames: bool
    eval_frame_stride: int
    eval_frame_format: str
    eval_jpg_quality: int
    write_manifest: bool


def read_prompts(prompt_file: str) -> list[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _ensure_cache_context(model):
    """
    Wan pipeline calls model.cache_context('cond'/'uncond').

    Some official WanTransformer3DModel builds do not expose cache_context.
    For normal generation, a no-op context keeps the pipeline runnable.
    """
    if model is None:
        return

    if not hasattr(model, "cache_context"):
        def cache_context(_name=""):
            return contextlib.nullcontext()

        model.cache_context = cache_context


def _load_pipeline(opts: SamplingOptions, device: str):
    model_path = opts.model_path

    sig = inspect.signature(load_wan_pipeline_fast)
    kwargs = {}
    if "patch_cache" in sig.parameters:
        kwargs["patch_cache"] = True

    pipe = load_wan_pipeline_fast(model_path, **kwargs)

    if opts.scheduler.lower() == "unipc":
        if UniPCMultistepScheduler is None:
            print("[WARN] UniPCMultistepScheduler is unavailable; keeping the loaded scheduler.")
        else:
            scheduler_kwargs = {}
            if opts.flow_shift is not None:
                scheduler_kwargs["flow_shift"] = float(opts.flow_shift)

            try:
                pipe.scheduler = UniPCMultistepScheduler.from_config(
                    pipe.scheduler.config,
                    **scheduler_kwargs,
                )
                print(f"[INFO] Using UniPC scheduler with flow_shift={opts.flow_shift}")
            except TypeError:
                pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
                print("[WARN] UniPC scheduler does not accept flow_shift in this diffusers version.")
    elif opts.scheduler.lower() not in ("default", "flowmatch", "euler"):
        print(f"[WARN] Unknown scheduler={opts.scheduler!r}; keeping the loaded scheduler.")

    _ensure_cache_context(getattr(pipe, "transformer", None))
    _ensure_cache_context(getattr(pipe, "transformer_2", None))

    if opts.cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        print("[load] enable_model_cpu_offload start")
        pipe.enable_model_cpu_offload()
        print("[load] enable_model_cpu_offload done")
    elif hasattr(pipe, "to"):
        print(f"[load] pipe.to({device}) start")
        pipe = pipe.to(device)
        print(f"[load] pipe.to({device}) done")

    return pipe


def _to_uint8_frame(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"), dtype=np.uint8)

    if isinstance(frame, np.ndarray):
        arr = frame

        if arr.dtype != np.uint8:
            if arr.size > 0 and arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return arr

    if torch.is_tensor(frame):
        t = frame.detach().float().cpu()

        if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW -> HWC
            t = t.permute(1, 2, 0)
        if t.ndim == 2:
            t = t.unsqueeze(-1).repeat(1, 1, 3)
        if t.numel() > 0 and t.max() <= 1.0:
            t = t * 255.0

        t = t.clamp(0, 255).to(torch.uint8)
        return t.numpy()

    raise TypeError(f"Unsupported frame type: {type(frame)}")


def _normalize_video_item_to_frames(video_item: Any) -> list[np.ndarray]:
    if isinstance(video_item, list):
        return [_to_uint8_frame(x) for x in video_item]

    if isinstance(video_item, np.ndarray):
        arr = video_item

        if arr.ndim == 4:
            if arr.shape[-1] in (1, 3):  # THWC
                return [_to_uint8_frame(arr[i]) for i in range(arr.shape[0])]
            if arr.shape[1] in (1, 3):  # TCHW
                return [_to_uint8_frame(arr[i]) for i in range(arr.shape[0])]

        raise ValueError(f"Unsupported numpy video shape: {arr.shape}")

    if torch.is_tensor(video_item):
        t = video_item.detach().cpu()

        if t.ndim == 4:
            return [_to_uint8_frame(t[i]) for i in range(t.shape[0])]

        raise ValueError(f"Unsupported tensor video shape: {tuple(t.shape)}")

    raise TypeError(f"Unsupported video item type: {type(video_item)}")


def _extract_video_batch(result: Any) -> list[list[np.ndarray]]:
    data = None

    if hasattr(result, "frames"):
        data = result.frames
    elif hasattr(result, "videos"):
        data = result.videos
    elif isinstance(result, tuple):
        data = result[0]
    else:
        data = result

    if isinstance(data, list):
        if len(data) == 0:
            return []
        if isinstance(data[0], list):
            return [_normalize_video_item_to_frames(v) for v in data]
        return [_normalize_video_item_to_frames(data)]

    if isinstance(data, np.ndarray):
        if data.ndim == 5:  # B,T,...
            return [_normalize_video_item_to_frames(data[i]) for i in range(data.shape[0])]
        if data.ndim == 4:
            return [_normalize_video_item_to_frames(data)]

    if torch.is_tensor(data):
        if data.ndim == 5:
            return [_normalize_video_item_to_frames(data[i]) for i in range(data.shape[0])]
        if data.ndim == 4:
            return [_normalize_video_item_to_frames(data)]

    raise TypeError(f"Unsupported pipeline result type: {type(result)}")


def _write_mp4(frames: list[np.ndarray], video_filename: Path, fps: int):
    video_filename.parent.mkdir(parents=True, exist_ok=True)

    writer = imageio.get_writer(
        str(video_filename),
        fps=fps,
        codec="libx264",
        format="FFMPEG",
        ffmpeg_params=["-pix_fmt", "yuv420p"],
    )

    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def _save_eval_frames(
    frames: list[np.ndarray],
    frame_dir: Path,
    frame_stride: int = 1,
    frame_format: str = "png",
    jpg_quality: int = 95,
):
    frame_dir.mkdir(parents=True, exist_ok=True)

    stride = max(1, int(frame_stride))
    indices = list(range(0, len(frames), stride))

    if len(frames) > 0 and indices[-1] != len(frames) - 1:
        indices.append(len(frames) - 1)

    saved_files = []

    for idx in indices:
        frame = frames[idx]
        img = Image.fromarray(frame)

        if frame_format.lower() in ("jpg", "jpeg"):
            out_path = frame_dir / f"{idx:04d}.jpg"
            img.save(out_path, quality=jpg_quality)
        else:
            out_path = frame_dir / f"{idx:04d}.png"
            img.save(out_path)

        saved_files.append(str(out_path))

    return indices, saved_files


def _append_manifest_line(manifest_path: Path, item: dict):
    with open(manifest_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(item, ensure_ascii=False) + "\n")


def preload_cfgcache_rho_table(proxy_tables_path: str, cfg_scale: float):
    """
    Load packed all-cfg rho table once, then pick the nearest cfg-specific rho_table_at.

    Expected npz:
        cfg_scales
        rho_table_at_all: [N_cfg, T, T]
        steps optional
    """
    data = np.load(proxy_tables_path)

    if "cfg_scales" not in data:
        raise ValueError("cfg_scales is required in proxy_tables_path")
    if "rho_table_at_all" not in data:
        raise ValueError("rho_table_at_all is required in proxy_tables_path")

    cfg_scales = np.asarray(data["cfg_scales"], dtype=np.float32)
    rho_table_at_all = np.asarray(data["rho_table_at_all"], dtype=np.float32)

    if rho_table_at_all.ndim != 3:
        raise ValueError(
            f"rho_table_at_all must have shape [N_cfg, T, T], got {rho_table_at_all.shape}"
        )
    if rho_table_at_all.shape[0] != len(cfg_scales):
        raise ValueError(
            f"rho_table_at_all.shape[0] ({rho_table_at_all.shape[0]}) "
            f"!= len(cfg_scales) ({len(cfg_scales)})"
        )

    cfg_scale = float(cfg_scale)
    cfg_idx = int(np.argmin(np.abs(cfg_scales - cfg_scale)))
    matched_cfg_scale = float(cfg_scales[cfg_idx])

    rho_table_at = rho_table_at_all[cfg_idx]

    out = {
        "steps": np.asarray(data["steps"], dtype=np.int64) if "steps" in data else None,
        "cfg_scales": cfg_scales,
        "cfg_idx": cfg_idx,
        "matched_cfg_scale": matched_cfg_scale,
        "rho_table_at": rho_table_at,
    }

    print(
        "[CFGCache] Preloaded rho table: "
        f"runtime_cfg_scale={cfg_scale:.4f}, "
        f"matched_cfg_scale={matched_cfg_scale:.4f}, "
        f"cfg_idx={cfg_idx}, "
        f"rho_table_shape={rho_table_at.shape}"
    )
    return out


def build_cfgcache_runtime_config(opts: SamplingOptions):
    """
    Build runtime cfgcache config once outside proxy init.
    Same pattern as CogVideoX.
    """
    proxy_tables_path = getattr(opts, "proxy_tables_path", None)
    if proxy_tables_path is None:
        return None

    proxy_tables = preload_cfgcache_rho_table(
        proxy_tables_path=proxy_tables_path,
        cfg_scale=opts.guidance_scale,
    )

    cfgcache_runtime = {
        "proxy_tables_path": proxy_tables_path,
        "proxy_tables": proxy_tables,
    }
    return cfgcache_runtime


def _infer_wan_num_layers(pipe) -> int:
    if getattr(pipe, "transformer", None) is not None and hasattr(pipe.transformer, "blocks"):
        return len(pipe.transformer.blocks)

    if getattr(pipe, "transformer_2", None) is not None and hasattr(pipe.transformer_2, "blocks"):
        return len(pipe.transformer_2.blocks)

    return 30


def _build_runtime_ctx(
    opts: SamplingOptions,
    batch_prompt_indices: list[int],
    cache_dic=None,
    current=None,
) -> dict:
    return {
        "prompt_indices": batch_prompt_indices,
        "taylor_method": opts.taylor_method,
        "cache_mode": opts.taylor_method,
        "interval": opts.interval,
        "max_order": opts.max_order,
        "first_enhance": opts.first_enhance,
        "hicache_scale": opts.hicache_scale,
        "rel_l1_thresh": opts.rel_l1_thresh,
        "proxy_tables_path": opts.proxy_tables_path,
        "true_cfg_scale": opts.guidance_scale,
        "guidance_scale": opts.guidance_scale,
        "cache_dic": cache_dic,
        "current": current,
    }


def _build_pipe_call_kwargs(
    pipe,
    *,
    batch_prompts: list[str],
    batch_negative_prompts: Optional[list[str]],
    fallback_negative_prompt: Optional[str],
    opts: SamplingOptions,
    generators,
    cache_dic=None,
    current=None,
    runtime_ctx: Optional[dict] = None,
):
    sig = inspect.signature(pipe.__call__)
    accepted = set(sig.parameters.keys())
    accepts_var_kwargs = any(
        p.kind == inspect.Parameter.VAR_KEYWORD
        for p in sig.parameters.values()
    )

    def can_pass(name: str) -> bool:
        return name in accepted or accepts_var_kwargs

    kwargs = {}

    if can_pass("prompt"):
        kwargs["prompt"] = batch_prompts

    neg_value = batch_negative_prompts if batch_negative_prompts is not None else fallback_negative_prompt
    if can_pass("negative_prompt") and neg_value is not None:
        kwargs["negative_prompt"] = neg_value

    if can_pass("guidance_scale"):
        kwargs["guidance_scale"] = opts.guidance_scale
    if can_pass("guidance_scale_2") and opts.guidance_scale_2 is not None:
        kwargs["guidance_scale_2"] = opts.guidance_scale_2

    if can_pass("height"):
        kwargs["height"] = opts.height
    if can_pass("width"):
        kwargs["width"] = opts.width

    if can_pass("num_inference_steps"):
        kwargs["num_inference_steps"] = opts.num_steps

    if can_pass("num_frames"):
        kwargs["num_frames"] = opts.num_frames
    elif can_pass("video_length"):
        kwargs["video_length"] = opts.num_frames
    elif can_pass("video_num_frames"):
        kwargs["video_num_frames"] = opts.num_frames

    if can_pass("num_videos_per_prompt"):
        kwargs["num_videos_per_prompt"] = opts.num_videos_per_prompt

    if can_pass("generator"):
        kwargs["generator"] = generators

    if can_pass("output_type"):
        kwargs["output_type"] = opts.output_type

    if can_pass("max_sequence_length"):
        kwargs["max_sequence_length"] = opts.max_sequence_length

    if can_pass("cache_dic") and cache_dic is not None:
        kwargs["cache_dic"] = cache_dic
    if can_pass("current") and current is not None:
        kwargs["current"] = current

    if can_pass("runtime_ctx") and runtime_ctx is not None:
        kwargs["runtime_ctx"] = runtime_ctx

    return kwargs


def _write_run_config(opts: SamplingOptions, out_dir: Path):
    import sys
    from datetime import datetime

    payload = {
        "timestamp": datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "working_directory": os.getcwd(),
        "model": {
            "name": opts.model_name,
            "path": opts.model_path,
        },
        "sampler": {
            "width": opts.width,
            "height": opts.height,
            "num_frames": opts.num_frames,
            "fps": opts.fps,
            "num_steps": opts.num_steps,
            "guidance_scale": opts.guidance_scale,
            "guidance_scale_2": opts.guidance_scale_2,
            "seed": opts.seed,
            "num_videos_per_prompt": opts.num_videos_per_prompt,
            "batch_size": opts.batch_size,
            "start_index": opts.start_index,
            "max_sequence_length": opts.max_sequence_length,
            "output_type": opts.output_type,
            "scheduler": opts.scheduler,
            "flow_shift": opts.flow_shift,
            "load_vae_fp32": opts.load_vae_fp32,
        },
        "cache": {
            "taylor_method": opts.taylor_method,
            "interval": opts.interval,
            "max_order": opts.max_order,
            "first_enhance": opts.first_enhance,
            "hicache_scale": opts.hicache_scale,
            "rel_l1_thresh": opts.rel_l1_thresh,
            "proxy_tables_path": opts.proxy_tables_path,
        },
        "prompts": {
            "count": len(opts.prompts),
            "negative_prompt": opts.negative_prompt,
            "has_negative_prompt_file": opts.negative_prompts is not None,
        },
        "output": {
            "output_dir": opts.output_dir,
            "save_eval_frames": opts.save_eval_frames,
            "write_manifest": opts.write_manifest,
        },
    }

    out_dir.mkdir(parents=True, exist_ok=True)

    with open(out_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def main(opts: SamplingOptions):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfgcache_runtime = None
    if opts.taylor_method == "CFGCache" and opts.proxy_tables_path is not None:
        t0 = time.perf_counter()
        cfgcache_runtime = build_cfgcache_runtime_config(opts)
        print(f"[CFGCache] preload done in {time.perf_counter() - t0:.3f}s")

    model_path = opts.model_path or os.environ.get("WAN_MODEL_PATH") or os.environ.get("WAN2_1_MODEL_PATH")
    if not model_path:
        raise ValueError(
            "Wan model path is not set. Pass --model_path or set WAN_MODEL_PATH / WAN2_1_MODEL_PATH."
        )
    opts.model_path = model_path

    prompts = opts.prompts
    if opts.limit is not None and opts.limit > 0:
        prompts = prompts[: opts.limit]

    negative_prompts = opts.negative_prompts
    if negative_prompts is not None and opts.limit is not None and opts.limit > 0:
        negative_prompts = negative_prompts[: opts.limit]

    if negative_prompts is not None and len(negative_prompts) != len(prompts):
        raise ValueError(
            f"negative_prompt_file line count ({len(negative_prompts)}) must match prompt count ({len(prompts)})."
        )

    final_output_path = Path(opts.output_dir)
    final_output_path.mkdir(parents=True, exist_ok=True)

    videos_dir = final_output_path / "videos"
    metadata_dir = final_output_path / "metadata"
    eval_frames_dir = final_output_path / "eval_frames"
    manifest_path = final_output_path / "manifest.jsonl"

    videos_dir.mkdir(parents=True, exist_ok=True)
    metadata_dir.mkdir(parents=True, exist_ok=True)

    if opts.save_eval_frames:
        eval_frames_dir.mkdir(parents=True, exist_ok=True)

    if opts.write_manifest and manifest_path.exists():
        manifest_path.unlink()

    _write_run_config(opts, final_output_path)

    print(f"Loading Wan pipeline from {model_path} on {device}...")
    print(
        f"[cache] taylor_method={opts.taylor_method}, "
        f"interval={opts.interval}, max_order={opts.max_order}, "
        f"first_enhance={opts.first_enhance}, "
        f"hicache_scale={opts.hicache_scale}, "
        f"rel_l1_thresh={opts.rel_l1_thresh}, "
        f"proxy_tables_path={opts.proxy_tables_path}"
    )

    pipe = _load_pipeline(opts, device=device)

    print(f"Loaded Wan pipeline from {model_path} on {device}")

    wan_num_layers = _infer_wan_num_layers(pipe)
    print(f"[cache] Wan num_layers={wan_num_layers}")

    total_prompts = len(prompts)
    progress_bar = tqdm(
        total=total_prompts * max(1, opts.num_videos_per_prompt),
        desc="Generating videos",
    )

    time_stats = {
        "e2e": 0.0,
        "sampling": 0.0,
        "save": 0.0,
        "total_videos": 0,
    }

    t_e2e_start = time.perf_counter()

    for batch_start in range(0, total_prompts, opts.batch_size):
        batch_end = min(batch_start + opts.batch_size, total_prompts)
        batch_prompts = prompts[batch_start:batch_end]

        batch_negative_prompts = None
        if negative_prompts is not None:
            batch_negative_prompts = negative_prompts[batch_start:batch_end]

        batch_size_actual = len(batch_prompts)

        if opts.seed is None:
            generators = None
        else:
            generators = [
                torch.Generator(device=device).manual_seed(
                    opts.seed + opts.start_index + batch_start + i
                )
                for i in range(batch_size_actual)
            ]

        if opts.taylor_method == "original":
            cache_dic, current = None, None
        else:
            model_kwargs = {
                "model_type": "wan",
                "num_layers": wan_num_layers,
                "num_steps": opts.num_steps,
                "interval": opts.interval,
                "max_order": opts.max_order,
                "first_enhance": opts.first_enhance,
                "hicache_scale": opts.hicache_scale,
                "rel_l1_thresh": opts.rel_l1_thresh,
                "use_true_cfg": opts.guidance_scale > 1.0,
            }

            if opts.taylor_method == "CFGCache":
                model_kwargs.update(
                    {
                        "true_cfg_scale": opts.guidance_scale,
                    }
                )

            cache_dic, current = cache_init(
                method=opts.taylor_method,
                model_kwargs=model_kwargs,
                cfgcache_runtime=cfgcache_runtime,
            )

        batch_prompt_indices = [
            opts.start_index + batch_start + i
            for i in range(batch_size_actual)
        ]

        runtime_ctx = _build_runtime_ctx(
            opts,
            batch_prompt_indices,
            cache_dic=cache_dic,
            current=current,
        )

        pipe_kwargs = _build_pipe_call_kwargs(
            pipe,
            batch_prompts=batch_prompts,
            batch_negative_prompts=batch_negative_prompts,
            fallback_negative_prompt=opts.negative_prompt,
            opts=opts,
            generators=generators,
            cache_dic=cache_dic,
            current=current,
            runtime_ctx=runtime_ctx,
        )

        t_sample0 = time.perf_counter()
        result = pipe(**pipe_kwargs)
        time_stats["sampling"] += time.perf_counter() - t_sample0

        videos = _extract_video_batch(result)
        t_save0 = time.perf_counter()

        for i, frames in enumerate(videos):
            global_video_idx = opts.start_index + batch_start + i
            video_stem = f"video_{global_video_idx:06d}"

            video_filename = videos_dir / f"{video_stem}.mp4"
            json_filename = metadata_dir / f"{video_stem}.json"

            _write_mp4(frames, video_filename, fps=opts.fps)

            saved_frame_indices = []
            frame_dir_rel = None

            if opts.save_eval_frames:
                frame_dir = eval_frames_dir / video_stem
                saved_frame_indices, _ = _save_eval_frames(
                    frames=frames,
                    frame_dir=frame_dir,
                    frame_stride=opts.eval_frame_stride,
                    frame_format=opts.eval_frame_format,
                    jpg_quality=opts.eval_jpg_quality,
                )
                frame_dir_rel = str(frame_dir.relative_to(final_output_path))

            metadata = {
                "video_id": video_stem,
                "prompt_index": global_video_idx,
                "prompt": batch_prompts[i] if i < len(batch_prompts) else "",
                "negative_prompt": (
                    batch_negative_prompts[i]
                    if batch_negative_prompts is not None and i < len(batch_negative_prompts)
                    else (opts.negative_prompt or "")
                ),
                "seed": None if opts.seed is None else opts.seed + global_video_idx,
                "width": opts.width,
                "height": opts.height,
                "num_frames": opts.num_frames,
                "fps": opts.fps,
                "num_steps": opts.num_steps,
                "guidance_scale": opts.guidance_scale,
                "guidance_scale_2": opts.guidance_scale_2,
                "model_name": opts.model_name,
                "model_path": opts.model_path,
                "scheduler": opts.scheduler,
                "flow_shift": opts.flow_shift,
                "taylor_method": opts.taylor_method,
                "interval": opts.interval,
                "max_order": opts.max_order,
                "first_enhance": opts.first_enhance,
                "hicache_scale": opts.hicache_scale,
                "rel_l1_thresh": opts.rel_l1_thresh,
                "proxy_tables_path": opts.proxy_tables_path,
                "video_path": str(video_filename.relative_to(final_output_path)),
                "metadata_path": str(json_filename.relative_to(final_output_path)),
                "eval_frame_dir": frame_dir_rel,
                "saved_frame_indices": saved_frame_indices,
            }

            with open(json_filename, "w", encoding="utf-8") as f:
                json.dump(metadata, f, ensure_ascii=False, indent=2)

            if opts.write_manifest:
                _append_manifest_line(
                    manifest_path,
                    {
                        "video_id": video_stem,
                        "prompt_index": global_video_idx,
                        "prompt": metadata["prompt"],
                        "video_path": metadata["video_path"],
                        "metadata_path": metadata["metadata_path"],
                        "eval_frame_dir": metadata["eval_frame_dir"],
                    },
                )

            time_stats["total_videos"] += 1
            progress_bar.update(1)

        time_stats["save"] += time.perf_counter() - t_save0

    time_stats["e2e"] = time.perf_counter() - t_e2e_start

    total_videos_done = max(1, time_stats["total_videos"])
    total_prompts_done = max(1, total_prompts)
    avg_sec_per_video = time_stats["e2e"] / total_videos_done
    avg_sec_per_prompt = avg_sec_per_video * max(1, opts.num_videos_per_prompt)
    avg_sec_sampling_per_video = time_stats["sampling"] / total_videos_done
    avg_sec_sampling_per_prompt = avg_sec_sampling_per_video * max(1, opts.num_videos_per_prompt)

    print("\n================ Timing Summary ================")
    print(f"Total elapsed:        {time_stats['e2e']:.3f} s")
    print(f"Sampling time:        {time_stats['sampling']:.3f} s")
    print(f"Saving time:          {time_stats['save']:.3f} s")
    print(f"Total videos:         {total_videos_done}")
    print(f"Total prompts:        {total_prompts_done}")
    print(f"Avg sec / video:      {avg_sec_per_video:.3f} s")
    print(
        f"Avg sec / prompt:     {avg_sec_per_prompt:.3f} s  "
        f"(includes {opts.num_videos_per_prompt} video/prompt)"
    )
    print(f"Avg sampling sec/video: {avg_sec_sampling_per_video:.3f} s")
    print(
        f"Avg sampling sec/prompt: {avg_sec_sampling_per_prompt:.3f} s  "
        f"(includes {opts.num_videos_per_prompt} video/prompt)"
    )
    print("================================================\n")

    timing_payload = {
        "total_sec": time_stats["e2e"],
        "sampling_sec": time_stats["sampling"],
        "save_sec": time_stats["save"],
        "total_videos": total_videos_done,
        "total_prompts": total_prompts_done,
        "avg_sec_per_video": avg_sec_per_video,
        "avg_sec_per_prompt": avg_sec_per_prompt,
    }

    with open(final_output_path / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing_payload, f, ensure_ascii=False, indent=2)

    progress_bar.close()

    print(f"Generated {total_videos_done} videos in {videos_dir}")
    print(f"Metadata saved in {metadata_dir}")

    if opts.save_eval_frames:
        print(f"Eval frames saved in {eval_frames_dir}")
    if opts.write_manifest:
        print(f"Manifest saved at {manifest_path}")


def app():
    import argparse

    parser = argparse.ArgumentParser(description="Generate videos using the Wan2.1 backend.")

    parser.add_argument(
        "--prompt_file",
        type=str,
        default="resources/prompts/prompt.txt",
        help="Path to the prompt text file.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=DEFAULT_NEGATIVE_PROMPT,
        help="Negative prompt for guidance.",
    )
    parser.add_argument(
        "--negative_prompt_file",
        type=str,
        default=None,
        help="Optional: file containing per-prompt negative prompts, line by line.",
    )

    parser.add_argument("--width", type=int, default=832, help="Video width. Wan2.1 480P common width: 832.")
    parser.add_argument("--height", type=int, default=480, help="Video height. Wan2.1 480P common height: 480.")
    parser.add_argument(
        "--num_frames",
        type=int,
        default=81,
        help="Number of video frames. Wan expects num_frames % 4 == 1.",
    )
    parser.add_argument("--fps", type=int, default=16, help="Video fps.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of sampling steps.")
    parser.add_argument("--guidance_scale", type=float, default=5.0, help="Guidance scale.")
    parser.add_argument(
        "--guidance_scale_2",
        type=float,
        default=None,
        help="Guidance scale for Wan2.2 low-noise transformer if boundary_ratio is used.",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed. Use negative value for random seed.")

    parser.add_argument(
        "--num_videos_per_prompt",
        type=int,
        default=1,
        help="Number of videos per prompt.",
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Prompt batch size.")

    parser.add_argument(
        "--model_name",
        type=str,
        default="wan2.1",
        choices=["wan2.1", "wan2.2", "wan"],
        help="Model name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help="Directory to save videos.",
    )
    parser.add_argument(
        "--model_path",
        type=str,
        default="",
        help="Path to the Wan model checkpoint. Or set WAN_MODEL_PATH / WAN2_1_MODEL_PATH.",
    )
    parser.add_argument(
        "--add_sampling_metadata",
        action="store_true",
        help="Whether to add sampling metadata to sidecar json. Sidecar json is always written.",
    )

    parser.add_argument("--cpu_offload", action="store_true", help="Enable model cpu offload.")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Text encoder max sequence length.")

    parser.add_argument(
        "--output_type",
        type=str,
        default="pil",
        choices=["pil", "np", "latent"],
        help="Pipeline output type before mp4 writing. Use pil/np for video saving.",
    )
    parser.add_argument(
        "--use_fast_loader",
        action="store_true",
        help="Use load_wan_pipeline_fast from util.py if it exists.",
    )

    parser.add_argument(
        "--scheduler",
        type=str,
        default="unipc",
        choices=["unipc", "default", "flowmatch", "euler"],
        help="Scheduler override. unipc follows the Wan example; default keeps loaded scheduler.",
    )
    parser.add_argument(
        "--flow_shift",
        type=float,
        default=3.0,
        help="Flow shift for UniPC scheduler. Use 3.0 for 480P and 5.0 for 720P.",
    )
    parser.add_argument(
        "--no_load_vae_fp32",
        action="store_true",
        help="Do not explicitly load the Wan VAE in fp32.",
    )

    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Global start index for multi-GPU launches.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Only use the first N prompts from prompt_file.",
    )

    # -------------------------------------------------------------------------
    # Cache / Taylor launcher compatibility arguments.
    # Keep this aligned with CogVideoX: CFGCache only needs proxy_tables_path.
    # -------------------------------------------------------------------------
    parser.add_argument(
        "--taylor_method",
        "--cache_mode",
        dest="taylor_method",
        type=str,
        default="original",
        choices=CACHE_METHODS,
        help="Choose cache method.",
    )
    parser.add_argument("--interval", type=int, default=3, help="Cache period length.")
    parser.add_argument("--max_order", type=int, default=1, help="Maximum order of Taylor expansion.")
    parser.add_argument("--first_enhance", type=int, default=3, help="Initial enhancement steps.")
    parser.add_argument(
        "--hicache_scale",
        type=float,
        default=0.5,
        help="Scaling factor for HiCache Hermite polynomials.",
    )
    parser.add_argument(
        "--rel_l1_thresh",
        type=float,
        default=0.2,
        help="TeaCache threshold.",
    )
    parser.add_argument(
        "--proxy_tables_path",
        type=str,
        default="/export/home/liuyiming54/RA-CFGCache/calibration/wan_rho_cfg5.npz",
        help="Path to packed all-cfg offline rho npz for CFGCache.",
    )

    parser.add_argument(
        "--no_save_eval_frames",
        action="store_true",
        help="Disable saving evaluation frames.",
    )
    parser.add_argument(
        "--eval_frame_stride",
        type=int,
        default=1,
        help="Save one frame every N frames for evaluation cache. Default saves all frames.",
    )
    parser.add_argument(
        "--eval_frame_format",
        type=str,
        default="png",
        choices=["png", "jpg"],
        help="Format for evaluation frame cache.",
    )
    parser.add_argument(
        "--eval_jpg_quality",
        type=int,
        default=95,
        help="JPEG quality when eval_frame_format=jpg.",
    )
    parser.add_argument(
        "--no_write_manifest",
        action="store_true",
        help="Disable writing manifest.jsonl.",
    )

    args = parser.parse_args()

    prompts = read_prompts(args.prompt_file)
    if args.limit is not None and args.limit > 0:
        prompts = prompts[: args.limit]

    negative_prompts = None
    if args.negative_prompt_file is not None:
        negative_prompts = read_prompts(args.negative_prompt_file)
        if args.limit is not None and args.limit > 0:
            negative_prompts = negative_prompts[: args.limit]

    seed = args.seed
    if seed is not None and seed < 0:
        seed = None

    opts = SamplingOptions(
        prompts=prompts,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        guidance_scale_2=args.guidance_scale_2,
        seed=seed,
        num_videos_per_prompt=args.num_videos_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        model_path=args.model_path,
        add_sampling_metadata=args.add_sampling_metadata,
        negative_prompt=args.negative_prompt,
        negative_prompts=negative_prompts,
        start_index=args.start_index,
        limit=args.limit,
        cpu_offload=args.cpu_offload,
        max_sequence_length=args.max_sequence_length,
        output_type=args.output_type,
        use_fast_loader=args.use_fast_loader,
        scheduler=args.scheduler,
        flow_shift=args.flow_shift,
        load_vae_fp32=not args.no_load_vae_fp32,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        taylor_method=args.taylor_method,
        hicache_scale=args.hicache_scale,
        rel_l1_thresh=args.rel_l1_thresh,
        proxy_tables_path=args.proxy_tables_path,
        save_eval_frames=not args.no_save_eval_frames,
        eval_frame_stride=args.eval_frame_stride,
        eval_frame_format=args.eval_frame_format,
        eval_jpg_quality=args.eval_jpg_quality,
        write_manifest=not args.no_write_manifest,
    )

    main(opts)


if __name__ == "__main__":
    app()
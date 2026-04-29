import os
import json
import time
import inspect
from pathlib import Path
from dataclasses import dataclass
from typing import Any, Optional

import imageio.v2 as imageio
import numpy as np
import torch
from PIL import Image
from tqdm import tqdm

from .pipeline_cogvideox import CogVideoXPipeline

from .cache_functions import cache_init
from .taylor_utils import pipeline_with_taylorseer

from .util import load_cogvideox_pipeline_fast



PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "cogvideox"


@dataclass
class SamplingOptions:
    prompts: list[str]
    width: int
    height: int
    num_frames: int
    fps: int
    num_steps: int
    guidance_scale: float
    seed: int | None
    num_videos_per_prompt: int
    batch_size: int
    model_name: str
    output_dir: str
    model_path: str
    add_sampling_metadata: bool
    negative_prompt: str | None
    negative_prompts: list[str] | None
    interval: int
    max_order: int
    first_enhance: int
    taylor_method: str
    hicache_scale: float
    rel_l1_thresh: float
    start_index: int
    limit: int | None
    cpu_offload: bool

    save_eval_frames: bool
    eval_frame_stride: int
    eval_frame_format: str
    eval_jpg_quality: int
    write_manifest: bool
    proxy_tables_path: str | None = None

def read_prompts(prompt_file: str) -> list[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_pipeline(model_path: str, use_fast_loader: bool = False):
    if use_fast_loader and load_cogvideox_pipeline_fast is not None:
        pipe = load_cogvideox_pipeline_fast(model_path, patch_cache=True)
        return pipe

    pipe = CogVideoXPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
        use_safetensors=True,
    )
    return pipe


def _to_uint8_frame(frame: Any) -> np.ndarray:
    if isinstance(frame, Image.Image):
        return np.array(frame.convert("RGB"), dtype=np.uint8)

    if isinstance(frame, np.ndarray):
        arr = frame
        if arr.dtype != np.uint8:
            if arr.max() <= 1.0:
                arr = (arr * 255.0).clip(0, 255).astype(np.uint8)
            else:
                arr = arr.clip(0, 255).astype(np.uint8)

        if arr.ndim == 3 and arr.shape[0] in (1, 3):  # CHW -> HWC
            arr = np.transpose(arr, (1, 2, 0))
        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)
        return arr

    if torch.is_tensor(frame):
        t = frame.detach().float().cpu()
        if t.ndim == 3 and t.shape[0] in (1, 3):  # CHW -> HWC
            t = t.permute(1, 2, 0)
        if t.ndim == 2:
            t = t.unsqueeze(-1).repeat(1, 1, 3)
        if t.max() <= 1.0:
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


def _write_mp4(frames: list[np.ndarray], video_filename: Path, fps: int):
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

def preload_cfgcache_rho_table(proxy_tables_path: str, cfg_scale: float):
    """
    Load packed all-cfg rho table once, then pick the nearest cfg-specific rho_table_at.
    Returns a dict directly consumable by proxy cfg["proxy_tables"].
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
            f"rho_table_at_all.shape[0] ({rho_table_at_all.shape[0]}) != len(cfg_scales) ({len(cfg_scales)})"
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

def _build_pipe_call_kwargs(
    pipe,
    *,
    batch_prompts: list[str],
    batch_negative_prompts: Optional[list[str]] = None,
    fallback_negative_prompt: Optional[str] = None,
    opts: SamplingOptions,
    generators,
    cache_dic,
    current,
    runtime_ctx=None,
):
    sig = inspect.signature(pipe.__call__)
    accepted = set(sig.parameters.keys())

    kwargs = {}

    if "prompt" in accepted:
        kwargs["prompt"] = batch_prompts

    neg_value = batch_negative_prompts if batch_negative_prompts is not None else fallback_negative_prompt
    if "negative_prompt" in accepted and neg_value is not None:
        kwargs["negative_prompt"] = neg_value

    if "guidance_scale" in accepted:
        kwargs["guidance_scale"] = opts.guidance_scale

    if "height" in accepted:
        kwargs["height"] = opts.height
    if "width" in accepted:
        kwargs["width"] = opts.width

    if "num_inference_steps" in accepted:
        kwargs["num_inference_steps"] = opts.num_steps

    if "num_frames" in accepted:
        kwargs["num_frames"] = opts.num_frames
    elif "video_length" in accepted:
        kwargs["video_length"] = opts.num_frames
    elif "video_num_frames" in accepted:
        kwargs["video_num_frames"] = opts.num_frames

    if "fps" in accepted:
        kwargs["fps"] = opts.fps
    elif "frame_rate" in accepted:
        kwargs["frame_rate"] = opts.fps

    if "num_videos_per_prompt" in accepted:
        kwargs["num_videos_per_prompt"] = opts.num_videos_per_prompt

    if "generator" in accepted:
        kwargs["generator"] = generators

    # CogVideoX 这里保持 PIL 输出更稳，后面统一转成 uint8 frame
    if "output_type" in accepted:
        kwargs["output_type"] = "pil"

    if "cache_dic" in accepted and cache_dic is not None:
        kwargs["cache_dic"] = cache_dic
    if "current" in accepted and current is not None:
        kwargs["current"] = current
    if "runtime_ctx" in accepted and runtime_ctx is not None:
        kwargs["runtime_ctx"] = runtime_ctx
    return kwargs


def main(opts: SamplingOptions):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    cfgcache_runtime = None
    if opts.taylor_method == "CFGCache" and opts.proxy_tables_path is not None:
        t0 = time.perf_counter()
        cfgcache_runtime = build_cfgcache_runtime_config(opts)
        print(f"[CFGCache] preload done in {time.perf_counter() - t0:.3f}s")

    model_path = opts.model_path or os.environ.get("COGVIDEOX_MODEL_PATH")
    if not model_path:
        raise ValueError(
            "CogVideoX model path is not set. "
            "Pass --model_path or set COGVIDEOX_MODEL_PATH."
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
    print(f"Loading CogVideoX pipeline from {model_path} on {device}...")
    pipe = _load_pipeline(model_path, use_fast_loader=True)

    if hasattr(pipe, "to") and not opts.cpu_offload:
        pipe = pipe.to(device)

    if opts.cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()

    print(f"Loaded CogVideoX pipeline from {model_path} on {device}")

    prompts = opts.prompts
    if opts.limit is not None and opts.limit > 0:
        prompts = prompts[: opts.limit]

    negative_prompts = opts.negative_prompts
    if negative_prompts is not None and opts.limit is not None and opts.limit > 0:
        negative_prompts = negative_prompts[: opts.limit]

    total_prompts = len(prompts)
    progress_bar = tqdm(total=total_prompts, desc="Generating videos")

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
                "num_steps": opts.num_steps,
                "interval": opts.interval,
                "max_order": opts.max_order,
                "first_enhance": opts.first_enhance,
                "hicache_scale": opts.hicache_scale,
                "rel_l1_thresh": opts.rel_l1_thresh,
            }

            if opts.taylor_method == "CFGCache":
                model_kwargs.update({
                    "true_cfg_scale": opts.guidance_scale,
                })

            cache_dic, current = cache_init(
                method=opts.taylor_method,
                model_kwargs=model_kwargs,
                cfgcache_runtime=cfgcache_runtime,
            )

        batch_prompt_indices = [
            opts.start_index + batch_start + i
            for i in range(batch_size_actual)
        ]
        runtime_ctx = {
            "prompt_indices": batch_prompt_indices,
            "dump_residual_cfg": {
                "enable": True,
                "root": "/export/home/liuyiming54/CFGCache-qwenimage/cfg3.5_cogvideox",
            },
        }
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
                "taylor_method": opts.taylor_method,
                "interval": opts.interval,
                "max_order": opts.max_order,
                "first_enhance": opts.first_enhance,
                "hicache_scale": opts.hicache_scale,
                "rel_l1_thresh": opts.rel_l1_thresh,
                "model_name": opts.model_name,
                "model_path": opts.model_path,
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
    avg_sec_per_video_sampling = time_stats["sampling"] / total_videos_done
    avg_sec_per_prompt_sampling = avg_sec_per_video_sampling * max(1, opts.num_videos_per_prompt)
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
    print(f"Avg sec / video sampling: {avg_sec_per_video_sampling:.3f} s")
    print(
        f"Avg sec / prompt sampling: {avg_sec_per_prompt_sampling:.3f} s  "
        f"(includes {opts.num_videos_per_prompt} video/prompt)"
    )
    print("================================================\n")

    progress_bar.close()
    print(f"Generated {total_videos_done} videos in {videos_dir}")
    print(f"Metadata saved in {metadata_dir}")
    if opts.save_eval_frames:
        print(f"Eval frames saved in {eval_frames_dir}")
    if opts.write_manifest:
        print(f"Manifest saved at {manifest_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate videos using the CogVideoX backend.")

    parser.add_argument(
        "--prompt_file",
        type=str,
        default="resources/prompts/prompt.txt",
        help="Path to the prompt text file.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default="",
        help="Negative prompt for guidance.",
    )
    parser.add_argument(
        "--negative_prompt_file",
        type=str,
        default=None,
        help="Optional: file containing per-prompt negative prompts, line by line.",
    )
    parser.add_argument("--width", type=int, default=720, help="Video width.")
    parser.add_argument("--height", type=int, default=480, help="Video height.")
    parser.add_argument("--num_frames", type=int, default=49, help="Number of video frames.")
    parser.add_argument("--fps", type=int, default=8, help="Video fps.")
    parser.add_argument("--num_steps", type=int, default=50, help="Number of sampling steps.")
    parser.add_argument("--guidance_scale", type=float, default=6.0, help="Guidance scale.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")
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
        default="cogvideox",
        choices=["cogvideox"],
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
        help="Path to the CogVideoX model checkpoint.",
    )
    parser.add_argument(
        "--add_sampling_metadata",
        action="store_true",
        help="Whether to add sampling metadata to sidecar json.",
    )
    parser.add_argument("--cpu_offload", action="store_true", help="Enable model cpu offload.")
    parser.add_argument("--interval", type=int, default=7, help="Cache period length.")
    parser.add_argument("--max_order", type=int, default=3, help="Maximum order of Taylor expansion.")
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
        default=0.6,
        help="TeaCache threshold.",
    )
    parser.add_argument(
        "--taylor_method",
        type=str,
        default="original",
        choices=[
            "original",
            "CFGCache",
            "ToCa",
            "Taylor",
            "GroupedTaylor",
            "Delta",
            "HiCache",
            "TeaCache",
            "DiCache",
            "MagCache",
        ],
        help="Choose cache method.",
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
    parser.add_argument(
        "--proxy_tables_path",
        type=str,
        default="/export/home/liuyiming54/CFGCache/calibration/cogvideox_rho_cfg6.npz",
        help="Path to packed all-cfg offline rho npz for CFGCache.",
    )
    # 默认就保存逐帧评测缓存 + manifest
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

    opts = SamplingOptions(
        prompts=prompts,
        width=args.width,
        height=args.height,
        num_frames=args.num_frames,
        fps=args.fps,
        num_steps=args.num_steps,
        guidance_scale=args.guidance_scale,
        seed=args.seed,
        num_videos_per_prompt=args.num_videos_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        model_path=args.model_path,
        add_sampling_metadata=args.add_sampling_metadata,
        negative_prompt=args.negative_prompt,
        negative_prompts=negative_prompts,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        taylor_method=args.taylor_method,
        hicache_scale=args.hicache_scale,
        rel_l1_thresh=args.rel_l1_thresh,
        start_index=args.start_index,
        limit=args.limit,
        cpu_offload=args.cpu_offload,
        save_eval_frames=not args.no_save_eval_frames,
        eval_frame_stride=args.eval_frame_stride,
        eval_frame_format=args.eval_frame_format,
        eval_jpg_quality=args.eval_jpg_quality,
        write_manifest=not args.no_write_manifest,
        proxy_tables_path=args.proxy_tables_path,
    )

    main(opts)
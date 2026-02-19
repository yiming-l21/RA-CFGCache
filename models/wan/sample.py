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
from .util import load_wan_pipeline_fast

try:
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
except Exception:
    UniPCMultistepScheduler = None


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "wan2.1"
DEFAULT_CALIBRATE_ROOT = PROJECT_ROOT / "calibration_rho" / "wan2.1"
DEFAULT_GT_CALIBRATE_ROOT = PROJECT_ROOT / "calibration_gt" / "wan2.1"

DEFAULT_NEGATIVE_PROMPT = (
    "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, "
    "images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, "
    "incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, "
    "misshapen limbs, fused fingers, still picture, messy background, three legs, many people "
    "in the background, walking backwards"
)


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

    calibrate_mode: str
    calibrate_root: str
    interval: int | None
    perturb_step: int | None

    save_eval_frames: bool
    eval_frame_stride: int
    eval_frame_format: str
    eval_jpg_quality: int
    write_manifest: bool


def read_prompts(prompt_file: str) -> list[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _ensure_branch_context(model):
    """Some local Wan pipelines call transformer.cache_context('cond'/'uncond').

    This is only a no-op compatibility shim for clean full inference / calibration.
    It does not enable any cache reuse behavior.
    """
    if model is None:
        return

    if not hasattr(model, "cache_context"):
        def cache_context(_name=""):
            return contextlib.nullcontext()

        model.cache_context = cache_context


def _load_pipeline(opts: SamplingOptions, device: str):
    model_path = opts.model_path

    if opts.use_fast_loader and load_wan_pipeline_fast is not None:
        sig = inspect.signature(load_wan_pipeline_fast)
        kwargs = {}
        # 关键：校准脚本不启用任何 cache patch。只允许 pipeline 内部已有的 calibration hook 工作。
        if "patch_cache" in sig.parameters:
            kwargs["patch_cache"] = False

        try:
            pipe = load_wan_pipeline_fast(model_path, **kwargs)
        except TypeError:
            pipe = load_wan_pipeline_fast(model_path)
    else:
        pipe = WanPipeline.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            local_files_only=True,
            use_safetensors=True,
        )

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

    _ensure_branch_context(getattr(pipe, "transformer", None))
    _ensure_branch_context(getattr(pipe, "transformer_2", None))

    if opts.cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    elif hasattr(pipe, "to"):
        pipe = pipe.to(device)

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
        arr = t.numpy()

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return arr

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

    if len(frames) > 0 and indices and indices[-1] != len(frames) - 1:
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


def build_runtime_ctx(
    *,
    opts: SamplingOptions,
    prompt_indices: list[int],
    gt_role: str | None = None,
    gt_run_name: str | None = None,
    gt_perturb_step: int | None = None,
):
    """Calibration-only runtime context.

    不向 pipeline 传任何 cache / Taylor / reuse 参数，只传 calibration 信息。

    rho:
        pipeline 内部保存 step_i_xt.pt / step_i_cond.pt / step_i_uncond.pt。

    gt:
        sample.py 外层负责 baseline + 多个 perturb run；
        pipeline 内部只负责在 perturb step 使用 guided_old 替换 guided_ref，
        并保存 guided_ref.pt / guided_old.pt / final_latents.pt。
    """
    if opts.calibrate_mode == "none":
        return None

    root = str(Path(opts.calibrate_root).expanduser())

    ctx = {
        "prompt_indices": prompt_indices,
        "calibrate_mode": opts.calibrate_mode,
        "calibrate_root": root,
        "model_name": opts.model_name,
        "guidance_scale": float(opts.guidance_scale),
    }

    if opts.calibrate_mode == "rho":
        ctx["dump_residual_cfg"] = {
            "enable": True,
            "root": root,
        }

    elif opts.calibrate_mode == "gt":
        if gt_role is None:
            gt_role = "baseline" if gt_perturb_step is None else "perturb"

        if gt_role not in {"baseline", "perturb"}:
            raise ValueError(f"Unsupported gt_role={gt_role!r}; expected baseline or perturb.")

        if gt_run_name is None:
            gt_run_name = "baseline" if gt_role == "baseline" else f"perturb_step_{int(gt_perturb_step):03d}"

        ctx["dump_gt_cfg"] = {
            "enable": True,
            "root": root,
            "role": gt_role,
            "run_name": gt_run_name,
            "run_id": gt_run_name,
            "is_baseline": gt_role == "baseline",
            "perturb_step": None if gt_perturb_step is None else int(gt_perturb_step),
        }

    else:
        raise ValueError(f"Unsupported calibrate_mode: {opts.calibrate_mode}")

    return ctx


def _build_pipe_call_kwargs(
    pipe,
    *,
    batch_prompts: list[str],
    batch_negative_prompts: Optional[list[str]],
    fallback_negative_prompt: Optional[str],
    opts: SamplingOptions,
    generators,
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
    if can_pass("true_cfg_scale"):
        kwargs["true_cfg_scale"] = opts.guidance_scale
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

    if can_pass("fps"):
        kwargs["fps"] = opts.fps
    elif can_pass("frame_rate"):
        kwargs["frame_rate"] = opts.fps

    if can_pass("num_videos_per_prompt"):
        kwargs["num_videos_per_prompt"] = opts.num_videos_per_prompt

    if can_pass("generator"):
        kwargs["generator"] = generators

    if can_pass("output_type"):
        kwargs["output_type"] = opts.output_type

    if can_pass("max_sequence_length"):
        kwargs["max_sequence_length"] = opts.max_sequence_length

    # calibration-only hook; no cache arguments are forwarded here.
    if can_pass("runtime_ctx") and runtime_ctx is not None:
        kwargs["runtime_ctx"] = runtime_ctx
    elif runtime_ctx is not None:
        print("[WARN] Pipeline __call__ does not accept runtime_ctx; calibration dump will be skipped.")

    return kwargs


def _format_perturb_step(step: int) -> str:
    return f"perturb_step_{int(step):03d}"


def build_gt_perturb_steps(
    num_denoise_steps: int,
    interval: int | None,
    perturb_step: int | None,
) -> list[int]:
    """构造 GT calibration 的扰动 step。

    step 是 0-based denoising loop index。
    step=0 没有上一轮 cond/uncond，因此不合法。
    如果 num_steps=50，合法范围是 1..49。
    """
    if interval is not None and perturb_step is not None:
        raise ValueError("calibrate_mode='gt' accepts only one of --interval or --perturb_step.")

    if num_denoise_steps <= 1:
        raise ValueError(f"GT calibration needs at least 2 steps, got {num_denoise_steps}.")

    if perturb_step is not None:
        perturb_step = int(perturb_step)
        if perturb_step <= 0 or perturb_step >= num_denoise_steps:
            raise ValueError(
                f"Invalid --perturb_step {perturb_step}. Valid range is [1, {num_denoise_steps - 1}]."
            )
        return [perturb_step]

    if interval is None:
        raise ValueError("calibrate_mode='gt' requires either --interval N or --perturb_step T.")

    interval = int(interval)
    if interval <= 0:
        raise ValueError(f"--interval must be positive, got {interval}.")

    steps = list(range(1, num_denoise_steps, interval))
    if len(steps) == 0:
        raise ValueError(f"--interval {interval} produced no perturb steps for {num_denoise_steps} steps.")
    return steps


def _cfg_trace_root(calibrate_root: str, guidance_scale: float) -> Path:
    root = Path(calibrate_root).expanduser()
    if root.name.startswith("cfg") and root.name.endswith("_traces"):
        return root
    return root / f"cfg{float(guidance_scale):.1f}_traces"


def _write_json_file(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _make_generators(
    *,
    seed: int | None,
    device: str,
    start_index: int,
    video_start: int,
    gen_count: int,
):
    """每次 baseline / perturb run 都重新构造 generator，保证初始噪声一致。"""
    if seed is None:
        return None

    return [
        torch.Generator(device=device).manual_seed(seed + start_index + video_start + i)
        for i in range(gen_count)
    ]


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
        "calibration": {
            "calibrate_mode": opts.calibrate_mode,
            "calibrate_root": opts.calibrate_root,
            "interval": opts.interval,
            "perturb_step": opts.perturb_step,
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

    model_path = opts.model_path or os.environ.get("WAN_MODEL_PATH") or os.environ.get("WAN2_1_MODEL_PATH")
    if not model_path:
        raise ValueError("Wan model path is not set. Pass --model_path or set WAN_MODEL_PATH / WAN2_1_MODEL_PATH.")
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

    total_prompts = len(prompts)
    if total_prompts == 0:
        raise ValueError("No prompts found.")

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

    Path(opts.calibrate_root).mkdir(parents=True, exist_ok=True)
    _write_run_config(opts, final_output_path)

    print(f"Loading Wan pipeline from {model_path} on {device}...")
    print(
        f"[calibration] mode={opts.calibrate_mode}, "
        f"root={opts.calibrate_root}, interval={opts.interval}, perturb_step={opts.perturb_step}"
    )

    pipe = _load_pipeline(opts, device=device)

    print(f"Loaded Wan pipeline from {model_path} on {device}")

    gt_steps = None
    if opts.calibrate_mode == "gt":
        gt_steps = build_gt_perturb_steps(
            num_denoise_steps=int(opts.num_steps),
            interval=opts.interval,
            perturb_step=opts.perturb_step,
        )

        trace_root = _cfg_trace_root(opts.calibrate_root, opts.guidance_scale)
        _write_json_file(
            trace_root / "gt_plan.json",
            {
                "model": opts.model_name,
                "num_denoise_steps": int(opts.num_steps),
                "perturb_steps": [int(x) for x in gt_steps],
                "step_indexing": (
                    "0-based denoising loop index; step 0 is invalid because "
                    "it has no previous cond/uncond prediction."
                ),
                "cfg_mode": "true_cfg_dual_branch",
                "guidance_scale": float(opts.guidance_scale),
                "guidance_scale_2": None if opts.guidance_scale_2 is None else float(opts.guidance_scale_2),
                "trace_root": str(trace_root),
            },
        )

    print(
        f"Start Wan full inference: "
        f"prompts={total_prompts}, "
        f"calibrate_mode={opts.calibrate_mode}, "
        f"output_dir={final_output_path}, "
        f"calibrate_root={opts.calibrate_root}"
    )

    progress_bar = tqdm(total=total_prompts, desc="Generating videos")

    time_stats = {
        "e2e": 0.0,
        "sampling": 0.0,
        "save": 0.0,
        "total_videos": 0,
    }

    t_e2e_start = time.perf_counter()
    videos_per_prompt = max(1, opts.num_videos_per_prompt)

    for batch_start in range(0, total_prompts, opts.batch_size):
        batch_end = min(batch_start + opts.batch_size, total_prompts)
        batch_prompts = prompts[batch_start:batch_end]
        batch_size_actual = len(batch_prompts)

        batch_negative_prompts = None
        if negative_prompts is not None:
            batch_negative_prompts = negative_prompts[batch_start:batch_end]

        gen_count = batch_size_actual * videos_per_prompt
        video_start = batch_start * videos_per_prompt

        batch_prompt_indices = [
            opts.start_index + batch_start + i
            for i in range(batch_size_actual)
        ]

        if opts.calibrate_mode == "gt":
            baseline_runtime_ctx = build_runtime_ctx(
                opts=opts,
                prompt_indices=batch_prompt_indices,
                gt_role="baseline",
                gt_run_name="baseline",
                gt_perturb_step=None,
            )

            baseline_generators = _make_generators(
                seed=opts.seed,
                device=device,
                start_index=opts.start_index,
                video_start=video_start,
                gen_count=gen_count,
            )

            baseline_kwargs = _build_pipe_call_kwargs(
                pipe,
                batch_prompts=batch_prompts,
                batch_negative_prompts=batch_negative_prompts,
                fallback_negative_prompt=opts.negative_prompt,
                opts=opts,
                generators=baseline_generators,
                runtime_ctx=baseline_runtime_ctx,
            )

            t_sample0 = time.perf_counter()
            with torch.inference_mode():
                result = pipe(**baseline_kwargs)
            time_stats["sampling"] += time.perf_counter() - t_sample0

            assert gt_steps is not None
            for step in gt_steps:
                run_name = _format_perturb_step(step)
                perturb_runtime_ctx = build_runtime_ctx(
                    opts=opts,
                    prompt_indices=batch_prompt_indices,
                    gt_role="perturb",
                    gt_run_name=run_name,
                    gt_perturb_step=step,
                )

                perturb_generators = _make_generators(
                    seed=opts.seed,
                    device=device,
                    start_index=opts.start_index,
                    video_start=video_start,
                    gen_count=gen_count,
                )

                perturb_kwargs = _build_pipe_call_kwargs(
                    pipe,
                    batch_prompts=batch_prompts,
                    batch_negative_prompts=batch_negative_prompts,
                    fallback_negative_prompt=opts.negative_prompt,
                    opts=opts,
                    generators=perturb_generators,
                    runtime_ctx=perturb_runtime_ctx,
                )

                t_sample0 = time.perf_counter()
                with torch.inference_mode():
                    _ = pipe(**perturb_kwargs)
                time_stats["sampling"] += time.perf_counter() - t_sample0

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

        else:
            generators = _make_generators(
                seed=opts.seed,
                device=device,
                start_index=opts.start_index,
                video_start=video_start,
                gen_count=gen_count,
            )

            runtime_ctx = build_runtime_ctx(
                opts=opts,
                prompt_indices=batch_prompt_indices,
            )

            pipe_kwargs = _build_pipe_call_kwargs(
                pipe,
                batch_prompts=batch_prompts,
                batch_negative_prompts=batch_negative_prompts,
                fallback_negative_prompt=opts.negative_prompt,
                opts=opts,
                generators=generators,
                runtime_ctx=runtime_ctx,
            )

            t_sample0 = time.perf_counter()
            with torch.inference_mode():
                result = pipe(**pipe_kwargs)
            time_stats["sampling"] += time.perf_counter() - t_sample0

        videos = _extract_video_batch(result)
        t_save0 = time.perf_counter()

        for i, frames in enumerate(videos):
            prompt_i = i // videos_per_prompt
            prompt_global_idx = opts.start_index + batch_start + prompt_i
            global_video_idx = opts.start_index + video_start + i
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

            if prompt_i < len(batch_prompts):
                prompt_text = batch_prompts[prompt_i]
            else:
                prompt_text = ""

            if batch_negative_prompts is not None and prompt_i < len(batch_negative_prompts):
                neg_text = batch_negative_prompts[prompt_i]
            else:
                neg_text = opts.negative_prompt or ""

            metadata = {
                "video_id": video_stem,
                "video_index": global_video_idx,
                "prompt_index": prompt_global_idx,
                "prompt": prompt_text,
                "negative_prompt": neg_text,
                "seed": None if opts.seed is None else opts.seed + global_video_idx,
                "width": opts.width,
                "height": opts.height,
                "num_frames": opts.num_frames,
                "fps": opts.fps,
                "num_steps": opts.num_steps,
                "guidance_scale": opts.guidance_scale,
                "guidance_scale_2": opts.guidance_scale_2,
                "calibrate_mode": opts.calibrate_mode,
                "calibrate_root": str(opts.calibrate_root),
                "model_name": opts.model_name,
                "model_path": opts.model_path,
                "scheduler": opts.scheduler,
                "flow_shift": opts.flow_shift,
                "max_sequence_length": opts.max_sequence_length,
                "output_type": opts.output_type,
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
                        "video_index": global_video_idx,
                        "prompt_index": prompt_global_idx,
                        "prompt": prompt_text,
                        "video_path": metadata["video_path"],
                        "metadata_path": metadata["metadata_path"],
                        "eval_frame_dir": metadata["eval_frame_dir"],
                    },
                )

            time_stats["total_videos"] += 1

        time_stats["save"] += time.perf_counter() - t_save0
        progress_bar.update(batch_size_actual)

    progress_bar.close()

    time_stats["e2e"] = time.perf_counter() - t_e2e_start

    total_videos_done = max(1, time_stats["total_videos"])
    avg_sec_per_video = time_stats["e2e"] / total_videos_done

    marker_path = final_output_path / ".full_output_dir"
    marker_path.write_text(str(final_output_path.resolve()), encoding="utf-8")

    print("\n================ Timing Summary ================")
    print(f"Total elapsed:        {time_stats['e2e']:.3f} s")
    print(f"Sampling time:        {time_stats['sampling']:.3f} s")
    print(f"Saving time:          {time_stats['save']:.3f} s")
    print(f"Total videos:         {time_stats['total_videos']}")
    print(f"Total prompts:        {total_prompts}")
    print(f"Avg sec / video:      {avg_sec_per_video:.3f} s")
    print("================================================\n")

    timing_payload = {
        "total_sec": time_stats["e2e"],
        "sampling_sec": time_stats["sampling"],
        "save_sec": time_stats["save"],
        "total_videos": time_stats["total_videos"],
        "total_prompts": total_prompts,
        "avg_sec_per_video": avg_sec_per_video,
    }
    with open(final_output_path / "timing.json", "w", encoding="utf-8") as f:
        json.dump(timing_payload, f, ensure_ascii=False, indent=2)

    print(f"Generated videos in:  {videos_dir}")
    print(f"Metadata saved in:    {metadata_dir}")

    if opts.save_eval_frames:
        print(f"Eval frames saved in: {eval_frames_dir}")

    if opts.write_manifest:
        print(f"Manifest saved at:    {manifest_path}")

    print(f"Full output marker:   {marker_path}")


def app():
    import argparse

    parser = argparse.ArgumentParser(
        description="Wan full inference + calibration dump. No cache logic."
    )

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
        help="Optional file containing per-prompt negative prompts.",
    )

    parser.add_argument(
        "--calibrate_mode",
        type=str,
        default="none",
        choices=["none", "rho", "gt"],
        help="Calibration mode: none | rho | gt.",
    )
    parser.add_argument(
        "--calibrate_root",
        type=str,
        default=str(DEFAULT_CALIBRATE_ROOT),
        help="Directory to save calibration tensors/logs.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="GT curve point interval. Only used when calibrate_mode=gt.",
    )
    parser.add_argument(
        "--perturb_step",
        type=int,
        default=None,
        help="Specific perturb step for GT curve calibration. Only used when calibrate_mode=gt.",
    )

    parser.add_argument("--width", type=int, default=832, help="Video width. Wan2.1 480P common width: 832.")
    parser.add_argument("--height", type=int, default=480, help="Video height. Wan2.1 480P common height: 480.")
    parser.add_argument("--num_frames", type=int, default=81, help="Number of video frames. Wan expects num_frames % 4 == 1.")
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

    parser.add_argument("--cpu_offload", action="store_true", help="Enable model CPU offload.")
    parser.add_argument("--max_sequence_length", type=int, default=512, help="Text encoder max sequence length.")

    parser.add_argument(
        "--output_type",
        type=str,
        default="pil",
        choices=["pil", "np", "latent"],
        help="Pipeline output type before mp4 writing. Use pil/np for video saving.",
    )
    parser.add_argument(
        "--no_fast_loader",
        action="store_true",
        help="Disable load_wan_pipeline_fast and use WanPipeline.from_pretrained directly.",
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
        help="Do not explicitly load the Wan VAE in fp32. Kept for launcher compatibility.",
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
        help="Format for evaluation frames.",
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

    if args.calibrate_mode == "gt":
        if args.interval is None and args.perturb_step is None:
            raise ValueError("--calibrate_mode gt requires either --interval N or --perturb_step N.")
        if args.interval is not None and args.perturb_step is not None:
            raise ValueError("--calibrate_mode gt accepts only one of --interval or --perturb_step.")

        # rho 默认路径不变；gt 若未显式传 calibrate_root，则自动切到 calibration_gt/wan2.1。
        if str(args.calibrate_root) == str(DEFAULT_CALIBRATE_ROOT):
            args.calibrate_root = str(DEFAULT_GT_CALIBRATE_ROOT)

    if args.calibrate_mode == "rho":
        if args.interval is not None or args.perturb_step is not None:
            print(
                "[WARN] calibrate_mode=rho only needs prompt_file. "
                "--interval / --perturb_step will not be used for rho."
            )

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
        negative_prompt=args.negative_prompt,
        negative_prompts=negative_prompts,
        start_index=args.start_index,
        limit=args.limit,
        cpu_offload=args.cpu_offload,
        max_sequence_length=args.max_sequence_length,
        output_type=args.output_type,
        use_fast_loader=not args.no_fast_loader,
        scheduler=args.scheduler,
        flow_shift=args.flow_shift,
        load_vae_fp32=not args.no_load_vae_fp32,
        calibrate_mode=args.calibrate_mode,
        calibrate_root=args.calibrate_root,
        interval=args.interval,
        perturb_step=args.perturb_step,
        save_eval_frames=not args.no_save_eval_frames,
        eval_frame_stride=args.eval_frame_stride,
        eval_frame_format=args.eval_frame_format,
        eval_jpg_quality=args.eval_jpg_quality,
        write_manifest=not args.no_write_manifest,
    )

    main(opts)


if __name__ == "__main__":
    app()

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
from .util import load_cogvideox_pipeline_fast


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "results" / "cogvideox"
DEFAULT_CALIBRATE_ROOT = PROJECT_ROOT / "calibration_rho" / "cogvideox"
DEFAULT_GT_CALIBRATE_ROOT = PROJECT_ROOT / "calibration_gt" / "cogvideox"


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

    negative_prompt: str | None
    negative_prompts: list[str] | None

    start_index: int
    limit: int | None
    cpu_offload: bool

    calibrate_mode: str
    calibrate_root: str
    interval: int | None
    perturb_step: int | None

    save_eval_frames: bool
    eval_frame_stride: int
    eval_frame_format: str
    eval_jpg_quality: int
    write_manifest: bool

    use_fast_loader: bool


def read_prompts(prompt_file: str) -> list[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]


def _load_pipeline(model_path: str, use_fast_loader: bool = True):
    """
    Clean full-inference loader.

    注意：
    这里不再显式启用任何 cache 运行参数。
    如果你的 load_cogvideox_pipeline_fast 内部仍然做了 monkey patch，
    需要保证它只用于 calibration hook，不做复用。
    """
    if use_fast_loader and load_cogvideox_pipeline_fast is not None:
        try:
            return load_cogvideox_pipeline_fast(model_path, patch_cache=False)
        except TypeError:
            return load_cogvideox_pipeline_fast(model_path)

    return CogVideoXPipeline.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        local_files_only=True,
        use_safetensors=True,
    )


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

        if arr.ndim == 3 and arr.shape[0] in (1, 3):
            arr = np.transpose(arr, (1, 2, 0))

        if arr.ndim == 2:
            arr = np.stack([arr] * 3, axis=-1)

        if arr.ndim == 3 and arr.shape[-1] == 1:
            arr = np.repeat(arr, 3, axis=-1)

        return arr

    if torch.is_tensor(frame):
        t = frame.detach().float().cpu()

        if t.ndim == 3 and t.shape[0] in (1, 3):
            t = t.permute(1, 2, 0)

        if t.ndim == 2:
            t = t.unsqueeze(-1).repeat(1, 1, 3)

        if t.max() <= 1.0:
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
            if arr.shape[-1] in (1, 3):
                return [_to_uint8_frame(arr[i]) for i in range(arr.shape[0])]

            if arr.shape[1] in (1, 3):
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
        if data.ndim == 5:
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


def build_runtime_ctx(
    *,
    opts: SamplingOptions,
    prompt_indices: list[int],
    gt_role: str | None = None,
    gt_run_name: str | None = None,
    gt_perturb_step: int | None = None,
):
    """
    Calibration-only runtime context.

    不改任何 cache / 复用逻辑，只把校准信息传给 pipeline。

    rho:
        pipeline 内部保存 step_i_xt.pt / step_i_cond.pt / step_i_uncond.pt

    gt:
        sample.py 外层负责 baseline + 多个 perturb run；
        pipeline 内部只负责在 perturb step 用 guided_old 替换 guided_ref，
        并保存 guided_ref.pt / guided_old.pt / final_latents.pt。
    """
    if opts.calibrate_mode == "none":
        return None

    root = str(Path(opts.calibrate_root).expanduser())

    ctx = {
        "prompt_indices": prompt_indices,
        "calibrate_mode": opts.calibrate_mode,
        "calibrate_root": root,
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

        # 注意：GT 模式不要打开 dump_residual_cfg，避免额外保存 rho 张量。

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
    runtime_ctx=None,
):
    sig = inspect.signature(pipe.__call__)
    accepted = set(sig.parameters.keys())

    kwargs = {}

    if "prompt" in accepted:
        kwargs["prompt"] = batch_prompts

    neg_value = (
        batch_negative_prompts
        if batch_negative_prompts is not None
        else fallback_negative_prompt
    )

    if "negative_prompt" in accepted and neg_value is not None:
        kwargs["negative_prompt"] = neg_value

    if "guidance_scale" in accepted:
        kwargs["guidance_scale"] = opts.guidance_scale

    # 兼容改过的 CogVideoX pipeline：如果它显式支持 true_cfg_scale，
    # 这里用原 CogVideoX 的 guidance_scale 作为 true CFG scale，
    # 不新增命令行参数，不改变原启动脚本。
    if "true_cfg_scale" in accepted:
        kwargs["true_cfg_scale"] = opts.guidance_scale

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

    if "output_type" in accepted:
        kwargs["output_type"] = "pil"

    if "runtime_ctx" in accepted and runtime_ctx is not None:
        kwargs["runtime_ctx"] = runtime_ctx

    return kwargs


def _format_perturb_step(step: int) -> str:
    return f"perturb_step_{int(step):03d}"


def build_gt_perturb_steps(
    num_denoise_steps: int,
    interval: int | None,
    perturb_step: int | None,
) -> list[int]:
    """
    构造 GT calibration 的扰动 step。

    step 是 0-based denoising loop index。
    step=0 没有上一轮 cond/uncond，所以不合法。
    如果 num_steps=50，合法范围是 1..49。
    """
    if interval is not None and perturb_step is not None:
        raise ValueError(
            "calibrate_mode='gt' accepts only one of --interval or --perturb_step."
        )

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
        raise ValueError(
            f"--interval {interval} produced no perturb steps for {num_denoise_steps} steps."
        )
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
    batch_start: int,
    gen_count: int,
):
    """每次 baseline / perturb run 都重新构造 generator，保证初始噪声一致。"""
    if seed is None:
        return None

    return [
        torch.Generator(device=device).manual_seed(seed + start_index + batch_start + i)
        for i in range(gen_count)
    ]


def main(opts: SamplingOptions):
    device = "cuda" if torch.cuda.is_available() else "cpu"

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

    Path(opts.calibrate_root).mkdir(parents=True, exist_ok=True)

    print(f"Loading CogVideoX pipeline from {model_path} on {device}...")
    pipe = _load_pipeline(model_path, use_fast_loader=opts.use_fast_loader)

    # 保持原有设备逻辑，不额外强制改动。
    if opts.cpu_offload and hasattr(pipe, "enable_model_cpu_offload"):
        pipe.enable_model_cpu_offload()
    elif hasattr(pipe, "to"):
        pipe = pipe.to(device)

    print(f"Loaded CogVideoX pipeline from {model_path} on {device}")

    prompts = opts.prompts
    if opts.limit is not None and opts.limit > 0:
        prompts = prompts[: opts.limit]

    negative_prompts = opts.negative_prompts
    if negative_prompts is not None and opts.limit is not None and opts.limit > 0:
        negative_prompts = negative_prompts[: opts.limit]

    total_prompts = len(prompts)
    if total_prompts == 0:
        raise ValueError("No prompts found.")

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
                "model": "cogvideox",
                "num_denoise_steps": int(opts.num_steps),
                "perturb_steps": [int(x) for x in gt_steps],
                "step_indexing": (
                    "0-based denoising loop index; step 0 is invalid because "
                    "it has no previous cond/uncond prediction."
                ),
                "cfg_mode": "true_cfg_dual_branch",
                "guidance_scale": float(opts.guidance_scale),
                "trace_root": str(trace_root),
            },
        )

    print(
        f"Start CogVideoX full inference: "
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

    for batch_start in range(0, total_prompts, opts.batch_size):
        batch_end = min(batch_start + opts.batch_size, total_prompts)
        batch_prompts = prompts[batch_start:batch_end]
        batch_size_actual = len(batch_prompts)

        batch_negative_prompts = None
        if negative_prompts is not None:
            batch_negative_prompts = negative_prompts[batch_start:batch_end]

        gen_count = batch_size_actual * max(1, opts.num_videos_per_prompt)

        batch_prompt_indices = [
            opts.start_index + batch_start + i
            for i in range(batch_size_actual)
        ]

        # ------------------------------------------------------------
        # Build and run pipeline.
        # none/rho: 保持原来单次调用逻辑。
        # gt: 仿照 Qwen sample.py，baseline 跑一次，随后每个 perturb_step 独立重跑。
        # ------------------------------------------------------------
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
                batch_start=batch_start,
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

            # perturb runs only generate calibration tensors in pipeline.
            # 不保存 perturb 视频，不改变原输出目录中的视频语义。
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
                    batch_start=batch_start,
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
                batch_start=batch_start,
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
            prompt_i = i // max(1, opts.num_videos_per_prompt)
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
                "prompt_index": global_video_idx,
                "prompt": prompt_text,
                "negative_prompt": neg_text,
                "seed": None if opts.seed is None else opts.seed + global_video_idx,
                "width": opts.width,
                "height": opts.height,
                "num_frames": opts.num_frames,
                "fps": opts.fps,
                "num_steps": opts.num_steps,
                "guidance_scale": opts.guidance_scale,
                "calibrate_mode": opts.calibrate_mode,
                "calibrate_root": str(opts.calibrate_root),
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

    print(f"Generated videos in:  {videos_dir}")
    print(f"Metadata saved in:    {metadata_dir}")

    if opts.save_eval_frames:
        print(f"Eval frames saved in: {eval_frames_dir}")

    if opts.write_manifest:
        print(f"Manifest saved at:    {manifest_path}")

    print(f"Full output marker:   {marker_path}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="CogVideoX full inference + calibration dump. No cache logic."
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
        default="",
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

    parser.add_argument("--cpu_offload", action="store_true", help="Enable model CPU offload.")

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
        "--no_fast_loader",
        action="store_true",
        help="Disable fast loader and use CogVideoXPipeline.from_pretrained directly.",
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
            raise ValueError(
                "--calibrate_mode gt requires either --interval N or --perturb_step N."
            )
        if args.interval is not None and args.perturb_step is not None:
            raise ValueError(
                "--calibrate_mode gt accepts only one of --interval or --perturb_step."
            )

        # 保持 rho 默认路径不变；仅当 gt 没显式传 calibrate_root 时，
        # 默认切到 calibration_gt/cogvideox，与 Qwen/FLUX GT 目录一致。
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
        negative_prompt=args.negative_prompt,
        negative_prompts=negative_prompts,
        start_index=args.start_index,
        limit=args.limit,
        cpu_offload=args.cpu_offload,
        calibrate_mode=args.calibrate_mode,
        calibrate_root=args.calibrate_root,
        interval=args.interval,
        perturb_step=args.perturb_step,
        save_eval_frames=not args.no_save_eval_frames,
        eval_frame_stride=args.eval_frame_stride,
        eval_frame_format=args.eval_frame_format,
        eval_jpg_quality=args.eval_jpg_quality,
        write_manifest=not args.no_write_manifest,
        use_fast_loader=not args.no_fast_loader,
    )

    main(opts)
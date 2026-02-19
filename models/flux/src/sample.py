import os
import time
import json
import inspect
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
import torch
from einops import rearrange
from PIL import ExifTags, Image
from transformers import pipeline
from tqdm import tqdm

from flux.sampling import get_noise, get_schedule, prepare, unpack
from flux.cache_denoise import denoise, denoise_cfg
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5


NSFW_THRESHOLD = 0.85


@dataclass
class SamplingOptions:
    prompts: list[str]
    width: int
    height: int
    num_steps: int | None
    guidance: float
    seed: int | None
    num_images_per_prompt: int
    batch_size: int
    model_name: str
    output_dir: str
    start_index: int
    add_sampling_metadata: bool
    true_cfg_scale: float
    negative_prompt: str | None
    negative_prompts: list[str] | None
    use_nsfw_filter: bool
    # Calibration options
    calibrate_mode: str          # none | rho | gt
    interval: int | None         # GT 曲线描点间隔
    perturb_step: int | None     # GT 曲线指定扰动步


@contextmanager
def temporary_env(env_updates: dict[str, str | None]):
    """Temporarily set environment variables for denoise-side calibration saving.

    sample.py should not assume the exact internal implementation of denoise_cfg.
    If denoise_cfg reads these env vars, it can save guided pred tensors into the
    same directory that sample.py uses for final latent/metric files.
    """
    old_values: dict[str, str | None] = {}
    for key, value in env_updates.items():
        old_values[key] = os.environ.get(key)
        if value is None:
            os.environ.pop(key, None)
        else:
            os.environ[key] = str(value)
    try:
        yield
    finally:
        for key, old_value in old_values.items():
            if old_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = old_value


def _clone_tensor_dict(inp: dict) -> dict:
    """Clone tensor values that may be mutated by denoise, keep embeddings shared otherwise."""
    out = {}
    for k, v in inp.items():
        out[k] = v.clone() if torch.is_tensor(v) else v
    return out


def _format_step(step: int) -> str:
    return f"step_{step:03d}"


def build_gt_perturb_steps(num_denoise_steps: int, interval: int | None, perturb_step: int | None) -> list[int]:
    """Build 0-based denoise step indices for GT calibration.

    A perturb at step t reuses the previous step's two branch predictions, so t=0
    is invalid. For a 50-step sampler, legal steps are 1..49.
    """
    if interval is not None and perturb_step is not None:
        raise ValueError(
            "Please pass only one of --interval or --perturb_step for calibrate_mode='gt'. "
            "--interval means sweep multiple perturb steps; --perturb_step means run one step."
        )

    if num_denoise_steps <= 1:
        raise ValueError(f"GT calibration needs at least 2 denoise steps, got {num_denoise_steps}.")

    if perturb_step is not None:
        if perturb_step <= 0 or perturb_step >= num_denoise_steps:
            raise ValueError(
                f"Invalid --perturb_step {perturb_step}. Valid range is "
                f"[1, {num_denoise_steps - 1}] because step 0 has no previous pred to reuse."
            )
        return [int(perturb_step)]

    if interval is None:
        raise ValueError("GT calibration requires --interval N or --perturb_step T.")
    if interval <= 0:
        raise ValueError(f"--interval must be positive, got {interval}.")

    steps = list(range(1, num_denoise_steps, int(interval)))
    if len(steps) == 0:
        raise ValueError(
            f"--interval {interval} produced no perturb steps for {num_denoise_steps} denoise steps. "
            f"Use an interval <= {num_denoise_steps - 1}."
        )
    return steps


def _call_denoise_cfg(
    model,
    inp_cond: dict,
    inp_uncond: dict,
    timesteps,
    guidance: float,
    true_cfg_scale: float,
    calibrate_mode: str,
    perturb_step: int | None,
    prompt_ids: list[int],
    gt_save_dir: str | None = None,
    gt_run_name: str | None = None,
):
    """Call denoise_cfg with backwards-compatible optional GT-save kwargs.

    If your flux.cache_denoise.denoise_cfg already supports gt_save_dir/gt_run_name,
    sample.py passes them directly. If not, this function silently falls back to the
    current signature and exposes the path via environment variables.
    """
    cond = _clone_tensor_dict(inp_cond)
    uncond = _clone_tensor_dict(inp_uncond)

    kwargs = dict(
        img=cond["img"],
        img_ids=cond["img_ids"],
        txt=cond["txt"],
        txt_ids=cond["txt_ids"],
        vec=cond["vec"],
        neg_txt=uncond["txt"],
        neg_txt_ids=uncond["txt_ids"],
        neg_vec=uncond["vec"],
        true_cfg_scale=true_cfg_scale,
        timesteps=timesteps,
        guidance=guidance,
        calibrate_mode=calibrate_mode,
        interval=None,
        perturb_step=perturb_step,
        prompt_ids=prompt_ids,
    )

    sig = inspect.signature(denoise_cfg)
    if gt_save_dir is not None and "gt_save_dir" in sig.parameters:
        kwargs["gt_save_dir"] = gt_save_dir
    if gt_run_name is not None and "gt_run_name" in sig.parameters:
        kwargs["gt_run_name"] = gt_run_name

    env_updates = {
        "CFG_CACHE_GT_SAVE_DIR": gt_save_dir,
        "CFG_CACHE_GT_RUN_NAME": gt_run_name,
    }
    with temporary_env(env_updates):
        return denoise_cfg(model, **kwargs)


def _save_batch_latents(latents: torch.Tensor, gt_root: str, prompt_ids: list[int], rel_dir: str, filename: str):
    latents_cpu = latents.detach().cpu()
    for local_i, prompt_id in enumerate(prompt_ids):
        out_dir = os.path.join(gt_root, f"prompt_{prompt_id:06d}", rel_dir)
        os.makedirs(out_dir, exist_ok=True)
        torch.save(latents_cpu[local_i: local_i + 1].contiguous(), os.path.join(out_dir, filename))


def _write_json(path: str, payload: dict):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

def _get_project_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../..")
    )

def run_gt_calibration(
    model,
    inp_cond: dict,
    inp_uncond: dict,
    timesteps,
    guidance: float,
    true_cfg_scale: float,
    output_dir: str,
    prompt_ids: list[int],
    interval: int | None,
    perturb_step: int | None,
) -> torch.Tensor:
    """Run full CFG baseline once, then run one CFG perturb inference for each GT step.

    GT perturb semantics are implemented inside denoise_cfg:
      at perturb step t, current cond/uncond branch preds are replaced by the cached
      cond/uncond branch preds from step t-1; denoise_cfg should save guided pred(t-1)
      and guided pred(t) after CFG composition.

    sample.py is responsible for:
      1) expanding --interval into multiple independent perturb runs;
      2) keeping the same prompt/seed/initial latent for every run;
      3) dumping full/perturbed final latents.

    No GT value or distance metric is computed here; sample.py only runs the required trajectories and dumps tensors.
    """
    gt_root = Path(_get_project_root()) / "calibration_gt" / "flux" / f"cfg{true_cfg_scale:.1f}_traces"
    os.makedirs(gt_root, exist_ok=True)

    num_denoise_steps = len(timesteps) - 1
    steps = build_gt_perturb_steps(num_denoise_steps, interval, perturb_step)

    plan = {
        "num_denoise_steps": int(num_denoise_steps),
        "perturb_steps": [int(s) for s in steps],
        "step_indexing": "0-based denoise loop index; interval sweep starts from 1, then 1+interval, 1+2*interval, ...; valid perturb step range is 1..num_denoise_steps-1",
        "prompt_ids": [int(x) for x in prompt_ids],
        "cfg_mode": "true_cfg_dual_branch",
        "true_cfg_scale": float(true_cfg_scale),
    }
    _write_json(os.path.join(gt_root, "gt_plan.json"), plan)

    print("[GT] Running full no-cache baseline once ...")
    baseline_latents = _call_denoise_cfg(
        model=model,
        inp_cond=inp_cond,
        inp_uncond=inp_uncond,
        timesteps=timesteps,
        guidance=guidance,
        true_cfg_scale=true_cfg_scale,
        calibrate_mode="none",
        perturb_step=None,
        prompt_ids=prompt_ids,
        gt_save_dir=os.path.join(gt_root, "baseline"),
        gt_run_name="baseline",
    )
    _save_batch_latents(
        baseline_latents,
        gt_root=gt_root,
        prompt_ids=prompt_ids,
        rel_dir="baseline",
        filename="final_latents.pt",
    )
    print(f"[GT] Baseline latents saved to: {gt_root}")

    for step in steps:
        step_name = _format_step(step)
        perturb_run_dir = os.path.join(gt_root, step_name)
        print(f"[GT] Running perturb inference for {step_name} ...")

        perturb_latents = _call_denoise_cfg(
            model=model,
            inp_cond=inp_cond,
            inp_uncond=inp_uncond,
            timesteps=timesteps,
            guidance=guidance,
            true_cfg_scale=true_cfg_scale,
            calibrate_mode="gt",
            perturb_step=step,
            prompt_ids=prompt_ids,
            gt_save_dir=perturb_run_dir,
            gt_run_name=step_name,
        )

        del perturb_latents
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"[GT] Calibration files saved to: {gt_root}")
    return baseline_latents


def main(opts: SamplingOptions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    # ------------------------------------------------------------
    # Calibration config
    # ------------------------------------------------------------
    if opts.calibrate_mode not in ["none", "rho", "gt"]:
        raise ValueError(
            f"Unsupported calibrate_mode: {opts.calibrate_mode}. "
            "Expected one of: none, rho, gt."
        )

    if opts.calibrate_mode == "gt":
        if opts.interval is None and opts.perturb_step is None:
            raise ValueError(
                "calibrate_mode='gt' requires either --interval or --perturb_step."
            )
        if opts.interval is not None and opts.perturb_step is not None:
            raise ValueError(
                "calibrate_mode='gt' accepts only one of --interval or --perturb_step. "
                "--interval sweeps multiple perturb steps; --perturb_step runs exactly one perturb inference."
            )
        if opts.true_cfg_scale <= 1.0:
            raise ValueError(
                "calibrate_mode='gt' is for CFG calibration and requires --true_cfg_scale > 1. "
                "Baseline and every perturb run will go through denoise_cfg dual-branch inference."
            )

    if opts.calibrate_mode == "rho":
        if opts.interval is not None or opts.perturb_step is not None:
            print(
                "[WARN] calibrate_mode='rho' only needs prompt_file. "
                "--interval / --perturb_step will be ignored unless your denoise code uses them."
            )

    if opts.calibrate_mode != "none":
        print("=============== Calibration Config ===============")
        print(f"calibrate_mode: {opts.calibrate_mode}")
        if opts.calibrate_mode == "gt":
            print(f"interval:       {opts.interval}")
            print(f"perturb_step:   {opts.perturb_step}")
            print(f"true_cfg_scale: {opts.true_cfg_scale}")
            print("GT mode:        dual-branch CFG baseline + dual-branch CFG perturb runs")
        print("==================================================")

    # ------------------------------------------------------------
    # Optional NSFW classifier
    # ------------------------------------------------------------
    if opts.use_nsfw_filter:
        try:
            from pathlib import Path

            project_root = Path(__file__).resolve().parents[3]
            nsfw_local_dir = project_root / "weights" / "Falconsai" / "nsfw_image_detection"
            model_id = str(nsfw_local_dir) if nsfw_local_dir.is_dir() else "Falconsai/nsfw_image_detection"

            pipeline_device = 0 if device.type == "cuda" else -1
            nsfw_classifier = pipeline(
                "image-classification",
                model=model_id,
                device=pipeline_device,
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize NSFW classifier, disabling NSFW filter: {e!r}")
            nsfw_classifier = None
    else:
        nsfw_classifier = None

    # ------------------------------------------------------------
    # Model config
    # ------------------------------------------------------------
    model_name = opts.model_name
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {model_name}, available options: {available}")

    if opts.num_steps is None:
        opts.num_steps = 4 if model_name == "flux-schnell" else 50

    opts.width = 16 * (opts.width // 16)
    opts.height = 16 * (opts.height // 16)

    os.makedirs(opts.output_dir, exist_ok=True)
    output_name = os.path.join(opts.output_dir, "img_{idx}.jpg")

    save_sampling_config(opts, opts.output_dir)

    # ------------------------------------------------------------
    # Load model components
    # ------------------------------------------------------------
    torch_device = device

    t5 = load_t5(torch_device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(torch_device)
    model = load_flow_model(model_name, device=torch_device)
    ae = load_ae(model_name, device=torch_device)

    # ------------------------------------------------------------
    # Seed
    # ------------------------------------------------------------
    if opts.seed is not None:
        base_seed = opts.seed
    else:
        base_seed = torch.randint(0, 2**32, (1,)).item()

    prompts = opts.prompts
    total_images = len(prompts) * opts.num_images_per_prompt
    progress_bar = tqdm(total=total_images, desc="Generating images")

    # ------------------------------------------------------------
    # Timing
    # ------------------------------------------------------------
    t_run_start = time.perf_counter()
    time_stats = {
        "prepare": 0.0,
        "denoise": 0.0,
        "decode": 0.0,
        "post": 0.0,
        "nsfw": 0.0,
        "save": 0.0,
        "total_images": 0,
        "total_prompts": len(prompts),
        "num_images_per_prompt": opts.num_images_per_prompt,
        "batch_size": opts.batch_size,
    }

    num_prompt_batches = (len(prompts) + opts.batch_size - 1) // opts.batch_size

    idx = opts.start_index

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(prompts))

        batch_prompts = prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Global prompt ids for rho calibration.
        batch_prompt_ids = list(range(
            opts.start_index + prompt_start,
            opts.start_index + prompt_end,
        ))

        for _ in range(opts.num_images_per_prompt):
            seed = base_seed + idx
            current_start_index = idx
            idx += num_prompts_in_batch

            batch_size = num_prompts_in_batch

            # Use image/sample ids for GT calibration so repeated images per prompt do not overwrite each other.
            batch_sample_ids = list(range(
                current_start_index,
                current_start_index + num_prompts_in_batch,
            ))

            x = get_noise(
                batch_size,
                opts.height,
                opts.width,
                device=torch_device,
                dtype=torch.bfloat16,
                seed=seed,
            )

            # ------------------------------------------------------------
            # Prepare text/image inputs
            # ------------------------------------------------------------
            t0 = time.perf_counter()

            inp_cond = prepare(t5, clip, x, prompt=batch_prompts)

            has_negative_prompt = (opts.negative_prompts is not None) or (opts.negative_prompt is not None)

            # GT calibration is specifically for CFG. Do not fall back to the single-branch
            # denoise() path in GT mode. Even if the caller does not pass a negative prompt,
            # use an empty unconditional prompt so that every GT run is a dual-branch CFG run.
            if opts.calibrate_mode == "gt":
                do_true_cfg = opts.true_cfg_scale > 1.0
            else:
                do_true_cfg = (opts.true_cfg_scale > 1.0) and has_negative_prompt

            if opts.calibrate_mode == "rho" and not do_true_cfg:
                raise ValueError(
                    "calibrate_mode='rho' requires true CFG, because rho calibration needs "
                    "both conditional and unconditional branches. Please set "
                    "--true_cfg_scale > 1 and provide --negative_prompt or --negative_prompt_file."
                )

            if opts.calibrate_mode == "gt" and not do_true_cfg:
                raise ValueError(
                    "calibrate_mode='gt' is CFG GT calibration. Please set --true_cfg_scale > 1."
                )

            if do_true_cfg:
                if opts.negative_prompts is not None:
                    batch_neg_prompts = opts.negative_prompts[prompt_start:prompt_end]
                elif opts.negative_prompt is not None:
                    batch_neg_prompts = [opts.negative_prompt] * len(batch_prompts)
                else:
                    batch_neg_prompts = [""] * len(batch_prompts)

                inp_uncond = prepare(t5, clip, x, prompt=batch_neg_prompts)
            else:
                inp_uncond = None

            time_stats["prepare"] += time.perf_counter() - t0

            timesteps = get_schedule(
                opts.num_steps,
                inp_cond["img"].shape[1],
                shift=(model_name != "flux-schnell"),
            )

            # ------------------------------------------------------------
            # Denoising
            # ------------------------------------------------------------
            with torch.no_grad():
                t0 = time.perf_counter()

                if not do_true_cfg:
                    x = denoise(
                        model,
                        img=inp_cond["img"],
                        img_ids=inp_cond["img_ids"],
                        txt=inp_cond["txt"],
                        txt_ids=inp_cond["txt_ids"],
                        vec=inp_cond["vec"],
                        timesteps=timesteps,
                        guidance=opts.guidance,
                    )
                elif opts.calibrate_mode == "gt":
                    x = run_gt_calibration(
                        model=model,
                        inp_cond=inp_cond,
                        inp_uncond=inp_uncond,
                        timesteps=timesteps,
                        guidance=opts.guidance,
                        true_cfg_scale=opts.true_cfg_scale,
                        output_dir=opts.output_dir,
                        prompt_ids=batch_sample_ids,
                        interval=opts.interval,
                        perturb_step=opts.perturb_step,
                    )
                else:
                    x = denoise_cfg(
                        model,
                        img=inp_cond["img"],
                        img_ids=inp_cond["img_ids"],
                        txt=inp_cond["txt"],
                        txt_ids=inp_cond["txt_ids"],
                        vec=inp_cond["vec"],
                        neg_txt=inp_uncond["txt"],
                        neg_txt_ids=inp_uncond["txt_ids"],
                        neg_vec=inp_uncond["vec"],
                        true_cfg_scale=opts.true_cfg_scale,
                        timesteps=timesteps,
                        guidance=opts.guidance,
                        calibrate_mode=opts.calibrate_mode,
                        interval=opts.interval,
                        perturb_step=opts.perturb_step,
                        prompt_ids=batch_prompt_ids,
                    )

                time_stats["denoise"] += time.perf_counter() - t0

                # ------------------------------------------------------------
                # Decode
                # ------------------------------------------------------------
                t0 = time.perf_counter()

                x = unpack(x.float(), opts.height, opts.width)

                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)

                time_stats["decode"] += time.perf_counter() - t0

            # ------------------------------------------------------------
            # Post-process
            # ------------------------------------------------------------
            t0 = time.perf_counter()

            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x, "b c h w -> b h w c")

            time_stats["post"] += time.perf_counter() - t0

            # ------------------------------------------------------------
            # Save images
            # ------------------------------------------------------------
            for i in range(batch_size):
                img_array = x[i]
                img = Image.fromarray(
                    (127.5 * (img_array + 1.0)).cpu().numpy().astype("uint8")
                )

                # NSFW filtering
                t0 = time.perf_counter()

                if opts.use_nsfw_filter and nsfw_classifier is not None:
                    nsfw_result = nsfw_classifier(img)
                    nsfw_score = next(
                        (res["score"] for res in nsfw_result if res["label"].lower() == "nsfw"),
                        0.0,
                    )
                else:
                    nsfw_score = 0.0

                time_stats["nsfw"] += time.perf_counter() - t0

                if nsfw_score < NSFW_THRESHOLD:
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                    exif_data[ExifTags.Base.Model] = model_name

                    if opts.add_sampling_metadata:
                        exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]

                    t0 = time.perf_counter()

                    fn = output_name.format(idx=current_start_index + i)
                    img.save(fn, exif=exif_data, quality=95, subsampling=0)

                    time_stats["save"] += time.perf_counter() - t0
                else:
                    print("Generated image may contain inappropriate content, skipped.")

                time_stats["total_images"] += 1
                progress_bar.update(1)

    progress_bar.close()

    # ------------------------------------------------------------
    # Timing summary
    # ------------------------------------------------------------
    t_run_end = time.perf_counter()
    total_sec = t_run_end - t_run_start

    total_images_done = max(1, time_stats["total_images"])
    total_prompts_done = max(1, len(opts.prompts))

    avg_sec_per_image = total_sec / total_images_done
    avg_sec_per_prompt = avg_sec_per_image * max(1, opts.num_images_per_prompt)

    print("\n================ Timing Summary ================")
    print(f"Total elapsed:        {total_sec:.3f} s")
    print(f"Total images:         {total_images_done}")
    print(f"Total prompts:        {total_prompts_done}")
    print(f"Avg sec / image:      {avg_sec_per_image:.3f} s")
    print(
        f"Avg sec / prompt:     {avg_sec_per_prompt:.3f} s "
        f"(includes {opts.num_images_per_prompt} img/prompt)"
    )
    print("--------------- Breakdown / image -------------")
    for k in ["prepare", "denoise", "decode", "post", "nsfw", "save"]:
        print(f"{k:>8}: {time_stats[k] / total_images_done:.3f} s")
    print("================================================\n")

    # ------------------------------------------------------------
    # Save timing
    # ------------------------------------------------------------
    try:
        timing_path = os.path.join(opts.output_dir, "timing.json")
        payload = {
            "total_sec": total_sec,
            "total_images": total_images_done,
            "total_prompts": total_prompts_done,
            "avg_sec_per_image": avg_sec_per_image,
            "avg_sec_per_prompt": avg_sec_per_prompt,
            "breakdown_sec_per_image": {
                k: time_stats[k] / total_images_done
                for k in ["prepare", "denoise", "decode", "post", "nsfw", "save"]
            },
            "config_hint": {
                "batch_size": opts.batch_size,
                "num_images_per_prompt": opts.num_images_per_prompt,
                "num_steps": opts.num_steps,
                "guidance": opts.guidance,
                "true_cfg_scale": opts.true_cfg_scale,
                "model_name": opts.model_name,
                "calibrate_mode": opts.calibrate_mode,
                "interval": opts.interval,
                "perturb_step": opts.perturb_step,
            },
        }

        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)

        print(f"[INFO] Timing stats saved to: {timing_path}")
    except Exception as e:
        print(f"[WARN] Failed to save timing.json: {e!r}")


def read_prompts(prompt_file: str) -> list[str]:
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def save_sampling_config(opts: SamplingOptions, out_dir: str):
    import sys
    from datetime import datetime

    os.makedirs(out_dir, exist_ok=True)

    cfg = {
        "timestamp": datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "working_directory": os.getcwd(),
        "sampler": {
            "width": int(opts.width),
            "height": int(opts.height),
            "num_steps": int(opts.num_steps),
            "guidance": float(opts.guidance),
            "true_cfg_scale": float(opts.true_cfg_scale),
            "seed": int(opts.seed) if isinstance(opts.seed, int) else opts.seed,
            "num_images_per_prompt": int(opts.num_images_per_prompt),
            "batch_size": int(opts.batch_size),
            "start_index": int(opts.start_index),
        },
        "model": {
            "name": opts.model_name,
        },
        "prompts": {
            "count": len(opts.prompts),
            "negative_prompt": opts.negative_prompt,
            "has_negative_prompt_file": opts.negative_prompts is not None,
        },
        "calibration": {
            "calibrate_mode": opts.calibrate_mode,
            "interval": opts.interval,
            "perturb_step": opts.perturb_step,
        },
        "output": {
            "dir": out_dir,
            "add_sampling_metadata": bool(opts.add_sampling_metadata),
            "use_nsfw_filter": bool(opts.use_nsfw_filter),
        },
    }

    dst = os.path.join(out_dir, "config.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)


def app():
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the FLUX model.")

    parser.add_argument(
        "--prompt_file",
        type=str,
        required=True,
        help="Path to the prompt text file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Maximum number of prompts to read from prompt_file. Use <=0 or omit to use all prompts.",
    )

    parser.add_argument("--width", type=int, default=1024, help="Width of the generated image.")
    parser.add_argument("--height", type=int, default=1024, help="Height of the generated image.")
    parser.add_argument("--num_steps", type=int, default=None, help="Number of sampling steps.")
    parser.add_argument("--guidance", type=float, default=3.5, help="Guidance value.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed.")

    parser.add_argument(
        "--num_images_per_prompt",
        type=int,
        default=1,
        help="Number of images per prompt.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=1,
        help="Batch size for prompt batching.",
    )

    parser.add_argument(
        "--model_name",
        type=str,
        default="flux-schnell",
        choices=["flux-dev", "flux-schnell"],
        help="Model name.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./samples",
        help="Directory to save generated images.",
    )
    parser.add_argument(
        "--start_index",
        type=int,
        default=0,
        help="Starting index offset for img_*.jpg numbering.",
    )

    parser.add_argument(
        "--add_sampling_metadata",
        action="store_true",
        help="Whether to add prompt metadata to image EXIF.",
    )
    parser.add_argument(
        "--use_nsfw_filter",
        action="store_true",
        help="Enable NSFW filter.",
    )

    # True CFG
    parser.add_argument(
        "--true_cfg_scale",
        type=float,
        default=1.0,
        help="True CFG scale. >1 enables dual-branch cond/uncond inference.",
    )
    parser.add_argument(
        "--negative_prompt",
        type=str,
        default=None,
        help="Negative prompt shared by all prompts.",
    )
    parser.add_argument(
        "--negative_prompt_file",
        type=str,
        default=None,
        help="Optional file containing per-prompt negative prompts, line by line.",
    )
    # Calibration
    parser.add_argument(
        "--calibrate_mode",
        type=str,
        default="none",
        choices=["none", "rho", "gt"],
        help="Calibration mode: none | rho | gt.",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="GT curve point interval. Only used when --calibrate_mode gt.",
    )
    parser.add_argument(
        "--perturb_step",
        type=int,
        default=None,
        help="Specific perturb step for GT curve calibration. Only used when --calibrate_mode gt.",
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
        if args.true_cfg_scale <= 1.0:
            raise ValueError(
                "--calibrate_mode gt is CFG GT calibration and requires --true_cfg_scale > 1."
            )

    if args.calibrate_mode == "rho":
        if args.interval is not None or args.perturb_step is not None:
            print(
                "[WARN] --calibrate_mode rho only needs --prompt_file. "
                "--interval / --perturb_step will be ignored unless your calibration code uses them."
            )

    prompts = read_prompts(args.prompt_file)

    if args.limit is not None and args.limit > 0:
        prompts = prompts[: args.limit]

    negative_prompts = None
    if args.negative_prompt_file is not None:
        negative_prompts = read_prompts(args.negative_prompt_file)
        if args.limit is not None and args.limit > 0:
            negative_prompts = negative_prompts[: args.limit]

    if negative_prompts is not None and len(negative_prompts) != len(prompts):
        raise ValueError(
            "negative_prompt_file must contain the same number of prompts as prompt_file "
            "after applying --limit: "
            f"got {len(negative_prompts)} negative prompts but {len(prompts)} prompts."
        )

    opts = SamplingOptions(
        prompts=prompts,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        guidance=args.guidance,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        start_index=args.start_index,
        add_sampling_metadata=args.add_sampling_metadata,
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt=args.negative_prompt,
        negative_prompts=negative_prompts,
        use_nsfw_filter=args.use_nsfw_filter,
        calibrate_mode=args.calibrate_mode,
        interval=args.interval,
        perturb_step=args.perturb_step,
    )

    main(opts)


if __name__ == "__main__":
    app()

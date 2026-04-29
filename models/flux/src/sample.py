import os
from dataclasses import dataclass
import time
import numpy as np
import torch
from einops import rearrange
from PIL import ExifTags, Image
from transformers import pipeline
from tqdm import tqdm

from flux.sampling import get_noise, get_schedule, prepare, unpack
from flux.ideas import denoise_cache,denoise_cache_cfg
from flux.util import configs, embed_watermark, load_ae, load_clip, load_flow_model, load_t5

NSFW_THRESHOLD = 0.85  # NSFW score threshold


@dataclass
class SamplingOptions:
    prompts: list[str]  # List of prompts
    width: int  # Image width
    height: int  # Image height
    num_steps: int  # Number of sampling steps
    guidance: float  # Guidance value
    seed: int | None  # Random seed
    num_images_per_prompt: int  # Number of images generated per prompt
    batch_size: int  # Batch size (batching of prompts)
    model_name: str  # Model name
    output_dir: str  # Output directory
    start_index: int  # Starting index offset for output numbering
    add_sampling_metadata: bool  # Whether to add metadata
    true_cfg_scale: float          # True CFG (>=1); 1 means off
    negative_prompt: str | None    # global negative prompt for all prompts
    negative_prompts: list[str] | None  # optional per-prompt negatives
    use_nsfw_filter: bool  # Whether to enable NSFW filter
    cache_mode: str  # Cache mode ('original', 'ToCa', 'Taylor', 'HiCache', 'Delta')
    interval: int  # Cache period length
    max_order: int  # Maximum order of Taylor expansion
    first_enhance: int  # Initial enhancement steps
    hicache_scale: float  # HiCache scaling factor
    rel_l1_thresh: float
    # ClusCa parameters
    clusca_fresh_threshold: int  # ClusCa fresh threshold
    clusca_cluster_num: int  # Number of clusters for ClusCa
    clusca_cluster_method: str  # Clustering method (kmeans/kmeans++/random)
    clusca_k: int  # Number of selected fresh tokens per cluster
    clusca_propagation_ratio: float  # Propagation ratio for cluster updates
    # Analytic HiCache (HiCache-Analytic) parameters
    analytic_sigma_alpha: float | None
    analytic_sigma_max: float | None
    analytic_sigma_beta: float | None
    analytic_sigma_eps: float | None
    analytic_sigma_q_quantile: float | None
    analytic_sigma_smooth: float | None
    proxy_tables_path: str | None = None

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
        cfg_scale=opts.true_cfg_scale,
    )

    cfgcache_runtime = {
        "proxy_tables_path": proxy_tables_path,
        "proxy_tables": proxy_tables,  
    }
    return cfgcache_runtime


def main(opts: SamplingOptions):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfgcache_runtime = None
    if opts.cache_mode == "CFGCache" and opts.proxy_tables_path is not None:
        t0 = time.perf_counter()
        cfgcache_runtime = build_cfgcache_runtime_config(opts)
        print(f"[CFGCache] preload done in {time.perf_counter() - t0:.3f}s")
    # Optional NSFW classifier
    if opts.use_nsfw_filter:
        try:
            from pathlib import Path

            project_root = Path(__file__).resolve().parents[3]
            nsfw_local_dir = project_root / "weights" / "Falconsai" / "nsfw_image_detection"
            model_id = str(nsfw_local_dir) if nsfw_local_dir.is_dir() else "Falconsai/nsfw_image_detection"
            nsfw_classifier = pipeline(
                "image-classification",
                model=model_id,
                device=device,
            )
        except Exception as e:
            print(f"[WARN] Failed to initialize NSFW classifier, disabling NSFW filter: {e!r}")
            nsfw_classifier = None
    else:
        nsfw_classifier = None

    # Load model
    model_name = opts.model_name
    if model_name not in configs:
        available = ", ".join(configs.keys())
        raise ValueError(f"Unknown model name: {model_name}, available options: {available}")

    if opts.num_steps is None:
        opts.num_steps = 4 if model_name == "flux-schnell" else 50

    # Ensure width and height are multiples of 16
    opts.width = 16 * (opts.width // 16)
    opts.height = 16 * (opts.height // 16)

    # Normal mode: use specified output_dir
    output_name = os.path.join(opts.output_dir, "img_{idx}.jpg")
    if not os.path.exists(opts.output_dir):
        os.makedirs(opts.output_dir)
    # Save a machine-readable config for reproducibility
    save_sampling_config(opts, opts.output_dir)

    idx = opts.start_index  # Image index offset for numbering

    # Initialize model components
    torch_device = device

    # Load T5 and CLIP models to GPU
    t5 = load_t5(torch_device, max_length=256 if model_name == "flux-schnell" else 512)
    clip = load_clip(torch_device)

    # Load model to GPU
    model = load_flow_model(model_name, device=torch_device)
    ae = load_ae(model_name, device=torch_device)

    # Set random seed
    if opts.seed is not None:
        base_seed = opts.seed
    else:
        base_seed = torch.randint(0, 2**32, (1,)).item()

    prompts = opts.prompts

    total_images = len(prompts) * opts.num_images_per_prompt

    progress_bar = tqdm(total=total_images, desc="Generating images")

    # ---------------- Timing stats ----------------
    t_run_start = time.perf_counter()
    time_stats = {
        "prepare": 0.0,   # t5/clip prepare (cond + optional uncond)
        "denoise": 0.0,   # denoise_cache / denoise_cache_cfg / denoise_test_FLOPs
        "decode": 0.0,    # unpack + ae.decode
        "post": 0.0,      # clamp/watermark/rearrange
        "nsfw": 0.0,      # nsfw classifier (if enabled)
        "save": 0.0,      # PIL save
        "total_images": 0,
        "total_prompts": len(prompts),
        "num_images_per_prompt": opts.num_images_per_prompt,
        "batch_size": opts.batch_size,
    }

    # Compute number of prompt batches
    num_prompt_batches = (len(prompts) + opts.batch_size - 1) // opts.batch_size

    for batch_idx in range(num_prompt_batches):
        prompt_start = batch_idx * opts.batch_size
        prompt_end = min(prompt_start + opts.batch_size, len(prompts))
        batch_prompts = prompts[prompt_start:prompt_end]
        num_prompts_in_batch = len(batch_prompts)

        # Generate corresponding number of images for each prompt
        for image_idx in range(opts.num_images_per_prompt):
            # Prepare random seed
            seed = base_seed + idx  # Assign a different seed for each image
            idx += num_prompts_in_batch  # Update image index

            # Prepare input
            batch_size = num_prompts_in_batch
            x = get_noise(
                batch_size,
                opts.height,
                opts.width,
                device=torch_device,
                dtype=torch.bfloat16,
                seed=seed,
            )

            # Prepare prompts
            # batch_prompts is a list containing the prompts in the current batch
            t0 = time.perf_counter()
            inp_cond = prepare(t5, clip, x, prompt=batch_prompts)

            do_true_cfg = (opts.true_cfg_scale > 1.0) and (
                (opts.negative_prompts is not None) or (opts.negative_prompt is not None)
            )
            if do_true_cfg:
                if opts.negative_prompts is not None:
                    batch_neg_prompts = opts.negative_prompts[prompt_start:prompt_end]
                else:
                    batch_neg_prompts = [opts.negative_prompt] * len(batch_prompts)
                inp_uncond = prepare(t5, clip, x, prompt=batch_neg_prompts)
            else:
                inp_uncond = None
            time_stats["prepare"] += time.perf_counter() - t0
            timesteps = get_schedule(
                opts.num_steps, inp_cond["img"].shape[1], shift=(model_name != "flux-schnell")
            )

            # Denoising
            with torch.no_grad():
                t0 = time.perf_counter()
                if not do_true_cfg:
                    x = denoise_cache(
                        model,
                        **inp_cond,
                        timesteps=timesteps,
                        guidance=opts.guidance,
                        cache_mode=opts.cache_mode,
                        interval=opts.interval,
                        max_order=opts.max_order,
                        first_enhance=opts.first_enhance,
                        hicache_scale=opts.hicache_scale,
                        rel_l1_thresh = opts.rel_l1_thresh,
                        # ClusCa parameters
                        clusca_fresh_threshold=opts.clusca_fresh_threshold,
                        clusca_cluster_num=opts.clusca_cluster_num,
                        clusca_cluster_method=opts.clusca_cluster_method,
                        clusca_k=opts.clusca_k,
                        clusca_propagation_ratio=opts.clusca_propagation_ratio,
                        analytic_sigma_alpha=opts.analytic_sigma_alpha,
                        analytic_sigma_max=opts.analytic_sigma_max,
                        analytic_sigma_beta=opts.analytic_sigma_beta,
                        analytic_sigma_eps=opts.analytic_sigma_eps,
                        analytic_sigma_q_quantile=opts.analytic_sigma_q_quantile,
                        analytic_sigma_smooth=opts.analytic_sigma_smooth,
                    )
                else:
                    x = denoise_cache_cfg(
                        model,
                        height=opts.height,
                        width=opts.width,
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
                        cache_mode=opts.cache_mode,
                        interval=opts.interval,
                        max_order=opts.max_order,
                        first_enhance=opts.first_enhance,
                        hicache_scale=opts.hicache_scale,
                        rel_l1_thresh = opts.rel_l1_thresh,
                        # ClusCa parameters
                        clusca_fresh_threshold=opts.clusca_fresh_threshold,
                        clusca_cluster_num=opts.clusca_cluster_num,
                        clusca_cluster_method=opts.clusca_cluster_method,
                        clusca_k=opts.clusca_k,
                        clusca_propagation_ratio=opts.clusca_propagation_ratio,
                        analytic_sigma_alpha=opts.analytic_sigma_alpha,
                        analytic_sigma_max=opts.analytic_sigma_max,
                        analytic_sigma_beta=opts.analytic_sigma_beta,
                        analytic_sigma_eps=opts.analytic_sigma_eps,
                        analytic_sigma_q_quantile=opts.analytic_sigma_q_quantile,
                        analytic_sigma_smooth=opts.analytic_sigma_smooth, 
                        cfgcache_runtime=cfgcache_runtime,
                    )
                    # x = search_denoise_cache(model, **inp, timesteps=timesteps, guidance=opts.guidance, interval=opts.interval, max_order=opts.max_order, first_enhance=opts.first_enhance)
                time_stats["denoise"] += time.perf_counter() - t0
                t0 = time.perf_counter()
                x = unpack(x.float(), opts.height, opts.width)
                with torch.autocast(device_type=torch_device.type, dtype=torch.bfloat16):
                    x = ae.decode(x)
                time_stats["decode"] += time.perf_counter() - t0

            # Convert to PIL format and save (skip if only collecting features)
            t0 = time.perf_counter()
            x = x.clamp(-1, 1)
            x = embed_watermark(x.float())
            x = rearrange(x, "b c h w -> b h w c")
            time_stats["post"] += time.perf_counter() - t0
            for i in range(batch_size):
                img_array = x[i]
                img = Image.fromarray((127.5 * (img_array + 1.0)).cpu().numpy().astype(np.uint8))

                # Optional NSFW filtering
                if opts.use_nsfw_filter:
                    nsfw_result = nsfw_classifier(img)
                    nsfw_score = next(
                        (res["score"] for res in nsfw_result if res["label"] == "nsfw"), 0.0
                    )
                else:
                    nsfw_score = 0.0  # If the filter is not enabled, assume safe

                if nsfw_score < NSFW_THRESHOLD:
                    exif_data = Image.Exif()
                    exif_data[ExifTags.Base.Software] = "AI generated;txt2img;flux"
                    exif_data[ExifTags.Base.Make] = "Black Forest Labs"
                    exif_data[ExifTags.Base.Model] = model_name
                    if opts.add_sampling_metadata:
                        exif_data[ExifTags.Base.ImageDescription] = batch_prompts[i]
                    # Save image
                    fn = output_name.format(idx=idx - num_prompts_in_batch + i)
                    img.save(fn, exif=exif_data, quality=95, subsampling=0)
                    time_stats["save"] += time.perf_counter() - t0
                else:
                    print("Generated image may contain inappropriate content, skipped.")
                time_stats["total_images"] += 1
                progress_bar.update(1)

    progress_bar.close()
    t_run_end = time.perf_counter()
    total_sec = t_run_end - t_run_start

    total_images_done = max(1, time_stats["total_images"])
    total_prompts_done = max(1, len(opts.prompts))  # total prompt count (not batched)

    avg_sec_per_image = total_sec / total_images_done
    avg_sec_per_prompt = avg_sec_per_image * max(1, opts.num_images_per_prompt)

    print("\n================ Timing Summary ================")
    print(f"Total elapsed:        {total_sec:.3f} s")
    print(f"Total images:         {total_images_done}")
    print(f"Total prompts:        {total_prompts_done}")
    print(f"Avg sec / image:      {avg_sec_per_image:.3f} s")
    print(f"Avg sec / prompt:     {avg_sec_per_prompt:.3f} s  (includes {opts.num_images_per_prompt} img/prompt)")
    print("--------------- Breakdown (sum) ---------------")
    for k in ["prepare", "denoise", "decode", "post", "nsfw", "save"]:
        print(f"{k:>8}: {time_stats[k] / total_images_done:.3f} s")
    print("================================================\n")

    # Optional: dump timing to json next to config.json
    try:
        import json
        out_dir = opts.output_dir
        timing_path = os.path.join(out_dir, "timing.json")
        payload = {
            "total_sec": total_sec,
            "total_images": total_images_done,
            "total_prompts": total_prompts_done,
            "avg_sec_per_image": avg_sec_per_image,
            "avg_sec_per_prompt": avg_sec_per_prompt,
            "breakdown_sec": {k: time_stats[k] / total_images_done for k in ["prepare","denoise","decode","post","nsfw","save"]},
            "config_hint": {
                "batch_size": opts.batch_size,
                "num_images_per_prompt": opts.num_images_per_prompt,
                "num_steps": opts.num_steps,
                "guidance": opts.guidance,
                "cache_mode": opts.cache_mode,
            },
        }
        with open(timing_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"[INFO] Timing stats saved to: {timing_path}")
    except Exception as e:
        print(f"[WARN] Failed to save timing.json: {e!r}")

def read_prompts(prompt_file: str):
    with open(prompt_file, "r", encoding="utf-8") as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts

def save_sampling_config(opts: SamplingOptions, out_dir: str):
    """Dump all key sampling parameters to a config.json in the output directory.

    This complements metadata and enables reproducible evaluation by capturing
    cache settings (interval, order, scales), sampler settings, and model info.
    """
    import json
    import sys
    from datetime import datetime

    os.makedirs(out_dir, exist_ok=True)

    # Build a structured config
    cfg = {
        "timestamp": datetime.now().isoformat(),
        "command_line": " ".join(sys.argv),
        "working_directory": os.getcwd(),
        "mode": opts.cache_mode,
        "sampler": {
            "width": int(opts.width),
            "height": int(opts.height),
            "num_steps": int(opts.num_steps),
            "guidance": float(opts.guidance),
            "seed": int(opts.seed) if isinstance(opts.seed, int) else opts.seed,
            "num_images_per_prompt": int(opts.num_images_per_prompt),
            "batch_size": int(opts.batch_size),
            "start_index": int(opts.start_index),
        },
        "model": {
            "name": opts.model_name,
        },
        "cache": {
            "interval": int(opts.interval),
            "max_order": int(opts.max_order),
            "first_enhance": int(opts.first_enhance),
            "hicache_scale": float(opts.hicache_scale),
            "analytic_sigma_alpha": float(opts.analytic_sigma_alpha)
            if opts.analytic_sigma_alpha is not None
            else None,
            "analytic_sigma_max": float(opts.analytic_sigma_max)
            if opts.analytic_sigma_max is not None
            else None,
            "analytic_sigma_beta": float(opts.analytic_sigma_beta)
            if opts.analytic_sigma_beta is not None
            else None,
            "analytic_sigma_eps": float(opts.analytic_sigma_eps)
            if opts.analytic_sigma_eps is not None
            else None,
            "analytic_sigma_q_quantile": float(opts.analytic_sigma_q_quantile)
            if opts.analytic_sigma_q_quantile is not None
            else None,
            "analytic_sigma_smooth": float(opts.analytic_sigma_smooth)
            if opts.analytic_sigma_smooth is not None
            else None,
        },
        "clusca": {
            "fresh_threshold": int(opts.clusca_fresh_threshold),
            "cluster_num": int(opts.clusca_cluster_num),
            "cluster_method": opts.clusca_cluster_method,
            "k": int(opts.clusca_k),
            "propagation_ratio": float(opts.clusca_propagation_ratio),
        },
        "prompts": {
            "count": len(opts.prompts),
        },
    }

    # Persist to JSON
    dst = os.path.join(out_dir, "config.json")
    with open(dst, "w", encoding="utf-8") as f:
        json.dump(cfg, f, ensure_ascii=False, indent=2)
    # Silent by default (avoid noisy logs under multi-gpu), caller can inspect file

def app():
    import argparse

    parser = argparse.ArgumentParser(description="Generate images using the flux model.")
    parser.add_argument("--prompt_file", type=str, required=True, help="Path to the prompt text file.")
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
    parser.add_argument("--num_images_per_prompt", type=int, default=1, help="Number of images per prompt.")
    parser.add_argument("--true_cfg_scale", type=float, default=1.0,
                    help="True CFG scale > 1 enables dual-branch CFG (cond+uncond).")
    parser.add_argument("--negative_prompt", type=str, default=None,
                        help="Negative prompt (shared for all prompts).")
    parser.add_argument("--negative_prompt_file", type=str, default=None,
                        help="Optional: file containing per-prompt negative prompts, line by line.")

    parser.add_argument("--batch_size", type=int, default=1, help="Batch size (prompt batching).")
    parser.add_argument(
        "--model_name",
        type=str,
        default="flux-schnell",
        choices=["flux-dev", "flux-schnell"],
        help="Model name.",
    )
    parser.add_argument("--output_dir", type=str, default="./samples", help="Directory to save images.")
    parser.add_argument(
        "--add_sampling_metadata", action="store_true", help="Whether to add prompt metadata to images."
    )
    parser.add_argument("--use_nsfw_filter", action="store_true", help="Enable NSFW filter.")
    parser.add_argument(
        "--cache_mode",
        type=str,
        default="original",
        choices=[
            "original",
            "CFGCache",
            "TeaCache",
            "MagCache",
            "DiCache",
            "ToCa",
            "Taylor",
            "Taylor-Scaled",
            "HiCache",
            "HiCache-Analytic",
            "Delta",
            "ClusCa",
            "Hi-ClusCa",
            "FasterCache",
        ],
        help="Cache mode for denoising.",
    )
    parser.add_argument("--interval", type=int, default=10, help="Cache period length.")
    parser.add_argument("--max_order", type=int, default=5, help="Maximum order of Taylor expansion.")
    parser.add_argument("--first_enhance", type=int, default=5, help="Initial enhancement steps.")
    parser.add_argument("--hicache_scale", type=float, default=1.0, help="HiCache scaling factor.")
    parser.add_argument("--rel_l1_thresh",type=float, default=0.6, help="TeaCache threshold." )
    # ClusCa arguments
    parser.add_argument(
        "--clusca_fresh_threshold",
        type=int,
        default=5,
        help="ClusCa fresh threshold.",
    )
    parser.add_argument(
        "--clusca_cluster_num",
        type=int,
        default=16,
        help="Number of clusters for ClusCa.",
    )
    parser.add_argument(
        "--clusca_cluster_method",
        type=str,
        default="kmeans",
        choices=["kmeans", "kmeans++", "random"],
        help="Clustering method for ClusCa.",
    )
    parser.add_argument(
        "--clusca_k",
        type=int,
        default=1,
        help="Number of selected fresh tokens per cluster.",
    )
    parser.add_argument(
        "--clusca_propagation_ratio",
        type=float,
        default=0.005,
        help="Propagation ratio for cluster updates.",
    )
    # Analytic HiCache (HiCache-Analytic) arguments
    parser.add_argument(
        "--analytic_sigma_alpha",
        type=float,
        default=None,
        help="Alpha factor for analytic sigma in HiCache-Analytic (default 1.28 → sigma≈0.9 when q≈1).",
    )
    parser.add_argument(
        "--analytic_sigma_max",
        type=float,
        default=None,
        help="Upper bound for analytic sigma in HiCache-Analytic (default 1.0).",
    )
    parser.add_argument(
        "--analytic_sigma_beta",
        type=float,
        default=None,
        help="EMA smoothing factor beta for analytic sigma statistics (default 0.01). "
        "Set to 0 to disable online updates and use the closed-form with q=1.",
    )
    parser.add_argument(
        "--analytic_sigma_eps",
        type=float,
        default=None,
        help="Epsilon added to the denominator in analytic sigma formula (default 1e-6).",
    )
    parser.add_argument(
        "--analytic_sigma_q_quantile",
        type=float,
        default=None,
        help="Optional quantile (e.g., 0.95) for robust q estimation; if unset, use mean.",
    )
    parser.add_argument(
        "--analytic_sigma_smooth",
        type=float,
        default=None,
        help="Gamma for log-domain EMA smoothing of sigma (0 disables smoothing).",
    )
    parser.add_argument(
        "--start_index", type=int, default=0, help="Starting index offset for img_*.jpg numbering."
    )
    parser.add_argument(
        "--proxy_tables_path",
        type=str,
        default=None,
        help="Path to packed all-cfg offline rho npz for CFGCache proxy.",
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

    if negative_prompts is not None and len(negative_prompts) != len(prompts):
        raise ValueError(
            f"negative_prompt_file must contain the same number of prompts as prompt_file after applying --limit: "
            f"got {len(negative_prompts)} negative prompts but {len(prompts)} prompts."
        )
    opts = SamplingOptions(
        prompts=prompts,
        width=args.width,
        height=args.height,
        num_steps=args.num_steps,
        true_cfg_scale=args.true_cfg_scale,
        negative_prompt=args.negative_prompt,
        negative_prompts=negative_prompts,
        guidance=args.guidance,
        seed=args.seed,
        num_images_per_prompt=args.num_images_per_prompt,
        batch_size=args.batch_size,
        model_name=args.model_name,
        output_dir=args.output_dir,
        start_index=args.start_index,
        add_sampling_metadata=args.add_sampling_metadata,
        use_nsfw_filter=args.use_nsfw_filter,
        cache_mode=args.cache_mode,
        interval=args.interval,
        max_order=args.max_order,
        first_enhance=args.first_enhance,
        hicache_scale=args.hicache_scale,
        rel_l1_thresh=args.rel_l1_thresh,
        # ClusCa parameters
        clusca_fresh_threshold=args.clusca_fresh_threshold,
        clusca_cluster_num=args.clusca_cluster_num,
        clusca_cluster_method=args.clusca_cluster_method,
        clusca_k=args.clusca_k,
        clusca_propagation_ratio=args.clusca_propagation_ratio,
        analytic_sigma_alpha=args.analytic_sigma_alpha,
        analytic_sigma_max=args.analytic_sigma_max,
        analytic_sigma_beta=args.analytic_sigma_beta,
        analytic_sigma_eps=args.analytic_sigma_eps,
        analytic_sigma_q_quantile=args.analytic_sigma_q_quantile,
        analytic_sigma_smooth=args.analytic_sigma_smooth,
        proxy_tables_path=args.proxy_tables_path,
    )

    main(opts)


if __name__ == "__main__":
    app()

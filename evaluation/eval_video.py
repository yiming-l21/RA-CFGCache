import re
import math
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional

import cv2
import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import torch.nn.functional as F
import lpips
import open_clip

# ---- SSIM ----
from skimage.metrics import structural_similarity as ssim

# ---- ImageReward ----
try:
    import ImageReward as ir
except Exception as e:
    ir = None
    _IMAGEREWARD_IMPORT_ERR = e


VIDEO_EXTS = {".mp4", ".avi", ".mov", ".mkv", ".webm", ".gif"}


def resolve_video_dir(path_str: str) -> Path:
    path = Path(path_str)
    if not path.exists():
        raise FileNotFoundError(f"Path not found: {path}")
    if (path / "videos").is_dir():
        return path / "videos"
    return path


def collect_video_files(video_dir: Path) -> Dict[str, Path]:
    files = {}
    for p in sorted(video_dir.iterdir()):
        if p.is_file() and p.suffix.lower() in VIDEO_EXTS:
            files[p.name] = p
    return files


def read_prompts(prompt_file: str) -> list[str]:
    prompts = []
    with open(prompt_file, "r", encoding="utf-8") as f:
        for line in f:
            s = line.strip()
            if s:
                prompts.append(s)
    if not prompts:
        raise RuntimeError(f"Empty prompt_file: {prompt_file}")
    return prompts


def parse_index_from_name(name: str) -> int:
    stem = Path(name).stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"Cannot parse index from filename: {name}")
    return int(nums[-1])


def load_video_rgb_frames(path: Path, max_frames: Optional[int] = None) -> List[np.ndarray]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {path}")

    frames = []
    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            break
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        frames.append(frame_rgb)
        if max_frames is not None and len(frames) >= max_frames:
            break
    cap.release()

    if not frames:
        raise RuntimeError(f"No frames decoded from video: {path}")
    return frames


def resize_rgb(img: np.ndarray, size_wh: Tuple[int, int]) -> np.ndarray:
    pil = Image.fromarray(img)
    pil = pil.resize(size_wh, Image.BICUBIC)
    return np.array(pil)


def align_video_frames(
    ref_frames: List[np.ndarray],
    cmp_frames: List[np.ndarray],
    resize_if_mismatch: bool = True,
    frame_align: str = "linspace_min",
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    if len(ref_frames) == 0 or len(cmp_frames) == 0:
        raise RuntimeError("Empty frame list encountered.")

    # spatial align
    h0, w0 = ref_frames[0].shape[:2]
    h1, w1 = cmp_frames[0].shape[:2]
    if (h0, w0) != (h1, w1):
        if not resize_if_mismatch:
            raise RuntimeError(f"Frame size mismatch: ref={(h0, w0)} cmp={(h1, w1)}")
        cmp_frames = [resize_rgb(x, (w0, h0)) for x in cmp_frames]

    # temporal align
    n0, n1 = len(ref_frames), len(cmp_frames)
    if n0 == n1:
        return ref_frames, cmp_frames

    if frame_align not in {"linspace_min", "truncate_min"}:
        raise ValueError(f"Unknown frame_align={frame_align}")

    tgt = min(n0, n1)
    if frame_align == "truncate_min":
        return ref_frames[:tgt], cmp_frames[:tgt]

    idx0 = np.linspace(0, n0 - 1, tgt).round().astype(int).tolist()
    idx1 = np.linspace(0, n1 - 1, tgt).round().astype(int).tolist()
    ref_aligned = [ref_frames[i] for i in idx0]
    cmp_aligned = [cmp_frames[i] for i in idx1]
    return ref_aligned, cmp_aligned


def psnr_u8(ref_u8: np.ndarray, cmp_u8: np.ndarray) -> float:
    ref = ref_u8.astype(np.float32) / 255.0
    cmp = cmp_u8.astype(np.float32) / 255.0
    mse = np.mean((ref - cmp) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))


def ssim_u8(ref_u8: np.ndarray, cmp_u8: np.ndarray) -> float:
    def rgb2gray_matlab(img_rgb: np.ndarray) -> np.ndarray:
        return np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float64)

    ref_gray = rgb2gray_matlab(ref_u8)
    cmp_gray = rgb2gray_matlab(cmp_u8)
    ssim_val = ssim(
        ref_gray,
        cmp_gray,
        data_range=255,
        win_size=11,
        gaussian_weights=True,
        sigma=1.5,
        K1=0.01,
        K2=0.03,
        use_sample_covariance=False,
        channel_axis=None,
    )
    return float(ssim_val)


def to_lpips_tensor(u8: np.ndarray, device: torch.device) -> torch.Tensor:
    img = u8.astype(np.float32) / 255.0
    img = img * 2.0 - 1.0
    img = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0).to(device, non_blocking=True)
    return img


@torch.no_grad()
def clip_text_image_score(
    pil_img: Image.Image,
    prompt: str,
    clip_model,
    preprocess,
    tokenizer,
    device: torch.device,
) -> float:
    img_in = preprocess(pil_img).unsqueeze(0).to(device, non_blocking=True)
    txt_in = tokenizer([prompt]).to(device, non_blocking=True)

    img_feat = clip_model.encode_image(img_in)
    txt_feat = clip_model.encode_text(txt_in)

    img_feat = F.normalize(img_feat, dim=-1)
    txt_feat = F.normalize(txt_feat, dim=-1)

    sim = (img_feat * txt_feat).sum(dim=-1).item()
    return float(sim)


def load_image_reward(device: torch.device):
    if ir is None:
        raise RuntimeError(
            f"ImageReward not available. Install with: pip install imagereward\n"
            f"Import error: {_IMAGEREWARD_IMPORT_ERR}"
        )
    model = ir.load("ImageReward-v1.0")
    model = model.to(device)
    model.eval()
    return model


@torch.no_grad()
def image_reward_score(pil_img: Image.Image, prompt: str, reward_model) -> float:
    s = reward_model.score(prompt, [pil_img])
    if isinstance(s, (list, tuple)):
        s = s[0]
    if hasattr(s, "item"):
        s = s.item()
    return float(s)


def make_device(device: str) -> torch.device:
    if device.startswith("cuda") and torch.cuda.is_available():
        dev = torch.device(device if ":" in device else "cuda:0")
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(dev)} ({dev})")
        return dev
    print("[INFO] Using CPU")
    return torch.device("cpu")


def summarize(values: List[float]) -> Tuple[Optional[float], Optional[float]]:
    if not values:
        return None, None
    return float(np.mean(values)), float(np.std(values))


def main(
    ref_dir: str,
    cmp_dir: str,
    prompt_file: str,
    out_json: str = "video_metrics.json",
    out_csv: str = "video_metrics.csv",
    device: str = "cuda:0",
    resize_if_mismatch: bool = True,
    prompt_align: str = "by_index",   # by_index | by_order
    frame_align: str = "linspace_min",  # linspace_min | truncate_min
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    enable_ssim: bool = True,
    enable_imagereward: bool = True,
    semantic_stride: int = 8,
    max_frames: Optional[int] = None,
):
    ref_dir = resolve_video_dir(ref_dir)
    cmp_dir = resolve_video_dir(cmp_dir)

    assert ref_dir.is_dir(), f"ref_dir not found: {ref_dir}"
    assert cmp_dir.is_dir(), f"cmp_dir not found: {cmp_dir}"

    device = make_device(device)

    ref_files = collect_video_files(ref_dir)
    cmp_files = collect_video_files(cmp_dir)
    names = sorted(set(ref_files.keys()) & set(cmp_files.keys()))
    if not names:
        raise RuntimeError("No matched video filenames between ref_dir and cmp_dir")

    prompts = read_prompts(prompt_file)

    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(clip_model_name)

    reward_model = None
    if enable_imagereward:
        reward_model = load_image_reward(device)

    per_video_rows = []

    all_video_psnr = []
    all_video_ssim = []
    all_video_lpips = []
    all_clip_ref = []
    all_clip_cmp = []
    all_ir_ref = []
    all_ir_cmp = []

    for order_i, name in enumerate(tqdm(names, desc="Video metrics")):
        ref_frames = load_video_rgb_frames(ref_files[name], max_frames=max_frames)
        cmp_frames = load_video_rgb_frames(cmp_files[name], max_frames=max_frames)
        ref_frames, cmp_frames = align_video_frames(
            ref_frames,
            cmp_frames,
            resize_if_mismatch=resize_if_mismatch,
            frame_align=frame_align,
        )

        if prompt_align == "by_index":
            idx = parse_index_from_name(name)
        elif prompt_align == "by_order":
            idx = order_i
        else:
            raise ValueError(f"Unknown prompt_align={prompt_align}")

        if idx < 0 or idx >= len(prompts):
            raise IndexError(f"Prompt index out of range for {name}: idx={idx}, num_prompts={len(prompts)}")

        prompt = prompts[idx]

        psnrs = []
        ssims = []
        lpipss = []
        clip_ref_scores = []
        clip_cmp_scores = []
        ir_ref_scores = []
        ir_cmp_scores = []

        for fi, (ref_u8, cmp_u8) in enumerate(zip(ref_frames, cmp_frames)):
            p = psnr_u8(ref_u8, cmp_u8)
            psnrs.append(p)

            if enable_ssim:
                ssims.append(ssim_u8(ref_u8, cmp_u8))

            with torch.no_grad():
                ref_t = to_lpips_tensor(ref_u8, device)
                cmp_t = to_lpips_tensor(cmp_u8, device)
                lpipss.append(float(lpips_model(ref_t, cmp_t).item()))

            if fi % max(1, semantic_stride) == 0 or fi == len(ref_frames) - 1:
                ref_pil = Image.fromarray(ref_u8)
                cmp_pil = Image.fromarray(cmp_u8)

                clip_ref_scores.append(
                    clip_text_image_score(ref_pil, prompt, clip_model, preprocess, tokenizer, device)
                )
                clip_cmp_scores.append(
                    clip_text_image_score(cmp_pil, prompt, clip_model, preprocess, tokenizer, device)
                )

                if enable_imagereward and reward_model is not None:
                    ir_ref_scores.append(image_reward_score(ref_pil, prompt, reward_model))
                    ir_cmp_scores.append(image_reward_score(cmp_pil, prompt, reward_model))

        video_row = {
            "name": name,
            "prompt_idx": idx,
            "prompt": prompt,
            "num_ref_frames": len(ref_frames),
            "num_cmp_frames": len(cmp_frames),
            "num_aligned_frames": len(psnrs),
            "psnr_mean": float(np.mean(psnrs)),
            "psnr_std": float(np.std(psnrs)),
            "ssim_mean": float(np.mean(ssims)) if enable_ssim and ssims else None,
            "ssim_std": float(np.std(ssims)) if enable_ssim and ssims else None,
            "lpips_mean": float(np.mean(lpipss)),
            "lpips_std": float(np.std(lpipss)),
            "clip_ref_mean": float(np.mean(clip_ref_scores)),
            "clip_ref_std": float(np.std(clip_ref_scores)),
            "clip_cmp_mean": float(np.mean(clip_cmp_scores)),
            "clip_cmp_std": float(np.std(clip_cmp_scores)),
            "ir_ref_mean": float(np.mean(ir_ref_scores)) if ir_ref_scores else None,
            "ir_ref_std": float(np.std(ir_ref_scores)) if ir_ref_scores else None,
            "ir_cmp_mean": float(np.mean(ir_cmp_scores)) if ir_cmp_scores else None,
            "ir_cmp_std": float(np.std(ir_cmp_scores)) if ir_cmp_scores else None,
        }
        per_video_rows.append(video_row)

        all_video_psnr.append(video_row["psnr_mean"])
        all_video_lpips.append(video_row["lpips_mean"])
        all_clip_ref.append(video_row["clip_ref_mean"])
        all_clip_cmp.append(video_row["clip_cmp_mean"])
        if video_row["ssim_mean"] is not None:
            all_video_ssim.append(video_row["ssim_mean"])
        if video_row["ir_ref_mean"] is not None:
            all_ir_ref.append(video_row["ir_ref_mean"])
        if video_row["ir_cmp_mean"] is not None:
            all_ir_cmp.append(video_row["ir_cmp_mean"])

    psnr_mean, psnr_std = summarize(all_video_psnr)
    lpips_mean, lpips_std = summarize(all_video_lpips)
    ssim_mean, ssim_std = summarize(all_video_ssim)
    clip_ref_mean, clip_ref_std = summarize(all_clip_ref)
    clip_cmp_mean, clip_cmp_std = summarize(all_clip_cmp)
    ir_ref_mean, ir_ref_std = summarize(all_ir_ref)
    ir_cmp_mean, ir_cmp_std = summarize(all_ir_cmp)

    summary = {
        "ref_dir": str(ref_dir),
        "cmp_dir": str(cmp_dir),
        "prompt_file": prompt_file,
        "prompt_align": prompt_align,
        "frame_align": frame_align,
        "num_video_pairs": len(names),
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "clip_model": clip_model_name,
        "clip_pretrained": clip_pretrained,
        "clip_ref_mean": clip_ref_mean,
        "clip_ref_std": clip_ref_std,
        "clip_cmp_mean": clip_cmp_mean,
        "clip_cmp_std": clip_cmp_std,
        "imagereward_ref_mean": ir_ref_mean,
        "imagereward_ref_std": ir_ref_std,
        "imagereward_cmp_mean": ir_cmp_mean,
        "imagereward_cmp_std": ir_cmp_std,
        "device": str(device),
        "semantic_stride": semantic_stride,
        "max_frames": max_frames,
    }

    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(
            {
                "summary": summary,
                "per_video": per_video_rows,
            },
            f,
            ensure_ascii=False,
            indent=2,
        )

    with open(out_csv, "w", encoding="utf-8") as f:
        f.write(
            "name,prompt_idx,num_aligned_frames,psnr_mean,psnr_std,ssim_mean,ssim_std,"
            "lpips_mean,lpips_std,clip_ref_mean,clip_ref_std,clip_cmp_mean,clip_cmp_std,"
            "ir_ref_mean,ir_ref_std,ir_cmp_mean,ir_cmp_std\n"
        )
        for r in per_video_rows:
            f.write(
                f"{r['name']},{r['prompt_idx']},{r['num_aligned_frames']},"
                f"{r['psnr_mean']},{r['psnr_std']},"
                f"{'' if r['ssim_mean'] is None else r['ssim_mean']},"
                f"{'' if r['ssim_std'] is None else r['ssim_std']},"
                f"{r['lpips_mean']},{r['lpips_std']},"
                f"{r['clip_ref_mean']},{r['clip_ref_std']},"
                f"{r['clip_cmp_mean']},{r['clip_cmp_std']},"
                f"{'' if r['ir_ref_mean'] is None else r['ir_ref_mean']},"
                f"{'' if r['ir_ref_std'] is None else r['ir_ref_std']},"
                f"{'' if r['ir_cmp_mean'] is None else r['ir_cmp_mean']},"
                f"{'' if r['ir_cmp_std'] is None else r['ir_cmp_std']}\n"
            )

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")

    print(f"\nSaved: {out_json}, {out_csv}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", type=str, required=True, help="Ref video dir or experiment root containing videos/")
    ap.add_argument("--cmp_dir", type=str, required=True, help="Cmp video dir or experiment root containing videos/")
    ap.add_argument("--prompt_file", type=str, required=True)

    ap.add_argument("--out_json", type=str, default="video_metrics.json")
    ap.add_argument("--out_csv", type=str, default="video_metrics.csv")
    ap.add_argument("--device", type=str, default="cuda:0")

    ap.add_argument("--no_resize_if_mismatch", action="store_true")
    ap.add_argument("--prompt_align", type=str, default="by_index", choices=["by_index", "by_order"])
    ap.add_argument("--frame_align", type=str, default="linspace_min", choices=["linspace_min", "truncate_min"])
    ap.add_argument("--max_frames", type=int, default=None)
    ap.add_argument("--semantic_stride", type=int, default=8)

    ap.add_argument("--no_ssim", action="store_true")
    ap.add_argument("--no_imagereward", action="store_true")

    ap.add_argument("--clip_model", type=str, default="ViT-B-32")
    ap.add_argument("--clip_pretrained", type=str, default="openai")

    args = ap.parse_args()

    main(
        ref_dir=args.ref_dir,
        cmp_dir=args.cmp_dir,
        prompt_file=args.prompt_file,
        out_json=args.out_json,
        out_csv=args.out_csv,
        device=args.device,
        resize_if_mismatch=not args.no_resize_if_mismatch,
        prompt_align=args.prompt_align,
        frame_align=args.frame_align,
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        enable_ssim=not args.no_ssim,
        enable_imagereward=not args.no_imagereward,
        semantic_stride=args.semantic_stride,
        max_frames=args.max_frames,
    )
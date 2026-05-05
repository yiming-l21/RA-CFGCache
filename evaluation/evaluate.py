import os
import re
import math
import json
from pathlib import Path
import cv2
os.environ["CLEANFID_CACHE_DIR"] = "/path/inception_model"
# ---- SSIM ----
from skimage.metrics import structural_similarity as ssim

# ---- ImageReward ----
try:
    import ImageReward as ir
except Exception as e:
    ir = None
    _IMAGEREWARD_IMPORT_ERR = e

import numpy as np
from PIL import Image
from tqdm import tqdm

import torch
import lpips

# ---- CLIP ----
import open_clip
import torch.nn.functional as F


def load_rgb(path: Path) -> np.ndarray:
    refimg = cv2.imread(path)
    return cv2.cvtColor(refimg, cv2.COLOR_BGR2RGB)


def psnr_u8(ref_u8: np.ndarray, cmp_u8: np.ndarray) -> float:
    ref = ref_u8.astype(np.float32) / 255.0
    cmp = cmp_u8.astype(np.float32) / 255.0
    mse = np.mean((ref - cmp) ** 2)
    if mse == 0:
        return float("inf")
    return 20.0 * math.log10(1.0 / math.sqrt(mse))

def ssim_u8(ref_u8: np.ndarray, cmp_u8: np.ndarray) -> float:
    """
    适配 MATLAB FR_MSSIM 的 skimage 版 SSIM 计算
    :param ref_u8: 参考图 (H,W,3) uint8
    :param cmp_u8: 对比图 (H,W,3) uint8
    :return: 均值 SSIM（和 MATLAB FR_MSSIM 基本一致）
    """
    # 1. 第一步：RGB 转灰度（严格对齐 MATLAB rgb2gray 公式）
    def rgb2gray_matlab(img_rgb: np.ndarray) -> np.ndarray:
        # MATLAB 官方公式：0.2989*R + 0.5870*G + 0.1140*B
        return np.dot(img_rgb[..., :3], [0.2989, 0.5870, 0.1140]).astype(np.float64)
    
    ref_gray = rgb2gray_matlab(ref_u8)
    cmp_gray = rgb2gray_matlab(cmp_u8)
    ssim_val = ssim(
        ref_gray,
        cmp_gray,
        data_range=255,          # 动态范围和 MATLAB 一致
        win_size=11,             # 对齐 MATLAB 11x11 窗口
        gaussian_weights=True,   # 高斯窗口（替代 skimage 默认的矩形窗口）
        sigma=1.5,
        K1=0.01,                 # 对齐 MATLAB K1
        K2=0.03,                 # 对齐 MATLAB K2
        use_sample_covariance=False,  # 对齐 MATLAB 协方差计算逻辑
        channel_axis=None        # 灰度图，无需通道轴
    )
    return float(ssim_val)

def to_lpips_tensor(u8: np.ndarray, device: torch.device) -> torch.Tensor:
    """
    严格对齐 LPIPS 官方预处理逻辑
    :param u8: RGB 图片 (H,W,3) uint8
    :param device: torch.device
    :return: LPIPS 输入张量 (1,3,H,W)，范围 [-1,1]，float32，RGB 通道
    """
    # 1. 转 float32（避免 uint8 转 float 的精度损耗）
    img = u8.astype(np.float32)
    
    # 2. 归一化到 [0,1]
    img = img / 255.0
    
    # 3. 应用 ImageNet 均值/std 归一化（LPIPS 模型训练时的标准）
    # 均值：[0.485, 0.456, 0.406]，std：[0.229, 0.224, 0.225]
    mean = np.array([0.485, 0.456, 0.406]).reshape(1, 1, 3)
    std = np.array([0.229, 0.224, 0.225]).reshape(1, 1, 3)
    img = (img - mean) / std
    
    # 4. 转 [-1, 1]（LPIPS 官方要求）
    img = img * 2.0 - 1.0
    
    # 5. 调整维度：(H,W,3) → (3,H,W) → (1,3,H,W)
    img = torch.from_numpy(img).permute(2, 0, 1).float()
    img = img.unsqueeze(0).to(device, non_blocking=True)
    
    # 6. 确保是 float32（避免 float64 导致的精度问题）
    return img.to(dtype=torch.float32)


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
    """
    Try to parse an integer index from filename like img_0007.png -> 7.
    Uses the LAST integer group in the stem.
    """
    stem = Path(name).stem
    nums = re.findall(r"\d+", stem)
    if not nums:
        raise ValueError(f"Cannot parse index from filename: {name}")
    return int(nums[-1])


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

    # cosine similarity
    sim = (img_feat * txt_feat).sum(dim=-1).item()
    return float(sim)

def load_image_reward(device: torch.device):
    if ir is None:
        raise RuntimeError(
            f"ImageReward not available. Install with: pip install imagereward\n"
            f"Import error: {_IMAGEREWARD_IMPORT_ERR}"
        )
    model = ir.load("ImageReward-v1.0")
    # 强制模型所有参数/缓冲区移到指定设备
    model = model.to(device)
    model.eval()
    # 确保所有子模块都在指定设备上
    for module in model.modules():
        for param in module.parameters(recurse=False):
            param.data = param.data.to(device)
        for buf in module.buffers(recurse=False):
            buf.data = buf.data.to(device)
    return model

@torch.no_grad()
def image_reward_score(pil_img: Image.Image, prompt: str, reward_model) -> float:
    # 常见 API：reward_model.score(prompt, [pil_img]) -> list/np/torch
    s = reward_model.score(prompt, [pil_img])
    if isinstance(s, (list, tuple)):
        s = s[0]
    if hasattr(s, "item"):
        s = s.item()
    return float(s)


def main(
    ref_dir: str,
    cmp_dir: str,
    prompt_file: str,
    out_json: str = "metrics.json",
    out_csv: str = "metrics.csv",
    device: str = "cuda",
    resize_if_mismatch: bool = True,
    prompt_align: str = "by_index",   # by_index | by_order
    clip_model_name: str = "ViT-B-32",
    clip_pretrained: str = "openai",
    enable_ssim: bool = True,
    enable_imagereward: bool = True,
):
    ref_dir = Path(ref_dir)
    cmp_dir = Path(cmp_dir)
    assert ref_dir.is_dir(), f"ref_dir not found: {ref_dir}"
    assert cmp_dir.is_dir(), f"cmp_dir not found: {cmp_dir}"

    # 修复1：正确的设备判断逻辑（优先用指定设备，无GPU则用CPU）
    if device.startswith("cuda") and torch.cuda.is_available():
        # 提取 GPU 序号（如 cuda:0），默认用 cuda:0
        device = torch.device(device if ":" in device else "cuda:0")
        # 打印当前使用的 GPU 信息，方便排查
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name(device)} (device: {device})")
    else:
        device = torch.device("cpu")
        print(f"[INFO] Using CPU (CUDA not available or device set to CPU)")

    # Collect matched filenames (by name)
    ref_files = {p.name: p for p in sorted(ref_dir.glob("*.jpg"))}
    cmp_files = {p.name: p for p in sorted(cmp_dir.glob("*.jpg"))}
    names = sorted(set(ref_files.keys()) & set(cmp_files.keys()))
    if not names:
        raise RuntimeError("No matched *.jpg filenames between ref_dir and cmp_dir")

    prompts = read_prompts(prompt_file)

    # LPIPS model - 确保移到指定设备
    lpips_model = lpips.LPIPS(net="alex").to(device)
    lpips_model.eval()

    # CLIP model - 确保移到指定设备
    clip_model, _, preprocess = open_clip.create_model_and_transforms(
        clip_model_name, pretrained=clip_pretrained
    )
    clip_model = clip_model.to(device).eval()
    tokenizer = open_clip.get_tokenizer(clip_model_name)
    
    reward_model = None
    # if enable_imagereward:
    #     reward_model = load_image_reward(device)

    rows = []
    psnrs, lpipss = [], []
    clip_ref_list, clip_cmp_list = [], []
    ssims = []
    ir_ref_list, ir_cmp_list = [], []
    
    for order_i, name in enumerate(tqdm(names, desc="Pairwise PSNR/LPIPS + CLIPScore")):
        ref_u8 = load_rgb(ref_files[name])
        cmp_u8 = load_rgb(cmp_files[name])

        if ref_u8.shape != cmp_u8.shape:
            if not resize_if_mismatch:
                raise RuntimeError(f"Shape mismatch for {name}: {ref_u8.shape} vs {cmp_u8.shape}")
            cmp_img = Image.fromarray(cmp_u8).resize((ref_u8.shape[1], ref_u8.shape[0]), Image.BICUBIC)
            cmp_u8 = np.array(cmp_img)

        # ---- prompt alignment ----
        if prompt_align == "by_index":
            idx = parse_index_from_name(name)
        elif prompt_align == "by_order":
            idx = order_i
        else:
            raise ValueError(f"Unknown prompt_align={prompt_align}")

        if idx < 0 or idx >= len(prompts):
            raise IndexError(f"Prompt index out of range for {name}: idx={idx}, num_prompts={len(prompts)}")

        prompt = prompts[idx]

        # ---- PSNR ----
        p = psnr_u8(ref_u8, cmp_u8)
        # ---- SSIM ----
        if enable_ssim:
            s = ssim_u8(ref_u8, cmp_u8)
        else:
            s = None
        # ---- LPIPS ----
        with torch.no_grad():
            ref_t = to_lpips_tensor(ref_u8, device)
            cmp_t = to_lpips_tensor(cmp_u8, device)
            l = float(lpips_model(ref_t, cmp_t).item())

        # ---- CLIPScore(text-image) for BOTH dirs ----
        ref_pil = Image.fromarray(ref_u8)
        cmp_pil = Image.fromarray(cmp_u8)
        cs_ref = clip_text_image_score(ref_pil, prompt, clip_model, preprocess, tokenizer, device)
        cs_cmp = clip_text_image_score(cmp_pil, prompt, clip_model, preprocess, tokenizer, device)
        
        # ---- ImageReward (text-image) ----
        ir_ref, ir_cmp = None, None
        if enable_imagereward and reward_model is not None:
            ir_ref = image_reward_score(ref_pil, prompt, reward_model)
            ir_cmp = image_reward_score(cmp_pil, prompt, reward_model)
        
        rows.append(
            {
                "name": name,
                "prompt_idx": idx,
                "prompt": prompt,
                "psnr": p,
                "lpips": l,
                "clip_ref": cs_ref,
                "clip_cmp": cs_cmp,
                "ssim": s,
                "ir_ref": ir_ref,
                "ir_cmp": ir_cmp,
            }
        )
        print(p)
        if np.isfinite(p): psnrs.append(p)
        lpipss.append(l)
        clip_ref_list.append(cs_ref)
        clip_cmp_list.append(cs_cmp)
        if enable_ssim:
            ssims.append(s)
        if enable_imagereward and ir_ref is not None:
            ir_ref_list.append(ir_ref)
            ir_cmp_list.append(ir_cmp)

    # Summary stats
    psnr_mean = float(np.mean(psnrs))
    psnr_std = float(np.std(psnrs))
    lpips_mean = float(np.mean(lpipss))
    lpips_std = float(np.std(lpipss))

    clip_ref_mean = float(np.mean(clip_ref_list))
    clip_ref_std = float(np.std(clip_ref_list))
    clip_cmp_mean = float(np.mean(clip_cmp_list))
    clip_cmp_std = float(np.std(clip_cmp_list))
    
    ssim_mean = float(np.mean(ssims)) if (enable_ssim and ssims) else None
    ssim_std = float(np.std(ssims)) if (enable_ssim and ssims) else None
    
    ir_ref_mean = float(np.mean(ir_ref_list)) if (enable_imagereward and ir_ref_list) else None
    ir_ref_std = float(np.std(ir_ref_list)) if (enable_imagereward and ir_ref_list) else None
    ir_cmp_mean = float(np.mean(ir_cmp_list)) if (enable_imagereward and ir_cmp_list) else None
    ir_cmp_std = float(np.std(ir_cmp_list)) if (enable_imagereward and ir_cmp_list) else None

    # FID (set-level)
    fid_value = None
    
    try:
        from cleanfid import fid
        fid_value = float(
            fid.compute_fid(
                str(ref_dir),
                str(cmp_dir),
                mode="clean",
                device=str(device),  # cleanfid 需要字符串格式的设备名
                use_dataparallel=False,
            )
        )
    except Exception as e:
        fid_value = None
        print(f"[WARN] FID not computed (cleanfid failed): {e}")

    summary = {
        "ref_dir": str(ref_dir),
        "cmp_dir": str(cmp_dir),
        "prompt_file": prompt_file,
        "prompt_align": prompt_align,
        "num_pairs": len(names),
        "psnr_mean": psnr_mean,
        "psnr_std": psnr_std,
        "lpips_mean": lpips_mean,
        "lpips_std": lpips_std,
        "fid": fid_value,
        "clip_model": clip_model_name,
        "clip_pretrained": clip_pretrained,
        "clip_ref_mean": clip_ref_mean,
        "clip_ref_std": clip_ref_std,
        "clip_cmp_mean": clip_cmp_mean,
        "clip_cmp_std": clip_cmp_std,
        "ssim_mean": ssim_mean,
        "ssim_std": ssim_std,
        "imagereward_ref_mean": ir_ref_mean,
        "imagereward_ref_std": ir_ref_std,
        "imagereward_cmp_mean": ir_cmp_mean,
        "imagereward_cmp_std": ir_cmp_std,
        "device": str(device),
    }

    # Save JSON
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump({"summary": summary, "per_image": rows}, f, ensure_ascii=False, indent=2)

    # Save CSV
    with open(out_csv, "w", encoding="utf-8") as f:
        f.write("name,prompt_idx,psnr,ssim,lpips,clip_ref,clip_cmp,ir_ref,ir_cmp\n")
        for r in rows:
            # 处理 None 值，避免 CSV 格式错误
            ssim_val = r.get('ssim', '') if r.get('ssim') is not None else ''
            ir_ref_val = r.get('ir_ref', '') if r.get('ir_ref') is not None else ''
            ir_cmp_val = r.get('ir_cmp', '') if r.get('ir_cmp') is not None else ''
            f.write(
                f"{r['name']},{r['prompt_idx']},{r['psnr']},{ssim_val},"
                f"{r['lpips']},{r['clip_ref']},{r['clip_cmp']},{ir_ref_val},{ir_cmp_val}\n"
            )

    print("\n=== Summary ===")
    for k, v in summary.items():
        print(f"{k}: {v}")
    print(f"\nSaved: {out_json}, {out_csv}")


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--ref_dir", type=str, required=True)
    ap.add_argument("--cmp_dir", type=str, required=True)
    ap.add_argument("--prompt_file", type=str, required=True)
    ap.add_argument("--out_json", type=str, default="metrics.json")
    ap.add_argument("--out_csv", type=str, default="metrics.csv")
    ap.add_argument("--device", type=str, default="cuda:0")  # 默认用 cuda:0，更明确
    ap.add_argument("--no_resize_if_mismatch", action="store_true")
    ap.add_argument("--no_ssim", action="store_true")
    ap.add_argument("--no_imagereward", action="store_true")

    ap.add_argument("--prompt_align", type=str, default="by_index", choices=["by_index", "by_order"])
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
        clip_model_name=args.clip_model,
        clip_pretrained=args.clip_pretrained,
        enable_ssim=not args.no_ssim,
        enable_imagereward=not args.no_imagereward,
    )
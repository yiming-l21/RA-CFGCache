import math
from typing import Callable

import numpy as np
import torch
from einops import rearrange, repeat
from PIL import Image
from torch import Tensor

from .model import Flux
from .modules.autoencoder import AutoEncoder
from .modules.conditioner import HFEmbedder
from .modules.image_embedders import CannyImageEncoder, DepthImageEncoder, ReduxImageEncoder
from .modules.cache_functions import cache_init,cal_type


def get_noise(
    num_samples: int,
    height: int,
    width: int,
    device: torch.device,
    dtype: torch.dtype,
    seed: int,
):
    return torch.randn(
        num_samples,
        16,
        # allow for packing
        2 * math.ceil(height / 16),
        2 * math.ceil(width / 16),
        device=device,
        dtype=dtype,
        generator=torch.Generator(device=device).manual_seed(seed),
    )


def prepare(t5: HFEmbedder, clip: HFEmbedder, img: Tensor, prompt: str | list[str]) -> dict[str, Tensor]:
    bs, c, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    # small_img_ids = torch.zeros((h // 2) // 2, (w // 2) // 2, 3)
    # small_img_ids[..., 1] = small_img_ids[..., 1] + torch.arange((h // 2) // 2)[:, None]
    # small_img_ids[..., 2] = small_img_ids[..., 2] + torch.arange((w // 2) // 2)[None, :]
    # small_img_ids = repeat(small_img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        # "img_ids": [img_ids.to(img.device), small_img_ids.to(img.device)],
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def prepare_control(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    encoder: DepthImageEncoder | CannyImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")

    width = w * 8
    height = h * 8
    img_cond = img_cond.resize((width, height), Image.LANCZOS)
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    with torch.no_grad():
        img_cond = encoder(img_cond)
        img_cond = ae.encode(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond
    return return_dict


def prepare_fill(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    ae: AutoEncoder,
    img_cond_path: str,
    mask_path: str,
) -> dict[str, Tensor]:
    # load and encode the conditioning image and the mask
    bs, _, _, _ = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    img_cond = np.array(img_cond)
    img_cond = torch.from_numpy(img_cond).float() / 127.5 - 1.0
    img_cond = rearrange(img_cond, "h w c -> 1 c h w")

    mask = Image.open(mask_path).convert("L")
    mask = np.array(mask)
    mask = torch.from_numpy(mask).float() / 255.0
    mask = rearrange(mask, "h w -> 1 1 h w")

    with torch.no_grad():
        img_cond = img_cond.to(img.device)
        mask = mask.to(img.device)
        img_cond = img_cond * (1 - mask)
        img_cond = ae.encode(img_cond)
        mask = mask[:, 0, :, :]
        mask = mask.to(torch.bfloat16)
        mask = rearrange(
            mask,
            "b (h ph) (w pw) -> b (ph pw) h w",
            ph=8,
            pw=8,
        )
        mask = rearrange(mask, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
        if mask.shape[0] == 1 and bs > 1:
            mask = repeat(mask, "1 ... -> bs ...", bs=bs)

    img_cond = img_cond.to(torch.bfloat16)
    img_cond = rearrange(img_cond, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img_cond = torch.cat((img_cond, mask), dim=-1)

    return_dict = prepare(t5, clip, img, prompt)
    return_dict["img_cond"] = img_cond.to(img.device)
    return return_dict


def prepare_redux(
    t5: HFEmbedder,
    clip: HFEmbedder,
    img: Tensor,
    prompt: str | list[str],
    encoder: ReduxImageEncoder,
    img_cond_path: str,
) -> dict[str, Tensor]:
    bs, _, h, w = img.shape
    if bs == 1 and not isinstance(prompt, str):
        bs = len(prompt)

    img_cond = Image.open(img_cond_path).convert("RGB")
    with torch.no_grad():
        img_cond = encoder(img_cond)

    img_cond = img_cond.to(torch.bfloat16)
    if img_cond.shape[0] == 1 and bs > 1:
        img_cond = repeat(img_cond, "1 ... -> bs ...", bs=bs)

    img = rearrange(img, "b c (h ph) (w pw) -> b (h w) (c ph pw)", ph=2, pw=2)
    if img.shape[0] == 1 and bs > 1:
        img = repeat(img, "1 ... -> bs ...", bs=bs)

    img_ids = torch.zeros(h // 2, w // 2, 3)
    img_ids[..., 1] = img_ids[..., 1] + torch.arange(h // 2)[:, None]
    img_ids[..., 2] = img_ids[..., 2] + torch.arange(w // 2)[None, :]
    img_ids = repeat(img_ids, "h w c -> b (h w) c", b=bs)

    if isinstance(prompt, str):
        prompt = [prompt]
    txt = t5(prompt)
    txt = torch.cat((txt, img_cond.to(txt)), dim=-2)
    if txt.shape[0] == 1 and bs > 1:
        txt = repeat(txt, "1 ... -> bs ...", bs=bs)
    txt_ids = torch.zeros(bs, txt.shape[1], 3)

    vec = clip(prompt)
    if vec.shape[0] == 1 and bs > 1:
        vec = repeat(vec, "1 ... -> bs ...", bs=bs)

    return {
        "img": img,
        "img_ids": img_ids.to(img.device),
        "txt": txt.to(img.device),
        "txt_ids": txt_ids.to(img.device),
        "vec": vec.to(img.device),
    }


def time_shift(mu: float, sigma: float, t: Tensor):
    return math.exp(mu) / (math.exp(mu) + (1 / t - 1) ** sigma)


def get_lin_function(
    x1: float = 256, y1: float = 0.5, x2: float = 4096, y2: float = 1.15
) -> Callable[[float], float]:
    m = (y2 - y1) / (x2 - x1)
    b = y1 - m * x1
    return lambda x: m * x + b


def get_schedule(
    num_steps: int,
    image_seq_len: int,
    base_shift: float = 0.5,
    max_shift: float = 1.15,
    shift: bool = True,
) -> list[float]:
    # extra step for zero
    timesteps = torch.linspace(1, 0, num_steps + 1)

    # shifting the schedule to favor high timesteps for higher signal images
    if shift:
        # estimate mu based on linear estimation between two points
        mu = get_lin_function(y1=base_shift, y2=max_shift)(image_seq_len)
        timesteps = time_shift(mu, 1.0, timesteps)

    return timesteps.tolist()


def denoise(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # extra img tokens
    img_cond: Tensor | None = None,
):
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        pred = model(
            img=torch.cat((img, img_cond), dim=-1) if img_cond is not None else img,
            # img_ids=img_ids[1] if small else img_ids[0],
            img_ids=img_ids[0],
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            guidance=guidance_vec,
        )

        img = img + (t_prev - t_curr) * pred

    return img


def unpack(x: Tensor, height: int, width: int) -> Tensor:
    return rearrange(
        x,
        "b (h w) (c ph pw) -> b c (h ph) (w pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )


####################################################################################################

from calflops import calculate_flops

def calc_flops_with_kwargs(model, inputs: dict):
    flops, macs, params = calculate_flops(
        model=model,
        kwargs=inputs,
        print_results=False,
    )
    return convert_flops(flops), flops, macs, params

def denoise_test_FLOPs(
    model: Flux,
    # model input
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    # sampling parameters
    timesteps: list[float],
    guidance: float = 4.0,
    # cache mode
    cache_mode: str = "Taylor",
):
    cache_dic, current = cache_init(timesteps, mode=cache_mode)
    # this is ignored for schnell
    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    current["step"] = 0
    current["num_steps"] = len(timesteps) - 1
    total_flops = 0
    for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
        current["t"] = t_curr
        inputs = dict(
            img=img,
            img_ids=img_ids,
            txt=txt,
            txt_ids=txt_ids,
            y=vec,
            timesteps=t_vec,
            cache_dic=cache_dic,
            current=current,
            guidance=guidance_vec,
        )
        flops, macs, params = calculate_flops(model=model, kwargs=inputs, print_results=False)
        total_flops += convert_flops(flops)
        current["step"] += 1

    print(f"Total {total_flops * 10 **(-12)} TFLOPs.")
    return img

def denoise_test_FLOPs_cfg(
    model: Flux,
    height: int,
    width: int,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    neg_txt: Tensor | None = None,
    neg_txt_ids: Tensor | None = None,
    neg_vec: Tensor | None = None,
    true_cfg_scale: float = 1.0,
    timesteps: list[float] | None = None,
    guidance: float = 4.0,
    cache_mode: str = "original",
    # cache params
    interval: int = 6,
    max_order: int = 1,
    first_enhance: int = 3,
    hicache_scale: float = 0.5,
    rel_l1_thresh: float = 0.6,
    # ClusCa
    clusca_fresh_threshold: int | None = None,
    clusca_cluster_num: int | None = None,
    clusca_cluster_method: str | None = None,
    clusca_k: int | None = None,
    clusca_propagation_ratio: float | None = None,
    # Analytic HiCache
    analytic_sigma_alpha: float | None = None,
    analytic_sigma_max: float | None = None,
    analytic_sigma_beta: float | None = None,
    analytic_sigma_eps: float | None = None,
    analytic_sigma_q_quantile: float | None = None,
    analytic_sigma_smooth: float | None = None,
):
    print("two branch")
    if timesteps is None:
        raise ValueError("timesteps must not be None")

    do_true_cfg = (true_cfg_scale > 1.0) and (neg_txt is not None) and (neg_vec is not None)
    if do_true_cfg and neg_txt_ids is None:
        raise ValueError("do_true_cfg=True but neg_txt_ids is None")

    # ---------- build model_kwargs ----------
    clusca_kwargs = None
    if cache_mode in {"ClusCa", "Hi-ClusCa"}:
        clusca_kwargs = {
            "clusca_fresh_threshold": clusca_fresh_threshold,
            "clusca_cluster_num": clusca_cluster_num,
            "clusca_cluster_method": clusca_cluster_method,
            "clusca_k": clusca_k,
            "clusca_propagation_ratio": clusca_propagation_ratio,
        }
        clusca_kwargs = {k: v for k, v in clusca_kwargs.items() if v is not None}

    analytic_kwargs = None
    if cache_mode == "HiCache-Analytic":
        analytic_kwargs = {
            "analytic_sigma_alpha": analytic_sigma_alpha,
            "analytic_sigma_max": analytic_sigma_max,
            "analytic_sigma_beta": analytic_sigma_beta,
            "analytic_sigma_eps": analytic_sigma_eps,
            "analytic_sigma_q_quantile": analytic_sigma_q_quantile,
            "analytic_sigma_smooth": analytic_sigma_smooth,
        }
        analytic_kwargs = {k: v for k, v in analytic_kwargs.items() if v is not None}

    if cache_mode in {"ClusCa", "Hi-ClusCa"}:
        model_kwargs = clusca_kwargs
    elif cache_mode == "HiCache-Analytic":
        model_kwargs = analytic_kwargs
    elif cache_mode == "TeaCache":
        model_kwargs = {"rel_l1_thresh": rel_l1_thresh}
    else:
        model_kwargs = None

    # ---------- init cache ----------
    def _init_one_cache(mode_name: str):
        cache_dic, current = cache_init(
            timesteps,
            model_kwargs=model_kwargs,
            mode=mode_name,
            interval=interval,
            max_order=max_order,
            first_enhance=first_enhance,
            hicache_scale=hicache_scale,
        )
        current["step"] = 0
        current["num_steps"] = len(timesteps) - 1
        return cache_dic, current

    cache_dic_cond, current_cond = _init_one_cache(cache_mode)

    if do_true_cfg:
        uncond_mode = "original" if cache_mode == "FasterCache" else cache_mode
        cache_dic_uncond, current_uncond = _init_one_cache(uncond_mode)
    else:
        cache_dic_uncond, current_uncond = None, None

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    # FasterCache cfg info
    fc = None
    if cache_mode == "FasterCache":
        fc = cache_dic_cond.get("fastercache", {}).get("cfg", None)
        if fc is not None:
            fc["enabled"] = True

    if cache_mode == "FasterCache" and fc is not None:
        T = int(current_cond.get("num_steps", 0) or 0)
        warmup = T // 3
        t0_ratio = float(fc.get("t0_ratio", 0.5))
        t0_step = warmup + int((T - warmup) * t0_ratio)
    else:
        t0_step = 0

    total_flops = 0.0
    per_step_flops = []

    for step_i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
        step_flops = 0.0
        t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

        # =====================================================
        # CFGCache: teacher-forcing oracle style => 每步固定 2 次 forward
        # =====================================================
        if cache_mode == "CFGCache" and do_true_cfg:
            current_cond["t"] = t_curr
            inputs_cond = dict(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                cache_dic=cache_dic_cond,
                current=current_cond,
                guidance=guidance_vec,
            )
            flops_num, flops_str, macs, params = calc_flops_with_kwargs(model, inputs_cond)
            step_flops += flops_num

            current_uncond["t"] = t_curr
            inputs_uncond = dict(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                cache_dic=cache_dic_uncond,
                current=current_uncond,
                guidance=guidance_vec,
            )
            flops_num, flops_str, macs, params = calc_flops_with_kwargs(model, inputs_uncond)
            step_flops += flops_num

            current_cond["step"] += 1
            current_uncond["step"] += 1

        # =====================================================
        # FasterCache
        # =====================================================
        elif cache_mode == "FasterCache" and do_true_cfg:
            cal_type(cache_dic_cond, current_cond)

            is_fastercache = (fc is not None) and fc.get("enabled", False)
            is_skip = is_fastercache and (current_cond.get("cfg_type") == "cfg_skip")

            # cond 一定算一次
            current_cond["t"] = t_curr
            inputs_cond = dict(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                cache_dic=cache_dic_cond,
                current=current_cond,
                guidance=guidance_vec,
            )
            flops_num, flops_str, macs, params = calc_flops_with_kwargs(model, inputs_cond)
            step_flops += flops_num

            # uncond 只有在没有 skip 时才算
            if not is_skip:
                current_uncond["t"] = t_curr
                inputs_uncond = dict(
                    img=img,
                    img_ids=img_ids,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    y=neg_vec,
                    timesteps=t_vec,
                    cache_dic=cache_dic_uncond,
                    current=current_uncond,
                    guidance=guidance_vec,
                )
                flops_num, flops_str, macs, params = calc_flops_with_kwargs(model, inputs_uncond)
                step_flops += flops_num

            current_cond["step"] += 1
            current_uncond["step"] += 1

        # =====================================================
        # 普通 true CFG
        # =====================================================
        elif do_true_cfg:
            current_cond["t"] = t_curr
            inputs_cond = dict(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                cache_dic=cache_dic_cond,
                current=current_cond,
                guidance=guidance_vec,
            )
            flops_num, flops_str, macs, params = calc_flops_with_kwargs(model, inputs_cond)
            step_flops += flops_num

            current_uncond["t"] = t_curr
            inputs_uncond = dict(
                img=img,
                img_ids=img_ids,
                txt=neg_txt,
                txt_ids=neg_txt_ids,
                y=neg_vec,
                timesteps=t_vec,
                cache_dic=cache_dic_uncond,
                current=current_uncond,
                guidance=guidance_vec,
            )
            flops_num, flops_str, macs, params = calc_flops_with_kwargs(model, inputs_uncond)
            step_flops += flops_num

            current_cond["step"] += 1
            current_uncond["step"] += 1

        # =====================================================
        # 单分支
        # =====================================================
        else:
            current_cond["t"] = t_curr
            inputs = dict(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                cache_dic=cache_dic_cond,
                current=current_cond,
                guidance=guidance_vec,
            )
            flops_num, flops_str, macs, params = calc_flops_with_kwargs(model, inputs)
            step_flops += flops_num
            current_cond["step"] += 1

        total_flops += step_flops
        per_step_flops.append(step_flops)

        print(
            f"[FLOPs] step={step_i:02d} "
            f"step_flops={step_flops / 1e12:.4f} TFLOPs "
            f"total={total_flops / 1e12:.4f} TFLOPs"
        )

    print(f"Total {total_flops * 1e-12:.4f} TFLOPs.")
    print(f"Total {total_flops * 1e-15:.4f} PFLOPs.")
    return img

import re

def convert_flops(flops_str: str) -> float:
    """
    将 '12.34 GFLOPS' / '1.2 TFLOPS' / '1.66 PFLOPS'
    转成对应的 FLOPs 数值（float）
    """
    match = re.match(r"([\d.]+)\s*([GPT]?FLOPS)", flops_str.strip(), re.IGNORECASE)
    if not match:
        raise ValueError(f"无法解析 FLOPS 字符串: {flops_str}")

    value = float(match.group(1))
    unit = match.group(2).upper()

    if unit == "GFLOPS":
        return value * 10**9
    elif unit == "TFLOPS":
        return value * 10**12
    elif unit == "PFLOPS":
        return value * 10**15
    else:
        raise ValueError(f"未知的 FLOPS 单位: {unit}")

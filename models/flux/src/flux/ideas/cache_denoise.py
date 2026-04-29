import torch
from torch import Tensor
from ..model import Flux
from ..modules.cache_functions import cache_init, cal_type
from flux.fastercache_utils import pack, _to_4d, _fft_split, _ifft_merge
from flux.sampling import unpack
from ..racfgcache_utils import _cfgcache_proxy_needs_preprobe, _prepare_cfgcache_probe_for_branch, _decide_cfgcache_joint, _make_joint_state

def denoise_cache(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    cache_mode: str = "Taylor",
    interval: int = 6,
    max_order: int = 1,
    first_enhance: int = 3,
    hicache_scale: float = 0.5,
    rel_l1_thresh: float = 0.6,
    clusca_fresh_threshold: int | None = None,
    clusca_cluster_num: int | None = None,
    clusca_cluster_method: str | None = None,
    clusca_k: int | None = None,
    clusca_propagation_ratio: float | None = None,
    analytic_sigma_alpha: float | None = None,
    analytic_sigma_max: float | None = None,
    analytic_sigma_beta: float | None = None,
    analytic_sigma_eps: float | None = None,
    analytic_sigma_q_quantile: float | None = None,
    analytic_sigma_smooth: float | None = None,
):
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

    cache_dic, current = cache_init(
        timesteps,
        model_kwargs=model_kwargs,
        mode=cache_mode,
        interval=interval,
        max_order=max_order,
        first_enhance=first_enhance,
        hicache_scale=hicache_scale,
    )

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)
    current["step"] = 0
    current["num_steps"] = len(timesteps) - 1

    with torch.inference_mode():
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)
            current["t"] = t_curr
            pred = model(
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
            img = img + (t_prev - t_curr) * pred
            current["step"] += 1

    model._last_cache_dic = cache_dic
    return img

def denoise_cache_cfg(
    model: "Flux",
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
    cache_mode: str = "Taylor",
    interval: int = 6,
    max_order: int = 1,
    first_enhance: int = 3,
    hicache_scale: float = 0.5,
    rel_l1_thresh: float = 0.6,
    clusca_fresh_threshold: int | None = None,
    clusca_cluster_num: int | None = None,
    clusca_cluster_method: str | None = None,
    clusca_k: int | None = None,
    clusca_propagation_ratio: float | None = None,
    analytic_sigma_alpha: float | None = None,
    analytic_sigma_max: float | None = None,
    analytic_sigma_beta: float | None = None,
    analytic_sigma_eps: float | None = None,
    analytic_sigma_q_quantile: float | None = None,
    analytic_sigma_smooth: float | None = None,
    cfgcache_runtime: dict | None = None,
):
    if timesteps is None:
        raise ValueError("timesteps must not be None")

    do_true_cfg = (true_cfg_scale > 1.0) and (neg_txt is not None) and (neg_vec is not None)
    if do_true_cfg and neg_txt_ids is None:
        raise ValueError("do_true_cfg=True but neg_txt_ids is None")

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
    elif cache_mode == "CFGCache":
        model_kwargs = {
            "proxy_name": "magcache_proxy",
        }
    else:
        model_kwargs = None
    
    if cache_mode == "CFGCache" and not do_true_cfg:
        raise ValueError("RA-CFGCache requires true CFG with both cond and uncond branches.")
    def _init_one_cache():
        cache_dic, current = cache_init(
            timesteps,
            model_kwargs=model_kwargs,
            mode=cache_mode,
            interval=interval,
            max_order=max_order,
            first_enhance=first_enhance,
            hicache_scale=hicache_scale,
            cfgcache_runtime=cfgcache_runtime if cache_mode == "CFGCache" else None,
        )
        current["step"] = 0
        current["num_steps"] = len(timesteps) - 1
        return cache_dic, current

    cache_dic_cond, current_cond = _init_one_cache()

    if do_true_cfg:
        uncond_mode = "original" if cache_mode == "FasterCache" else cache_mode
        cache_dic_uncond, current_uncond = cache_init(
            timesteps,
            model_kwargs=model_kwargs,
            mode=uncond_mode,
            interval=interval,
            max_order=max_order,
            first_enhance=first_enhance,
            hicache_scale=hicache_scale,
            cfgcache_runtime=cfgcache_runtime if cache_mode == "CFGCache" else None,
        )
        current_uncond["step"] = 0
        current_uncond["num_steps"] = len(timesteps) - 1
    else:
        cache_dic_uncond, current_uncond = None, None

    if cache_mode == "CFGCache" and not do_true_cfg:
        raise ValueError("CFGCache joint drift requires true CFG with both cond and uncond branches.")

    joint_state = None
    if cache_mode == "CFGCache" and do_true_cfg:
        joint_state = _make_joint_state(
            cache_dic_cond=cache_dic_cond,
            cache_dic_uncond=cache_dic_uncond,
            true_cfg_scale=true_cfg_scale,
            num_steps=len(timesteps) - 1,
        )

    guidance_vec = torch.full((img.shape[0],), guidance, device=img.device, dtype=img.dtype)

    fc = None
    if cache_mode == "FasterCache":
        fc = cache_dic_cond.get("fastercache", {}).get("cfg", None)
        if fc is not None:
            fc["enabled"] = True

    if cache_mode == "FasterCache" and fc is not None:
        T = int(current_cond.get("num_steps", 0) or 0)
        warmup = T // 3
        alpha1 = float(fc.get("alpha1", 0.2))
        alpha2 = float(fc.get("alpha2", 0.2))
        t0_ratio = float(fc.get("t0_ratio", 0.5))
        t0_step = warmup + int((T - warmup) * t0_ratio)
    else:
        T = warmup = t0_step = 0
        alpha1 = alpha2 = 0.0

    with torch.inference_mode():
        for i, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full((img.shape[0],), t_curr, dtype=img.dtype, device=img.device)

            current_cond["t"] = t_curr
            if do_true_cfg:
                current_uncond["t"] = t_curr

            preencoded_cond = None
            preencoded_uncond = None
            cfgcache_decision = None

            if cache_mode == "CFGCache" and do_true_cfg:
                cc = cache_dic_cond["cfgcache"]
                step_i = int(current_cond.get("step", 0))
                warmup_steps = int(cc.get("warmup_steps", 5))
                cond_ready = cc.get("previous_residual", None) is not None
                uncond_ready = cache_dic_uncond["cfgcache"].get("previous_residual", None) is not None

                forced_full = (step_i <= warmup_steps) or (not cond_ready) or (not uncond_ready)

                if forced_full:
                    cfgcache_decision = {
                        "action": "full_both",
                        "dhat": None,
                        "ghat": 1.0,
                        "rhat": None,
                    }
                    current_cond["cfgcache_joint_action"] = "full_both"
                    current_uncond["cfgcache_joint_action"] = "full_both"
                    current_cond["cfgcache_probe_prepared"] = False
                    current_uncond["cfgcache_probe_prepared"] = False
                else:
                    need_joint_preprobe = _cfgcache_proxy_needs_preprobe(cache_dic_cond)

                    if need_joint_preprobe:
                        preencoded_cond = _prepare_cfgcache_probe_for_branch(
                            model=model,
                            img=img,
                            img_ids=img_ids,
                            txt=txt,
                            txt_ids=txt_ids,
                            vec=vec,
                            t_vec=t_vec,
                            guidance_vec=guidance_vec,
                            cache_dic=cache_dic_cond,
                            current=current_cond,
                        )

                        preencoded_uncond = _prepare_cfgcache_probe_for_branch(
                            model=model,
                            img=img,
                            img_ids=img_ids,
                            txt=neg_txt,
                            txt_ids=neg_txt_ids,
                            vec=neg_vec,
                            t_vec=t_vec,
                            guidance_vec=guidance_vec,
                            cache_dic=cache_dic_uncond,
                            current=current_uncond,
                        )

                    current_cond["cfgcache_probe_prepared"] = preencoded_cond is not None
                    current_uncond["cfgcache_probe_prepared"] = preencoded_uncond is not None

                    cfgcache_decision = _decide_cfgcache_joint(
                        cache_dic_cond=cache_dic_cond,
                        cache_dic_uncond=cache_dic_uncond,
                        current_cond=current_cond,
                        true_cfg_scale=true_cfg_scale,
                    )

                    action = cfgcache_decision["action"]
                    current_cond["cfgcache_joint_action"] = action
                    current_uncond["cfgcache_joint_action"] = action

            if cache_mode == "FasterCache":
                cal_type(cache_dic_cond, current_cond)
                is_fastercache = (
                    (cache_mode == "FasterCache")
                    and do_true_cfg
                    and (fc is not None)
                    and fc.get("enabled", False)
                )
                is_skip = is_fastercache and (current_cond.get("cfg_type") == "cfg_skip")

            pred_cond = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                cache_dic=cache_dic_cond,
                current=current_cond,
                guidance=guidance_vec,
                preencoded=preencoded_cond,
            )

            if do_true_cfg:
                if cache_mode == "FasterCache":
                    if is_skip:
                        cfg_mode = fc.get("cfg_mode", "fft")
                        s_fc = float(fc.get("scale", 1.0))

                        if cfg_mode == "delta":
                            delta = fc.get("delta", None)
                            has_delta = fc.get("has_delta", False) and (delta is not None)
                            if not has_delta:
                                is_skip = False
                            else:
                                pred_uncond = pred_cond + s_fc * delta
                        else:
                            delta_lf = fc.get("delta_lf", None)
                            delta_hf = fc.get("delta_hf", None)
                            has_fft = (
                                fc.get("has_delta_fft", False)
                                and (delta_lf is not None)
                                and (delta_hf is not None)
                            )
                            if not has_fft:
                                is_skip = False
                            else:
                                predc_4d = _to_4d(pred_cond, height, width, unpack).float()
                                lf_c, hf_c = _fft_split(predc_4d)
                                step = int(current_cond["step"])
                                is_early = (step < t0_step)
                                w1 = 1.0 + (alpha1 if is_early else 0.0)
                                w2 = 1.0 + (alpha2 if (not is_early) else 0.0)
                                pred_uncond_4d = _ifft_merge(lf_c + w1 * delta_lf, hf_c + w2 * delta_hf)
                                pred_uncond = pack(
                                    pred_uncond_4d.to(dtype=pred_cond.dtype),
                                    height=height,
                                    width=width,
                                )

                    if not is_skip:
                        current_uncond["t"] = t_curr
                        pred_uncond = model(
                            img=img,
                            img_ids=img_ids,
                            txt=neg_txt,
                            txt_ids=neg_txt_ids,
                            y=neg_vec,
                            timesteps=t_vec,
                            cache_dic=cache_dic_uncond,
                            current=current_uncond,
                            guidance=guidance_vec,
                            preencoded=preencoded_uncond,
                        )
                        if is_fastercache:
                            cfg_mode = fc.get("cfg_mode", "fft")
                            sdelta = (pred_uncond - pred_cond).detach()
                            if cfg_mode == "delta":
                                fc["delta"] = sdelta
                                fc["has_delta"] = True
                            else:
                                delta = sdelta.float()
                                delta_4d = _to_4d(delta, height, width, unpack)
                                dlf, dhf = _fft_split(delta_4d)
                                fc["delta_lf"] = dlf
                                fc["delta_hf"] = dhf
                                fc["has_delta_fft"] = True
                else:
                    current_uncond["t"] = t_curr
                    pred_uncond = model(
                        img=img,
                        img_ids=img_ids,
                        txt=neg_txt,
                        txt_ids=neg_txt_ids,
                        y=neg_vec,
                        timesteps=t_vec,
                        cache_dic=cache_dic_uncond,
                        current=current_uncond,
                        guidance=guidance_vec,
                        preencoded=preencoded_uncond,
                    )
                pred = pred_uncond + true_cfg_scale * (pred_cond - pred_uncond)
            else:
                pred = pred_cond

            if cache_mode == "CFGCache":
                action = current_cond.get("cfgcache_joint_action", "full_both")

                if action == "reuse_both":
                    joint_state["accumulated_risk"] += float(cfgcache_decision["rhat"])
                    joint_state["consecutive_reuse"] += 1
                    joint_state["last_action"] = "reuse_both"
                else:
                    joint_state["anchor_step"] = int(current_cond["step"])
                    joint_state["accumulated_risk"] = 0.0
                    joint_state["consecutive_reuse"] = 0
                    joint_state["last_action"] = "full_both"

                joint_state["log"].append({
                    "step": int(current_cond["step"]),
                    "anchor_step": int(joint_state["anchor_step"]),
                    "action": joint_state["last_action"],
                    "dhat": None if cfgcache_decision["dhat"] is None else float(cfgcache_decision["dhat"]),
                    "ghat": float(cfgcache_decision["ghat"]),
                    "rhat": None if cfgcache_decision["rhat"] is None else float(cfgcache_decision["rhat"]),
                    "accumulated_risk": float(joint_state["accumulated_risk"]),
                })

            img = img + (t_prev - t_curr) * pred
            current_cond["step"] += 1
            if do_true_cfg:
                current_uncond["step"] += 1

    if do_true_cfg:
        model._last_cache_dic = {"cond": cache_dic_cond, "uncond": cache_dic_uncond}
    else:
        model._last_cache_dic = cache_dic_cond
    return img
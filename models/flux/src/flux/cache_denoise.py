import os
import json
from pathlib import Path

import torch
from torch import Tensor

from .model import Flux


def denoise(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
) -> Tensor:
    """
    Pure full denoising without any DiT cache logic.

    This function performs standard FLUX denoising:
        x_{t_prev} = x_t + (t_prev - t_curr) * model(x_t, t_curr)

    No cache_dic.
    No current.
    No cache_mode.
    No Taylor / TeaCache / FasterCache / CFGCache / ClusCa.
    """

    guidance_vec = torch.full(
        (img.shape[0],),
        guidance,
        device=img.device,
        dtype=img.dtype,
    )

    with torch.inference_mode():
        for t_curr, t_prev in zip(timesteps[:-1], timesteps[1:]):
            t_vec = torch.full(
                (img.shape[0],),
                t_curr,
                device=img.device,
                dtype=img.dtype,
            )

            pred = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            img = img + (t_prev - t_curr) * pred

    return img


def _get_project_root() -> str:
    return os.path.abspath(
        os.path.join(os.path.dirname(__file__), "../../../..")
    )


def _ensure_prompt_ids(prompt_ids: list[int] | None, batch_size: int) -> list[int]:
    if prompt_ids is None:
        return list(range(batch_size))

    if len(prompt_ids) != batch_size:
        raise ValueError(
            f"len(prompt_ids) must match batch size, got "
            f"len(prompt_ids)={len(prompt_ids)}, batch={batch_size}"
        )

    return [int(x) for x in prompt_ids]


def _save_tensor_per_prompt(
    *,
    tensor: Tensor,
    prompt_ids: list[int],
    out_dirs: list[Path],
    filename: str,
):
    """
    tensor shape 默认第一维是 batch。
    保存时保留原 dtype，只 detach + cpu。
    """
    if tensor.shape[0] != len(prompt_ids):
        raise ValueError(
            f"Tensor batch size mismatch when saving {filename}: "
            f"tensor.shape[0]={tensor.shape[0]}, len(prompt_ids)={len(prompt_ids)}"
        )

    for b, out_dir in enumerate(out_dirs):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(
            tensor[b].detach().cpu(),
            out_dir / filename,
        )


def _write_json(path: Path, data: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def _resolve_runtime_ctx(
    *,
    runtime_ctx: dict | None,
    calibrate_mode: str,
    calibrate_root: str | None,
    perturb_step: int | None,
    prompt_ids: list[int] | None,
):
    """
    兼容两种调用方式：

    1. 旧方式：
       denoise_cfg(..., calibrate_mode="gt", perturb_step=10, prompt_ids=[...])

    2. 新方式：
       pipeline 外层传 runtime_ctx：
       runtime_ctx["dump_gt_cfg"] = {
           "root": ...,
           "role": "baseline" / "perturb",
           "run_id": "baseline" / "perturb_step_010",
           "perturb_step": 10,
       }

    如果 runtime_ctx 存在，则优先使用 runtime_ctx。
    """
    if runtime_ctx is None:
        return {
            "calibrate_mode": calibrate_mode,
            "calibrate_root": calibrate_root,
            "perturb_step": perturb_step,
            "prompt_ids": prompt_ids,
            "gt_role": None,
            "gt_run_id": None,
            "dump_gt_cfg": None,
            "dump_residual_cfg": None,
        }

    mode = runtime_ctx.get("calibrate_mode", calibrate_mode)
    ctx_prompt_ids = runtime_ctx.get("prompt_indices", prompt_ids)

    dump_gt_cfg = runtime_ctx.get("dump_gt_cfg", None)
    dump_residual_cfg = runtime_ctx.get("dump_residual_cfg", None)

    root = runtime_ctx.get("calibrate_root", calibrate_root)

    gt_role = None
    gt_run_id = None
    gt_perturb_step = perturb_step

    if dump_gt_cfg is not None:
        root = dump_gt_cfg.get("root", root)
        gt_role = dump_gt_cfg.get("role", None)
        gt_run_id = dump_gt_cfg.get("run_id", None)
        gt_perturb_step = dump_gt_cfg.get("perturb_step", perturb_step)

    return {
        "calibrate_mode": mode,
        "calibrate_root": root,
        "perturb_step": gt_perturb_step,
        "prompt_ids": ctx_prompt_ids,
        "gt_role": gt_role,
        "gt_run_id": gt_run_id,
        "dump_gt_cfg": dump_gt_cfg,
        "dump_residual_cfg": dump_residual_cfg,
    }


def denoise_cfg(
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    timesteps: list[float],
    guidance: float = 4.0,
    neg_txt: Tensor | None = None,
    neg_txt_ids: Tensor | None = None,
    neg_vec: Tensor | None = None,
    true_cfg_scale: float = 1.0,

    # old style calibration args
    calibrate_mode: str = "none",
    interval: int | None = None,
    perturb_step: int | None = None,
    prompt_ids: list[int] | None = None,
    calibrate_root: str | None = None,

    # new style: 外层 pipeline 直接透传 runtime_ctx
    runtime_ctx: dict | None = None,
) -> Tensor:
    """
    FLUX denoise with optional true CFG and calibration dump.

    Supported modes:

    1. calibrate_mode="none"
       正常推理，不保存。

    2. calibrate_mode="rho"
       保存每一步:
           step_xxx_xt.pt
           step_xxx_cond.pt
           step_xxx_uncond.pt

    3. calibrate_mode="gt"
       只保存后续计算 GT 需要的 tensor，不在这里计算 GT。

       baseline run:
           calibration_root/gt/prompt_xxxxxx/baseline/final_latents.pt

       perturb run:
           calibration_root/gt/prompt_xxxxxx/perturb_step_xxx/
               final_latents.pt
               guided_ref.pt
               guided_old.pt

       其中：
           guided_ref = 当前 step 正常计算得到的 guided pred
           guided_old = 上一个 step 的 cond/uncond pred 组合出来的 guided pred

       在 perturb_step 处，真正用于更新 img 的 pred 被替换为 guided_old。
       其他 step 正常计算。
    """

    if timesteps is None:
        raise ValueError("timesteps must not be None")

    ctx = _resolve_runtime_ctx(
        runtime_ctx=runtime_ctx,
        calibrate_mode=calibrate_mode,
        calibrate_root=calibrate_root,
        perturb_step=perturb_step,
        prompt_ids=prompt_ids,
    )

    calibrate_mode = ctx["calibrate_mode"]
    calibrate_root = ctx["calibrate_root"]
    perturb_step = ctx["perturb_step"]
    prompt_ids = ctx["prompt_ids"]
    gt_role = ctx["gt_role"]
    gt_run_id = ctx["gt_run_id"]
    dump_gt_cfg = ctx["dump_gt_cfg"]

    if calibrate_mode not in ["none", "rho", "gt"]:
        raise ValueError(f"Unsupported calibrate_mode: {calibrate_mode}")

    batch_size = img.shape[0]
    prompt_ids = _ensure_prompt_ids(prompt_ids, batch_size)

    do_true_cfg = (
        true_cfg_scale > 1.0
        and neg_txt is not None
        and neg_txt_ids is not None
        and neg_vec is not None
    )

    if calibrate_mode in ["rho", "gt"] and not do_true_cfg:
        raise ValueError(
            f"calibrate_mode='{calibrate_mode}' requires true CFG. "
            f"Please set true_cfg_scale > 1.0 and provide negative prompt inputs."
        )

    num_denoise_steps = len(timesteps) - 1

    if calibrate_mode == "gt":
        if gt_role is None:
            # 兼容旧调用方式：只要 calibrate_mode=gt，但没传 role，则默认认为是 perturb。
            gt_role = "perturb" if perturb_step is not None else "baseline"

        if gt_run_id is None:
            if gt_role == "baseline":
                gt_run_id = "baseline"
            elif gt_role == "perturb":
                if perturb_step is None:
                    raise ValueError(
                        "GT perturb run requires perturb_step."
                    )
                gt_run_id = f"perturb_step_{int(perturb_step):03d}"

        if gt_role not in ["baseline", "perturb"]:
            raise ValueError(
                f"Unsupported gt_role={gt_role}. Expected 'baseline' or 'perturb'."
            )

        if gt_role == "perturb":
            if perturb_step is None:
                raise ValueError("GT perturb run requires perturb_step.")

            if perturb_step <= 0:
                raise ValueError(
                    "GT perturb_step must be >= 1 because step 0 has no previous pred."
                )

            if perturb_step >= num_denoise_steps:
                raise ValueError(
                    f"GT perturb_step must be < num_denoise_steps={num_denoise_steps}, "
                    f"got {perturb_step}."
                )

    guidance_vec = torch.full(
        (batch_size,),
        guidance,
        device=img.device,
        dtype=img.dtype,
    )

    # -----------------------------
    # rho root
    # -----------------------------
    if calibrate_mode == "rho":
        project_root = _get_project_root()

        cfg_tag = f"cfg{float(true_cfg_scale):g}_traces"
        if calibrate_root is None:
            trace_root = Path(project_root) / "calibration_rho" / "flux" / cfg_tag
        else:
            trace_root = Path(calibrate_root) / cfg_tag
    else:
        trace_root = None

    # -----------------------------
    # gt root
    # -----------------------------
    if calibrate_mode == "gt":
        if calibrate_root is None:
            gt_root = Path(_get_project_root()) / "calibration_gt" / "flux"
        else:
            gt_root = Path(calibrate_root)

        if gt_role == "baseline":
            gt_out_dirs = [
                gt_root / f"cfg{float(true_cfg_scale):.1f}_traces" / f"prompt_{pid:06d}" / "baseline"
                for pid in prompt_ids
            ]
        else:
            gt_out_dirs = [
                gt_root / f"cfg{float(true_cfg_scale):.1f}_traces" / f"prompt_{pid:06d}" / str(gt_run_id)
                for pid in prompt_ids
            ]
    else:
        gt_root = None
        gt_out_dirs = None

    prev_cond: Tensor | None = None
    prev_uncond: Tensor | None = None
    perturb_done = False

    with torch.inference_mode():
        for step_idx, (t_curr, t_prev) in enumerate(zip(timesteps[:-1], timesteps[1:])):
            t_vec = torch.full(
                (batch_size,),
                t_curr,
                device=img.device,
                dtype=img.dtype,
            )

            # -----------------------------
            # cond branch
            # -----------------------------
            pred_cond = model(
                img=img,
                img_ids=img_ids,
                txt=txt,
                txt_ids=txt_ids,
                y=vec,
                timesteps=t_vec,
                guidance=guidance_vec,
            )

            # -----------------------------
            # uncond branch + true CFG
            # -----------------------------
            if do_true_cfg:
                pred_uncond = model(
                    img=img,
                    img_ids=img_ids,
                    txt=neg_txt,
                    txt_ids=neg_txt_ids,
                    y=neg_vec,
                    timesteps=t_vec,
                    guidance=guidance_vec,
                )

                if calibrate_mode == "rho":
                    assert trace_root is not None

                    for b in range(batch_size):
                        prompt_id = int(prompt_ids[b])
                        prompt_dir = trace_root / f"prompt_{prompt_id:06d}"
                        prompt_dir.mkdir(parents=True, exist_ok=True)

                        torch.save(
                            img[b].detach().cpu(),
                            prompt_dir / f"step_{step_idx}_xt.pt",
                        )
                        torch.save(
                            pred_cond[b].detach().cpu(),
                            prompt_dir / f"step_{step_idx}_cond.pt",
                        )
                        torch.save(
                            pred_uncond[b].detach().cpu(),
                            prompt_dir / f"step_{step_idx}_uncond.pt",
                        )

                guided_ref = pred_uncond + true_cfg_scale * (pred_cond - pred_uncond)
                pred_for_update = guided_ref

                if (
                    calibrate_mode == "gt"
                    and gt_role == "perturb"
                    and int(step_idx) == int(perturb_step)
                ):
                    if prev_cond is None or prev_uncond is None:
                        raise RuntimeError(
                            f"Cannot perturb step {step_idx}: previous cond/uncond pred is None."
                        )

                    guided_old = prev_uncond + true_cfg_scale * (prev_cond - prev_uncond)

                    assert gt_out_dirs is not None
                    _save_tensor_per_prompt(
                        tensor=guided_ref,
                        prompt_ids=prompt_ids,
                        out_dirs=gt_out_dirs,
                        filename="guided_ref.pt",
                    )
                    _save_tensor_per_prompt(
                        tensor=guided_old,
                        prompt_ids=prompt_ids,
                        out_dirs=gt_out_dirs,
                        filename="guided_old.pt",
                    )
                    pred_for_update = guided_old
                    perturb_done = True

                # 更新 prev：保存的是当前 step 正常计算出来的双分支 pred
                # 注意：即使当前 step 被 perturb，也要用当前正常 pred 更新 prev。
                prev_cond = pred_cond.detach()
                prev_uncond = pred_uncond.detach()

            else:
                pred_for_update = pred_cond

            # -----------------------------
            # Euler / flow update
            # -----------------------------
            img = img + (t_prev - t_curr) * pred_for_update

    # -----------------------------
    # 保存 GT final latents
    # -----------------------------
    if calibrate_mode == "gt":
        assert gt_out_dirs is not None

        if gt_role == "perturb" and not perturb_done:
            raise RuntimeError(
                f"GT perturb_step={perturb_step} was not executed. "
                f"num_denoise_steps={num_denoise_steps}."
            )

        _save_tensor_per_prompt(
            tensor=img,
            prompt_ids=prompt_ids,
            out_dirs=gt_out_dirs,
            filename="final_latents.pt",
        )

        for b, out_dir in enumerate(gt_out_dirs):
            meta_path = out_dir / "run_meta.json"
            _write_json(
                meta_path,
                {
                    "prompt_id": int(prompt_ids[b]),
                    "calibrate_mode": calibrate_mode,
                    "gt_role": gt_role,
                    "gt_run_id": gt_run_id,
                    "perturb_step": None if perturb_step is None else int(perturb_step),
                    "num_denoise_steps": int(num_denoise_steps),
                    "true_cfg_scale": float(true_cfg_scale),
                    "guidance": float(guidance),
                    "saved_tensors": (
                        ["final_latents.pt"]
                        if gt_role == "baseline"
                        else [
                            "guided_ref.pt",
                            "guided_old.pt",
                            "cond_ref.pt",
                            "uncond_ref.pt",
                            "cond_old.pt",
                            "uncond_old.pt",
                            "final_latents.pt",
                        ]
                    ),
                },
            )

    return img
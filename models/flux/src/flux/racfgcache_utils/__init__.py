from .proxies import ProxyContext
from ..model import Flux
from torch import Tensor
def build_prop_weight_schedule(T: int, a: float, alpha: float, b: float):
    T = max(int(T), 1)
    if T == 1:
        return [1.0]

    raws = []
    for k in range(T):
        xk = (T - 1 - k) / float(T - 1)
        raws.append(a * (xk ** alpha) + b)

    raw_mean = max(sum(raws) / len(raws), 1e-12)
    return [float(r / raw_mean) for r in raws]


def _cfgcache_proxy_needs_preprobe(cache_dic) -> bool:
    if cache_dic is None or cache_dic.get("mode") != "CFGCache":
        return False

    cc = cache_dic.get("cfgcache", {})
    proxy = cc.get("_proxy_obj", None)
    if proxy is None:
        return False

    probe_mode = getattr(proxy, "probe_mode", cc.get("probe_mode", "none"))
    has_custom_prepare = hasattr(proxy, "prepare_forward_probe")

    return bool(has_custom_prepare or (probe_mode not in [None, "none"]))


def _prepare_cfgcache_probe_for_branch(
    *,
    model: Flux,
    img: Tensor,
    img_ids: Tensor,
    txt: Tensor,
    txt_ids: Tensor,
    vec: Tensor,
    t_vec: Tensor,
    guidance_vec: Tensor | None,
    cache_dic: dict,
    current: dict,
):
    if not _cfgcache_proxy_needs_preprobe(cache_dic):
        return None

    if not hasattr(model, "_cfgcache_encode_inputs"):
        raise AttributeError(
            "Model does not implement _cfgcache_encode_inputs; "
            "please use the modified Flux class."
        )

    encoded = model._cfgcache_encode_inputs(
        img=img,
        img_ids=img_ids,
        txt=txt,
        txt_ids=txt_ids,
        timesteps=t_vec,
        y=vec,
        guidance=guidance_vec,
    )

    model._cfgcache_prepare_probe(
        img=encoded["img"],
        txt=encoded["txt"],
        vec=encoded["vec"],
        pe=encoded["pe"],
        cache_dic=cache_dic,
        current=current,
    )
    return encoded



def _make_joint_state(cache_dic_cond, cache_dic_uncond, true_cfg_scale: float, num_steps: int):
    cc_cond = cache_dic_cond["cfgcache"]
    cc_uncond = cache_dic_uncond["cfgcache"]

    joint_state = {
        "anchor_step": 0,
        "accumulated_risk": 0.0,
        "consecutive_reuse": 0,
        "last_action": "full_both",
        "true_cfg_scale": float(true_cfg_scale),
        "tau": float(cc_cond.get("cfgcache_thresh", 0.24)),
        "log": [],
        "last_metrics": None,
        "prop_weight_schedule": build_prop_weight_schedule(
            T=max(1, int(num_steps)),
            a=float(cc_cond.get("prop_a", 0.4806165972188018)),
            alpha=float(cc_cond.get("prop_alpha", 0.4782565130260521)),
            b=float(cc_cond.get("prop_b", 0.06411702055286407)),
        ),
    }

    cc_cond["joint_state"] = joint_state
    cc_uncond["joint_state"] = joint_state

    cc_cond["true_cfg_scale"] = float(true_cfg_scale)
    cc_uncond["true_cfg_scale"] = float(true_cfg_scale)

    return joint_state


def _decide_cfgcache_joint(
    *,
    cache_dic_cond,
    cache_dic_uncond,
    current_cond,
    true_cfg_scale: float,
):
    cc = cache_dic_cond["cfgcache"]
    joint_state = cc["joint_state"]
    proxy = cc["_proxy_obj"]

    step_i = int(current_cond.get("step", 0))
    num_steps = int(current_cond.get("num_steps", 0)) + 1

    ctx = ProxyContext(
        step_i=step_i,
        num_steps=num_steps,
        cfg_scale=float(true_cfg_scale),
        anchor_step=int(joint_state["anchor_step"]),
        img=None,
        current=current_cond,
        cache_dic_cond=cache_dic_cond,
        cache_dic_uncond=cache_dic_uncond,
    )

    scores = proxy.score(ctx)
    dhat = float(scores.reuse_both)

    if bool(cc.get("use_prop_weight", True)):
        ghat = float(joint_state["prop_weight_schedule"][step_i])
    else:
        ghat = 1.0

    rhat = max(float(dhat) * float(ghat), 0.0)

    joint_state["last_metrics"] = {
        "dhat": float(dhat),
        "ghat": float(ghat),
        "rhat": float(rhat),
    }

    tau = float(joint_state["tau"])
    reuse_budget_ok = (joint_state["accumulated_risk"] + rhat <= tau)
    action = "reuse_both" if reuse_budget_ok else "full_both"

    return {
        "action": action,
        "dhat": dhat,
        "ghat": float(ghat),
        "rhat": rhat,
    }

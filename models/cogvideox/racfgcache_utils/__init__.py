def _cfgcache_enabled(cache_dic, guidance_scale: float) -> bool:
    return (
        cache_dic is not None
        and cache_dic.get("mode") == "CFGCache"
        and float(guidance_scale) > 1.0
    )


def _cfgcache_branch_state(cache_dic, branch: str):
    return cache_dic["cfgcache"][branch]


def _cfgcache_branch_ready(cache_dic, branch: str) -> bool:
    bc = _cfgcache_branch_state(cache_dic, branch)
    return bc.get("previous_postnorm_hidden", None) is not None


def _cfgcache_proxy_preprobe_kind(cache_dic) -> str:
    """
    Return one of:
      - "none":        no probe is needed before the joint decision
      - "light":       only timestep-conditioned Tea-style signal is needed
      - "transformer": shallow transformer probe is needed (Di-style)
    """
    if cache_dic is None or cache_dic.get("mode") != "CFGCache":
        return "none"

    cc = cache_dic.get("cfgcache", {})
    proxy = cc.get("_proxy_obj", None)
    if proxy is None:
        return "none"

    probe_mode = getattr(proxy, "probe_mode", cc.get("probe_mode", "none"))
    probe_mode = "none" if probe_mode is None else str(probe_mode).lower()

    has_custom_prepare = hasattr(proxy, "prepare_forward_probe")
    if has_custom_prepare:
        return "transformer"

    if probe_mode in {"none", "off", "false"}:
        return "none"

    if probe_mode in {"modulated_inp", "emb", "light", "light_signal", "tea", "teacache"}:
        return "light"

    return "transformer"


def _decide_cfgcache_joint_single_cache(
    *,
    cache_dic,
    current,
    true_cfg_scale: float,
):
    from .proxies import ProxyContext

    cc = cache_dic["cfgcache"]
    joint_state = cc["joint_state"]
    proxy = cc["_proxy_obj"]

    step_i = int(current.get("step", 0))
    num_steps = int(current.get("num_steps", 0)) + 1

    ctx = ProxyContext(
        step_i=step_i,
        num_steps=num_steps,
        cfg_scale=float(true_cfg_scale),
        anchor_step=int(joint_state["anchor_step"]),
        img=None,
        current=current,
        cache_dic_cond=cache_dic,
        cache_dic_uncond=cache_dic,
    )

    scores = proxy.score(ctx)
    dhat = float(scores.reuse_both)

    if bool(cc.get("use_prop_weight", True)):
        ghat = float(joint_state["prop_weight_schedule"][step_i])
    else:
        ghat = 1.0

    rhat = max(float(dhat) * float(ghat), 0.0)
    tau = float(joint_state["tau"])
    action = "reuse_both" if (joint_state["accumulated_risk"] + rhat <= tau) else "full_both"

    joint_state["last_metrics"] = {
        "dhat": float(dhat),
        "ghat": float(ghat),
        "rhat": float(rhat),
    }

    return {
        "action": action,
        "dhat": float(dhat),
        "ghat": float(ghat),
        "rhat": float(rhat),
    }


def _commit_cfgcache_joint_single_cache(
    *,
    cache_dic,
    current,
    decision,
):
    if cache_dic is None or cache_dic.get("mode") != "CFGCache":
        return
    if decision is None:
        return

    cc = cache_dic["cfgcache"]
    joint_state = cc["joint_state"]
    action = current.get("cfgcache_joint_action", "full_both")

    if action == "reuse_both":
        joint_state["accumulated_risk"] += float(decision["rhat"])
        joint_state["consecutive_reuse"] += 1
        joint_state["last_action"] = "reuse_both"
    else:
        joint_state["anchor_step"] = int(current["step"])
        joint_state["accumulated_risk"] = 0.0
        joint_state["consecutive_reuse"] = 0
        joint_state["last_action"] = "full_both"

    joint_state["log"].append({
        "step": int(current["step"]),
        "anchor_step": int(joint_state["anchor_step"]),
        "action": joint_state["last_action"],
        "dhat": None if decision["dhat"] is None else float(decision["dhat"]),
        "ghat": float(decision["ghat"]),
        "rhat": None if decision["rhat"] is None else float(decision["rhat"]),
        "accumulated_risk": float(joint_state["accumulated_risk"]),
    })
def _cfgcache_enabled(cache_dic, guidance_scale: float) -> bool:
    return (
        cache_dic is not None
        and cache_dic.get("mode") == "CFGCache"
        and float(guidance_scale) > 1.0
    )


def _cfgcache_branch_state(cache_dic, branch: str):
    return cache_dic["cfgcache"][branch]


def _cfgcache_branch_ready(cache_dic, branch: str) -> bool:
    """
    Wan-only readiness.

    Wan cache reuses hidden residual:
        hidden_states = base_hidden_states + previous_residual

    So branch readiness should be based on previous_residual, not
    previous_postnorm_hidden.
    """
    bc = _cfgcache_branch_state(cache_dic, branch)
    return bc.get("previous_residual", None) is not None


def _cfgcache_both_branches_ready(cache_dic) -> bool:
    if cache_dic is None or cache_dic.get("mode") != "CFGCache":
        return False

    cc = cache_dic.get("cfgcache", {})
    return (
        cc.get("cond", {}).get("previous_residual", None) is not None
        and cc.get("uncond", {}).get("previous_residual", None) is not None
    )


def _cfgcache_proxy_preprobe_kind(cache_dic) -> str:
    """
    Return one of:
      - "none":        no probe is needed before the joint decision
      - "light":       only timestep-conditioned Tea-style signal is needed
      - "transformer": shallow transformer probe is needed, Di-style
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
    """
    Decide joint CFGCache action for Wan.

    Wan uses one shared decision for the CFG pair:
        - full_both
        - reuse_both

    Important:
    If either cond/uncond branch does not have previous_residual yet,
    force full_both. Otherwise the controller may record reuse risk even
    though local cal_type later falls back to full.
    """
    from .proxies import ProxyContext

    cc = cache_dic["cfgcache"]
    joint_state = cc["joint_state"]
    proxy = cc["_proxy_obj"]

    step_i = int(current.get("step", 0))
    num_steps = int(cc.get("num_steps", current.get("num_steps", 0) or 0))
    num_steps = max(num_steps, 1)

    both_ready = _cfgcache_both_branches_ready(cache_dic)

    # First valid refresh / cache not ready: must full.
    if not both_ready:
        joint_state["last_metrics"] = {
            "dhat": None,
            "ghat": 1.0,
            "rhat": None,
            "reason": "branch_not_ready",
        }

        return {
            "action": "full_both",
            "dhat": None,
            "ghat": 1.0,
            "rhat": None,
            "reason": "branch_not_ready",
        }

    ctx = ProxyContext(
        step_i=step_i,
        num_steps=num_steps,
        cfg_scale=float(true_cfg_scale),
        anchor_step=int(joint_state.get("anchor_step", 0)),
        img=None,
        current=current,
        cache_dic_cond=cache_dic,
        cache_dic_uncond=cache_dic,
    )

    scores = proxy.score(ctx)
    dhat = float(scores.reuse_both)

    if bool(cc.get("use_prop_weight", True)):
        schedule = joint_state.get("prop_weight_schedule", None)
        if schedule is None or len(schedule) == 0:
            ghat = 1.0
        else:
            safe_step_i = min(max(step_i, 0), len(schedule) - 1)
            ghat = float(schedule[safe_step_i])
    else:
        ghat = 1.0

    rhat = max(float(dhat) * float(ghat), 0.0)
    tau = float(joint_state["tau"])
    accumulated_risk = float(joint_state.get("accumulated_risk", 0.0))
    action = "reuse_both" if accumulated_risk + rhat <= tau else "full_both"
    print(f"CFGCache step {step_i}: dhat={dhat:.4f}, ghat={ghat:.4f}, rhat={rhat:.4f}, tau={tau:.4f}, accumulated_risk={accumulated_risk:.4f}, action={action}")
    joint_state["last_metrics"] = {
        "dhat": float(dhat),
        "ghat": float(ghat),
        "rhat": float(rhat),
        "reason": "ok",
    }

    return {
        "action": action,
        "dhat": float(dhat),
        "ghat": float(ghat),
        "rhat": float(rhat),
        "reason": "ok",
    }


def _commit_cfgcache_joint_single_cache(
    *,
    cache_dic,
    current,
    decision,
):
    """
    Commit joint CFGCache decision after the CFG pair finishes.

    For Wan:
    - reuse_both accumulates estimated risk.
    - full_both resets anchor and accumulated risk.
    """
    if cache_dic is None or cache_dic.get("mode") != "CFGCache":
        return

    if decision is None:
        return

    cc = cache_dic["cfgcache"]
    joint_state = cc["joint_state"]

    action = current.get("cfgcache_joint_action", decision.get("action", "full_both"))

    if action == "reuse_both":
        rhat = decision.get("rhat", None)
        if rhat is not None:
            joint_state["accumulated_risk"] = float(joint_state.get("accumulated_risk", 0.0)) + float(rhat)

        joint_state["consecutive_reuse"] = int(joint_state.get("consecutive_reuse", 0)) + 1
        joint_state["last_action"] = "reuse_both"

    else:
        joint_state["anchor_step"] = int(current.get("step", 0))
        joint_state["accumulated_risk"] = 0.0
        joint_state["consecutive_reuse"] = 0
        joint_state["last_action"] = "full_both"

    joint_state.setdefault("log", []).append(
        {
            "step": int(current.get("step", 0)),
            "anchor_step": int(joint_state.get("anchor_step", 0)),
            "action": joint_state["last_action"],
            "dhat": None if decision.get("dhat", None) is None else float(decision["dhat"]),
            "ghat": None if decision.get("ghat", None) is None else float(decision["ghat"]),
            "rhat": None if decision.get("rhat", None) is None else float(decision["rhat"]),
            "accumulated_risk": float(joint_state.get("accumulated_risk", 0.0)),
            "reason": decision.get("reason", "ok"),
        }
    )
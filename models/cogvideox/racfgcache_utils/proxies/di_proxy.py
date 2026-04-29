import math
import numpy as np

from .base import ProxyBase, ProxyContext, ActionScores
from .registry import register_proxy


@register_proxy
class DiCacheSqExpOfflineRhoProxy(ProxyBase):
    """
    Hybrid online CFG proxy for CogVideoX single-cache two-branch setup:
      - e_u / e_c: online, from branch-local shallow probe payload
      - rho_hat   : offline, from rho_table_at[a, t]
      - combine   : CFG square expansion

    Expected layout:
      cache_dic["cfgcache"]["cond"]["_probe_payload"]
      cache_dic["cfgcache"]["uncond"]["_probe_payload"]
      cache_dic["cfgcache"]["proxy_tables"]["rho_table_at"]
    """

    name = "dicache_proxy"
    action_space = "joint"
    probe_mode = "both"

    def init_state(self, cfgcache_runtime: dict, cfg: dict) -> None:
        if cfgcache_runtime is None:
            raise ValueError(
                "cfgcache_runtime must be provided for dicache_sqexp_offline_rho"
            )

        if "proxy_tables" not in cfgcache_runtime:
            raise ValueError(
                "proxy_tables is required in cfgcache_runtime for dicache_sqexp_offline_rho"
            )

        tb = cfgcache_runtime["proxy_tables"]
        if "rho_table_at" not in tb:
            raise ValueError("rho_table_at is required in cfgcache_runtime['proxy_tables']")

        rho_table_at = np.asarray(tb["rho_table_at"], dtype=np.float32)

        cfg["proxy_tables"] = {
            "steps": tb["steps"] if "steps" in tb else None,
            "cfg_scales": tb.get("cfg_scales", None),
            "cfg_idx": tb.get("cfg_idx", None),
            "matched_cfg_scale": tb.get("matched_cfg_scale", None),
            "rho_table_at": rho_table_at,
        }

    @staticmethod
    def _extract_branch_error(payload: dict, error_choice: str):
        dx = payload.get("dicache_dx", None)
        dy = payload.get("dicache_dy", None)

        if dy is None:
            return None

        if error_choice == "delta_minus":
            if dx is None:
                return None
            return abs(float(dy) - float(dx))

        return float(dy)

    def score(self, ctx: ProxyContext) -> ActionScores:
        if ctx.cache_dic_uncond is None:
            return ActionScores(full=0.0, reuse_both=float("inf"))

        # single-cache two-branch layout
        cfg_shared = ctx.cache_dic_cond["cfgcache"]
        cfg_c = cfg_shared["cond"]
        cfg_u = cfg_shared["uncond"]

        payload_c = cfg_c.get("_probe_payload", {}) or {}
        payload_u = cfg_u.get("_probe_payload", {}) or {}

        error_choice = cfg_shared.get("proxy_error_choice", "delta_y")

        ec = self._extract_branch_error(payload_c, error_choice)
        eu = self._extract_branch_error(payload_u, error_choice)

        if ec is None or eu is None:
            return ActionScores(full=0.0, reuse_both=float("inf"))

        tb = cfg_shared["proxy_tables"]
        rho_table = tb["rho_table_at"]

        a = int(ctx.anchor_step)
        t = int(ctx.step_i)
        s = float(ctx.cfg_scale)

        a = max(0, min(a, rho_table.shape[0] - 1))
        t = max(0, min(t, rho_table.shape[1] - 1))

        rho = float(rho_table[a, t])

        term_u = (1.0 - s) ** 2 * (eu ** 2)
        term_c = s ** 2 * (ec ** 2)
        term_cross = 2.0 * s * (1.0 - s) * eu * ec * rho

        val_raw = term_u + term_c + term_cross
        val_clamped = max(val_raw, 0.0)
        ehat = float(math.sqrt(val_clamped))
        return ActionScores(
            full=0.0,
            reuse_both=ehat,
        )
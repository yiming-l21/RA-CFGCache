import math
import numpy as np

from .base import ProxyBase, ProxyContext, ActionScores
from .registry import register_proxy


@register_proxy
class DiCacheSqExpOfflineRhoProxy(ProxyBase):
    """
    Wan CFGCache Di-style proxy with offline rho.

    Wan setup:
      - one shared cache_dic
      - two CFG branches:
          cache_dic["cfgcache"]["cond"]
          cache_dic["cfgcache"]["uncond"]
      - each branch stores a shallow-probe payload:
          branch_cfg["_probe_payload"] = {
              "dicache_dx": ...,
              "dicache_dy": ...,
          }

    Branch error:
      e_c / e_u are estimated online from shallow transformer probe.

    Cross-branch correlation:
      rho_hat is loaded offline from:
          cache_dic["cfgcache"]["proxy_tables"]["rho_table_at"][anchor_step, step_i]

    CFG composition:
      guided = uncond + s * (cond - uncond)
             = (1 - s) * uncond + s * cond

      e_guided^2 =
          (1 - s)^2 e_u^2
        + s^2 e_c^2
        + 2 s (1 - s) rho e_u e_c
    """

    name = "dicache_proxy"
    action_space = "joint"

    # Not light. The controller should prepare shallow-transformer probe before score().
    # Your _cfgcache_proxy_preprobe_kind() will treat this as transformer probe.
    probe_mode = "transformer"

    def init_state(self, cfgcache_runtime: dict, cfg: dict) -> None:
        if cfgcache_runtime is None:
            raise ValueError(
                "cfgcache_runtime must be provided for dicache_proxy because offline rho_table_at is required."
            )

        if "proxy_tables" not in cfgcache_runtime:
            raise ValueError(
                "proxy_tables is required in cfgcache_runtime for dicache_proxy."
            )

        tb = cfgcache_runtime["proxy_tables"]

        if "rho_table_at" not in tb:
            raise ValueError("rho_table_at is required in cfgcache_runtime['proxy_tables'].")

        rho_table_at = np.asarray(tb["rho_table_at"], dtype=np.float32)

        if rho_table_at.ndim != 2:
            raise ValueError(
                f"rho_table_at should be a 2D array [anchor_step, step_i], got shape={rho_table_at.shape}."
            )

        cfg["proxy_tables"] = {
            "steps": tb["steps"] if "steps" in tb else None,
            "cfg_scales": tb.get("cfg_scales", None),
            "cfg_idx": tb.get("cfg_idx", None),
            "matched_cfg_scale": tb.get("matched_cfg_scale", None),
            "rho_table_at": rho_table_at,
        }

    @staticmethod
    def _extract_branch_error(payload: dict, error_choice: str):
        """
        Extract one scalar branch error from Wan shallow-probe payload.

        Expected payload:
            {
                "dicache_dx": input drift,
                "dicache_dy": shallow-output drift,
            }

        error_choice:
            - "delta_y": use dy
            - "delta_minus": use |dy - dx|
        """
        if not isinstance(payload, dict):
            return None

        dx = payload.get("dicache_dx", None)
        dy = payload.get("dicache_dy", None)

        if dy is None:
            return None

        if error_choice == "delta_minus":
            if dx is None:
                return None
            return abs(float(dy) - float(dx))

        return float(dy)

    @staticmethod
    def _get_shared_cfg(ctx: ProxyContext):
        """
        In Wan CFGCache, ctx.cache_dic_cond and ctx.cache_dic_uncond usually point
        to the same shared cache_dic.
        """
        if ctx.cache_dic_cond is None:
            raise ValueError("ctx.cache_dic_cond is None.")

        shared = ctx.cache_dic_cond["cfgcache"]

        if "cond" not in shared or "uncond" not in shared:
            raise KeyError("Expected shared cfgcache with 'cond' and 'uncond' branches.")

        return shared

    def score(self, ctx: ProxyContext) -> ActionScores:
        if ctx.cache_dic_uncond is None:
            return ActionScores(full=0.0, reuse_both=float("inf"))

        shared = self._get_shared_cfg(ctx)

        cfg_c = shared["cond"]
        cfg_u = shared["uncond"]

        payload_c = cfg_c.get("_probe_payload", {}) or {}
        payload_u = cfg_u.get("_probe_payload", {}) or {}

        error_choice = shared.get("proxy_error_choice", "delta_y")

        ec = self._extract_branch_error(payload_c, error_choice)
        eu = self._extract_branch_error(payload_u, error_choice)

        if ec is None or eu is None:
            shared["last_proxy_metrics"] = {
                "proxy_name": self.name,
                "step_i": int(ctx.step_i),
                "anchor_step": int(ctx.anchor_step),
                "cfg_scale": float(ctx.cfg_scale),
                "ec": None,
                "eu": None,
                "rho": None,
                "ehat": float("inf"),
                "reason": "missing_branch_error",
            }
            return ActionScores(full=0.0, reuse_both=float("inf"))

        tb = shared.get("proxy_tables", None)
        if tb is None or "rho_table_at" not in tb:
            shared["last_proxy_metrics"] = {
                "proxy_name": self.name,
                "step_i": int(ctx.step_i),
                "anchor_step": int(ctx.anchor_step),
                "cfg_scale": float(ctx.cfg_scale),
                "ec": float(ec),
                "eu": float(eu),
                "rho": None,
                "ehat": float("inf"),
                "reason": "missing_rho_table",
            }
            return ActionScores(full=0.0, reuse_both=float("inf"))

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
        val_clamped = max(float(val_raw), 0.0)
        ehat = float(math.sqrt(val_clamped))

        shared["last_proxy_metrics"] = {
            "proxy_name": self.name,
            "step_i": int(ctx.step_i),
            "anchor_step": int(ctx.anchor_step),
            "cfg_scale": float(ctx.cfg_scale),
            "ec": float(ec),
            "eu": float(eu),
            "rho": float(rho),
            "term_u": float(term_u),
            "term_c": float(term_c),
            "term_cross": float(term_cross),
            "val_raw": float(val_raw),
            "ehat": float(ehat),
            "reason": "ok",
        }

        return ActionScores(
            full=0.0,
            reuse_both=ehat,
        )
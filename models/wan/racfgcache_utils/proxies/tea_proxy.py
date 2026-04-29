import math
import numpy as np

from .base import ProxyBase, ProxyContext, ActionScores
from .registry import register_proxy


@register_proxy
class TeaCacheSqExpOfflineRhoProxy(ProxyBase):
    """
    Tea-style CFG proxy for CogVideoX shared-cfgcache two-branch setup.

    Design:
      - branch error e_u / e_c:
          computed from Tea-style current-vs-reference scalar drift
          with optional polynomial calibration
      - rho_hat:
          offline, from rho_table_at[a, t]
      - combine:
          CFG square expansion

    Expected shared layout:
      cache_dic["cfgcache"]["cond"]
      cache_dic["cfgcache"]["uncond"]
      cache_dic["cfgcache"]["proxy_tables"]["rho_table_at"]

    Current signal priority:
      1) _probe_modulated_inp
      2) _probe_summary["modulated_inp"]
      3) _probe_hidden_states

    Reference signal priority:
      prev mode:
        proxy_prev_modulated_inp
        proxy_prev_probe_states
      anchor mode:
        proxy_anchor_modulated_inp
        proxy_anchor_probe_states
    """

    name = "teacache_proxy"
    action_space = "joint"
    probe_mode = "light"

    def init_state(self, cfgcache_runtime: dict, cfg: dict) -> None:
        if cfgcache_runtime is None:
            raise ValueError(
                "cfgcache_runtime must be provided for teacache_sqexp_offline_rho"
            )

        if "proxy_tables" not in cfgcache_runtime:
            raise ValueError(
                "proxy_tables is required in cfgcache_runtime for teacache_sqexp_offline_rho"
            )

        tb = cfgcache_runtime["proxy_tables"]
        if "rho_table_at" not in tb:
            raise ValueError("rho_table_at is required in cfgcache_runtime['proxy_tables']")

        rho_table_at = np.asarray(tb["rho_table_at"], dtype=np.float32)

        out = {
            "steps": tb["steps"] if "steps" in tb else None,
            "cfg_scales": tb.get("cfg_scales", None),
            "cfg_idx": tb.get("cfg_idx", None),
            "matched_cfg_scale": tb.get("matched_cfg_scale", None),
            "rho_table_at": rho_table_at,
        }

        # 同时兼容两种命名
        coeff = None
        if "teacache_poly_coeffs" in tb and tb["teacache_poly_coeffs"] is not None:
            coeff = np.asarray(tb["teacache_poly_coeffs"], dtype=np.float32)
        elif "teacache_coefficients" in tb and tb["teacache_coefficients"] is not None:
            coeff = np.asarray(tb["teacache_coefficients"], dtype=np.float32)

        if coeff is not None:
            out["teacache_poly_coeffs"] = coeff
            out["teacache_coefficients"] = coeff

        cfg["proxy_tables"] = out

    @staticmethod
    def _rel_l1(curr, ref, eps: float = 1e-6):
        if curr is None or ref is None:
            return None

        denom = ref.abs().mean()
        if float(denom) == 0.0:
            return 0.0

        val = (curr - ref).abs().mean() / (denom + eps)
        return float(val.detach().cpu())

    @staticmethod
    def _poly_calibrate(x: float, coeff):
        if x is None:
            return None
        if coeff is None:
            return float(x)

        coeff = np.asarray(coeff, dtype=np.float64).reshape(-1)
        if coeff.size == 0:
            return float(x)

        y = 0.0
        for c in coeff:
            y = y * float(x) + float(c)
        return float(y)

    @staticmethod
    def _get_shared_cfg(ctx: ProxyContext):
        shared = ctx.cache_dic_cond["cfgcache"]
        if "cond" not in shared or "uncond" not in shared:
            raise KeyError("Expected shared cfgcache with 'cond' and 'uncond' branches.")
        return shared

    @staticmethod
    def _pick_current_signal(branch_cfg: dict):
        cur = branch_cfg.get("_probe_modulated_inp", None)
        if cur is not None:
            return cur

        probe_summary = branch_cfg.get("_probe_summary", None)
        if isinstance(probe_summary, dict):
            cur = probe_summary.get("modulated_inp", None)
            if cur is not None:
                return cur

        cur = branch_cfg.get("_probe_hidden_states", None)
        if cur is not None:
            return cur

        return None

    @staticmethod
    def _pick_reference_signal(branch_cfg: dict, ref_mode: str):
        if ref_mode == "anchor":
            ref = branch_cfg.get("proxy_anchor_modulated_inp", None)
            if ref is not None:
                return ref

            ref = branch_cfg.get("proxy_anchor_probe_states", None)
            if ref is not None:
                return ref
        else:
            ref = branch_cfg.get("proxy_prev_modulated_inp", None)
            if ref is not None:
                return ref

            ref = branch_cfg.get("proxy_prev_probe_states", None)
            if ref is not None:
                return ref

        return None

    @staticmethod
    def _get_poly_coeff(branch_cfg: dict, shared_cfg: dict):
        # branch-local 优先
        coeff = branch_cfg.get("teacache_poly_coeffs", None)
        if coeff is not None:
            return coeff

        coeff = branch_cfg.get("teacache_coefficients", None)
        if coeff is not None:
            return coeff

        # shared cfg 其次
        coeff = shared_cfg.get("teacache_poly_coeffs", None)
        if coeff is not None:
            return coeff

        coeff = shared_cfg.get("teacache_coefficients", None)
        if coeff is not None:
            return coeff

        # proxy_tables 最后
        tb = shared_cfg.get("proxy_tables", {})
        coeff = tb.get("teacache_poly_coeffs", None)
        if coeff is not None:
            return coeff

        coeff = tb.get("teacache_coefficients", None)
        if coeff is not None:
            return coeff

        return None

    @classmethod
    def _extract_branch_error(cls, branch_cfg: dict, shared_cfg: dict):
        ref_mode = branch_cfg.get(
            "proxy_ref_mode",
            shared_cfg.get("proxy_ref_mode", "anchor"),
        )
        eps = float(shared_cfg.get("proxy_eps", 1e-6))

        curr = cls._pick_current_signal(branch_cfg)
        ref = cls._pick_reference_signal(branch_cfg, ref_mode)

        raw = cls._rel_l1(curr, ref, eps=eps)

        coeff = cls._get_poly_coeff(branch_cfg, shared_cfg)
        cal = cls._poly_calibrate(raw, coeff)

        if cal is None:
            return None, raw

        cal = max(float(cal), 0.0)
        return cal, raw

    def score(self, ctx: ProxyContext) -> ActionScores:
        if ctx.cache_dic_uncond is None:
            return ActionScores(full=0.0, reuse_both=float("inf"))

        shared = self._get_shared_cfg(ctx)
        cfg_c = shared["cond"]
        cfg_u = shared["uncond"]

        ec, ec_raw = self._extract_branch_error(cfg_c, shared)
        eu, eu_raw = self._extract_branch_error(cfg_u, shared)

        if ec is None or eu is None:
            return ActionScores(full=0.0, reuse_both=float("inf"))

        tb = shared.get("proxy_tables", None)
        if tb is None or "rho_table_at" not in tb:
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
        val_clamped = max(val_raw, 0.0)
        ehat = float(math.sqrt(val_clamped))
        shared["last_proxy_metrics"] = {
            "proxy_name": self.name,
            "step_i": int(ctx.step_i),
            "anchor_step": int(ctx.anchor_step),
            "cfg_scale": float(ctx.cfg_scale),
            "ec_raw": None if ec_raw is None else float(ec_raw),
            "eu_raw": None if eu_raw is None else float(eu_raw),
            "ec": float(ec),
            "eu": float(eu),
            "rho": float(rho),
            "ehat": float(ehat),
        }

        return ActionScores(
            full=0.0,
            reuse_both=ehat,
        )
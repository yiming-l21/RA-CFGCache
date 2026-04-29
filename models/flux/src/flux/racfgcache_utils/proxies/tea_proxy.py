import math
import numpy as np
import torch

from .base import ProxyBase, ProxyContext, ActionScores
from .registry import register_proxy


@register_proxy
class TeaCacheSqExpOfflineRhoProxy(ProxyBase):
    """
    TeaCache-style base proxy + CFG-aware guided composition.

    branch proxy:
        e_u, e_c = rel_l1(current_modulated_input, previous_modulated_input)
        optional polynomial rescaling on each branch

    CFG composition:
        d_g^2 = (1-s)^2 e_u^2 + s^2 e_c^2 + 2 s (1-s) rho * e_u * e_c

    Notes:
      - This proxy outputs local guided drift estimate \hat d_g(t),
        NOT the final propagation-revised risk.
      - Propagation gain \hat g_t should still be applied outside
        in _decide_cfgcache_joint(), consistent with your current code.
    """
    name = "teacache_proxy"
    action_space = "joint"
    probe_mode = "teacache_modulated"   # use built-in probe path in Flux._cfgcache_prepare_probe

    def init_state(self, cfg: dict) -> None:
        table_path = cfg.get("proxy_tables_path", None)
        if table_path is None:
            raise ValueError("proxy_tables_path is required for teacache_sqexp_offline_rho")

        data = np.load(table_path)
        if "rho_table_at" not in data:
            raise ValueError("rho_table_at is required in proxy_tables_path for teacache_sqexp_offline_rho")

        cfg["proxy_tables"] = {
            "steps": data["steps"] if "steps" in data else None,
            "rho_table_at": data["rho_table_at"],
        }

        cfg["probe_mode"] = "teacache_modulated"

        cfg.setdefault("anchor_step", 0)
        cfg.setdefault("proxy_eps", 1e-6)

        # TeaCache branch proxy options
        cfg.setdefault("teacache_poly_coeffs", None)

        # adjacent-step states
        cfg.setdefault("proxy_prev_input", None)
        cfg.setdefault("proxy_prev_probe_states", None)

    @staticmethod
    def _rel_l1(curr: torch.Tensor, prev: torch.Tensor, eps: float = 1e-6) -> float:
        denom = prev.abs().mean()
        if float(denom) == 0.0:
            return 0.0
        val = (curr - prev).abs().mean() / (denom + eps)
        return float(val.detach().cpu())

    @staticmethod
    def _apply_poly(x: float, coeffs) -> float:
        """
        coeffs follow np.polyfit / np.poly1d convention:
        highest degree -> constant term
        """
        if coeffs is None:
            return float(x)

        x = float(x)
        y = 0.0
        for c in coeffs:
            y = y * x + float(c)
        return max(float(y), 0.0)

    @staticmethod
    def _lookup_rho(rho_table, a: int, t: int) -> float:
        """
        Support both:
          - rho_table[t]
          - rho_table[a, t]
        """
        if rho_table.ndim == 1:
            t = max(0, min(t, rho_table.shape[0] - 1))
            return float(rho_table[t])

        if rho_table.ndim == 2:
            a = max(0, min(a, rho_table.shape[0] - 1))
            t = max(0, min(t, rho_table.shape[1] - 1))
            return float(rho_table[a, t])

        raise ValueError(f"Unsupported rho_table_at ndim={rho_table.ndim}")

    def _extract_branch_error(self, cfg: dict):
        """
        TeaCache branch error:
          rel_l1(curr_modulated_input, prev_modulated_input)
          + optional polynomial mapping
        """
        curr = cfg.get("_probe_modulated_inp", None)
        prev = cfg.get("proxy_prev_probe_states", None)

        if curr is None or prev is None:
            return None

        eps = float(cfg.get("proxy_eps", 1e-6))
        coeffs = cfg.get("teacache_poly_coeffs", None)

        err = self._rel_l1(curr.detach(), prev, eps=eps)
        err = self._apply_poly(err, coeffs)
        return float(err)

    def score(self, ctx: ProxyContext) -> ActionScores:
        if ctx.cache_dic_uncond is None:
            return ActionScores(full=0.0, reuse_both=float("inf"))

        cfg_c = ctx.cache_dic_cond["cfgcache"]
        cfg_u = ctx.cache_dic_uncond["cfgcache"]

        ec = self._extract_branch_error(cfg_c)   # conditional branch proxy error
        eu = self._extract_branch_error(cfg_u)   # unconditional branch proxy error

        # current-step probe not ready on either branch -> force full refresh
        if ec is None or eu is None:
            return ActionScores(full=0.0, reuse_both=float("inf"))

        tb = cfg_c["proxy_tables"]
        rho_table = tb["rho_table_at"]

        a = int(cfg_c.get("anchor_step", 0))
        t = int(ctx.step_i)
        s = float(ctx.cfg_scale)

        rho = self._lookup_rho(rho_table, a=a, t=t)

        term_u = (1.0 - s) ** 2 * (eu ** 2)
        term_c = s ** 2 * (ec ** 2)
        term_cross = 2.0 * s * (1.0 - s) * eu * ec * rho

        val_raw = term_u + term_c + term_cross
        val_clamped = max(val_raw, 0.0)
        ehat = float(math.sqrt(val_clamped))

        # optional debug
        # print(
        #     "[TeaCacheSqExpOfflineRho] "
        #     f"step={t}/{ctx.num_steps-1} "
        #     f"anchor={a} "
        #     f"cfg={s:.3f} "
        #     f"eu={eu:.6f} "
        #     f"ec={ec:.6f} "
        #     f"rho={rho:.6f} "
        #     f"term_u={term_u:.6f} "
        #     f"term_c={term_c:.6f} "
        #     f"term_cross={term_cross:.6f} "
        #     f"ehat={ehat:.6f}"
        # )

        return ActionScores(
            full=0.0,
            reuse_both=ehat,
        )
import math
import numpy as np

from .base import ProxyBase, ProxyContext, ActionScores
from .registry import register_proxy


@register_proxy
class MagCacheSqExpOfflineRhoProxy(ProxyBase):
    """
    Offline MagCache-style CFG proxy:
      - e_u / e_c: offline branch-wise skip error from calibrated gamma curves
      - rho_hat   : offline, from rho_table_at[a, t]
      - combine   : CFG square expansion

    Required proxy_tables entries:
      - rho_table_at                         : [T, T]
      - one of:
          mag_gamma_cond / mag_gamma_uncond  : [T]
        or
          mag_gamma_shared                   : [T]

    Supported aliases:
      - mag_ratios_cond / mag_ratios_uncond / mag_ratios
    """

    name = "magcache_proxy"
    action_space = "joint"
    probe_mode = "none"
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
        gamma_cond = np.array([
            1.00000, 1.15589, 1.03062, 1.06443, 1.05367, 1.03530, 1.03932, 1.02461, 1.02777,
            1.01864, 1.02197, 1.01830, 1.01566, 1.01181, 1.01239, 1.00975, 1.01173, 1.00840,
            1.00098, 1.00147, 1.00974, 1.00173, 1.01344, 1.00718, 0.99618, 1.00715, 1.00924,
            1.00119, 1.00006, 1.00295, 0.99880, 1.01076, 0.99010, 1.00541, 1.00079, 0.99685,
            0.99504, 0.99612, 0.98827, 0.99784, 0.99364, 0.99299, 0.98553, 0.98508, 0.98461,
            0.96980, 0.96539, 0.95062, 0.92728, 0.92891,
        ], dtype=np.float64)
        gamma_uncond = np.array([
            1.00000, 1.15589, 1.03062, 1.06443, 1.05367, 1.03530, 1.03932, 1.02461, 1.02777,
            1.01864, 1.02197, 1.01830, 1.01566, 1.01181, 1.01239, 1.00975, 1.01173, 1.00840,
            1.00098, 1.00147, 1.00974, 1.00173, 1.01344, 1.00718, 0.99618, 1.00715, 1.00924,
            1.00119, 1.00006, 1.00295, 0.99880, 1.01076, 0.99010, 1.00541, 1.00079, 0.99685,
            0.99504, 0.99612, 0.98827, 0.99784, 0.99364, 0.99299, 0.98553, 0.98508, 0.98461,
            0.96980, 0.96539, 0.95062, 0.92728, 0.92891,
        ], dtype=np.float64)
        skip_err_table_cond = self._build_skip_error_table(gamma_cond)
        skip_err_table_uncond = self._build_skip_error_table(gamma_uncond)
        cfg["proxy_tables"] = {
            "steps": tb["steps"] if "steps" in tb else None,
            "cfg_scales": tb.get("cfg_scales", None),
            "cfg_idx": tb.get("cfg_idx", None),
            "matched_cfg_scale": tb.get("matched_cfg_scale", None),
            "rho_table_at": rho_table_at,
            "mag_gamma_cond": gamma_cond,
            "mag_gamma_uncond": gamma_uncond,
            "skip_err_table_cond": skip_err_table_cond,
            "skip_err_table_uncond": skip_err_table_uncond,
        }
    @staticmethod
    def _pick_gamma(tb: dict, primary_keys=()):
        for k in primary_keys:
            if k in tb and tb[k] is not None:
                arr = np.asarray(tb[k], dtype=np.float32).reshape(-1)
                return arr
        return None

    @staticmethod
    def _normalize_gamma(gamma: np.ndarray) -> np.ndarray:
        gamma = np.asarray(gamma, dtype=np.float64).reshape(-1)

        if gamma.size == 0:
            raise ValueError("gamma curve must be non-empty")

        # step-0 has no previous step; force gamma[0] = 1
        gamma = gamma.copy()
        gamma[0] = 1.0

        # numerical guard
        gamma = np.clip(gamma, 1e-8, 1e8)
        return gamma

    @classmethod
    def _build_skip_error_table(cls, gamma: np.ndarray) -> np.ndarray:
        """
        Build table[a, t] = |1 - prod_{i=a+1..t} gamma_i|
        This matches the MagCache accumulated-ratio style more robustly than max(0, 1-prod).
        """
        gamma = cls._normalize_gamma(gamma)
        T = gamma.shape[0]

        prefix = np.ones(T, dtype=np.float64)
        for i in range(1, T):
            prefix[i] = prefix[i - 1] * gamma[i]

        table = np.zeros((T, T), dtype=np.float32)
        for a in range(T):
            for t in range(a + 1, T):
                prod = prefix[t] / max(prefix[a], 1e-12)
                table[a, t] = float(abs(1.0 - prod))
        return table

    def score(self, ctx: ProxyContext) -> ActionScores:
        if ctx.cache_dic_uncond is None:
            return ActionScores(full=0.0, reuse_both=float("inf"))

        cfg_c = ctx.cache_dic_cond["cfgcache"]

        tb = cfg_c["proxy_tables"]
        rho_table = tb["rho_table_at"]
        err_table_c = tb["skip_err_table_cond"]
        err_table_u = tb["skip_err_table_uncond"]

        a = int(ctx.anchor_step)
        t = int(ctx.step_i)
        s = float(ctx.cfg_scale)

        # safe clamp
        a = max(0, min(a, rho_table.shape[0] - 1))
        t = max(0, min(t, rho_table.shape[1] - 1))

        # no reuse on anchor/current invalid order
        if t <= a:
            return ActionScores(full=0.0, reuse_both=float("inf"))

        ec = float(err_table_c[a, t])
        eu = float(err_table_u[a, t])
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
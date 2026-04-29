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
        official_13b_mag_ratios = np.array(
            [1.0] * 2 + [
                1.0124, 1.02213,
                1.00166, 1.0041,
                0.99791, 1.00061,
                0.99682, 0.99762,
                0.99634, 0.99685,
                0.99567, 0.99586,
                0.99416, 0.99422,
                0.99578, 0.99575,
                0.9957, 0.99563,
                0.99511, 0.99506,
                0.99535, 0.99531,
                0.99552, 0.99549,
                0.99541, 0.99539,
                0.9954, 0.99536,
                0.99489, 0.99485,
                0.99518, 0.99514,
                0.99484, 0.99478,
                0.99481, 0.99479,
                0.99415, 0.99413,
                0.99419, 0.99416,
                0.99396, 0.99393,
                0.99388, 0.99386,
                0.99349, 0.99349,
                0.99309, 0.99304,
                0.9927, 0.9927,
                0.99228, 0.99226,
                0.99171, 0.9917,
                0.99137, 0.99135,
                0.99068, 0.99063,
                0.99005, 0.99003,
                0.98944, 0.98942,
                0.98849, 0.98849,
                0.98758, 0.98757,
                0.98644, 0.98643,
                0.98504, 0.98503,
                0.9836, 0.98359,
                0.98202, 0.98201,
                0.97977, 0.97978,
                0.97717, 0.97718,
                0.9741, 0.97411,
                0.97003, 0.97002,
                0.96538, 0.96541,
                0.9593, 0.95933,
                0.95086, 0.95089,
                0.94013, 0.94019,
                0.92402, 0.92414,
                0.90241, 0.9026,
                0.86821, 0.86868,
                0.81838, 0.81939,
            ],
            dtype=np.float64,
        )

        ratios_in = np.asarray(
            model_kwargs.get("mag_ratios", official_13b_mag_ratios),
            dtype=np.float64,
        )
        gamma_cond = ratios_in[0::2]
        gamma_uncond = ratios_in[1::2]
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
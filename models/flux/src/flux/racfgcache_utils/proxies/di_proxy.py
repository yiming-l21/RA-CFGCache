import math
import numpy as np
import torch
import torch.nn.functional as F

from .base import ProxyBase, ProxyContext, ActionScores
from .registry import register_proxy


@register_proxy
class DiCacheSqExpOfflineRhoProxy(ProxyBase):
    """
    Hybrid online CFG proxy:
      - e_u / e_c: online, from DiCache-style shallow probe
      - rho_hat   : offline, from rho_table_at[a, t]
      - combine   : CFG square expansion

    e_branch:
      - delta_y       : rel_l1(current_probe, previous_probe)
      - delta_minus   : |delta_y - delta_x|
    """
    name = "dicache_proxy"
    action_space = "joint"
    probe_mode = "none"   # use custom prepare_forward_probe()

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
    def _rel_l1(curr: torch.Tensor, prev: torch.Tensor, eps: float = 1e-6) -> float:
        denom = prev.abs().mean()
        if float(denom) == 0.0:
            return 0.0
        val = (curr - prev).abs().mean() / (denom + eps)
        return float(val.detach().cpu())

    def prepare_forward_probe(
        self,
        *,
        model,
        img,
        txt,
        vec,
        pe,
        cache_dic,
        current,
    ):
        """
        Run DiCache-style shallow probe before cal_type / joint decision.

        Returns a LIGHT payload:
          - dicache_dx
          - dicache_dy
          - probe_depth

        Large tensors are NOT stored inside payload anymore.
        They are stored separately by _cfgcache_prepare_probe().
        """
        cc = cache_dic["cfgcache"]

        prev_inp = cc.get("proxy_prev_input", None)
        prev_probe = cc.get("proxy_prev_probe_states", None)

        depth = min(int(cc.get("probe_depth", 2)), len(model.double_blocks))

        test_img = img.clone()
        test_txt = txt.clone()

        old_type = current.get("type", None)
        old_layer = current.get("layer", None)

        for probe_i in range(depth):
            current["type"] = "full"
            current["layer"] = probe_i
            test_img, test_txt = model.double_blocks[probe_i](
                img=test_img,
                txt=test_txt,
                vec=vec,
                pe=pe,
                cache_dic=cache_dic,
                current=current,
            )

        if old_type is not None:
            current["type"] = old_type
        else:
            current.pop("type", None)

        if old_layer is not None:
            current["layer"] = old_layer
        else:
            current.pop("layer", None)

        eps = float(cc.get("proxy_eps", 1e-6))

        dx = None
        dy = None

        if prev_inp is not None:
            dx = self._rel_l1(img.detach(), prev_inp, eps)

        if prev_probe is not None:
            dy = self._rel_l1(test_img.detach(), prev_probe, eps)

        # store heavy tensors separately for resume
        cc["_probe_img"] = test_img.detach()
        cc["_probe_txt"] = test_txt.detach()
        cc["_probe_depth"] = depth

        payload = {
            "probe_depth": depth,
            "dicache_dx": dx,
            "dicache_dy": dy,
        }
        return payload

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

        # default: delta_y
        return float(dy)

    def score(self, ctx: ProxyContext) -> ActionScores:
        if ctx.cache_dic_uncond is None:
            return ActionScores(full=0.0, reuse_both=float("inf"))

        cfg_c = ctx.cache_dic_cond["cfgcache"]
        cfg_u = ctx.cache_dic_uncond["cfgcache"]

        payload_c = cfg_c.get("_probe_payload", {}) or {}
        payload_u = cfg_u.get("_probe_payload", {}) or {}

        error_choice = cfg_c.get("proxy_error_choice", "delta_y")

        ec = self._extract_branch_error(payload_c, error_choice)
        eu = self._extract_branch_error(payload_u, error_choice)

        # current-step probe not ready on either branch -> force full
        if ec is None or eu is None:
            return ActionScores(full=0.0, reuse_both=float("inf"))

        tb = cfg_c["proxy_tables"]
        rho_table = tb["rho_table_at"]

        # use shared anchor directly from ctx, no need to mirror-write into cfgcache
        a = int(ctx.anchor_step)
        t = int(ctx.step_i)
        s = float(ctx.cfg_scale)

        # safe clamp
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
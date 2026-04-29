from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class ProxyContext:
    step_i: int
    num_steps: int
    cfg_scale: float
    anchor_step: int

    img: Any
    current: dict
    cache_dic_cond: dict
    cache_dic_uncond: Optional[dict] = None

    probe_cond: Any = None
    probe_uncond: Any = None
    probe_joint: Any = None


@dataclass
class ActionScores:
    full: float = 0.0
    reuse_both: Optional[float] = None


class ProxyBase:
    name = "base"
    action_space = "joint"
    probe_mode = "none"

    def init_state(self, cfgcache_runtime: dict, cfg: dict) -> None:
        pass

    def prepare_runtime_probe(self, model, ctx: ProxyContext):
        return None

    def score(self, ctx: ProxyContext) -> ActionScores:
        raise NotImplementedError

    def on_full_refresh(self, cfg: dict, step_i: int):
        cfg["anchor_step"] = int(step_i)
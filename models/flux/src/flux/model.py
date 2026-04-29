from dataclasses import dataclass

import torch
from torch import Tensor, nn

from flux.modules.layers import (
    DoubleStreamBlock,
    EmbedND,
    LastLayer,
    MLPEmbedder,
    SingleStreamBlock,
    timestep_embedding,
)
from flux.modules.lora import LinearLora, replace_linear_with_lora
from flux.modules.cache_functions import cal_type


@dataclass
class FluxParams:
    in_channels: int
    out_channels: int
    vec_in_dim: int
    context_in_dim: int
    hidden_size: int
    mlp_ratio: float
    num_heads: int
    depth: int
    depth_single_blocks: int
    axes_dim: list[int]
    theta: int
    qkv_bias: bool
    guidance_embed: bool


class Flux(nn.Module):
    """
    Transformer model for flow matching on sequences.
    """

    def __init__(self, params: FluxParams):
        super().__init__()

        self.params = params
        self.in_channels = params.in_channels
        self.out_channels = params.out_channels
        if params.hidden_size % params.num_heads != 0:
            raise ValueError(
                f"Hidden size {params.hidden_size} must be divisible by num_heads {params.num_heads}"
            )
        pe_dim = params.hidden_size // params.num_heads
        if sum(params.axes_dim) != pe_dim:
            raise ValueError(f"Got {params.axes_dim} but expected positional dim {pe_dim}")
        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads
        self.pe_embedder = EmbedND(dim=pe_dim, theta=params.theta, axes_dim=params.axes_dim)
        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)
        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size) if params.guidance_embed else nn.Identity()
        )
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.double_blocks = nn.ModuleList(
            [
                DoubleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                    qkv_bias=params.qkv_bias,
                )
                for _ in range(params.depth)
            ]
        )

        self.single_blocks = nn.ModuleList(
            [
                SingleStreamBlock(self.hidden_size, self.num_heads, mlp_ratio=params.mlp_ratio)
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(self.hidden_size, 1, self.out_channels)

    @staticmethod
    def _summarize_probe_feat(x: Tensor) -> Tensor:
        # x: [B, N, D]
        return torch.stack(
            [
                x.abs().mean(dim=(1, 2)),
                x.pow(2).mean(dim=(1, 2)).sqrt(),
                x.mean(dim=(1, 2)),
                x.std(dim=(1, 2)),
            ],
            dim=-1,
        )  # [B, 4]

    @staticmethod
    def _cfgcache_clear_probe_state(cc: dict) -> None:
        # shrink clear set to only what CFGCache resume / scoring actually needs
        for k in [
            "_probe_payload",
            "_probe_img",
            "_probe_txt",
            "_probe_depth",
            "_probe_modulated_inp",
        ]:
            cc.pop(k, None)

    def _cfgcache_encode_inputs(
        self,
        *,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
    ) -> dict:
        """
        Shared encoder for:
          - preprobe path in denoise_cache_cfg()
          - optional reuse inside forward()
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        img_h = self.img_in(img)
        vec_h = self.time_in(timestep_embedding(timesteps, 256))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec_h = vec_h + self.guidance_in(timestep_embedding(guidance, 256))
        vec_h = vec_h + self.vector_in(y)
        txt_h = self.txt_in(txt)

        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        return {
            "img": img_h,
            "txt": txt_h,
            "vec": vec_h,
            "pe": pe,
        }

    def _cfgcache_prepare_probe(
        self,
        *,
        img: Tensor,
        txt: Tensor,
        vec: Tensor,
        pe: Tensor,
        cache_dic: dict,
        current: dict,
    ) -> None:
        cc = cache_dic.get("cfgcache", None)
        if cc is None:
            return

        self._cfgcache_clear_probe_state(cc)

        proxy = cc.get("_proxy_obj", None)
        probe_mode = getattr(proxy, "probe_mode", cc.get("probe_mode", "none"))

        # --------------------------------------------------
        # 1) custom proxy hook has highest priority
        # --------------------------------------------------
        if proxy is not None and hasattr(proxy, "prepare_forward_probe"):
            payload = proxy.prepare_forward_probe(
                model=self,
                img=img,
                txt=txt,
                vec=vec,
                pe=pe,
                cache_dic=cache_dic,
                current=current,
            )
            if isinstance(payload, dict):
                cc["_probe_payload"] = payload
                if "probe_depth" in payload:
                    cc["_probe_depth"] = int(payload["probe_depth"])
            return

        # --------------------------------------------------
        # 2) built-in probe modes
        # --------------------------------------------------
        if probe_mode == "none":
            return

        # TeaCache-like: block-0 modulated input
        if probe_mode == "teacache_modulated":
            blk0 = self.double_blocks[0]
            with torch.no_grad():
                img_mod1, _ = blk0.img_mod(vec)
                img_normed = blk0.img_norm1(img)
                img_modulated = (1 + img_mod1.scale) * img_normed + img_mod1.shift

            cc["_probe_modulated_inp"] = img_modulated.detach()
            cc["_probe_payload"] = {
                "modulated_inp": True,
            }
            return

        # DiCache-like: shallow probe on first m double blocks
        if probe_mode == "shallow_blocks":
            m = int(cc.get("probe_depth", 4))
            depth = min(m, len(self.double_blocks))

            test_img = img.clone()
            test_txt = txt.clone()

            old_layer = current.get("layer", None)

            for probe_i in range(depth):
                old_type = current.get("type", None)
                current["type"] = "full"
                current["layer"] = probe_i
                test_img, test_txt = self.double_blocks[probe_i](
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

            cc["_probe_img"] = test_img.detach()
            cc["_probe_txt"] = test_txt.detach()
            cc["_probe_depth"] = depth
            cc["_probe_payload"] = {
                "probe_depth": depth,
            }
            return

        raise ValueError(f"Unsupported CFGCache probe_mode: {probe_mode}")

    def forward(
        self,
        img: Tensor,
        img_ids: Tensor,
        txt: Tensor,
        txt_ids: Tensor,
        timesteps: Tensor,
        y: Tensor,
        guidance: Tensor | None = None,
        *args,
        **kwargs,
    ) -> Tensor:
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        cache_dic = kwargs.get("cache_dic", None)
        current = kwargs.get("current", None)
        preencoded = kwargs.get("preencoded", None)

        is_cfgcache = cache_dic is not None and cache_dic.get("mode") == "CFGCache"

        # running on sequences img
        if preencoded is not None:
            img = preencoded["img"]
            txt = preencoded["txt"]
            vec = preencoded["vec"]
            pe = preencoded["pe"]
            img_base = img
            img0 = img.detach() if is_cfgcache else None
        else:
            img = self.img_in(img)
            img_base = img
            img0 = img.detach() if is_cfgcache else None

            vec = self.time_in(timestep_embedding(timesteps, 256))
            if self.params.guidance_embed:
                if guidance is None:
                    raise ValueError("Didn't get guidance strength for guidance distilled model.")
                vec = vec + self.guidance_in(timestep_embedding(guidance, 256))
            vec = vec + self.vector_in(y)
            txt = self.txt_in(txt)

            ids = torch.cat((txt_ids, img_ids), dim=1)
            pe = self.pe_embedder(ids)

        # -------------------------
        # TeaCache: compute gating signal BEFORE cal_type (paper-faithful)
        # -------------------------
        if cache_dic is not None and cache_dic.get("mode") == "TeaCache":
            tc = cache_dic.get("teacache", None)
            if tc is not None and cache_dic.get("teacache_enable", True):
                blk0 = self.double_blocks[0]
                with torch.no_grad():
                    img_mod1, _ = blk0.img_mod(vec)
                    img_normed = blk0.img_norm1(img)
                    img_modulated = (1 + img_mod1.scale) * img_normed + img_mod1.shift
                current["teacache_modulated_inp"] = img_modulated.detach()

        # -------------------------
        # DiCache: compute probe BEFORE cal_type
        # -------------------------
        if cache_dic is not None and cache_dic.get("mode") == "DiCache":
            dc = cache_dic.get("dicache", None)
            if dc is not None and cache_dic.get("dicache_enable", True):
                if current is not None and int(current.get("step", 0)) == 0:
                    dc["cnt"] = 0
                    dc["accumulated_rel_l1_distance"] = 0.0
                    dc["resume_flag"] = False
                    dc["previous_input"] = None
                    dc["previous_probe_states"] = None
                    dc["previous_residual"] = None
                    dc["previous_probe_residual"] = None
                    dc["residual_window"] = []
                    dc["probe_residual_window"] = []

        # -------------------------
        # DiCache: compute probe BEFORE cal_type (official-style OPP)
        # -------------------------
        if cache_dic is not None and cache_dic.get("mode") == "DiCache":
            dc = cache_dic.get("dicache", None)
            if dc is not None and cache_dic.get("dicache_enable", True):

                cnt = int(dc.get("cnt", 0))
                T = int(dc.get("num_steps", current.get("num_steps", 0) or 0))
                ret_ratio = float(dc.get("ret_ratio", 0.2))
                need_probe = not ((cnt <= int(ret_ratio * T)) or (T > 0 and cnt == T - 1))

                current["dicache_delta_x"] = None
                current["dicache_delta_y"] = None

                dc["_probe_img"] = None
                dc["_probe_txt"] = None
                dc["base_img"] = None

                if need_probe:
                    prev_inp = dc.get("previous_input", None)
                    prev_probe = dc.get("previous_probe_states", None)

                    m = int(dc.get("probe_depth", 1))
                    test_img = img.clone()
                    test_txt = txt.clone()

                    for probe_i in range(min(m, len(self.double_blocks))):
                        old_type = current.get("type", None)
                        current["type"] = "full"
                        current["layer"] = probe_i
                        test_img, test_txt = self.double_blocks[probe_i](
                            img=test_img, txt=test_txt, vec=vec, pe=pe, cache_dic=cache_dic, current=current
                        )
                        if old_type is not None:
                            current["type"] = old_type

                    if (prev_inp is not None) and (prev_probe is not None):
                        eps = 1e-6
                        dx = (
                            test_img.new_tensor(0.0)
                            if prev_inp.abs().mean() == 0
                            else (img - prev_inp).abs().mean() / (prev_inp.abs().mean() + eps)
                        )
                        dy = (
                            test_img.new_tensor(0.0)
                            if prev_probe.abs().mean() == 0
                            else (test_img - prev_probe).abs().mean() / (prev_probe.abs().mean() + eps)
                        )
                        current["dicache_delta_x"] = float(dx.detach().cpu())
                        current["dicache_delta_y"] = float(dy.detach().cpu())

                    dc["_probe_img"] = test_img.detach()
                    dc["_probe_txt"] = test_txt.detach()
                    dc["base_img"] = img_base.detach()

        if is_cfgcache:
            cc = cache_dic["cfgcache"]
            probe_prepared = bool(current.get("cfgcache_probe_prepared", False))
            if (not probe_prepared) and cc.get("_probe_payload", None) is None:
                self._cfgcache_prepare_probe(
                    img=img,
                    txt=txt,
                    vec=vec,
                    pe=pe,
                    cache_dic=cache_dic,
                    current=current,
                )

        if cache_dic is not None:
            cal_type(cache_dic=cache_dic, current=current)

        # -------------------------
        # TeaCache: if skip, reuse residual and return
        # -------------------------
        if cache_dic is not None and cache_dic.get("mode") == "TeaCache":
            tc = cache_dic["teacache"]

            if current.get("type") == "TeaCacheSkip" and tc.get("previous_residual", None) is not None:
                img = img + tc["previous_residual"]
                img = self.final_layer(img, vec)
                return img
            else:
                tc["base_img"] = img.detach()

        if cache_dic is not None and cache_dic.get("mode") == "MagCache":
            mc = cache_dic["magcache"]

            if current is not None and int(current.get("step", 0)) == 0:
                mc["cnt"] = 0
                mc["accumulated_ratio"] = 1.0
                mc["accumulated_err"] = 0.0
                mc["accumulated_steps"] = 0
                mc["previous_residual"] = None

            if current.get("type") == "MagCacheSkip" and mc.get("previous_residual", None) is not None:
                img = img + mc["previous_residual"]
                img = self.final_layer(img, vec)
                return img
            else:
                mc["base_img"] = img

        # -------------------------
        # CFGCache: residual reuse at hidden level
        # -------------------------
        if is_cfgcache:
            cc = cache_dic["cfgcache"]

            if current.get("type") == "CFGCacheSkip":
                if cc.get("previous_residual", None) is not None:
                    cc["proxy_prev_input"] = img0
                    if cc.get("_probe_modulated_inp", None) is not None:
                        cc["proxy_prev_probe_states"] = cc["_probe_modulated_inp"].detach()
                    elif cc.get("_probe_img", None) is not None:
                        cc["proxy_prev_probe_states"] = cc["_probe_img"].detach()

                    img = img + cc["previous_residual"]
                    out = self.final_layer(img, vec)
                    self._cfgcache_clear_probe_state(cc)
                    return out
                else:
                    current["type"] = "full"

            cc["base_img"] = img0

        # -------------------------
        # DiCache: if skip, reuse residual with DCTA and return
        # -------------------------
        if cache_dic is not None and cache_dic.get("mode") == "DiCache":
            dc = cache_dic["dicache"]

            if current.get("type") == "DiCacheSkip" and dc.get("previous_residual", None) is not None:
                ori_img = img_base.detach()

                if (
                    len(dc.get("residual_window", [])) >= 2
                    and len(dc.get("probe_residual_window", [])) >= 2
                    and dc.get("_probe_img", None) is not None
                ):
                    current_residual_indicator = dc["_probe_img"] - ori_img
                    denom = (dc["probe_residual_window"][-1] - dc["probe_residual_window"][-2]).abs().mean()
                    if float(denom) > 0:
                        gamma = (
                            (current_residual_indicator - dc["probe_residual_window"][-2]).abs().mean() / denom
                        ).clamp(1.0, 1.5)
                    else:
                        gamma = ori_img.new_tensor(1.0)

                    aligned_residual = (
                        dc["residual_window"][-2]
                        + gamma * (dc["residual_window"][-1] - dc["residual_window"][-2])
                    )
                    img = ori_img + aligned_residual
                else:
                    img = ori_img + dc["previous_residual"]

                out = self.final_layer(img, vec)

                dc["previous_input"] = ori_img
                if dc.get("_probe_img", None) is not None:
                    dc["previous_probe_states"] = dc["_probe_img"].detach()

                dc["_probe_img"] = None
                dc["_probe_txt"] = None
                return out

            dc["base_img"] = img_base.detach()

        start_i = 0
        if is_cfgcache:
            cc = cache_dic["cfgcache"]
            probe_img = cc.get("_probe_img", None)
            probe_txt = cc.get("_probe_txt", None)
            probe_depth = int(cc.get("_probe_depth", 0) or 0)

            if (
                current.get("type") != "CFGCacheSkip"
                and probe_img is not None
                and probe_txt is not None
                and probe_depth > 0
            ):
                img = probe_img
                txt = probe_txt
                start_i = probe_depth

        elif cache_dic is not None and cache_dic.get("mode") == "DiCache":
            dc = cache_dic["dicache"]
            if current.get("dicache_resume", False) and dc.get("_probe_img", None) is not None:
                img = dc["_probe_img"]
                txt = dc["_probe_txt"]
                start_i = int(dc.get("probe_depth", 1))

        for i in range(start_i, len(self.double_blocks)):
            block = self.double_blocks[i]
            current["layer"] = i
            img, txt = block(img=img, txt=txt, vec=vec, pe=pe, cache_dic=cache_dic, current=current)

            if cache_dic is not None and cache_dic.get("mode") == "DiCache":
                dc = cache_dic["dicache"]
                probe_depth = int(dc.get("probe_depth", 1))
                if i == probe_depth - 1:
                    cnt = int(dc.get("cnt", 0))
                    T = int(dc.get("num_steps", current.get("num_steps", 0) or 0))
                    ret_ratio = float(dc.get("ret_ratio", 0.2))
                    if (cnt <= int(ret_ratio * T)) or (T > 0 and cnt == T - 1):
                        dc["previous_probe_states"] = img.detach()
                    elif dc.get("_probe_img", None) is not None:
                        dc["previous_probe_states"] = dc["_probe_img"].detach()

        img = torch.cat((txt, img), 1)

        if cache_dic is not None and cache_dic["Delta-DiT"]:
            delta_base = img

        double_block_depth = len(self.double_blocks)
        for i, block in enumerate(self.single_blocks):
            current["layer"] = i + double_block_depth
            img = block(img, vec=vec, pe=pe, cache_dic=cache_dic, current=current)

        if cache_dic is not None and cache_dic["Delta-DiT"]:
            if current["type"] == "Delta-Cache":
                img = delta_base + cache_dic["Delta-Cache"]
            else:
                cache_dic["Delta-Cache"] = img - delta_base

        img = img[:, txt.shape[1] :, ...]

        if cache_dic is not None and cache_dic.get("mode") == "TeaCache":
            tc = cache_dic["teacache"]
            base = tc.pop("base_img")
            tc["previous_residual"] = (img - base).detach()

        if is_cfgcache:
            cc = cache_dic["cfgcache"]
            base = cc.pop("base_img", None)
            if base is not None:
                cc["previous_residual"] = (img - base).detach()

            cc["proxy_prev_input"] = img0

            if cc.get("_probe_modulated_inp", None) is not None:
                cc["proxy_prev_probe_states"] = cc["_probe_modulated_inp"].detach()
            elif cc.get("_probe_img", None) is not None:
                cc["proxy_prev_probe_states"] = cc["_probe_img"].detach()

            self._cfgcache_clear_probe_state(cc)

        if cache_dic is not None and cache_dic.get("mode") == "MagCache":
            mc = cache_dic["magcache"]
            base = mc.pop("base_img", None)
            if base is not None:
                r = (img - base).detach()
                mc["previous_residual"] = r

        if cache_dic is not None and cache_dic.get("mode") == "DiCache":
            dc = cache_dic["dicache"]
            base = dc.pop("base_img", None)

            if base is not None:
                dc["previous_residual"] = (img - base).detach()
                dc["previous_input"] = base.detach()

                if dc.get("previous_probe_states", None) is not None:
                    dc["previous_probe_residual"] = (dc["previous_probe_states"] - base).detach()

                if dc.get("previous_residual", None) is not None:
                    dc.setdefault("residual_window", []).append(dc["previous_residual"])
                    if len(dc["residual_window"]) > 2:
                        dc["residual_window"].pop(0)

                if dc.get("previous_probe_residual", None) is not None:
                    dc.setdefault("probe_residual_window", []).append(dc["previous_probe_residual"])
                    if len(dc["probe_residual_window"]) > 2:
                        dc["probe_residual_window"].pop(0)

            dc["_probe_img"] = None
            dc["_probe_txt"] = None
        img = self.final_layer(img, vec)
        return img


class FluxLoraWrapper(Flux):
    def __init__(
        self,
        lora_rank: int = 128,
        lora_scale: float = 1.0,
        *args,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)

        self.lora_rank = lora_rank

        replace_linear_with_lora(
            self,
            max_rank=lora_rank,
            scale=lora_scale,
        )

    def set_lora_scale(self, scale: float) -> None:
        for module in self.modules():
            if isinstance(module, LinearLora):
                module.set_scale(scale=scale)
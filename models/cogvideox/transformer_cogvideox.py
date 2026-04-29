# Copyright 2025 The CogVideoX team, Tsinghua University & ZhipuAI and The HuggingFace Team.
# All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from typing import Any
import os
from pathlib import Path
import torch
from torch import nn

from .cache_functions import cal_type
from .cache_functions.force_init import force_init
from .taylor_utils import taylor_formula, derivative_approximation, taylor_cache_init

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import PeftAdapterMixin
from diffusers.utils import apply_lora_scale, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models.attention import Attention, AttentionMixin, FeedForward
from diffusers.models.attention_processor import CogVideoXAttnProcessor2_0, FusedCogVideoXAttnProcessor2_0
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import CogVideoXPatchEmbed, TimestepEmbedding, Timesteps
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import AdaLayerNorm, CogVideoXLayerNormZero


logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


@maybe_allow_in_graph
class CogVideoXBlock(nn.Module):
    r"""
    Transformer block used in [CogVideoX](https://github.com/THUDM/CogVideo) model.

    Parameters:
        dim (`int`):
            The number of channels in the input and output.
        num_attention_heads (`int`):
            The number of heads to use for multi-head attention.
        attention_head_dim (`int`):
            The number of channels in each head.
        time_embed_dim (`int`):
            The number of channels in timestep embedding.
        dropout (`float`, defaults to `0.0`):
            The dropout probability to use.
        activation_fn (`str`, defaults to `"gelu-approximate"`):
            Activation function to be used in feed-forward.
        attention_bias (`bool`, defaults to `False`):
            Whether or not to use bias in attention projection layers.
        qk_norm (`bool`, defaults to `True`):
            Whether or not to use normalization after query and key projections in Attention.
        norm_elementwise_affine (`bool`, defaults to `True`):
            Whether to use learnable elementwise affine parameters for normalization.
        norm_eps (`float`, defaults to `1e-5`):
            Epsilon value for normalization layers.
        final_dropout (`bool` defaults to `False`):
            Whether to apply a final dropout after the last feed-forward layer.
        ff_inner_dim (`int`, *optional*, defaults to `None`):
            Custom hidden dimension of Feed-forward layer. If not provided, `4 * dim` is used.
        ff_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Feed-forward layer.
        attention_out_bias (`bool`, defaults to `True`):
            Whether or not to use bias in Attention output projection layer.
    """

    def __init__(
        self,
        dim: int,
        num_attention_heads: int,
        attention_head_dim: int,
        time_embed_dim: int,
        dropout: float = 0.0,
        activation_fn: str = "gelu-approximate",
        attention_bias: bool = False,
        qk_norm: bool = True,
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        final_dropout: bool = True,
        ff_inner_dim: int | None = None,
        ff_bias: bool = True,
        attention_out_bias: bool = True,
    ):
        super().__init__()

        # 1. Self Attention
        self.norm1 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.attn1 = Attention(
            query_dim=dim,
            dim_head=attention_head_dim,
            heads=num_attention_heads,
            qk_norm="layer_norm" if qk_norm else None,
            eps=1e-6,
            bias=attention_bias,
            out_bias=attention_out_bias,
            processor=CogVideoXAttnProcessor2_0(),
        )

        # 2. Feed Forward
        self.norm2 = CogVideoXLayerNormZero(time_embed_dim, dim, norm_elementwise_affine, norm_eps, bias=True)

        self.ff = FeedForward(
            dim,
            dropout=dropout,
            activation_fn=activation_fn,
            final_dropout=final_dropout,
            inner_dim=ff_inner_dim,
            bias=ff_bias,
        )

    def _forward_original(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
            hidden_states, encoder_hidden_states, temb
        )

        # attention
        attn_hidden_states, attn_encoder_hidden_states = self.attn1(
            hidden_states=norm_hidden_states,
            encoder_hidden_states=norm_encoder_hidden_states,
            image_rotary_emb=image_rotary_emb,
            **attention_kwargs,
        )

        hidden_states = hidden_states + gate_msa * attn_hidden_states
        encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

        # norm & modulate
        norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
            hidden_states, encoder_hidden_states, temb
        )

        # feed-forward
        norm_hidden_states = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
        ff_output = self.ff(norm_hidden_states)

        hidden_states = hidden_states + gate_ff * ff_output[:, text_seq_length:]
        encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_output[:, :text_seq_length]

        return hidden_states, encoder_hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        cache_dic: dict[str, Any] | None = None,
        current: dict[str, Any] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        text_seq_length = encoder_hidden_states.size(1)
        attention_kwargs = attention_kwargs or {}

        # --------------------------------------------------
        # no cache / non-Taylor cache modes
        # --------------------------------------------------
        if cache_dic is None or current is None:
            return self._forward_original(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )

        use_taylor_block = bool(
            cache_dic.get("taylor_cache", False)
            or cache_dic.get("use_grouped_taylor", False)
            or cache_dic.get("use_hicache", False)
        )

        if not use_taylor_block:
            return self._forward_original(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=temb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
            )

        current["stream"] = "double_stream"
        block_type = current.get("type", "full")

        # ==================================================
        # full path: compute true outputs and update Taylor cache
        # ==================================================
        if block_type in {"full", "Delta-Cache"}:
            # -------- attn --------
            norm_hidden_states, norm_encoder_hidden_states, gate_msa, enc_gate_msa = self.norm1(
                hidden_states, encoder_hidden_states, temb
            )

            attn_hidden_states, attn_encoder_hidden_states = self.attn1(
                hidden_states=norm_hidden_states,
                encoder_hidden_states=norm_encoder_hidden_states,
                image_rotary_emb=image_rotary_emb,
                **attention_kwargs,
            )

            # image branch attention output
            current["block"] = "img_attn"
            force_init(cache_dic=cache_dic, current=current, tokens=hidden_states)
            taylor_cache_init(cache_dic=cache_dic, current=current)
            if cache_dic.get("taylor_cache", False) or cache_dic.get("enable_feature_collection", False):
                derivative_approximation(
                    cache_dic=cache_dic,
                    current=current,
                    feature=attn_hidden_states,
                )

            # text branch attention output
            current["block"] = "txt_attn"
            force_init(cache_dic=cache_dic, current=current, tokens=encoder_hidden_states)
            taylor_cache_init(cache_dic=cache_dic, current=current)
            if cache_dic.get("taylor_cache", False) or cache_dic.get("enable_feature_collection", False):
                derivative_approximation(
                    cache_dic=cache_dic,
                    current=current,
                    feature=attn_encoder_hidden_states,
                )

            hidden_states = hidden_states + gate_msa * attn_hidden_states
            encoder_hidden_states = encoder_hidden_states + enc_gate_msa * attn_encoder_hidden_states

            # -------- ff --------
            norm_hidden_states, norm_encoder_hidden_states, gate_ff, enc_gate_ff = self.norm2(
                hidden_states, encoder_hidden_states, temb
            )

            ff_input = torch.cat([norm_encoder_hidden_states, norm_hidden_states], dim=1)
            ff_output = self.ff(ff_input)

            ff_hidden_states = ff_output[:, text_seq_length:]
            ff_encoder_hidden_states = ff_output[:, :text_seq_length]

            # image branch mlp output
            current["block"] = "img_mlp"
            force_init(cache_dic=cache_dic, current=current, tokens=hidden_states)
            taylor_cache_init(cache_dic=cache_dic, current=current)
            if cache_dic.get("taylor_cache", False) or cache_dic.get("enable_feature_collection", False):
                derivative_approximation(
                    cache_dic=cache_dic,
                    current=current,
                    feature=ff_hidden_states,
                )

            # text branch mlp output
            current["block"] = "txt_mlp"
            force_init(cache_dic=cache_dic, current=current, tokens=encoder_hidden_states)
            taylor_cache_init(cache_dic=cache_dic, current=current)
            if cache_dic.get("taylor_cache", False) or cache_dic.get("enable_feature_collection", False):
                derivative_approximation(
                    cache_dic=cache_dic,
                    current=current,
                    feature=ff_encoder_hidden_states,
                )

            hidden_states = hidden_states + gate_ff * ff_hidden_states
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * ff_encoder_hidden_states

            return hidden_states, encoder_hidden_states

        # ==================================================
        # taylor cache path: reuse Taylor predictions
        # ==================================================
        elif block_type in {"taylor_cache", "grouped_taylor_cache"}:
            # gate comes from norm, so we still need current-step norms
            _, _, gate_msa, enc_gate_msa = self.norm1(
                hidden_states, encoder_hidden_states, temb
            )

            current["block"] = "img_attn"
            hidden_states = hidden_states + gate_msa * taylor_formula(
                cache_dic=cache_dic, current=current
            )

            current["block"] = "txt_attn"
            encoder_hidden_states = encoder_hidden_states + enc_gate_msa * taylor_formula(
                cache_dic=cache_dic, current=current
            )

            _, _, gate_ff, enc_gate_ff = self.norm2(
                hidden_states, encoder_hidden_states, temb
            )

            current["block"] = "img_mlp"
            hidden_states = hidden_states + gate_ff * taylor_formula(
                cache_dic=cache_dic, current=current
            )

            current["block"] = "txt_mlp"
            encoder_hidden_states = encoder_hidden_states + enc_gate_ff * taylor_formula(
                cache_dic=cache_dic, current=current
            )

            return hidden_states, encoder_hidden_states

        return self._forward_original(
            hidden_states=hidden_states,
            encoder_hidden_states=encoder_hidden_states,
            temb=temb,
            image_rotary_emb=image_rotary_emb,
            attention_kwargs=attention_kwargs,
        )


class CogVideoXTransformer3DModel(ModelMixin, AttentionMixin, ConfigMixin, PeftAdapterMixin, CacheMixin):
    """
    A Transformer model for video-like data in [CogVideoX](https://github.com/THUDM/CogVideo).
    """

    _skip_layerwise_casting_patterns = ["patch_embed", "norm"]
    _supports_gradient_checkpointing = True
    _no_split_modules = ["CogVideoXBlock", "CogVideoXPatchEmbed"]

    @register_to_config
    def __init__(
        self,
        num_attention_heads: int = 30,
        attention_head_dim: int = 64,
        in_channels: int = 16,
        out_channels: int | None = 16,
        flip_sin_to_cos: bool = True,
        freq_shift: int = 0,
        time_embed_dim: int = 512,
        ofs_embed_dim: int | None = None,
        text_embed_dim: int = 4096,
        num_layers: int = 30,
        dropout: float = 0.0,
        attention_bias: bool = True,
        sample_width: int = 90,
        sample_height: int = 60,
        sample_frames: int = 49,
        patch_size: int = 2,
        patch_size_t: int | None = None,
        temporal_compression_ratio: int = 4,
        max_text_seq_length: int = 226,
        activation_fn: str = "gelu-approximate",
        timestep_activation_fn: str = "silu",
        norm_elementwise_affine: bool = True,
        norm_eps: float = 1e-5,
        spatial_interpolation_scale: float = 1.875,
        temporal_interpolation_scale: float = 1.0,
        use_rotary_positional_embeddings: bool = False,
        use_learned_positional_embeddings: bool = False,
        patch_bias: bool = True,
    ):
        super().__init__()
        inner_dim = num_attention_heads * attention_head_dim

        if not use_rotary_positional_embeddings and use_learned_positional_embeddings:
            raise ValueError(
                "There are no CogVideoX checkpoints available with disable rotary embeddings and learned positional "
                "embeddings. If you're using a custom model and/or believe this should be supported, please open an "
                "issue at https://github.com/huggingface/diffusers/issues."
            )

        # 1. Patch embedding
        self.patch_embed = CogVideoXPatchEmbed(
            patch_size=patch_size,
            patch_size_t=patch_size_t,
            in_channels=in_channels,
            embed_dim=inner_dim,
            text_embed_dim=text_embed_dim,
            bias=patch_bias,
            sample_width=sample_width,
            sample_height=sample_height,
            sample_frames=sample_frames,
            temporal_compression_ratio=temporal_compression_ratio,
            max_text_seq_length=max_text_seq_length,
            spatial_interpolation_scale=spatial_interpolation_scale,
            temporal_interpolation_scale=temporal_interpolation_scale,
            use_positional_embeddings=not use_rotary_positional_embeddings,
            use_learned_positional_embeddings=use_learned_positional_embeddings,
        )
        self.embedding_dropout = nn.Dropout(dropout)

        # 2. Time embeddings and ofs embedding
        self.time_proj = Timesteps(inner_dim, flip_sin_to_cos, freq_shift)
        self.time_embedding = TimestepEmbedding(inner_dim, time_embed_dim, timestep_activation_fn)

        self.ofs_proj = None
        self.ofs_embedding = None
        if ofs_embed_dim:
            self.ofs_proj = Timesteps(ofs_embed_dim, flip_sin_to_cos, freq_shift)
            self.ofs_embedding = TimestepEmbedding(
                ofs_embed_dim, ofs_embed_dim, timestep_activation_fn
            )

        # 3. Transformer blocks
        self.transformer_blocks = nn.ModuleList(
            [
                CogVideoXBlock(
                    dim=inner_dim,
                    num_attention_heads=num_attention_heads,
                    attention_head_dim=attention_head_dim,
                    time_embed_dim=time_embed_dim,
                    dropout=dropout,
                    activation_fn=activation_fn,
                    attention_bias=attention_bias,
                    norm_elementwise_affine=norm_elementwise_affine,
                    norm_eps=norm_eps,
                )
                for _ in range(num_layers)
            ]
        )
        self.norm_final = nn.LayerNorm(inner_dim, norm_eps, norm_elementwise_affine)

        # 4. Output blocks
        self.norm_out = AdaLayerNorm(
            embedding_dim=time_embed_dim,
            output_dim=2 * inner_dim,
            norm_elementwise_affine=norm_elementwise_affine,
            norm_eps=norm_eps,
            chunk_dim=1,
        )

        if patch_size_t is None:
            output_dim = patch_size * patch_size * out_channels
        else:
            output_dim = patch_size * patch_size * patch_size_t * out_channels

        self.proj_out = nn.Linear(inner_dim, output_dim)
        self.gradient_checkpointing = False

    def fuse_qkv_projections(self):
        self.original_attn_processors = None

        for _, attn_processor in self.attn_processors.items():
            if "Added" in str(attn_processor.__class__.__name__):
                raise ValueError("`fuse_qkv_projections()` is not supported for models having added KV projections.")

        self.original_attn_processors = self.attn_processors

        for module in self.modules():
            if isinstance(module, Attention):
                module.fuse_projections(fuse=True)

        self.set_attn_processor(FusedCogVideoXAttnProcessor2_0())

    def unfuse_qkv_projections(self):
        if self.original_attn_processors is not None:
            self.set_attn_processor(self.original_attn_processors)

    @staticmethod
    def _cfgcache_rel_l1(curr: torch.Tensor, prev: torch.Tensor, eps: float = 1e-6) -> float:
        denom = prev.abs().mean()
        if float(denom) == 0.0:
            return 0.0
        val = (curr - prev).abs().mean() / (denom + eps)
        return float(val.detach().cpu())

    def _cfgcache_get_branch_state(self, cache_dic: dict[str, Any], current: dict[str, Any]):
        cc = cache_dic["cfgcache"]
        branch = current.get("model", "cond")
        return branch, cc[branch], cc

    def _cfgcache_prepare_light_probe(
        self,
        *,
        emb: torch.Tensor,
        cache_dic: dict[str, Any],
        current: dict[str, Any],
    ):
        """
        Lightweight probe path for Tea-style proxies.

        It only prepares the timestep-conditioned signal used by the proxy and
        does NOT run patch embedding or shallow transformer blocks.
        """
        branch, bc, cc = self._cfgcache_get_branch_state(cache_dic, current)

        probe_modulated_inp = emb.detach()
        bc["_probe_modulated_inp"] = probe_modulated_inp
        bc["_probe_emb"] = probe_modulated_inp

        # Tea-style summary/fallback payload
        bc["_probe_summary"] = {
            "modulated_inp": probe_modulated_inp,
        }
        bc["_probe_payload"] = {
            "probe_depth": 0,
            "dicache_dx": None,
            "dicache_dy": None,
        }

        # No shallow hidden probe is materialized in the light path.
        bc["_probe_hidden_states"] = None
        bc["_probe_encoder_hidden_states"] = None
        bc["_probe_img"] = None
        bc["_probe_txt"] = None
        bc["_probe_depth"] = 0

        return bc["_probe_payload"]

    def cfgcache_prepare_proxy_signal(
        self,
        *,
        timestep: int | float | torch.LongTensor,
        hidden_dtype: torch.dtype,
        cache_dic: dict[str, Any],
        current: dict[str, Any],
        timestep_cond: torch.Tensor | None = None,
        ofs: int | float | torch.LongTensor | None = None,
    ):
        """
        Public helper for pipeline-side lightweight proxy preparation.

        This computes the CFGCache proxy signal directly from the timestep
        embedding, without invoking patch embedding or transformer blocks.
        """
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None and ofs is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        return self._cfgcache_prepare_light_probe(
            emb=emb,
            cache_dic=cache_dic,
            current=current,
        )

    def _cfgcache_prepare_probe(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        emb: torch.Tensor,
        image_rotary_emb,
        attention_kwargs,
        cache_dic: dict[str, Any],
        current: dict[str, Any],
    ):
        branch, bc, cc = self._cfgcache_get_branch_state(cache_dic, current)

        proxy = cc.get("_proxy_obj", None)
        probe_mode = getattr(proxy, "probe_mode", cc.get("probe_mode", "both"))
        probe_mode = "none" if probe_mode is None else str(probe_mode).lower()

        # Tea-like proxies only need the lightweight modulated signal.
        if probe_mode in {"modulated_inp", "emb", "light", "light_signal", "tea", "teacache"}:
            return self._cfgcache_prepare_light_probe(
                emb=emb,
                cache_dic=cache_dic,
                current=current,
            )

        # --------------------------------------------------
        # Tea-style current signal
        # --------------------------------------------------
        bc["_probe_modulated_inp"] = emb.detach()

        # --------------------------------------------------
        # Di-style reference states
        # --------------------------------------------------
        prev_inp = bc.get("proxy_prev_input", None)
        prev_probe = bc.get("proxy_prev_probe_states", None)

        depth = min(int(cc.get("probe_depth", 2)), len(self.transformer_blocks))

        test_hidden_states = hidden_states.clone()
        test_encoder_hidden_states = encoder_hidden_states.clone()

        old_type = current.get("type", None)
        old_layer = current.get("layer", None)

        for probe_i in range(depth):
            current["type"] = "full"
            current["layer"] = probe_i
            test_hidden_states, test_encoder_hidden_states = self.transformer_blocks[probe_i](
                hidden_states=test_hidden_states,
                encoder_hidden_states=test_encoder_hidden_states,
                temb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
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
            dx = self._cfgcache_rel_l1(hidden_states.detach(), prev_inp, eps)

        if prev_probe is not None:
            dy = self._cfgcache_rel_l1(test_hidden_states.detach(), prev_probe, eps)

        # --------------------------------------------------
        # save probe states for both Di / Tea
        # --------------------------------------------------
        bc["_probe_hidden_states"] = test_hidden_states.detach()
        bc["_probe_encoder_hidden_states"] = test_encoder_hidden_states.detach()
        bc["_probe_img"] = bc["_probe_hidden_states"]
        bc["_probe_txt"] = bc["_probe_encoder_hidden_states"]
        bc["_probe_depth"] = depth

        # Tea fallback summary
        bc["_probe_summary"] = {
            "modulated_inp": bc["_probe_modulated_inp"],
        }

        payload = {
            "probe_depth": depth,
            "dicache_dx": dx,
            "dicache_dy": dy,
        }
        bc["_probe_payload"] = payload
        return payload

    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep: int | float | torch.LongTensor,
        timestep_cond: torch.Tensor | None = None,
        ofs: int | float | torch.LongTensor | None = None,
        image_rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        attention_kwargs: dict[str, Any] | None = None,
        cache_dic: dict[str, Any] | None = None,
        current: dict[str, Any] | None = None,
        cfgcache_only_probe: bool = False,
        return_dict: bool = True,
    ) -> tuple[torch.Tensor] | Transformer2DModelOutput:
        batch_size, num_frames, channels, height, width = hidden_states.shape

        # 1. Time embedding
        timesteps = timestep
        t_emb = self.time_proj(timesteps)
        t_emb = t_emb.to(dtype=hidden_states.dtype)
        emb = self.time_embedding(t_emb, timestep_cond)

        if self.ofs_embedding is not None:
            ofs_emb = self.ofs_proj(ofs)
            ofs_emb = ofs_emb.to(dtype=hidden_states.dtype)
            ofs_emb = self.ofs_embedding(ofs_emb)
            emb = emb + ofs_emb

        # 2. Patch embedding
        hidden_states = self.patch_embed(encoder_hidden_states, hidden_states)
        hidden_states = self.embedding_dropout(hidden_states)

        text_seq_length = encoder_hidden_states.shape[1]
        encoder_hidden_states = hidden_states[:, :text_seq_length]
        hidden_states = hidden_states[:, text_seq_length:]

        cache_mode = None if cache_dic is None else cache_dic.get("mode", None)

        # probe-only path for CFGCache
        if (
            cache_mode == "CFGCache"
            and cache_dic is not None
            and current is not None
            and cfgcache_only_probe
        ):
            self._cfgcache_prepare_probe(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                emb=emb,
                image_rotary_emb=image_rotary_emb,
                attention_kwargs=attention_kwargs,
                cache_dic=cache_dic,
                current=current,
            )
            if not return_dict:
                return (hidden_states,)
            return Transformer2DModelOutput(sample=hidden_states)

        # =========================================================
        # common execution controls
        # =========================================================
        run_transformer_blocks = True
        start_block_idx = 0

        # ---------- TeaCache runtime ----------
        save_teacache_residual = False
        teacache_branch = None
        teacache_state = None
        tea_ori_hidden_states = None
        tea_ori_encoder_hidden_states = None

        # ---------- DiCache runtime ----------
        save_dicache_residual = False
        dicache_branch = None
        dicache_state = None
        dcta_used = False

        di_base_hidden_states = None
        di_base_encoder_hidden_states = None

        di_ori_hidden_states = None
        di_ori_encoder_hidden_states = None

        di_probe_hidden_states = None
        di_probe_encoder_hidden_states = None
        di_probe_depth = 0
        di_capture_probe_at = None

        # ---------- MagCache runtime ----------
        save_magcache_residual = False
        magcache_branch = None
        magcache_state = None

        mag_base_hidden_states = None
        mag_base_encoder_hidden_states = None

        mag_ori_hidden_states = None
        mag_ori_encoder_hidden_states = None

        # ---------- CFGCache runtime ----------
        save_cfgcache_postnorm = False
        cfgcache_branch = None
        cfgcache_state = None

        cfg_base_hidden_states = None
        cfg_base_encoder_hidden_states = None

        cfg_ori_hidden_states = None
        cfg_ori_encoder_hidden_states = None

        cfg_probe_depth = 0
        cfg_capture_probe_at = None

        # =========================================================
        # cache dispatch
        # =========================================================
        if cache_dic is not None and current is not None:
            # -----------------------------------------------------
            # TeaCache dispatch
            # -----------------------------------------------------
            if cache_mode == "TeaCache" and cache_dic.get("teacache_enable", True):
                teacache_branch = current.get("model", "cond")
                teacache_state = cache_dic["teacache"][teacache_branch]

                current["teacache_modulated_inp"] = emb.detach()
                cal_type(cache_dic=cache_dic, current=current)

                if current.get("type") == "TeaCacheSkip":
                    has_residual = (
                        teacache_state.get("previous_residual", None) is not None
                        and teacache_state.get("previous_residual_encoder", None) is not None
                    )

                    if has_residual:
                        hidden_states = hidden_states + teacache_state["previous_residual"]
                        encoder_hidden_states = (
                            encoder_hidden_states + teacache_state["previous_residual_encoder"]
                        )
                        run_transformer_blocks = False
                    else:
                        current["type"] = "full"
                        run_transformer_blocks = True

                if run_transformer_blocks:
                    tea_ori_hidden_states = hidden_states.clone()
                    tea_ori_encoder_hidden_states = encoder_hidden_states.clone()
                    save_teacache_residual = True

            # -----------------------------------------------------
            # DiCache dispatch
            # -----------------------------------------------------
            elif cache_mode == "DiCache" and cache_dic.get("dicache_enable", True):
                dicache_branch = current.get("model", "cond")
                dicache_state = cache_dic["dicache"][dicache_branch]

                T = int(dicache_state.get("num_steps", current.get("num_steps", 0) or 0))
                cnt = int(dicache_state.get("cnt", 0))
                ret_ratio = float(dicache_state.get("ret_ratio", 0.2))

                di_base_hidden_states = hidden_states.detach()
                di_base_encoder_hidden_states = encoder_hidden_states.detach()

                dicache_state["base_img"] = di_base_hidden_states
                dicache_state["base_hidden_states"] = di_base_hidden_states
                dicache_state["base_encoder_hidden_states"] = di_base_encoder_hidden_states

                current["dicache_delta_x"] = None
                current["dicache_delta_y"] = None

                dicache_state["_probe_img"] = None
                dicache_state["_probe_txt"] = None
                dicache_state["_probe_hidden_states"] = None
                dicache_state["_probe_encoder_hidden_states"] = None

                di_probe_depth = min(int(dicache_state.get("probe_depth", 1)), len(self.transformer_blocks))
                need_probe = not ((cnt <= int(ret_ratio * T)) or (T > 0 and cnt == T - 1))

                if need_probe and di_probe_depth > 0:
                    test_hidden_states = hidden_states.clone()
                    test_encoder_hidden_states = encoder_hidden_states.clone()

                    for probe_i in range(di_probe_depth):
                        test_hidden_states, test_encoder_hidden_states = self.transformer_blocks[probe_i](
                            hidden_states=test_hidden_states,
                            encoder_hidden_states=test_encoder_hidden_states,
                            temb=emb,
                            image_rotary_emb=image_rotary_emb,
                            attention_kwargs=attention_kwargs,
                        )

                    di_probe_hidden_states = test_hidden_states.detach()
                    di_probe_encoder_hidden_states = test_encoder_hidden_states.detach()

                    dicache_state["_probe_img"] = di_probe_hidden_states
                    dicache_state["_probe_txt"] = di_probe_encoder_hidden_states
                    dicache_state["_probe_hidden_states"] = di_probe_hidden_states
                    dicache_state["_probe_encoder_hidden_states"] = di_probe_encoder_hidden_states

                    prev_inp = dicache_state.get("previous_input", None)
                    prev_probe = dicache_state.get("previous_probe_states", None)

                    if (prev_inp is not None) and (prev_probe is not None):
                        eps = 1e-6

                        prev_inp_mean = prev_inp.abs().mean()
                        if float(prev_inp_mean) == 0.0:
                            dx = test_hidden_states.new_tensor(0.0)
                        else:
                            dx = (hidden_states - prev_inp).abs().mean() / (prev_inp_mean + eps)

                        prev_probe_mean = prev_probe.abs().mean()
                        if float(prev_probe_mean) == 0.0:
                            dy = test_hidden_states.new_tensor(0.0)
                        else:
                            dy = (test_hidden_states - prev_probe).abs().mean() / (prev_probe_mean + eps)

                        current["dicache_delta_x"] = float(dx.detach().cpu())
                        current["dicache_delta_y"] = float(dy.detach().cpu())

                cal_type(cache_dic=cache_dic, current=current)

                if current.get("type") == "DiCacheSkip":
                    has_residual = dicache_state.get("previous_residual", None) is not None

                    if has_residual:
                        use_dcta = (
                            len(dicache_state.get("residual_window", [])) >= 2
                            and len(dicache_state.get("probe_residual_window", [])) >= 2
                            and dicache_state.get("_probe_hidden_states", None) is not None
                        )

                        if use_dcta:
                            current_probe_residual = dicache_state["_probe_hidden_states"] - di_base_hidden_states
                            denom = (
                                dicache_state["probe_residual_window"][-1]
                                - dicache_state["probe_residual_window"][-2]
                            ).abs().mean()

                            if float(denom) > 0:
                                gamma = (
                                    (current_probe_residual - dicache_state["probe_residual_window"][-2]).abs().mean()
                                    / denom
                                ).clamp(1.0, 1.5)
                            else:
                                gamma = di_base_hidden_states.new_tensor(1.0)

                            aligned_hidden_residual = (
                                dicache_state["residual_window"][-2]
                                + gamma * (dicache_state["residual_window"][-1] - dicache_state["residual_window"][-2])
                            )
                            hidden_states = di_base_hidden_states + aligned_hidden_residual

                            if len(dicache_state.get("residual_window_encoder", [])) >= 2:
                                aligned_encoder_residual = (
                                    dicache_state["residual_window_encoder"][-2]
                                    + gamma * (
                                        dicache_state["residual_window_encoder"][-1]
                                        - dicache_state["residual_window_encoder"][-2]
                                    )
                                )
                                encoder_hidden_states = di_base_encoder_hidden_states + aligned_encoder_residual
                            else:
                                prev_residual_encoder = dicache_state.get("previous_residual_encoder", None)
                                if prev_residual_encoder is not None:
                                    encoder_hidden_states = di_base_encoder_hidden_states + prev_residual_encoder

                            dcta_used = True
                        else:
                            hidden_states = di_base_hidden_states + dicache_state["previous_residual"]

                            prev_residual_encoder = dicache_state.get("previous_residual_encoder", None)
                            if prev_residual_encoder is not None:
                                encoder_hidden_states = di_base_encoder_hidden_states + prev_residual_encoder

                        dicache_state["previous_input"] = di_base_hidden_states
                        dicache_state["previous_input_encoder"] = di_base_encoder_hidden_states

                        if dicache_state.get("_probe_hidden_states", None) is not None:
                            dicache_state["previous_probe_states"] = dicache_state["_probe_hidden_states"].detach()
                        if dicache_state.get("_probe_encoder_hidden_states", None) is not None:
                            dicache_state["previous_probe_states_encoder"] = (
                                dicache_state["_probe_encoder_hidden_states"].detach()
                            )

                        dicache_state["_probe_img"] = None
                        dicache_state["_probe_txt"] = None
                        dicache_state["_probe_hidden_states"] = None
                        dicache_state["_probe_encoder_hidden_states"] = None

                        run_transformer_blocks = False
                    else:
                        current["type"] = "full"
                        current["dicache_resume"] = False
                        run_transformer_blocks = True

                if run_transformer_blocks:
                    di_ori_hidden_states = di_base_hidden_states.clone()
                    di_ori_encoder_hidden_states = di_base_encoder_hidden_states.clone()
                    save_dicache_residual = True

                    if (
                        current.get("dicache_resume", False)
                        and dicache_state.get("_probe_hidden_states", None) is not None
                        and dicache_state.get("_probe_encoder_hidden_states", None) is not None
                    ):
                        hidden_states = dicache_state["_probe_hidden_states"]
                        encoder_hidden_states = dicache_state["_probe_encoder_hidden_states"]
                        start_block_idx = di_probe_depth
                    else:
                        start_block_idx = 0

                    if di_probe_depth > 0 and start_block_idx == 0:
                        di_capture_probe_at = di_probe_depth - 1

            # -----------------------------------------------------
            # MagCache dispatch
            # -----------------------------------------------------
            elif cache_mode == "MagCache" and cache_dic.get("magcache_enable", True):
                magcache_branch = current.get("model", "cond")
                magcache_state = cache_dic["magcache"][magcache_branch]

                mag_base_hidden_states = hidden_states.detach()
                mag_base_encoder_hidden_states = encoder_hidden_states.detach()

                magcache_state["base_img"] = mag_base_hidden_states
                magcache_state["base_hidden_states"] = mag_base_hidden_states
                magcache_state["base_encoder_hidden_states"] = mag_base_encoder_hidden_states

                cal_type(cache_dic=cache_dic, current=current)

                if current.get("type") == "MagCacheSkip":
                    has_residual = (
                        magcache_state.get("previous_residual", None) is not None
                        and magcache_state.get("previous_residual_encoder", None) is not None
                    )

                    if has_residual:
                        hidden_states = mag_base_hidden_states + magcache_state["previous_residual"]
                        encoder_hidden_states = (
                            mag_base_encoder_hidden_states + magcache_state["previous_residual_encoder"]
                        )
                        run_transformer_blocks = False
                    else:
                        current["type"] = "full"
                        run_transformer_blocks = True

                if run_transformer_blocks:
                    mag_ori_hidden_states = mag_base_hidden_states.clone()
                    mag_ori_encoder_hidden_states = mag_base_encoder_hidden_states.clone()
                    save_magcache_residual = True

            # -----------------------------------------------------
            # CFGCache dispatch
            # -----------------------------------------------------
            elif cache_mode == "CFGCache" and cache_dic.get("cfgcache_enable", True):
                cfgcache_branch, cfgcache_state, cc = self._cfgcache_get_branch_state(cache_dic, current)

                cfg_base_hidden_states = hidden_states.detach()
                cfg_base_encoder_hidden_states = encoder_hidden_states.detach()

                cfgcache_state["base_hidden_states"] = cfg_base_hidden_states
                cfgcache_state["base_encoder_hidden_states"] = cfg_base_encoder_hidden_states
                cfgcache_state["base_img"] = cfg_base_hidden_states

                current["dicache_delta_x"] = None
                current["dicache_delta_y"] = None

                if cfgcache_state.get("_probe_hidden_states", None) is None:
                    cfgcache_state["_probe_img"] = None
                    cfgcache_state["_probe_txt"] = None
                    cfgcache_state["_probe_hidden_states"] = None
                    cfgcache_state["_probe_encoder_hidden_states"] = None
                    cfgcache_state["_probe_modulated_inp"] = None
                    cfgcache_state["_probe_summary"] = None
                    cfgcache_state["_probe_payload"] = None

                cal_type(cache_dic=cache_dic, current=current)

                if current.get("type") == "CFGCacheSkip":
                    has_residual = (
                        cfgcache_state.get("previous_residual", None) is not None
                        and cfgcache_state.get("previous_residual_encoder", None) is not None
                    )

                    if has_residual:
                        hidden_states = cfg_base_hidden_states + cfgcache_state["previous_residual"]
                        encoder_hidden_states = (
                            cfg_base_encoder_hidden_states + cfgcache_state["previous_residual_encoder"]
                        )

                        # ------------------------------
                        # Di-style prev references
                        # ------------------------------
                        cfgcache_state["proxy_prev_input"] = cfg_base_hidden_states
                        cfgcache_state["proxy_prev_input_encoder"] = cfg_base_encoder_hidden_states

                        if cfgcache_state.get("_probe_hidden_states", None) is not None:
                            cfgcache_state["proxy_prev_probe_states"] = (
                                cfgcache_state["_probe_hidden_states"].detach()
                            )
                        if cfgcache_state.get("_probe_encoder_hidden_states", None) is not None:
                            cfgcache_state["proxy_prev_probe_states_encoder"] = (
                                cfgcache_state["_probe_encoder_hidden_states"].detach()
                            )

                        # ------------------------------
                        # Tea-style prev references
                        # ------------------------------
                        if cfgcache_state.get("_probe_modulated_inp", None) is not None:
                            cfgcache_state["proxy_prev_modulated_inp"] = (
                                cfgcache_state["_probe_modulated_inp"].detach()
                            )
                        elif cfgcache_state.get("proxy_prev_modulated_inp", None) is None:
                            cfgcache_state["proxy_prev_modulated_inp"] = emb.detach()

                        # cleanup temp probe
                        cfgcache_state["_probe_img"] = None
                        cfgcache_state["_probe_txt"] = None
                        cfgcache_state["_probe_hidden_states"] = None
                        cfgcache_state["_probe_encoder_hidden_states"] = None
                        cfgcache_state["_probe_modulated_inp"] = None
                        cfgcache_state["_probe_summary"] = None
                        cfgcache_state["_probe_payload"] = None

                        run_transformer_blocks = False
                    else:
                        current["type"] = "full"
                        current["cfgcache_resume"] = False
                        run_transformer_blocks = True

                if run_transformer_blocks:
                    cfg_ori_hidden_states = cfg_base_hidden_states.clone()
                    cfg_ori_encoder_hidden_states = cfg_base_encoder_hidden_states.clone()
                    save_cfgcache_postnorm = True

                    cfg_probe_depth = int(cfgcache_state.get("_probe_depth", 0) or 0)

                    if (
                        current.get("cfgcache_resume", False)
                        and cfgcache_state.get("_probe_hidden_states", None) is not None
                        and cfgcache_state.get("_probe_encoder_hidden_states", None) is not None
                    ):
                        hidden_states = cfgcache_state["_probe_hidden_states"]
                        encoder_hidden_states = cfgcache_state["_probe_encoder_hidden_states"]
                        start_block_idx = cfg_probe_depth
                    else:
                        start_block_idx = 0

                    if cfg_probe_depth > 0 and start_block_idx == 0:
                        cfg_capture_probe_at = cfg_probe_depth - 1

        # -----------------------------------------------------
        # Taylor / HiCache / Taylor-Scaled / ClusCa style dispatch
        # -----------------------------------------------------
        if (
            cache_dic is not None
            and current is not None
            and (
                cache_dic.get("taylor_cache", False)
                or cache_dic.get("use_grouped_taylor", False)
                or cache_dic.get("use_hicache", False)
            )
        ):
            cal_type(cache_dic=cache_dic, current=current)

        # =========================================================
        # 3. Transformer blocks
        # =========================================================
        if run_transformer_blocks:
            for i in range(start_block_idx, len(self.transformer_blocks)):
                if current is not None:
                    current["layer"] = i

                block = self.transformer_blocks[i]

                if torch.is_grad_enabled() and self.gradient_checkpointing:
                    hidden_states, encoder_hidden_states = self._gradient_checkpointing_func(
                        block,
                        hidden_states,
                        encoder_hidden_states,
                        emb,
                        image_rotary_emb,
                        attention_kwargs,
                    )
                else:
                    hidden_states, encoder_hidden_states = block(
                        hidden_states=hidden_states,
                        encoder_hidden_states=encoder_hidden_states,
                        temb=emb,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        cache_dic=cache_dic,
                        current=current,
                    )

                if (
                    cache_mode == "DiCache"
                    and dicache_state is not None
                    and di_capture_probe_at is not None
                    and i == di_capture_probe_at
                ):
                    dicache_state["previous_probe_states"] = hidden_states.detach()
                    dicache_state["previous_probe_states_encoder"] = encoder_hidden_states.detach()

                if (
                    cache_mode == "CFGCache"
                    and cfgcache_state is not None
                    and cfg_capture_probe_at is not None
                    and i == cfg_capture_probe_at
                ):
                    cfgcache_state["previous_probe_states"] = hidden_states.detach()
                    cfgcache_state["previous_probe_states_encoder"] = encoder_hidden_states.detach()

        # =========================================================
        # post-transformer residual save
        # =========================================================
        if save_teacache_residual and teacache_state is not None:
            teacache_state["previous_residual"] = (hidden_states - tea_ori_hidden_states).detach()
            teacache_state["previous_residual_encoder"] = (
                encoder_hidden_states - tea_ori_encoder_hidden_states
            ).detach()

        if save_dicache_residual and dicache_state is not None:
            dicache_state["previous_residual"] = (hidden_states - di_ori_hidden_states).detach()
            dicache_state["previous_residual_encoder"] = (
                encoder_hidden_states - di_ori_encoder_hidden_states
            ).detach()

            dicache_state["previous_input"] = di_base_hidden_states
            dicache_state["previous_input_encoder"] = di_base_encoder_hidden_states

            if dicache_state.get("_probe_hidden_states", None) is not None:
                dicache_state["previous_probe_states"] = dicache_state["_probe_hidden_states"].detach()
            if dicache_state.get("_probe_encoder_hidden_states", None) is not None:
                dicache_state["previous_probe_states_encoder"] = (
                    dicache_state["_probe_encoder_hidden_states"].detach()
                )

            if dicache_state.get("previous_probe_states", None) is not None:
                dicache_state["previous_probe_residual"] = (
                    dicache_state["previous_probe_states"] - di_base_hidden_states
                ).detach()

            if dicache_state.get("previous_probe_states_encoder", None) is not None:
                dicache_state["previous_probe_residual_encoder"] = (
                    dicache_state["previous_probe_states_encoder"] - di_base_encoder_hidden_states
                ).detach()

            dicache_state.setdefault("residual_window", []).append(dicache_state["previous_residual"])
            if len(dicache_state["residual_window"]) > 2:
                dicache_state["residual_window"].pop(0)

            dicache_state.setdefault("residual_window_encoder", []).append(
                dicache_state["previous_residual_encoder"]
            )
            if len(dicache_state["residual_window_encoder"]) > 2:
                dicache_state["residual_window_encoder"].pop(0)

            if dicache_state.get("previous_probe_residual", None) is not None:
                dicache_state.setdefault("probe_residual_window", []).append(
                    dicache_state["previous_probe_residual"]
                )
                if len(dicache_state["probe_residual_window"]) > 2:
                    dicache_state["probe_residual_window"].pop(0)

            if dicache_state.get("previous_probe_residual_encoder", None) is not None:
                dicache_state.setdefault("probe_residual_window_encoder", []).append(
                    dicache_state["previous_probe_residual_encoder"]
                )
                if len(dicache_state["probe_residual_window_encoder"]) > 2:
                    dicache_state["probe_residual_window_encoder"].pop(0)

            dicache_state["_probe_img"] = None
            dicache_state["_probe_txt"] = None
            dicache_state["_probe_hidden_states"] = None
            dicache_state["_probe_encoder_hidden_states"] = None

        if save_magcache_residual and magcache_state is not None:
            magcache_state["previous_residual"] = (
                hidden_states - mag_ori_hidden_states
            ).detach()
            magcache_state["previous_residual_encoder"] = (
                encoder_hidden_states - mag_ori_encoder_hidden_states
            ).detach()

            magcache_state["base_img"] = None
            magcache_state["base_hidden_states"] = None
            magcache_state["base_encoder_hidden_states"] = None

        if save_cfgcache_postnorm and cfgcache_state is not None:
            # --------------------------------------------------
            # core reusable residuals for CFGCache skip path
            # --------------------------------------------------
            cfgcache_state["previous_residual"] = (hidden_states - cfg_ori_hidden_states).detach()
            cfgcache_state["previous_residual_encoder"] = (
                encoder_hidden_states - cfg_ori_encoder_hidden_states
            ).detach()

            # optional snapshot
            cfgcache_state["previous_postnorm_hidden"] = hidden_states.detach()

            # ------------------------------
            # Di-style prev references
            # ------------------------------
            cfgcache_state["proxy_prev_input"] = cfg_base_hidden_states
            cfgcache_state["proxy_prev_input_encoder"] = cfg_base_encoder_hidden_states

            if cfgcache_state.get("_probe_hidden_states", None) is not None:
                cfgcache_state["proxy_prev_probe_states"] = cfgcache_state["_probe_hidden_states"].detach()
            elif cfgcache_state.get("previous_probe_states", None) is not None:
                cfgcache_state["proxy_prev_probe_states"] = cfgcache_state["previous_probe_states"].detach()

            if cfgcache_state.get("_probe_encoder_hidden_states", None) is not None:
                cfgcache_state["proxy_prev_probe_states_encoder"] = (
                    cfgcache_state["_probe_encoder_hidden_states"].detach()
                )
            elif cfgcache_state.get("previous_probe_states_encoder", None) is not None:
                cfgcache_state["proxy_prev_probe_states_encoder"] = (
                    cfgcache_state["previous_probe_states_encoder"].detach()
                )

            # ------------------------------
            # Tea-style prev references
            # ------------------------------
            if cfgcache_state.get("_probe_modulated_inp", None) is not None:
                cfgcache_state["proxy_prev_modulated_inp"] = (
                    cfgcache_state["_probe_modulated_inp"].detach()
                )
            else:
                cfgcache_state["proxy_prev_modulated_inp"] = emb.detach()

            # --------------------------------------------------
            # full refresh means new anchor
            # keep both Tea / Di-style anchor references
            # --------------------------------------------------
            cfgcache_state["proxy_anchor_modulated_inp"] = cfgcache_state["proxy_prev_modulated_inp"]

            if cfgcache_state.get("proxy_prev_probe_states", None) is not None:
                cfgcache_state["proxy_anchor_probe_states"] = (
                    cfgcache_state["proxy_prev_probe_states"].detach()
                )

            if cfgcache_state.get("proxy_prev_probe_states_encoder", None) is not None:
                cfgcache_state["proxy_anchor_probe_states_encoder"] = (
                    cfgcache_state["proxy_prev_probe_states_encoder"].detach()
                )

            # cleanup temp probe
            cfgcache_state["_probe_img"] = None
            cfgcache_state["_probe_txt"] = None
            cfgcache_state["_probe_hidden_states"] = None
            cfgcache_state["_probe_encoder_hidden_states"] = None
            cfgcache_state["_probe_modulated_inp"] = None
            cfgcache_state["_probe_summary"] = None
            cfgcache_state["_probe_payload"] = None

        # official CogVideoX norm_final path
        if not self.config.use_rotary_positional_embeddings:
            hidden_states = self.norm_final(hidden_states)
        else:
            hidden_states = torch.cat([encoder_hidden_states, hidden_states], dim=1)
            hidden_states = self.norm_final(hidden_states)
            hidden_states = hidden_states[:, text_seq_length:]

        # 4. Final block
        hidden_states = self.norm_out(hidden_states, temb=emb)
        hidden_states = self.proj_out(hidden_states)

        # 5. Unpatchify
        p = self.config.patch_size
        p_t = self.config.patch_size_t

        if p_t is None:
            output = hidden_states.reshape(batch_size, num_frames, height // p, width // p, -1, p, p)
            output = output.permute(0, 1, 4, 2, 5, 3, 6).flatten(5, 6).flatten(3, 4)
        else:
            output = hidden_states.reshape(
                batch_size, (num_frames + p_t - 1) // p_t, height // p, width // p, -1, p_t, p, p
            )
            output = output.permute(0, 1, 5, 4, 2, 6, 3, 7).flatten(6, 7).flatten(4, 5).flatten(1, 2)

        if not return_dict:
            return (output,)
        return Transformer2DModelOutput(sample=output)
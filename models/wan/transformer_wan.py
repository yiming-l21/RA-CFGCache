# Copyright 2025 The Wan Team and The HuggingFace Team. All rights reserved.
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

import math
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.configuration_utils import ConfigMixin, register_to_config
from diffusers.loaders import FromOriginalModelMixin, PeftAdapterMixin
from diffusers.utils import apply_lora_scale, deprecate, logging
from diffusers.utils.torch_utils import maybe_allow_in_graph
from diffusers.models._modeling_parallel import ContextParallelInput, ContextParallelOutput
from diffusers.models.attention import AttentionMixin, AttentionModuleMixin, FeedForward
from diffusers.models.attention_dispatch import dispatch_attention_fn
from diffusers.models.cache_utils import CacheMixin
from diffusers.models.embeddings import PixArtAlphaTextProjection, TimestepEmbedding, Timesteps, get_1d_rotary_pos_embed
from diffusers.models.modeling_outputs import Transformer2DModelOutput
from diffusers.models.modeling_utils import ModelMixin
from diffusers.models.normalization import FP32LayerNorm
from .cache_functions import cal_type
from .cache_functions.force_init import force_init
from .taylor_utils import taylor_formula, derivative_approximation, taylor_cache_init

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


def _get_qkv_projections(attn: "WanAttention", hidden_states: torch.Tensor, encoder_hidden_states: torch.Tensor):
    # encoder_hidden_states is only passed for cross-attention
    if encoder_hidden_states is None:
        encoder_hidden_states = hidden_states

    if attn.fused_projections:
        if not attn.is_cross_attention:
            # In self-attention layers, we can fuse the entire QKV projection into a single linear
            query, key, value = attn.to_qkv(hidden_states).chunk(3, dim=-1)
        else:
            # In cross-attention layers, we can only fuse the KV projections into a single linear
            query = attn.to_q(hidden_states)
            key, value = attn.to_kv(encoder_hidden_states).chunk(2, dim=-1)
    else:
        query = attn.to_q(hidden_states)
        key = attn.to_k(encoder_hidden_states)
        value = attn.to_v(encoder_hidden_states)
    return query, key, value


def _get_added_kv_projections(attn: "WanAttention", encoder_hidden_states_img: torch.Tensor):
    if attn.fused_projections:
        key_img, value_img = attn.to_added_kv(encoder_hidden_states_img).chunk(2, dim=-1)
    else:
        key_img = attn.add_k_proj(encoder_hidden_states_img)
        value_img = attn.add_v_proj(encoder_hidden_states_img)
    return key_img, value_img


class WanAttnProcessor:
    _attention_backend = None
    _parallel_config = None

    def __init__(self):
        if not hasattr(F, "scaled_dot_product_attention"):
            raise ImportError(
                "WanAttnProcessor requires PyTorch 2.0. To use it, please upgrade PyTorch to version 2.0 or higher."
            )

    def __call__(
        self,
        attn: "WanAttention",
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> torch.Tensor:
        encoder_hidden_states_img = None
        if attn.add_k_proj is not None:
            # 512 is the context length of the text encoder, hardcoded for now
            image_context_length = encoder_hidden_states.shape[1] - 512
            encoder_hidden_states_img = encoder_hidden_states[:, :image_context_length]
            encoder_hidden_states = encoder_hidden_states[:, image_context_length:]

        query, key, value = _get_qkv_projections(attn, hidden_states, encoder_hidden_states)

        query = attn.norm_q(query)
        key = attn.norm_k(key)

        query = query.unflatten(2, (attn.heads, -1))
        key = key.unflatten(2, (attn.heads, -1))
        value = value.unflatten(2, (attn.heads, -1))

        if rotary_emb is not None:

            def apply_rotary_emb(
                hidden_states: torch.Tensor,
                freqs_cos: torch.Tensor,
                freqs_sin: torch.Tensor,
            ):
                x1, x2 = hidden_states.unflatten(-1, (-1, 2)).unbind(-1)
                cos = freqs_cos[..., 0::2]
                sin = freqs_sin[..., 1::2]
                out = torch.empty_like(hidden_states)
                out[..., 0::2] = x1 * cos - x2 * sin
                out[..., 1::2] = x1 * sin + x2 * cos
                return out.type_as(hidden_states)

            query = apply_rotary_emb(query, *rotary_emb)
            key = apply_rotary_emb(key, *rotary_emb)

        # I2V task
        hidden_states_img = None
        if encoder_hidden_states_img is not None:
            key_img, value_img = _get_added_kv_projections(attn, encoder_hidden_states_img)
            key_img = attn.norm_added_k(key_img)

            key_img = key_img.unflatten(2, (attn.heads, -1))
            value_img = value_img.unflatten(2, (attn.heads, -1))

            hidden_states_img = dispatch_attention_fn(
                query,
                key_img,
                value_img,
                attn_mask=None,
                dropout_p=0.0,
                is_causal=False,
                backend=self._attention_backend,
                # Reference: https://github.com/huggingface/diffusers/pull/12909
                parallel_config=None,
            )
            hidden_states_img = hidden_states_img.flatten(2, 3)
            hidden_states_img = hidden_states_img.type_as(query)

        hidden_states = dispatch_attention_fn(
            query,
            key,
            value,
            attn_mask=attention_mask,
            dropout_p=0.0,
            is_causal=False,
            backend=self._attention_backend,
            # Reference: https://github.com/huggingface/diffusers/pull/12909
            parallel_config=(self._parallel_config if encoder_hidden_states is None else None),
        )
        hidden_states = hidden_states.flatten(2, 3)
        hidden_states = hidden_states.type_as(query)

        if hidden_states_img is not None:
            hidden_states = hidden_states + hidden_states_img

        hidden_states = attn.to_out[0](hidden_states)
        hidden_states = attn.to_out[1](hidden_states)
        return hidden_states


class WanAttnProcessor2_0:
    def __new__(cls, *args, **kwargs):
        deprecation_message = (
            "The WanAttnProcessor2_0 class is deprecated and will be removed in a future version. "
            "Please use WanAttnProcessor instead. "
        )
        deprecate("WanAttnProcessor2_0", "1.0.0", deprecation_message, standard_warn=False)
        return WanAttnProcessor(*args, **kwargs)


class WanAttention(torch.nn.Module, AttentionModuleMixin):
    _default_processor_cls = WanAttnProcessor
    _available_processors = [WanAttnProcessor]

    def __init__(
        self,
        dim: int,
        heads: int = 8,
        dim_head: int = 64,
        eps: float = 1e-5,
        dropout: float = 0.0,
        added_kv_proj_dim: int | None = None,
        cross_attention_dim_head: int | None = None,
        processor=None,
        is_cross_attention=None,
    ):
        super().__init__()

        self.inner_dim = dim_head * heads
        self.heads = heads
        self.added_kv_proj_dim = added_kv_proj_dim
        self.cross_attention_dim_head = cross_attention_dim_head
        self.kv_inner_dim = self.inner_dim if cross_attention_dim_head is None else cross_attention_dim_head * heads

        self.to_q = torch.nn.Linear(dim, self.inner_dim, bias=True)
        self.to_k = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_v = torch.nn.Linear(dim, self.kv_inner_dim, bias=True)
        self.to_out = torch.nn.ModuleList(
            [
                torch.nn.Linear(self.inner_dim, dim, bias=True),
                torch.nn.Dropout(dropout),
            ]
        )
        self.norm_q = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)
        self.norm_k = torch.nn.RMSNorm(dim_head * heads, eps=eps, elementwise_affine=True)

        self.add_k_proj = self.add_v_proj = None
        if added_kv_proj_dim is not None:
            self.add_k_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.add_v_proj = torch.nn.Linear(added_kv_proj_dim, self.inner_dim, bias=True)
            self.norm_added_k = torch.nn.RMSNorm(dim_head * heads, eps=eps)

        if is_cross_attention is not None:
            self.is_cross_attention = is_cross_attention
        else:
            self.is_cross_attention = cross_attention_dim_head is not None

        self.set_processor(processor)

    def fuse_projections(self):
        if getattr(self, "fused_projections", False):
            return

        if not self.is_cross_attention:
            concatenated_weights = torch.cat([self.to_q.weight.data, self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_q.bias.data, self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_qkv = nn.Linear(in_features, out_features, bias=True)
            self.to_qkv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )
        else:
            concatenated_weights = torch.cat([self.to_k.weight.data, self.to_v.weight.data])
            concatenated_bias = torch.cat([self.to_k.bias.data, self.to_v.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        if self.added_kv_proj_dim is not None:
            concatenated_weights = torch.cat([self.add_k_proj.weight.data, self.add_v_proj.weight.data])
            concatenated_bias = torch.cat([self.add_k_proj.bias.data, self.add_v_proj.bias.data])
            out_features, in_features = concatenated_weights.shape
            with torch.device("meta"):
                self.to_added_kv = nn.Linear(in_features, out_features, bias=True)
            self.to_added_kv.load_state_dict(
                {"weight": concatenated_weights, "bias": concatenated_bias}, strict=True, assign=True
            )

        self.fused_projections = True

    @torch.no_grad()
    def unfuse_projections(self):
        if not getattr(self, "fused_projections", False):
            return

        if hasattr(self, "to_qkv"):
            delattr(self, "to_qkv")
        if hasattr(self, "to_kv"):
            delattr(self, "to_kv")
        if hasattr(self, "to_added_kv"):
            delattr(self, "to_added_kv")

        self.fused_projections = False

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        rotary_emb: tuple[torch.Tensor, torch.Tensor] | None = None,
        **kwargs,
    ) -> torch.Tensor:
        return self.processor(self, hidden_states, encoder_hidden_states, attention_mask, rotary_emb, **kwargs)


class WanImageEmbedding(torch.nn.Module):
    def __init__(self, in_features: int, out_features: int, pos_embed_seq_len=None):
        super().__init__()

        self.norm1 = FP32LayerNorm(in_features)
        self.ff = FeedForward(in_features, out_features, mult=1, activation_fn="gelu")
        self.norm2 = FP32LayerNorm(out_features)
        if pos_embed_seq_len is not None:
            self.pos_embed = nn.Parameter(torch.zeros(1, pos_embed_seq_len, in_features))
        else:
            self.pos_embed = None

    def forward(self, encoder_hidden_states_image: torch.Tensor) -> torch.Tensor:
        if self.pos_embed is not None:
            batch_size, seq_len, embed_dim = encoder_hidden_states_image.shape
            encoder_hidden_states_image = encoder_hidden_states_image.view(-1, 2 * seq_len, embed_dim)
            encoder_hidden_states_image = encoder_hidden_states_image + self.pos_embed

        hidden_states = self.norm1(encoder_hidden_states_image)
        hidden_states = self.ff(hidden_states)
        hidden_states = self.norm2(hidden_states)
        return hidden_states


class WanTimeTextImageEmbedding(nn.Module):
    def __init__(
        self,
        dim: int,
        time_freq_dim: int,
        time_proj_dim: int,
        text_embed_dim: int,
        image_embed_dim: int | None = None,
        pos_embed_seq_len: int | None = None,
    ):
        super().__init__()

        self.timesteps_proj = Timesteps(num_channels=time_freq_dim, flip_sin_to_cos=True, downscale_freq_shift=0)
        self.time_embedder = TimestepEmbedding(in_channels=time_freq_dim, time_embed_dim=dim)
        self.act_fn = nn.SiLU()
        self.time_proj = nn.Linear(dim, time_proj_dim)
        self.text_embedder = PixArtAlphaTextProjection(text_embed_dim, dim, act_fn="gelu_tanh")

        self.image_embedder = None
        if image_embed_dim is not None:
            self.image_embedder = WanImageEmbedding(image_embed_dim, dim, pos_embed_seq_len=pos_embed_seq_len)

    def forward(
        self,
        timestep: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        timestep_seq_len: int | None = None,
    ):
        timestep = self.timesteps_proj(timestep)
        if timestep_seq_len is not None:
            timestep = timestep.unflatten(0, (-1, timestep_seq_len))

        time_embedder_dtype = next(iter(self.time_embedder.parameters())).dtype
        if timestep.dtype != time_embedder_dtype and time_embedder_dtype != torch.int8:
            timestep = timestep.to(time_embedder_dtype)
        temb = self.time_embedder(timestep).type_as(encoder_hidden_states)
        timestep_proj = self.time_proj(self.act_fn(temb))

        encoder_hidden_states = self.text_embedder(encoder_hidden_states)
        if encoder_hidden_states_image is not None:
            encoder_hidden_states_image = self.image_embedder(encoder_hidden_states_image)

        return temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image


class WanRotaryPosEmbed(nn.Module):
    def __init__(
        self,
        attention_head_dim: int,
        patch_size: tuple[int, int, int],
        max_seq_len: int,
        theta: float = 10000.0,
    ):
        super().__init__()

        self.attention_head_dim = attention_head_dim
        self.patch_size = patch_size
        self.max_seq_len = max_seq_len

        h_dim = w_dim = 2 * (attention_head_dim // 6)
        t_dim = attention_head_dim - h_dim - w_dim

        self.t_dim = t_dim
        self.h_dim = h_dim
        self.w_dim = w_dim

        freqs_dtype = torch.float32 if torch.backends.mps.is_available() else torch.float64

        freqs_cos = []
        freqs_sin = []

        for dim in [t_dim, h_dim, w_dim]:
            freq_cos, freq_sin = get_1d_rotary_pos_embed(
                dim,
                max_seq_len,
                theta,
                use_real=True,
                repeat_interleave_real=True,
                freqs_dtype=freqs_dtype,
            )
            freqs_cos.append(freq_cos)
            freqs_sin.append(freq_sin)

        self.register_buffer("freqs_cos", torch.cat(freqs_cos, dim=1), persistent=False)
        self.register_buffer("freqs_sin", torch.cat(freqs_sin, dim=1), persistent=False)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.patch_size
        ppf, pph, ppw = num_frames // p_t, height // p_h, width // p_w

        split_sizes = [self.t_dim, self.h_dim, self.w_dim]

        freqs_cos = self.freqs_cos.split(split_sizes, dim=1)
        freqs_sin = self.freqs_sin.split(split_sizes, dim=1)

        freqs_cos_f = freqs_cos[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_h = freqs_cos[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_cos_w = freqs_cos[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_sin_f = freqs_sin[0][:ppf].view(ppf, 1, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_h = freqs_sin[1][:pph].view(1, pph, 1, -1).expand(ppf, pph, ppw, -1)
        freqs_sin_w = freqs_sin[2][:ppw].view(1, 1, ppw, -1).expand(ppf, pph, ppw, -1)

        freqs_cos = torch.cat([freqs_cos_f, freqs_cos_h, freqs_cos_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)
        freqs_sin = torch.cat([freqs_sin_f, freqs_sin_h, freqs_sin_w], dim=-1).reshape(1, ppf * pph * ppw, 1, -1)

        return freqs_cos, freqs_sin

@maybe_allow_in_graph
class WanTransformerBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        ffn_dim: int,
        num_heads: int,
        qk_norm: str = "rms_norm_across_heads",
        cross_attn_norm: bool = False,
        eps: float = 1e-6,
        added_kv_proj_dim: int | None = None,
    ):
        super().__init__()

        # 1. Self-attention
        self.norm1 = FP32LayerNorm(dim, eps, elementwise_affine=False)
        self.attn1 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            cross_attention_dim_head=None,
            processor=WanAttnProcessor(),
        )

        # 2. Cross-attention
        self.attn2 = WanAttention(
            dim=dim,
            heads=num_heads,
            dim_head=dim // num_heads,
            eps=eps,
            added_kv_proj_dim=added_kv_proj_dim,
            cross_attention_dim_head=dim // num_heads,
            processor=WanAttnProcessor(),
        )
        self.norm2 = FP32LayerNorm(dim, eps, elementwise_affine=True) if cross_attn_norm else nn.Identity()

        # 3. Feed-forward
        self.ffn = FeedForward(dim, inner_dim=ffn_dim, activation_fn="gelu-approximate")
        self.norm3 = FP32LayerNorm(dim, eps, elementwise_affine=False)

        self.scale_shift_table = nn.Parameter(torch.randn(1, 6, dim) / dim**0.5)

    def _get_modulation(self, temb: torch.Tensor):
        if temb.ndim == 4:
            # temb: batch_size, seq_len, 6, inner_dim
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table.unsqueeze(0) + temb.float()
            ).chunk(6, dim=2)

            shift_msa = shift_msa.squeeze(2)
            scale_msa = scale_msa.squeeze(2)
            gate_msa = gate_msa.squeeze(2)
            c_shift_msa = c_shift_msa.squeeze(2)
            c_scale_msa = c_scale_msa.squeeze(2)
            c_gate_msa = c_gate_msa.squeeze(2)
        else:
            # temb: batch_size, 6, inner_dim
            shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = (
                self.scale_shift_table + temb.float()
            ).chunk(6, dim=1)

        return shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa

    def _forward_original(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
    ) -> torch.Tensor:
        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self._get_modulation(temb)

        # 1. Self-attention
        norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(hidden_states)
        attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)
        hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

        # 2. Cross-attention
        norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
        attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)
        hidden_states = hidden_states + attn_output

        # 3. Feed-forward
        norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
            hidden_states
        )
        ff_output = self.ffn(norm_hidden_states)
        hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)

        return hidden_states

    def forward(
        self,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        temb: torch.Tensor,
        rotary_emb: torch.Tensor,
        cache_dic: dict[str, Any] | None = None,
        current: dict[str, Any] | None = None,
    ) -> torch.Tensor:
        # No cache or whole-transformer cache modes:
        # TeaCache / DiCache / MagCache / CFGCache are handled at transformer level.
        if cache_dic is None or current is None:
            return self._forward_original(hidden_states, encoder_hidden_states, temb, rotary_emb)

        use_taylor_block = bool(
            cache_dic.get("taylor_cache", False)
            or cache_dic.get("use_grouped_taylor", False)
            or cache_dic.get("use_hicache", False)
        )

        if not use_taylor_block:
            return self._forward_original(hidden_states, encoder_hidden_states, temb, rotary_emb)

        current["stream"] = "single_stream"
        block_type = current.get("type", "full")

        shift_msa, scale_msa, gate_msa, c_shift_msa, c_scale_msa, c_gate_msa = self._get_modulation(temb)

        # ==================================================
        # full path: compute true outputs and update Taylor cache
        # ==================================================
        if block_type in {"full", "Delta-Cache"}:
            # -------- self-attn --------
            norm_hidden_states = (self.norm1(hidden_states.float()) * (1 + scale_msa) + shift_msa).type_as(
                hidden_states
            )
            attn_output = self.attn1(norm_hidden_states, None, None, rotary_emb)

            current["block"] = "img_attn"
            force_init(cache_dic=cache_dic, current=current, tokens=hidden_states)
            taylor_cache_init(cache_dic=cache_dic, current=current)
            if cache_dic.get("taylor_cache", False) or cache_dic.get("enable_feature_collection", False):
                derivative_approximation(
                    cache_dic=cache_dic,
                    current=current,
                    feature=attn_output,
                )

            hidden_states = (hidden_states.float() + attn_output * gate_msa).type_as(hidden_states)

            # -------- cross-attn --------
            norm_hidden_states = self.norm2(hidden_states.float()).type_as(hidden_states)
            cross_attn_output = self.attn2(norm_hidden_states, encoder_hidden_states, None, None)

            current["block"] = "img_cross_attn"
            force_init(cache_dic=cache_dic, current=current, tokens=hidden_states)
            taylor_cache_init(cache_dic=cache_dic, current=current)
            if cache_dic.get("taylor_cache", False) or cache_dic.get("enable_feature_collection", False):
                derivative_approximation(
                    cache_dic=cache_dic,
                    current=current,
                    feature=cross_attn_output,
                )

            hidden_states = hidden_states + cross_attn_output

            # -------- mlp --------
            norm_hidden_states = (self.norm3(hidden_states.float()) * (1 + c_scale_msa) + c_shift_msa).type_as(
                hidden_states
            )
            ff_output = self.ffn(norm_hidden_states)

            current["block"] = "img_mlp"
            force_init(cache_dic=cache_dic, current=current, tokens=hidden_states)
            taylor_cache_init(cache_dic=cache_dic, current=current)
            if cache_dic.get("taylor_cache", False) or cache_dic.get("enable_feature_collection", False):
                derivative_approximation(
                    cache_dic=cache_dic,
                    current=current,
                    feature=ff_output,
                )

            hidden_states = (hidden_states.float() + ff_output.float() * c_gate_msa).type_as(hidden_states)
            return hidden_states

        # ==================================================
        # Taylor cache path: reuse Taylor predictions
        # ==================================================
        if block_type in {"taylor_cache", "grouped_taylor_cache"}:
            # Taylor path must preserve the incoming hidden_states dtype.
            # gate_msa / c_gate_msa are fp32 because _get_modulation uses temb.float().
            # Without explicit casting, hidden_states will be promoted to fp32,
            # then proj_out bf16 will crash with Float vs BFloat16.
            hs_dtype = hidden_states.dtype
            hs_device = hidden_states.device

            current["block"] = "img_attn"
            pred_attn = taylor_formula(
                cache_dic=cache_dic,
                current=current,
            ).to(device=hs_device)
            hidden_states = (
                hidden_states.float()
                + gate_msa.to(device=hs_device).float() * pred_attn.float()
            ).to(dtype=hs_dtype)

            current["block"] = "img_cross_attn"
            pred_cross = taylor_formula(
                cache_dic=cache_dic,
                current=current,
            ).to(device=hs_device)
            hidden_states = (
                hidden_states.float()
                + pred_cross.float()
            ).to(dtype=hs_dtype)

            current["block"] = "img_mlp"
            pred_mlp = taylor_formula(
                cache_dic=cache_dic,
                current=current,
            ).to(device=hs_device)
            hidden_states = (
                hidden_states.float()
                + c_gate_msa.to(device=hs_device).float() * pred_mlp.float()
            ).to(dtype=hs_dtype)

            return hidden_states

        return self._forward_original(hidden_states, encoder_hidden_states, temb, rotary_emb)


class WanTransformer3DModel(
    ModelMixin, ConfigMixin, PeftAdapterMixin, FromOriginalModelMixin, CacheMixin, AttentionMixin
):
    r"""
    A Transformer model for video-like data used in the Wan model.

    Args:
        patch_size (`tuple[int]`, defaults to `(1, 2, 2)`):
            3D patch dimensions for video embedding (t_patch, h_patch, w_patch).
        num_attention_heads (`int`, defaults to `40`):
            Fixed length for text embeddings.
        attention_head_dim (`int`, defaults to `128`):
            The number of channels in each head.
        in_channels (`int`, defaults to `16`):
            The number of channels in the input.
        out_channels (`int`, defaults to `16`):
            The number of channels in the output.
        text_dim (`int`, defaults to `512`):
            Input dimension for text embeddings.
        freq_dim (`int`, defaults to `256`):
            Dimension for sinusoidal time embeddings.
        ffn_dim (`int`, defaults to `13824`):
            Intermediate dimension in feed-forward network.
        num_layers (`int`, defaults to `40`):
            The number of layers of transformer blocks to use.
        window_size (`tuple[int]`, defaults to `(-1, -1)`):
            Window size for local attention (-1 indicates global attention).
        cross_attn_norm (`bool`, defaults to `True`):
            Enable cross-attention normalization.
        qk_norm (`bool`, defaults to `True`):
            Enable query/key normalization.
        eps (`float`, defaults to `1e-6`):
            Epsilon value for normalization layers.
        add_img_emb (`bool`, defaults to `False`):
            Whether to use img_emb.
        added_kv_proj_dim (`int`, *optional*, defaults to `None`):
            The number of channels to use for the added key and value projections. If `None`, no projection is used.
    """

    _supports_gradient_checkpointing = True
    _skip_layerwise_casting_patterns = ["patch_embedding", "condition_embedder", "norm"]
    _no_split_modules = ["WanTransformerBlock"]
    _keep_in_fp32_modules = ["time_embedder", "scale_shift_table", "norm1", "norm2", "norm3"]
    _keys_to_ignore_on_load_unexpected = ["norm_added_q"]
    _repeated_blocks = ["WanTransformerBlock"]
    _cp_plan = {
        "rope": {
            0: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
            1: ContextParallelInput(split_dim=1, expected_dims=4, split_output=True),
        },
        "blocks.0": {
            "hidden_states": ContextParallelInput(split_dim=1, expected_dims=3, split_output=False),
        },
        # Reference: https://github.com/huggingface/diffusers/pull/12909
        # We need to disable the splitting of encoder_hidden_states because the image_encoder
        # (Wan 2.1 I2V) consistently generates 257 tokens for image_embed. This causes the shape
        # of encoder_hidden_states—whose token count is always 769 (512 + 257) after concatenation
        # —to be indivisible by the number of devices in the CP.
        "proj_out": ContextParallelOutput(gather_dim=1, expected_dims=3),
        "": {
            "timestep": ContextParallelInput(split_dim=1, expected_dims=2, split_output=False),
        },
    }

    @register_to_config
    def __init__(
        self,
        patch_size: tuple[int, ...] = (1, 2, 2),
        num_attention_heads: int = 40,
        attention_head_dim: int = 128,
        in_channels: int = 16,
        out_channels: int = 16,
        text_dim: int = 4096,
        freq_dim: int = 256,
        ffn_dim: int = 13824,
        num_layers: int = 40,
        cross_attn_norm: bool = True,
        qk_norm: str | None = "rms_norm_across_heads",
        eps: float = 1e-6,
        image_dim: int | None = None,
        added_kv_proj_dim: int | None = None,
        rope_max_seq_len: int = 1024,
        pos_embed_seq_len: int | None = None,
    ) -> None:
        super().__init__()

        inner_dim = num_attention_heads * attention_head_dim
        out_channels = out_channels or in_channels

        # 1. Patch & position embedding
        self.rope = WanRotaryPosEmbed(attention_head_dim, patch_size, rope_max_seq_len)
        self.patch_embedding = nn.Conv3d(in_channels, inner_dim, kernel_size=patch_size, stride=patch_size)

        # 2. Condition embeddings
        # image_embedding_dim=1280 for I2V model
        self.condition_embedder = WanTimeTextImageEmbedding(
            dim=inner_dim,
            time_freq_dim=freq_dim,
            time_proj_dim=inner_dim * 6,
            text_embed_dim=text_dim,
            image_embed_dim=image_dim,
            pos_embed_seq_len=pos_embed_seq_len,
        )

        # 3. Transformer blocks
        self.blocks = nn.ModuleList(
            [
                WanTransformerBlock(
                    inner_dim, ffn_dim, num_attention_heads, qk_norm, cross_attn_norm, eps, added_kv_proj_dim
                )
                for _ in range(num_layers)
            ]
        )

        # 4. Output norm & projection
        self.norm_out = FP32LayerNorm(inner_dim, eps, elementwise_affine=False)
        self.proj_out = nn.Linear(inner_dim, out_channels * math.prod(patch_size))
        self.scale_shift_table = nn.Parameter(torch.randn(1, 2, inner_dim) / inner_dim**0.5)

        self.gradient_checkpointing = False

    def _cache_rel_l1(self,curr: torch.Tensor, prev: torch.Tensor, eps: float = 1e-6) -> float:
        denom = prev.abs().mean()
        if float(denom) == 0.0:
            return 0.0
        val = (curr - prev).abs().mean() / (denom + eps)
        return float(val.detach().cpu())

    def _cache_get_branch_state(self, cache_dic: dict[str, Any], current: dict[str, Any], key: str):
        branch = current.get("model", "cond")
        return branch, cache_dic[key][branch]

    def _cfgcache_get_branch_state(self, cache_dic: dict[str, Any], current: dict[str, Any]):
        cc = cache_dic["cfgcache"]
        branch = current.get("model", "cond")
        return branch, cc[branch], cc

    def _cfgcache_prepare_light_probe(
        self,
        *,
        timestep_proj: torch.Tensor,
        cache_dic: dict[str, Any],
        current: dict[str, Any],
    ):
        branch, bc, cc = self._cfgcache_get_branch_state(cache_dic, current)

        probe_modulated_inp = timestep_proj.detach()
        bc["_probe_modulated_inp"] = probe_modulated_inp
        bc["_probe_emb"] = probe_modulated_inp
        bc["_probe_summary"] = {"modulated_inp": probe_modulated_inp}
        bc["_probe_payload"] = {
            "probe_depth": 0,
            "dicache_dx": None,
            "dicache_dy": None,
        }

        bc["_probe_hidden_states"] = None
        bc["_probe_img"] = None
        bc["_probe_depth"] = 0

        return bc["_probe_payload"]

    def _cfgcache_prepare_probe(
        self,
        *,
        hidden_states: torch.Tensor,
        encoder_hidden_states: torch.Tensor,
        timestep_proj: torch.Tensor,
        rotary_emb,
        cache_dic: dict[str, Any],
        current: dict[str, Any],
    ):
        branch, bc, cc = self._cfgcache_get_branch_state(cache_dic, current)

        proxy = cc.get("_proxy_obj", None)
        probe_mode = getattr(proxy, "probe_mode", cc.get("probe_mode", "light"))
        probe_mode = "light" if probe_mode is None else str(probe_mode).lower()

        if probe_mode in {"modulated_inp", "emb", "light", "light_signal", "tea", "teacache"}:
            return self._cfgcache_prepare_light_probe(
                timestep_proj=timestep_proj,
                cache_dic=cache_dic,
                current=current,
            )

        bc["_probe_modulated_inp"] = timestep_proj.detach()

        prev_inp = bc.get("proxy_prev_input", None)
        prev_probe = bc.get("proxy_prev_probe_states", None)

        depth = min(int(cc.get("probe_depth", 2)), len(self.blocks))
        test_hidden_states = hidden_states.clone()

        old_type = current.get("type", None)
        old_layer = current.get("layer", None)

        for probe_i in range(depth):
            current["type"] = "full"
            current["layer"] = probe_i
            test_hidden_states = self.blocks[probe_i](
                hidden_states=test_hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                temb=timestep_proj,
                rotary_emb=rotary_emb,
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
            dx = self._cache_rel_l1(hidden_states.detach(), prev_inp, eps)

        if prev_probe is not None:
            dy = self._cache_rel_l1(test_hidden_states.detach(), prev_probe, eps)

        bc["_probe_hidden_states"] = test_hidden_states.detach()
        bc["_probe_img"] = bc["_probe_hidden_states"]
        bc["_probe_depth"] = depth
        bc["_probe_summary"] = {"modulated_inp": bc["_probe_modulated_inp"]}
        bc["_probe_payload"] = {
            "probe_depth": depth,
            "dicache_dx": dx,
            "dicache_dy": dy,
        }

        return bc["_probe_payload"]
    
    @apply_lora_scale("attention_kwargs")
    def forward(
        self,
        hidden_states: torch.Tensor,
        timestep: torch.LongTensor,
        encoder_hidden_states: torch.Tensor,
        encoder_hidden_states_image: torch.Tensor | None = None,
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        cache_dic: dict[str, Any] | None = None,
        current: dict[str, Any] | None = None,
        cfgcache_only_probe: bool = False,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        batch_size, num_channels, num_frames, height, width = hidden_states.shape
        p_t, p_h, p_w = self.config.patch_size
        post_patch_num_frames = num_frames // p_t
        post_patch_height = height // p_h
        post_patch_width = width // p_w

        rotary_emb = self.rope(hidden_states)

        hidden_states = self.patch_embedding(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)

        # timestep shape: batch_size, or batch_size, seq_len
        if timestep.ndim == 2:
            ts_seq_len = timestep.shape[1]
            timestep = timestep.flatten()
        else:
            ts_seq_len = None

        temb, timestep_proj, encoder_hidden_states, encoder_hidden_states_image = self.condition_embedder(
            timestep,
            encoder_hidden_states,
            encoder_hidden_states_image,
            timestep_seq_len=ts_seq_len,
        )

        if ts_seq_len is not None:
            timestep_proj = timestep_proj.unflatten(2, (6, -1))
        else:
            timestep_proj = timestep_proj.unflatten(1, (6, -1))

        if encoder_hidden_states_image is not None:
            encoder_hidden_states = torch.concat([encoder_hidden_states_image, encoder_hidden_states], dim=1)

        cache_mode = None if cache_dic is None else cache_dic.get("mode", None)

        # Probe-only path for CFGCache.
        if (
            cache_mode == "CFGCache"
            and cache_dic is not None
            and current is not None
            and cfgcache_only_probe
        ):
            self._cfgcache_prepare_probe(
                hidden_states=hidden_states,
                encoder_hidden_states=encoder_hidden_states,
                timestep_proj=timestep_proj,
                rotary_emb=rotary_emb,
                cache_dic=cache_dic,
                current=current,
            )
            if not return_dict:
                return (hidden_states,)
            return Transformer2DModelOutput(sample=hidden_states)

        run_transformer_blocks = True
        start_block_idx = 0

        # TeaCache
        save_teacache_residual = False
        teacache_state = None
        tea_ori_hidden_states = None

        # DiCache
        save_dicache_residual = False
        dicache_state = None
        di_base_hidden_states = None
        di_ori_hidden_states = None
        di_probe_depth = 0
        di_capture_probe_at = None

        # MagCache
        save_magcache_residual = False
        magcache_state = None
        mag_base_hidden_states = None
        mag_ori_hidden_states = None

        # CFGCache
        save_cfgcache_residual = False
        cfgcache_state = None
        cfg_base_hidden_states = None
        cfg_ori_hidden_states = None
        cfg_probe_depth = 0
        cfg_capture_probe_at = None

        # =========================================================
        # Whole-transformer cache dispatch
        # =========================================================
        if cache_dic is not None and current is not None:
            # -------------------------
            # TeaCache
            # -------------------------
            if cache_mode == "TeaCache" and cache_dic.get("teacache_enable", True):
                branch, teacache_state = self._cache_get_branch_state(cache_dic, current, "teacache")
                current["teacache_modulated_inp"] = temb.float().detach()

                cal_type(cache_dic=cache_dic, current=current)

                if current.get("type") == "TeaCacheSkip":
                    if teacache_state.get("previous_residual", None) is not None:
                        hidden_states = hidden_states + teacache_state["previous_residual"]
                        run_transformer_blocks = False
                    else:
                        current["type"] = "full"
                        run_transformer_blocks = True

                if run_transformer_blocks:
                    tea_ori_hidden_states = hidden_states.clone()
                    save_teacache_residual = True

            # -------------------------
            # DiCache
            # -------------------------
            elif cache_mode == "DiCache" and cache_dic.get("dicache_enable", True):
                branch, dicache_state = self._cache_get_branch_state(cache_dic, current, "dicache")

                T = int(dicache_state.get("num_steps", current.get("num_steps", 0) or 0))
                cnt = int(dicache_state.get("cnt", 0))
                ret_ratio = float(dicache_state.get("ret_ratio", 0.2))

                di_base_hidden_states = hidden_states.detach()
                dicache_state["base_img"] = di_base_hidden_states
                dicache_state["base_hidden_states"] = di_base_hidden_states

                current["dicache_delta_x"] = None
                current["dicache_delta_y"] = None

                dicache_state["_probe_img"] = None
                dicache_state["_probe_hidden_states"] = None

                di_probe_depth = min(int(dicache_state.get("probe_depth", 1)), len(self.blocks))
                need_probe = not ((cnt <= int(ret_ratio * T)) or (T > 0 and cnt == T - 1))

                if need_probe and di_probe_depth > 0:
                    test_hidden_states = hidden_states.clone()

                    for probe_i in range(di_probe_depth):
                        test_hidden_states = self.blocks[probe_i](
                            hidden_states=test_hidden_states,
                            encoder_hidden_states=encoder_hidden_states,
                            temb=timestep_proj,
                            rotary_emb=rotary_emb,
                        )

                    dicache_state["_probe_img"] = test_hidden_states.detach()
                    dicache_state["_probe_hidden_states"] = test_hidden_states.detach()

                    prev_inp = dicache_state.get("previous_input", None)
                    prev_probe = dicache_state.get("previous_probe_states", None)

                    if prev_inp is not None and prev_probe is not None:
                        eps = 1e-6
                        current["dicache_delta_x"] = self._cache_rel_l1(hidden_states.detach(), prev_inp, eps)
                        current["dicache_delta_y"] = self._cache_rel_l1(test_hidden_states.detach(), prev_probe, eps)

                cal_type(cache_dic=cache_dic, current=current)

                if current.get("type") == "DiCacheSkip":
                    if dicache_state.get("previous_residual", None) is not None:
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
                                ).clamp(1.0, 2.0)
                            else:
                                gamma = di_base_hidden_states.new_tensor(1.0)

                            aligned_residual = (
                                dicache_state["residual_window"][-2]
                                + gamma * (
                                    dicache_state["residual_window"][-1]
                                    - dicache_state["residual_window"][-2]
                                )
                            )
                            hidden_states = di_base_hidden_states + aligned_residual
                        else:
                            hidden_states = di_base_hidden_states + dicache_state["previous_residual"]

                        dicache_state["previous_input"] = di_base_hidden_states

                        if dicache_state.get("_probe_hidden_states", None) is not None:
                            dicache_state["previous_probe_states"] = dicache_state["_probe_hidden_states"].detach()

                        dicache_state["_probe_img"] = None
                        dicache_state["_probe_hidden_states"] = None

                        run_transformer_blocks = False
                    else:
                        current["type"] = "full"
                        current["dicache_resume"] = False
                        run_transformer_blocks = True

                if run_transformer_blocks:
                    di_ori_hidden_states = di_base_hidden_states.clone()
                    save_dicache_residual = True

                    if (
                        current.get("dicache_resume", False)
                        and dicache_state.get("_probe_hidden_states", None) is not None
                    ):
                        hidden_states = dicache_state["_probe_hidden_states"]
                        start_block_idx = di_probe_depth
                    else:
                        start_block_idx = 0

                    if di_probe_depth > 0 and start_block_idx == 0:
                        di_capture_probe_at = di_probe_depth - 1

            # -------------------------
            # MagCache
            # -------------------------
            elif cache_mode == "MagCache" and cache_dic.get("magcache_enable", True):
                branch, magcache_state = self._cache_get_branch_state(cache_dic, current, "magcache")

                mag_base_hidden_states = hidden_states.detach()
                magcache_state["base_img"] = mag_base_hidden_states
                magcache_state["base_hidden_states"] = mag_base_hidden_states

                cal_type(cache_dic=cache_dic, current=current)

                if current.get("type") == "MagCacheSkip":
                    if magcache_state.get("previous_residual", None) is not None:
                        hidden_states = mag_base_hidden_states + magcache_state["previous_residual"]
                        run_transformer_blocks = False
                    else:
                        current["type"] = "full"
                        run_transformer_blocks = True

                if run_transformer_blocks:
                    mag_ori_hidden_states = mag_base_hidden_states.clone()
                    save_magcache_residual = True

            # -------------------------
            # CFGCache
            # -------------------------
            elif cache_mode == "CFGCache" and cache_dic.get("cfgcache_enable", True):
                branch, cfgcache_state, cc = self._cfgcache_get_branch_state(cache_dic, current)

                cfg_base_hidden_states = hidden_states.detach()
                cfgcache_state["base_hidden_states"] = cfg_base_hidden_states
                cfgcache_state["base_img"] = cfg_base_hidden_states

                current["dicache_delta_x"] = None
                current["dicache_delta_y"] = None

                self._cfgcache_prepare_probe(
                    hidden_states=hidden_states,
                    encoder_hidden_states=encoder_hidden_states,
                    timestep_proj=timestep_proj,
                    rotary_emb=rotary_emb,
                    cache_dic=cache_dic,
                    current=current,
                )

                cal_type(cache_dic=cache_dic, current=current)

                if current.get("type") == "CFGCacheSkip":
                    if cfgcache_state.get("previous_residual", None) is not None:
                        hidden_states = cfg_base_hidden_states + cfgcache_state["previous_residual"]

                        cfgcache_state["proxy_prev_input"] = cfg_base_hidden_states

                        if cfgcache_state.get("_probe_hidden_states", None) is not None:
                            cfgcache_state["proxy_prev_probe_states"] = (
                                cfgcache_state["_probe_hidden_states"].detach()
                            )

                        if cfgcache_state.get("_probe_modulated_inp", None) is not None:
                            cfgcache_state["proxy_prev_modulated_inp"] = (
                                cfgcache_state["_probe_modulated_inp"].detach()
                            )
                        elif cfgcache_state.get("proxy_prev_modulated_inp", None) is None:
                            cfgcache_state["proxy_prev_modulated_inp"] = timestep_proj.detach()

                        cfgcache_state["_probe_img"] = None
                        cfgcache_state["_probe_hidden_states"] = None
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
                    save_cfgcache_residual = True

                    cfg_probe_depth = int(cfgcache_state.get("_probe_depth", 0) or 0)

                    if (
                        current.get("cfgcache_resume", False)
                        and cfgcache_state.get("_probe_hidden_states", None) is not None
                    ):
                        hidden_states = cfgcache_state["_probe_hidden_states"]
                        start_block_idx = cfg_probe_depth
                    else:
                        start_block_idx = 0

                    if cfg_probe_depth > 0 and start_block_idx == 0:
                        cfg_capture_probe_at = cfg_probe_depth - 1

        # -------------------------
        # Taylor / HiCache / GroupedTaylor
        # -------------------------
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
        # Transformer blocks
        # =========================================================
        if run_transformer_blocks:
            if torch.is_grad_enabled() and self.gradient_checkpointing:
                for i in range(start_block_idx, len(self.blocks)):
                    if current is not None:
                        current["layer"] = i

                    hidden_states = self._gradient_checkpointing_func(
                        self.blocks[i],
                        hidden_states,
                        encoder_hidden_states,
                        timestep_proj,
                        rotary_emb,
                    )
            else:
                for i in range(start_block_idx, len(self.blocks)):
                    if current is not None:
                        current["layer"] = i

                    hidden_states = self.blocks[i](
                        hidden_states,
                        encoder_hidden_states,
                        timestep_proj,
                        rotary_emb,
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

                    if (
                        cache_mode == "CFGCache"
                        and cfgcache_state is not None
                        and cfg_capture_probe_at is not None
                        and i == cfg_capture_probe_at
                    ):
                        cfgcache_state["previous_probe_states"] = hidden_states.detach()

        # =========================================================
        # Save residuals
        # =========================================================
        if save_teacache_residual and teacache_state is not None:
            teacache_state["previous_residual"] = (hidden_states - tea_ori_hidden_states).detach()

        if save_dicache_residual and dicache_state is not None:
            dicache_state["previous_residual"] = (hidden_states - di_ori_hidden_states).detach()
            dicache_state["previous_input"] = di_base_hidden_states

            if dicache_state.get("_probe_hidden_states", None) is not None:
                dicache_state["previous_probe_states"] = dicache_state["_probe_hidden_states"].detach()

            if dicache_state.get("previous_probe_states", None) is not None:
                dicache_state["previous_probe_residual"] = (
                    dicache_state["previous_probe_states"] - di_base_hidden_states
                ).detach()

            dicache_state.setdefault("residual_window", []).append(dicache_state["previous_residual"])
            if len(dicache_state["residual_window"]) > 2:
                dicache_state["residual_window"].pop(0)

            if dicache_state.get("previous_probe_residual", None) is not None:
                dicache_state.setdefault("probe_residual_window", []).append(
                    dicache_state["previous_probe_residual"]
                )
                if len(dicache_state["probe_residual_window"]) > 2:
                    dicache_state["probe_residual_window"].pop(0)

            dicache_state["_probe_img"] = None
            dicache_state["_probe_hidden_states"] = None

        if save_magcache_residual and magcache_state is not None:
            magcache_state["previous_residual"] = (hidden_states - mag_ori_hidden_states).detach()
            magcache_state["base_img"] = None
            magcache_state["base_hidden_states"] = None

        if save_cfgcache_residual and cfgcache_state is not None:
            cfgcache_state["previous_residual"] = (hidden_states - cfg_ori_hidden_states).detach()
            cfgcache_state["previous_postnorm_hidden"] = hidden_states.detach()

            cfgcache_state["proxy_prev_input"] = cfg_base_hidden_states

            if cfgcache_state.get("_probe_hidden_states", None) is not None:
                cfgcache_state["proxy_prev_probe_states"] = cfgcache_state["_probe_hidden_states"].detach()
            elif cfgcache_state.get("previous_probe_states", None) is not None:
                cfgcache_state["proxy_prev_probe_states"] = cfgcache_state["previous_probe_states"].detach()

            if cfgcache_state.get("_probe_modulated_inp", None) is not None:
                cfgcache_state["proxy_prev_modulated_inp"] = cfgcache_state["_probe_modulated_inp"].detach()
            else:
                cfgcache_state["proxy_prev_modulated_inp"] = timestep_proj.detach()

            cfgcache_state["proxy_anchor_modulated_inp"] = cfgcache_state["proxy_prev_modulated_inp"]

            if cfgcache_state.get("proxy_prev_probe_states", None) is not None:
                cfgcache_state["proxy_anchor_probe_states"] = cfgcache_state["proxy_prev_probe_states"].detach()

            cfgcache_state["_probe_img"] = None
            cfgcache_state["_probe_hidden_states"] = None
            cfgcache_state["_probe_modulated_inp"] = None
            cfgcache_state["_probe_summary"] = None
            cfgcache_state["_probe_payload"] = None

        # =========================================================
        # Output norm, projection & unpatchify
        # =========================================================
        if temb.ndim == 3:
            shift, scale = (self.scale_shift_table.unsqueeze(0).to(temb.device) + temb.unsqueeze(2)).chunk(2, dim=2)
            shift = shift.squeeze(2)
            scale = scale.squeeze(2)
        else:
            shift, scale = (self.scale_shift_table.to(temb.device) + temb.unsqueeze(1)).chunk(2, dim=1)

        shift = shift.to(hidden_states.device)
        scale = scale.to(hidden_states.device)

        hidden_states = (self.norm_out(hidden_states.float()) * (1 + scale) + shift).type_as(hidden_states)
        hidden_states = self.proj_out(hidden_states)

        hidden_states = hidden_states.reshape(
            batch_size,
            post_patch_num_frames,
            post_patch_height,
            post_patch_width,
            p_t,
            p_h,
            p_w,
            -1,
        )
        hidden_states = hidden_states.permute(0, 7, 1, 4, 2, 5, 3, 6)
        output = hidden_states.flatten(6, 7).flatten(4, 5).flatten(2, 3)

        if not return_dict:
            return (output,)

        return Transformer2DModelOutput(sample=output)
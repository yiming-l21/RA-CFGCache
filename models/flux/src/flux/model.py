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
    Minimal FLUX transformer model for standard inference.
    All cache-related logic has been removed.
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
            raise ValueError(f"Got axes_dim={params.axes_dim}, but expected positional dim {pe_dim}")

        self.hidden_size = params.hidden_size
        self.num_heads = params.num_heads

        self.pe_embedder = EmbedND(
            dim=pe_dim,
            theta=params.theta,
            axes_dim=params.axes_dim,
        )

        self.img_in = nn.Linear(self.in_channels, self.hidden_size, bias=True)
        self.txt_in = nn.Linear(params.context_in_dim, self.hidden_size)

        self.time_in = MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
        self.vector_in = MLPEmbedder(params.vec_in_dim, self.hidden_size)

        self.guidance_in = (
            MLPEmbedder(in_dim=256, hidden_dim=self.hidden_size)
            if params.guidance_embed
            else nn.Identity()
        )

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
                SingleStreamBlock(
                    self.hidden_size,
                    self.num_heads,
                    mlp_ratio=params.mlp_ratio,
                )
                for _ in range(params.depth_single_blocks)
            ]
        )

        self.final_layer = LastLayer(
            self.hidden_size,
            1,
            self.out_channels,
        )

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
        """
        Standard FLUX forward pass.

        Args:
            img: image latent tokens, shape [B, N_img, C_img]
            img_ids: image position ids, shape [B, N_img, ...]
            txt: text tokens, shape [B, N_txt, C_txt]
            txt_ids: text position ids, shape [B, N_txt, ...]
            timesteps: diffusion timesteps, shape [B]
            y: pooled vector conditioning
            guidance: guidance strength, required when params.guidance_embed=True

        Returns:
            Predicted velocity/noise tokens, shape [B, N_img, out_channels]
        """
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input img and txt tensors must have 3 dimensions.")

        # 1. Input projection
        img = self.img_in(img)
        txt = self.txt_in(txt)

        # 2. Time / guidance / vector conditioning
        vec = self.time_in(timestep_embedding(timesteps, 256))

        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Didn't get guidance strength for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256))

        vec = vec + self.vector_in(y)

        # 3. Positional embedding
        ids = torch.cat((txt_ids, img_ids), dim=1)
        pe = self.pe_embedder(ids)

        # 4. Double-stream blocks
        for block in self.double_blocks:
            img, txt = block(
                img=img,
                txt=txt,
                vec=vec,
                pe=pe,
            )

        # 5. Merge text/image streams
        txt_len = txt.shape[1]
        img = torch.cat((txt, img), dim=1)

        # 6. Single-stream blocks
        for block in self.single_blocks:
            img = block(
                img,
                vec=vec,
                pe=pe,
            )

        # 7. Keep image tokens only
        img = img[:, txt_len:, ...]

        # 8. Final projection
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
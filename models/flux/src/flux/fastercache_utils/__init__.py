import torch.fft
import math
from einops import rearrange
import torch
from torch import Tensor
def pack(x: Tensor, height: int, width: int) -> Tensor:
    # x: (b, c, (h*ph), (w*pw))  where ph=pw=2
    return rearrange(
        x,
        "b c (h ph) (w pw) -> b (h w) (c ph pw)",
        h=math.ceil(height / 16),
        w=math.ceil(width / 16),
        ph=2,
        pw=2,
    )
@torch.no_grad()
def _fft_masks(H, W, device, radius_div=5):
    radius = min(H, W) // radius_div
    yy = torch.arange(H, device=device)
    xx = torch.arange(W, device=device)
    Y, X = torch.meshgrid(yy, xx, indexing="ij")
    cy, cx = H // 2, W // 2
    mask = (X - cx) ** 2 + (Y - cy) ** 2 <= radius ** 2
    low = mask[None, None, :, :]          # (1,1,H,W)
    high = (~mask)[None, None, :, :]
    return low, high

@torch.no_grad()
def _to_4d(x, height, width, unpack_fn):
    # x: packed (B, L, D) or already 4D (B,C,h,w)
    if x.ndim == 4:
        return x
    if x.ndim == 3:
        return unpack_fn(x.float(), height, width)  # -> (B,C,h,w)
    raise ValueError(f"Unsupported ndim={x.ndim}")

@torch.no_grad()
def _fft_split(x_4d, radius_div=5):
    # x_4d: (B,C,H,W) real
    B, C, H, W = x_4d.shape
    X = torch.fft.fftshift(torch.fft.fft2(x_4d), dim=(-2, -1))  # complex
    low, high = _fft_masks(H, W, x_4d.device, radius_div=radius_div)
    return X * low, X * high  # complex

@torch.no_grad()
def _ifft_merge(lf, hf):
    X = lf + hf
    x = torch.fft.ifft2(torch.fft.ifftshift(X, dim=(-2, -1)), dim=(-2, -1)).real
    return x

def _get_fr(cache_dic):
    return None if cache_dic is None else cache_dic.get("fastercache", {}).get("reuse", None)

def _layer_key(current: dict, extra: str = "") -> tuple:
    stream = current.get("stream", "unknown")
    layer = int(current.get("layer", -1))
    extra = str(extra) if extra else ""
    return (stream, layer, extra) if extra else (stream, layer)

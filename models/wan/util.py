import os
import time
import contextlib
from types import MethodType

import torch

from .pipeline_wan import WanPipeline

try:
    from .transformer_wan import WanTransformer3DModel as LocalWanTransformer
except Exception:
    try:
        from diffusers.models import WanTransformer3DModel as LocalWanTransformer
    except Exception:
        LocalWanTransformer = None

try:
    from diffusers import AutoencoderKLWan
except Exception:
    AutoencoderKLWan = None

try:
    from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler
except Exception:
    UniPCMultistepScheduler = None


def _bind_method(obj, cls, name: str):
    fn = getattr(cls, name, None)
    if fn is None:
        raise AttributeError(f"{cls.__name__} has no method '{name}'")
    setattr(obj, name, MethodType(fn, obj))


def _ensure_cache_context(model):
    """Wan pipeline calls model.cache_context('cond'/'uncond').

    Some official WanTransformer3DModel builds do not expose cache_context.
    For normal generation / calibration dump, a no-op context keeps the pipeline runnable.
    """
    if model is None:
        return
    if not hasattr(model, "cache_context"):
        def cache_context(_name=""):
            return contextlib.nullcontext()
        setattr(model, "cache_context", cache_context)


def _tensor_sample_stats(tensor: torch.Tensor, max_rows: int = 1024):
    """Compute cheap stats on a small slice to avoid materializing huge UMT5 embeddings."""
    with torch.no_grad():
        t = tensor.detach()
        if t.ndim >= 2:
            t = t[: min(max_rows, t.shape[0])]
        t = t.float().cpu()
        return {
            "mean": t.mean().item(),
            "std": t.std().item(),
            "abs_mean": t.abs().mean().item(),
        }


def _fix_umt5_tied_embeddings(pipe):
    """Fix UMT5 tied embedding mismatch for converted Wan diffusers checkpoints.

    Some converted Wan diffusers checkpoints store token embeddings as
    text_encoder.shared.weight, while text_encoder.encoder.embed_tokens.weight is
    missing / newly initialized. If not fixed, prompt encoding becomes broken and
    generated videos can become gray/blurry.
    """
    text_encoder = getattr(pipe, "text_encoder", None)
    if text_encoder is None:
        return pipe

    if not (
        hasattr(text_encoder, "shared")
        and hasattr(text_encoder, "encoder")
        and hasattr(text_encoder.encoder, "embed_tokens")
    ):
        return pipe

    shared = text_encoder.shared
    embed = text_encoder.encoder.embed_tokens

    try:
        same_ptr = shared.weight.data_ptr() == embed.weight.data_ptr()
    except Exception:
        same_ptr = False

    try:
        shared_stats = _tensor_sample_stats(shared.weight)
        embed_stats = _tensor_sample_stats(embed.weight)
        sample_rows = min(1024, shared.weight.shape[0], embed.weight.shape[0])
        mean_diff = (
            shared.weight[:sample_rows].detach().float().cpu()
            - embed.weight[:sample_rows].detach().float().cpu()
        ).abs().mean().item()
    except Exception as e:
        print(f"[WARN] Failed to compute UMT5 embedding stats: {e!r}", flush=True)
        shared_stats = {"std": float("nan"), "abs_mean": float("nan")}
        embed_stats = {"std": float("nan"), "abs_mean": float("nan")}
        mean_diff = float("inf")

    print(
        "[text-encoder-check] "
        f"shared/embed same_ptr={same_ptr}, "
        f"sample_mean_abs_diff={mean_diff:.6e}, "
        f"shared_std={shared_stats.get('std', float('nan')):.6e}, "
        f"embed_std={embed_stats.get('std', float('nan')):.6e}, "
        f"shared_abs_mean={shared_stats.get('abs_mean', float('nan')):.6e}, "
        f"embed_abs_mean={embed_stats.get('abs_mean', float('nan')):.6e}",
        flush=True,
    )

    need_fix = False
    if not same_ptr:
        need_fix = True
    try:
        if embed_stats.get("std", 0.0) < 1e-8 or embed_stats.get("abs_mean", 0.0) < 1e-8:
            need_fix = True
    except Exception:
        pass

    if need_fix:
        print(
            "[text-encoder-fix] tying text_encoder.encoder.embed_tokens to text_encoder.shared",
            flush=True,
        )
        text_encoder.encoder.embed_tokens = shared

    if hasattr(text_encoder, "tie_weights"):
        try:
            text_encoder.tie_weights()
        except Exception as e:
            print(f"[WARN] text_encoder.tie_weights() failed: {e!r}", flush=True)

    # Some tie_weights implementations may not reassign encoder.embed_tokens.
    # Enforce the tie one more time if needed.
    try:
        if text_encoder.shared.weight.data_ptr() != text_encoder.encoder.embed_tokens.weight.data_ptr():
            text_encoder.encoder.embed_tokens = text_encoder.shared
    except Exception:
        pass

    try:
        shared = text_encoder.shared
        embed = text_encoder.encoder.embed_tokens
        after_same_ptr = shared.weight.data_ptr() == embed.weight.data_ptr()
        after_shared_stats = _tensor_sample_stats(shared.weight)
        after_embed_stats = _tensor_sample_stats(embed.weight)
        print(
            "[text-encoder-check-after] "
            f"shared/embed same_ptr={after_same_ptr}, "
            f"shared_std={after_shared_stats.get('std', float('nan')):.6e}, "
            f"embed_std={after_embed_stats.get('std', float('nan')):.6e}, "
            f"shared_abs_mean={after_shared_stats.get('abs_mean', float('nan')):.6e}, "
            f"embed_abs_mean={after_embed_stats.get('abs_mean', float('nan')):.6e}",
            flush=True,
        )
    except Exception as e:
        print(f"[WARN] Failed to re-check UMT5 tied embeddings: {e!r}", flush=True)

    return pipe


def _patch_one_wan_transformer(
    tr,
    *,
    require_cfgcache: bool = False,
):
    if tr is None:
        return None

    # If there is no local transformer implementation, only make cache_context safe.
    if LocalWanTransformer is None:
        _ensure_cache_context(tr)
        return tr

    # Required for using the local execution path.
    required_methods = [
        "forward",
    ]

    # Required only when you already implemented CFGCache-specific methods
    # in transformer_wan.py.
    cfgcache_methods = [
        "_cfgcache_encode_inputs",
        "_cfgcache_prepare_probe",
    ]

    for name in required_methods:
        if hasattr(LocalWanTransformer, name):
            _bind_method(tr, LocalWanTransformer, name)

    if require_cfgcache:
        for name in cfgcache_methods:
            _bind_method(tr, LocalWanTransformer, name)
    else:
        for name in cfgcache_methods:
            if hasattr(LocalWanTransformer, name):
                _bind_method(tr, LocalWanTransformer, name)

    # Optional helpers: bind when present in the local class.
    for maybe_name in [
        "cache_context",
        "_ensure_teacache_runtime",
        "_ensure_dicache_runtime",
        "_ensure_magcache_runtime",
        "_ensure_cfgcache_runtime",
        "_clear_cfgcache_probe_state",
    ]:
        if hasattr(LocalWanTransformer, maybe_name):
            _bind_method(tr, LocalWanTransformer, maybe_name)

    # Wan pipeline always expects cache_context. If the local transformer does not provide it,
    # add a no-op version for original / calibration runs.
    _ensure_cache_context(tr)

    missing = [name for name in required_methods if not hasattr(tr, name)]
    if missing:
        raise RuntimeError(f"Wan transformer patch failed, missing methods: {missing}")

    if require_cfgcache:
        missing_cfg = [name for name in cfgcache_methods if not hasattr(tr, name)]
        if missing_cfg:
            raise RuntimeError(f"Wan CFGCache patch failed, missing methods: {missing_cfg}")

    return tr


def _patch_local_wan_modules(
    pipe: WanPipeline,
    require_cfgcache: bool = False,
) -> WanPipeline:
    """
    Ensure the runtime pipeline uses the local Wan pipeline / transformer implementation,
    even when from_pretrained() materializes a vanilla diffusers instance.
    """
    # Keep the local pipeline class in case HF returns a vanilla parent instance.
    if not isinstance(pipe, WanPipeline):
        pipe.__class__ = WanPipeline

    if hasattr(pipe, "transformer"):
        _patch_one_wan_transformer(
            pipe.transformer,
            require_cfgcache=require_cfgcache,
        )

    # Wan2.2 may have transformer_2. Wan2.1 usually does not.
    if hasattr(pipe, "transformer_2"):
        _patch_one_wan_transformer(
            pipe.transformer_2,
            require_cfgcache=require_cfgcache,
        )

    return pipe


def _maybe_load_vae_fp32(
    model_path: str,
    *,
    local_files_only: bool = True,
    use_safetensors: bool = True,
):
    if AutoencoderKLWan is None:
        return None

    try:
        return AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
            use_safetensors=use_safetensors,
        )
    except TypeError:
        # Older diffusers may not accept use_safetensors here.
        return AutoencoderKLWan.from_pretrained(
            model_path,
            subfolder="vae",
            torch_dtype=torch.float32,
            local_files_only=local_files_only,
        )
    except Exception as e:
        print(f"[WARN] Failed to load Wan VAE in fp32, falling back to pipeline default: {e!r}", flush=True)
        return None


def _maybe_set_unipc_scheduler(pipe, flow_shift: float | None = 3.0):
    if UniPCMultistepScheduler is None:
        print("[WARN] UniPCMultistepScheduler is unavailable; keeping loaded scheduler.", flush=True)
        return pipe

    scheduler_kwargs = {}
    if flow_shift is not None:
        scheduler_kwargs["flow_shift"] = float(flow_shift)

    try:
        pipe.scheduler = UniPCMultistepScheduler.from_config(
            pipe.scheduler.config,
            **scheduler_kwargs,
        )
        print(f"[load] scheduler: UniPCMultistepScheduler, flow_shift={flow_shift}", flush=True)
    except TypeError:
        pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
        print("[WARN] UniPC scheduler does not accept flow_shift in this diffusers version.", flush=True)

    return pipe


def load_wan_pipeline_fast(
    model_path: str,
    patch_cache: bool = True,
    require_cfgcache: bool = False,
    torch_dtype: torch.dtype = torch.bfloat16,
    device_map: str | None = None,
    local_files_only: bool = True,
    use_safetensors: bool = True,
    load_vae_fp32: bool = True,
    set_unipc_scheduler: bool = False,
    flow_shift: float | None = 3.0,
):
    # 让 diffusers 并行加载分片
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "YES")

    t0 = time.perf_counter()
    print(f"[load] start from_pretrained: {model_path}", flush=True)

    vae = None
    if load_vae_fp32:
        vae = _maybe_load_vae_fp32(
            model_path,
            local_files_only=local_files_only,
            use_safetensors=use_safetensors,
        )

    kwargs = {
        "torch_dtype": torch_dtype,
        "local_files_only": local_files_only,
    }
    if use_safetensors is not None:
        kwargs["use_safetensors"] = use_safetensors
    if device_map is not None:
        kwargs["device_map"] = device_map
    if vae is not None:
        kwargs["vae"] = vae

    try:
        pipe = WanPipeline.from_pretrained(model_path, **kwargs)
    except TypeError:
        # Some local pipeline classes may not accept use_safetensors / device_map.
        kwargs.pop("use_safetensors", None)
        kwargs.pop("device_map", None)
        pipe = WanPipeline.from_pretrained(model_path, **kwargs)

    print(f"[load] from_pretrained done in {time.perf_counter() - t0:.2f}s", flush=True)

    # Fix UMT5 text encoder tied embedding mismatch before any inference.
    # Without this, encoder.embed_tokens may stay zero/random even when shared.weight is loaded,
    # causing prompt encoding failure and gray/blurry videos.
    pipe = _fix_umt5_tied_embeddings(pipe)

    if set_unipc_scheduler:
        pipe = _maybe_set_unipc_scheduler(pipe, flow_shift=flow_shift)

    if patch_cache:
        t1 = time.perf_counter()

        # Then force-bind the local transformer methods.
        pipe = _patch_local_wan_modules(
            pipe,
            require_cfgcache=require_cfgcache,
        )

        print(f"[load] patch done in {time.perf_counter() - t1:.2f}s", flush=True)
        print(f"[patch-check] pipeline class: {pipe.__class__}", flush=True)
        print(f"[patch-check] transformer class: {getattr(pipe, 'transformer', None).__class__}", flush=True)
        print(f"[patch-check] transformer_2 class: {getattr(pipe, 'transformer_2', None).__class__}", flush=True)
        print(
            f"[patch-check] transformer has cache_context: {hasattr(getattr(pipe, 'transformer', None), 'cache_context')}",
            flush=True,
        )
        print(
            f"[patch-check] transformer has _cfgcache_encode_inputs: {hasattr(getattr(pipe, 'transformer', None), '_cfgcache_encode_inputs')}",
            flush=True,
        )
        print(
            f"[patch-check] transformer has _cfgcache_prepare_probe: {hasattr(getattr(pipe, 'transformer', None), '_cfgcache_prepare_probe')}",
            flush=True,
        )
        if getattr(pipe, "transformer_2", None) is not None:
            print(
                f"[patch-check] transformer_2 has cache_context: {hasattr(pipe.transformer_2, 'cache_context')}",
                flush=True,
            )

    else:
        _ensure_cache_context(getattr(pipe, "transformer", None))
        _ensure_cache_context(getattr(pipe, "transformer_2", None))

    return pipe
import os
import time
import torch

from .pipeline_cogvideox import CogVideoXPipeline
from .transformer_cogvideox import CogVideoXTransformer3DModel
from .taylor_utils import pipeline_with_taylorseer


def load_cogvideox_pipeline_fast(model_path: str, patch_cache: bool = True):
    os.environ.setdefault("HF_ENABLE_PARALLEL_LOADING", "YES")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    t0 = time.perf_counter()
    print(f"[load] start from_pretrained: {model_path}", flush=True)

    # 1) 先显式加载你本地改过的 transformer
    transformer = CogVideoXTransformer3DModel.from_pretrained(
        model_path,
        subfolder="transformer",
        torch_dtype=dtype,
        local_files_only=True,
    )

    # 2) 再把这个 transformer 注入 pipeline
    load_kwargs = dict(
        transformer=transformer,
        torch_dtype=dtype,
        use_safetensors=True,
        local_files_only=True,
    )

    # 某些版本支持 device_map，有些不支持，保留 fallback
    if torch.cuda.is_available():
        load_kwargs["device_map"] = "cuda"

    try:
        pipe = CogVideoXPipeline.from_pretrained(
            model_path,
            **load_kwargs,
        )
    except TypeError:
        load_kwargs.pop("device_map", None)
        pipe = CogVideoXPipeline.from_pretrained(
            model_path,
            **load_kwargs,
        )

    print(f"[load] from_pretrained done in {time.perf_counter() - t0:.2f}s", flush=True)
    print(f"[load] transformer type = {type(pipe.transformer)}", flush=True)

    if hasattr(pipe, "vae") and pipe.vae is not None:
        if hasattr(pipe.vae, "enable_slicing"):
            pipe.vae.enable_slicing()
        if hasattr(pipe.vae, "enable_tiling"):
            pipe.vae.enable_tiling()

    if patch_cache:
        t1 = time.perf_counter()
        pipe = pipeline_with_taylorseer(pipe)
        print(f"[load] patch done in {time.perf_counter() - t1:.2f}s", flush=True)

    return pipe
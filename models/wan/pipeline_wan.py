import contextlib
import html
import json
from pathlib import Path
from typing import Any, Callable

import regex as re
import torch
from transformers import AutoTokenizer, UMT5EncoderModel

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import WanLoraLoaderMixin
from diffusers.models import AutoencoderKLWan, WanTransformer3DModel
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils import is_ftfy_available, is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.pipelines.wan.pipeline_output import WanPipelineOutput


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name

if is_ftfy_available():
    import ftfy


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers.utils import export_to_video
        >>> from diffusers import AutoencoderKLWan, WanPipeline
        >>> from diffusers.schedulers.scheduling_unipc_multistep import UniPCMultistepScheduler

        >>> # Available models: Wan-AI/Wan2.1-T2V-14B-Diffusers, Wan-AI/Wan2.1-T2V-1.3B-Diffusers
        >>> model_id = "Wan-AI/Wan2.1-T2V-14B-Diffusers"
        >>> vae = AutoencoderKLWan.from_pretrained(model_id, subfolder="vae", torch_dtype=torch.float32)
        >>> pipe = WanPipeline.from_pretrained(model_id, vae=vae, torch_dtype=torch.bfloat16)
        >>> flow_shift = 5.0  # 5.0 for 720P, 3.0 for 480P
        >>> pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config, flow_shift=flow_shift)
        >>> pipe.to("cuda")

        >>> prompt = "A cat and a dog baking a cake together in a kitchen. The cat is carefully measuring flour, while the dog is stirring the batter with a wooden spoon. The kitchen is cozy, with sunlight streaming through the window."
        >>> negative_prompt = "Bright tones, overexposed, static, blurred details, subtitles, style, works, paintings, images, static, overall gray, worst quality, low quality, JPEG compression residue, ugly, incomplete, extra fingers, poorly drawn hands, poorly drawn faces, deformed, disfigured, misshapen limbs, fused fingers, still picture, messy background, three legs, many people in the background, walking backwards"

        >>> output = pipe(
        ...     prompt=prompt,
        ...     negative_prompt=negative_prompt,
        ...     height=720,
        ...     width=1280,
        ...     num_frames=81,
        ...     guidance_scale=5.0,
        ... ).frames[0]
        >>> export_to_video(output, "output.mp4", fps=16)
        ```
"""


def basic_clean(text):
    if is_ftfy_available():
        text = ftfy.fix_text(text)
    text = html.unescape(html.unescape(text))
    return text.strip()


def whitespace_clean(text):
    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    return text


def prompt_clean(text):
    text = whitespace_clean(basic_clean(text))
    return text


# ============================================================
# Calibration helpers
# ============================================================

def _is_cfg_trace_root(path: Path) -> bool:
    name = path.name
    return name.startswith("cfg") and name.endswith("_traces")


def _resolve_cfg_trace_root(root: str | Path, cfg_scale: float) -> Path:
    root = Path(root).expanduser()
    if _is_cfg_trace_root(root):
        return root
    return root / f"cfg{float(cfg_scale):.1f}_traces"


def _normalize_prompt_indices(prompt_indices, batch_size: int) -> list[int]:
    """
    Make prompt/sample ids match the latent batch dimension.

    runtime_ctx["prompt_indices"] normally has length = prompt batch size.
    When num_videos_per_prompt > 1, latent batch may be larger. We expand ids.
    """
    if prompt_indices is None:
        return list(range(batch_size))

    prompt_indices = [int(x) for x in prompt_indices]

    if len(prompt_indices) == batch_size:
        return prompt_indices

    if len(prompt_indices) > 0 and batch_size % len(prompt_indices) == 0:
        base = prompt_indices
        repeat = batch_size // len(base)
        expanded = []
        for r in range(repeat):
            offset = r * len(base)
            expanded.extend([int(x) + offset for x in base])
        return expanded

    raise ValueError(
        f"prompt_indices length mismatch: len(prompt_indices)={len(prompt_indices)}, "
        f"batch_size={batch_size}"
    )


def _save_tensor_per_prompt(*, tensor: torch.Tensor, out_dirs: list[Path], filename: str):
    if tensor.shape[0] != len(out_dirs):
        raise ValueError(
            f"Tensor batch mismatch when saving {filename}: "
            f"tensor.shape[0]={tensor.shape[0]}, len(out_dirs)={len(out_dirs)}"
        )

    tensor_cpu = tensor.detach().float().cpu()

    for b, out_dir in enumerate(out_dirs):
        out_dir.mkdir(parents=True, exist_ok=True)
        torch.save(tensor_cpu[b].contiguous(), out_dir / filename)


def _write_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def _get_gt_cfg(runtime_ctx: dict[str, Any] | None) -> dict[str, Any] | None:
    if runtime_ctx is None:
        return None

    dump_gt_cfg = runtime_ctx.get("dump_gt_cfg", None)
    calibrate_mode = runtime_ctx.get("calibrate_mode", "none")

    if isinstance(dump_gt_cfg, dict) and dump_gt_cfg.get("enable", False):
        return dump_gt_cfg

    if calibrate_mode == "gt":
        return {"enable": True}

    return None


def _resolve_gt_run(
    *,
    runtime_ctx: dict[str, Any] | None,
    cfg_scale: float,
    batch_size: int,
    num_denoise_steps: int,
) -> dict[str, Any]:
    """
    Resolve Wan GT calibration metadata from runtime_ctx.

    Expected layout:
        calibration_gt/wan2.1/cfg5.0_traces/prompt_000000/baseline/
            final_latents.pt
            run_meta.json

        calibration_gt/wan2.1/cfg5.0_traces/prompt_000000/perturb_step_003/
            guided_ref.pt
            guided_old.pt
            final_latents.pt
            run_meta.json
    """
    gt_cfg = _get_gt_cfg(runtime_ctx)
    enabled = gt_cfg is not None

    if not enabled:
        return {
            "enabled": False,
            "role": None,
            "run_name": None,
            "perturb_step": None,
            "prompt_indices": None,
            "out_dirs": None,
            "trace_root": None,
        }

    root = (
        gt_cfg.get("root", None)
        or (runtime_ctx or {}).get("calibrate_root", None)
        or (runtime_ctx or {}).get("run_dir", None)
    )
    if root is None:
        raise ValueError(
            "GT calibration requires runtime_ctx['dump_gt_cfg']['root'] "
            "or runtime_ctx['calibrate_root']."
        )

    trace_root = _resolve_cfg_trace_root(root, cfg_scale)

    prompt_indices = _normalize_prompt_indices(
        (runtime_ctx or {}).get("prompt_indices", None),
        batch_size,
    )

    perturb_step = gt_cfg.get("perturb_step", None)
    if perturb_step is not None:
        perturb_step = int(perturb_step)

    role = gt_cfg.get("role", None)
    if role is None:
        is_baseline = bool(gt_cfg.get("is_baseline", perturb_step is None))
        role = "baseline" if is_baseline else "perturb"

    if role not in {"baseline", "perturb"}:
        raise ValueError(f"Unsupported GT role={role!r}; expected 'baseline' or 'perturb'.")

    run_name = gt_cfg.get("run_name", None) or gt_cfg.get("run_id", None)
    if run_name is None:
        run_name = "baseline" if role == "baseline" else f"perturb_step_{int(perturb_step):03d}"
    else:
        run_name = str(run_name)
        if role == "perturb" and run_name.startswith("step_"):
            run_name = f"perturb_{run_name}"

    if role == "perturb":
        if perturb_step is None:
            raise ValueError("GT perturb run requires dump_gt_cfg['perturb_step'].")
        if perturb_step <= 0:
            raise ValueError("GT perturb_step must be >= 1 because step 0 has no previous branch pred.")
        if perturb_step >= num_denoise_steps:
            raise ValueError(
                f"GT perturb_step must be < num_denoise_steps={num_denoise_steps}, "
                f"got {perturb_step}."
            )

    out_dirs = [trace_root / f"prompt_{pid:06d}" / str(run_name) for pid in prompt_indices]

    return {
        "enabled": True,
        "role": role,
        "run_name": str(run_name),
        "perturb_step": perturb_step,
        "prompt_indices": prompt_indices,
        "out_dirs": out_dirs,
        "trace_root": trace_root,
    }


def _model_cache_context(model, name: str):
    """
    Some local Wan transformer builds expose cache_context("cond"/"uncond"),
    while official diffusers builds may not. Calibration does not depend on cache logic,
    so we fallback to nullcontext.
    """
    if model is not None and hasattr(model, "cache_context"):
        return model.cache_context(name)
    return contextlib.nullcontext()


class WanPipeline(DiffusionPipeline, WanLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using Wan.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods
    implemented for all pipelines.

    Args:
        tokenizer ([`T5Tokenizer`]):
            Tokenizer from T5, specifically the google/umt5-xxl variant.
        text_encoder ([`T5EncoderModel`]):
            T5 encoder, specifically the google/umt5-xxl variant.
        transformer ([`WanTransformer3DModel`]):
            Conditional Transformer to denoise the input latents.
        scheduler ([`FlowMatchEulerDiscreteScheduler`]):
            Scheduler used with the transformer.
        vae ([`AutoencoderKLWan`]):
            VAE model to encode/decode videos.
        transformer_2 ([`WanTransformer3DModel`], *optional*):
            Conditional Transformer for low-noise stage in Wan2.2.
        boundary_ratio (`float`, *optional*):
            Boundary ratio for switching between transformer and transformer_2.
        expand_timesteps (`bool`, defaults to `False`):
            Wan2.2 ti2v expanded timestep mode.
    """

    model_cpu_offload_seq = "text_encoder->transformer->transformer_2->vae"
    _callback_tensor_inputs = ["latents", "prompt_embeds", "negative_prompt_embeds"]
    _optional_components = ["transformer", "transformer_2"]

    def __init__(
        self,
        tokenizer: AutoTokenizer,
        text_encoder: UMT5EncoderModel,
        vae: AutoencoderKLWan,
        scheduler: FlowMatchEulerDiscreteScheduler,
        transformer: WanTransformer3DModel | None = None,
        transformer_2: WanTransformer3DModel | None = None,
        boundary_ratio: float | None = None,
        expand_timesteps: bool = False,
    ):
        super().__init__()

        self.register_modules(
            vae=vae,
            text_encoder=text_encoder,
            tokenizer=tokenizer,
            transformer=transformer,
            scheduler=scheduler,
            transformer_2=transformer_2,
        )
        self.register_to_config(boundary_ratio=boundary_ratio)
        self.register_to_config(expand_timesteps=expand_timesteps)

        self.vae_scale_factor_temporal = self.vae.config.scale_factor_temporal if getattr(self, "vae", None) else 4
        self.vae_scale_factor_spatial = self.vae.config.scale_factor_spatial if getattr(self, "vae", None) else 8
        self.video_processor = VideoProcessor(vae_scale_factor=self.vae_scale_factor_spatial)

    def _get_t5_prompt_embeds(
        self,
        prompt: str | list[str] = None,
        num_videos_per_prompt: int = 1,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        device = device or self._execution_device
        dtype = dtype or self.text_encoder.dtype

        prompt = [prompt] if isinstance(prompt, str) else prompt
        prompt = [prompt_clean(u) for u in prompt]
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_attention_mask=True,
            return_tensors="pt",
        )
        text_input_ids, mask = text_inputs.input_ids, text_inputs.attention_mask
        seq_lens = mask.gt(0).sum(dim=1).long()

        prompt_embeds = self.text_encoder(text_input_ids.to(device), mask.to(device)).last_hidden_state
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        prompt_embeds = [u[:v] for u, v in zip(prompt_embeds, seq_lens)]
        prompt_embeds = torch.stack(
            [torch.cat([u, u.new_zeros(max_sequence_length - u.size(0), u.size(1))]) for u in prompt_embeds],
            dim=0,
        )

        _, seq_len, _ = prompt_embeds.shape
        prompt_embeds = prompt_embeds.repeat(1, num_videos_per_prompt, 1)
        prompt_embeds = prompt_embeds.view(batch_size * num_videos_per_prompt, seq_len, -1)

        return prompt_embeds

    def encode_prompt(
        self,
        prompt: str | list[str],
        negative_prompt: str | list[str] | None = None,
        do_classifier_free_guidance: bool = True,
        num_videos_per_prompt: int = 1,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        max_sequence_length: int = 226,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ):
        r"""
        Encodes the prompt into text encoder hidden states.
        """
        device = device or self._execution_device

        prompt = [prompt] if isinstance(prompt, str) else prompt

        if prompt is not None:
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        if prompt_embeds is None:
            prompt_embeds = self._get_t5_prompt_embeds(
                prompt=prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        if do_classifier_free_guidance and negative_prompt_embeds is None:
            negative_prompt = negative_prompt or ""
            negative_prompt = batch_size * [negative_prompt] if isinstance(negative_prompt, str) else negative_prompt

            if prompt is not None and type(prompt) is not type(negative_prompt):
                raise TypeError(
                    f"`negative_prompt` should be the same type to `prompt`, but got {type(negative_prompt)} !="
                    f" {type(prompt)}."
                )
            elif batch_size != len(negative_prompt):
                raise ValueError(
                    f"`negative_prompt`: {negative_prompt} has batch size {len(negative_prompt)}, but `prompt`:"
                    f" {prompt} has batch size {batch_size}. Please make sure that passed `negative_prompt` matches"
                    " the batch size of `prompt`."
                )

            negative_prompt_embeds = self._get_t5_prompt_embeds(
                prompt=negative_prompt,
                num_videos_per_prompt=num_videos_per_prompt,
                max_sequence_length=max_sequence_length,
                device=device,
                dtype=dtype,
            )

        return prompt_embeds, negative_prompt_embeds

    def check_inputs(
        self,
        prompt,
        negative_prompt,
        height,
        width,
        prompt_embeds=None,
        negative_prompt_embeds=None,
        callback_on_step_end_tensor_inputs=None,
        guidance_scale_2=None,
    ):
        if height % 16 != 0 or width % 16 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 16 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, "
                f"but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )

        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and "
                f"`negative_prompt_embeds`: {negative_prompt_embeds}. "
                "Please make sure to only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")
        elif negative_prompt is not None and (
            not isinstance(negative_prompt, str) and not isinstance(negative_prompt, list)
        ):
            raise ValueError(f"`negative_prompt` has to be of type `str` or `list` but is {type(negative_prompt)}")

        if self.config.boundary_ratio is None and guidance_scale_2 is not None:
            raise ValueError("`guidance_scale_2` is only supported when the pipeline's `boundary_ratio` is not None.")

    def prepare_latents(
        self,
        batch_size: int,
        num_channels_latents: int = 16,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        dtype: torch.dtype | None = None,
        device: torch.device | None = None,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if latents is not None:
            return latents.to(device=device, dtype=dtype)

        num_latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1
        shape = (
            batch_size,
            num_channels_latents,
            num_latent_frames,
            int(height) // self.vae_scale_factor_spatial,
            int(width) // self.vae_scale_factor_spatial,
        )

        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        return latents

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def do_classifier_free_guidance(self):
        return self._guidance_scale > 1.0

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    def _dump_rho_calibration_step(
        self,
        *,
        runtime_ctx: dict[str, Any] | None,
        step_idx: int,
        latents: torch.Tensor,
        noise_pred_text: torch.Tensor,
        noise_pred_uncond: torch.Tensor,
        guidance_scale: float | None = None,
    ):
        """
        Dump rho-calibration tensors:

            cfgX.X_traces/prompt_xxxxxx/
                step_i_xt.pt
                step_i_cond.pt
                step_i_uncond.pt

        Saved before scheduler.step.
        GT mode does not call this function.
        """
        if runtime_ctx is None:
            return

        dump_cfg = runtime_ctx.get("dump_residual_cfg", None)
        calibrate_mode = runtime_ctx.get("calibrate_mode", "none")

        rho_enabled = (
            calibrate_mode == "rho"
            or (isinstance(dump_cfg, dict) and dump_cfg.get("enable", False))
        )

        if not rho_enabled:
            return

        root = None
        if isinstance(dump_cfg, dict):
            root = dump_cfg.get("root", None)

        root = root or runtime_ctx.get("calibrate_root", None)

        if root is None:
            raise ValueError(
                "runtime_ctx['dump_residual_cfg']['root'] or runtime_ctx['calibrate_root'] "
                "must be provided for rho calibration."
            )

        if guidance_scale is None:
            guidance_scale = 1.0

        trace_root = _resolve_cfg_trace_root(root, float(guidance_scale))
        prompt_indices = _normalize_prompt_indices(
            runtime_ctx.get("prompt_indices", None),
            latents.shape[0],
        )

        for b in range(latents.shape[0]):
            prompt_id = int(prompt_indices[b])
            prompt_dir = trace_root / f"prompt_{prompt_id:06d}"
            prompt_dir.mkdir(parents=True, exist_ok=True)

            torch.save(
                latents[b].detach().float().cpu(),
                prompt_dir / f"step_{step_idx}_xt.pt",
            )
            torch.save(
                noise_pred_text[b].detach().float().cpu(),
                prompt_dir / f"step_{step_idx}_cond.pt",
            )
            torch.save(
                noise_pred_uncond[b].detach().float().cpu(),
                prompt_dir / f"step_{step_idx}_uncond.pt",
            )

    @torch.no_grad()
    @replace_example_docstring(EXAMPLE_DOC_STRING)
    def __call__(
        self,
        prompt: str | list[str] = None,
        negative_prompt: str | list[str] = None,
        height: int = 480,
        width: int = 832,
        num_frames: int = 81,
        num_inference_steps: int = 50,
        guidance_scale: float = 5.0,
        guidance_scale_2: float | None = None,
        num_videos_per_prompt: int | None = 1,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.Tensor | None = None,
        prompt_embeds: torch.Tensor | None = None,
        negative_prompt_embeds: torch.Tensor | None = None,
        output_type: str | None = "np",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 512,
        runtime_ctx: dict[str, Any] | None = None,
    ):
        r"""
        The call function to the pipeline for generation.

        Args:
            prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to guide the image generation. If not defined, pass `prompt_embeds` instead.
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts to avoid during image generation. If not defined, pass `negative_prompt_embeds`
                instead. Ignored when not using guidance (`guidance_scale` < `1`).
            height (`int`, defaults to `480`):
                The height in pixels of the generated image.
            width (`int`, defaults to `832`):
                The width in pixels of the generated image.
            num_frames (`int`, defaults to `81`):
                The number of frames in the generated video.
            num_inference_steps (`int`, defaults to `50`):
                The number of denoising steps. More denoising steps usually lead to a higher quality image at the
                expense of slower inference.
            guidance_scale (`float`, defaults to `5.0`):
                Guidance scale as defined in [Classifier-Free Diffusion
                Guidance](https://huggingface.co/papers/2207.12598). `guidance_scale` is defined as `w` of equation 2.
                of [Imagen Paper](https://huggingface.co/papers/2205.11487). Guidance scale is enabled by setting
                `guidance_scale > 1`. Higher guidance scale encourages to generate images that are closely linked to
                the text `prompt`, usually at the expense of lower image quality.
            guidance_scale_2 (`float`, *optional*, defaults to `None`):
                Guidance scale for the low-noise stage transformer (`transformer_2`). If `None` and the pipeline's
                `boundary_ratio` is not None, uses the same value as `guidance_scale`. Only used when `transformer_2`
                and the pipeline's `boundary_ratio` are not None.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                The number of images to generate per prompt.
            generator (`torch.Generator` or `list[torch.Generator]`, *optional*):
                A [`torch.Generator`](https://pytorch.org/docs/stable/generated/torch.Generator.html) to make
                generation deterministic.
            latents (`torch.Tensor`, *optional*):
                Pre-generated noisy latents sampled from a Gaussian distribution, to be used as inputs for image
                generation. Can be used to tweak the same generation with different prompts. If not provided, a latents
                tensor is generated by sampling using the supplied random `generator`.
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs (prompt weighting). If not
                provided, text embeddings are generated from the `prompt` input argument.
            output_type (`str`, *optional*, defaults to `"np"`):
                The output format of the generated image. Choose between `PIL.Image` or `np.array`.
            return_dict (`bool`, *optional*, defaults to `True`):
                Whether or not to return a [`WanPipelineOutput`] instead of a plain tuple.
            attention_kwargs (`dict`, *optional*):
                A kwargs dictionary that if specified is passed along to the `AttentionProcessor` as defined under
                `self.processor` in
                [diffusers.models.attention_processor](https://github.com/huggingface/diffusers/blob/main/src/diffusers/models/attention_processor.py).
            callback_on_step_end (`Callable`, `PipelineCallback`, `MultiPipelineCallbacks`, *optional*):
                A function or a subclass of `PipelineCallback` or `MultiPipelineCallbacks` that is called at the end of
                each denoising step during the inference. with the following arguments: `callback_on_step_end(self:
                DiffusionPipeline, step: int, timestep: int, callback_kwargs: Dict)`. `callback_kwargs` will include a
                list of all tensors as specified by `callback_on_step_end_tensor_inputs`.
            callback_on_step_end_tensor_inputs (`list`, *optional*):
                The list of tensor inputs for the `callback_on_step_end` function. The tensors specified in the list
                will be passed as `callback_kwargs` argument. You will only be able to include variables listed in the
                `._callback_tensor_inputs` attribute of your pipeline class.
            max_sequence_length (`int`, defaults to `512`):
                The maximum sequence length of the text encoder. If the prompt is longer than this, it will be
                truncated. If the prompt is shorter, it will be padded to this length.

        Examples:

        Returns:
            [`~WanPipelineOutput`] or `tuple`:
                If `return_dict` is `True`, [`WanPipelineOutput`] is returned, otherwise a `tuple` is returned where
                the first element is a list with the generated images and the second element is a list of `bool`s
                indicating whether the corresponding generated image contains "not-safe-for-work" (nsfw) content.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        # 1. Check inputs.
        self.check_inputs(
            prompt,
            negative_prompt,
            height,
            width,
            prompt_embeds,
            negative_prompt_embeds,
            callback_on_step_end_tensor_inputs,
            guidance_scale_2,
        )

        if num_frames % self.vae_scale_factor_temporal != 1:
            logger.warning(
                f"`num_frames - 1` has to be divisible by {self.vae_scale_factor_temporal}. "
                "Rounding to the nearest number."
            )
            num_frames = num_frames // self.vae_scale_factor_temporal * self.vae_scale_factor_temporal + 1

        num_frames = max(num_frames, 1)

        patch_size = (
            self.transformer.config.patch_size
            if self.transformer is not None
            else self.transformer_2.config.patch_size
        )
        h_multiple_of = self.vae_scale_factor_spatial * patch_size[1]
        w_multiple_of = self.vae_scale_factor_spatial * patch_size[2]
        calc_height = height // h_multiple_of * h_multiple_of
        calc_width = width // w_multiple_of * w_multiple_of

        if height != calc_height or width != calc_width:
            logger.warning(
                f"`height` and `width` must be multiples of ({h_multiple_of}, {w_multiple_of}) "
                f"for proper patchification. Adjusting ({height}, {width}) -> ({calc_height}, {calc_width})."
            )
            height, width = calc_height, calc_width

        if self.config.boundary_ratio is not None and guidance_scale_2 is None:
            guidance_scale_2 = guidance_scale

        base_cfg_scale = float(guidance_scale)

        self._guidance_scale = base_cfg_scale
        self._guidance_scale_2 = guidance_scale_2
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        device = self._execution_device

        # 2. Define call parameters.
        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        do_classifier_free_guidance = self.do_classifier_free_guidance

        # Calibration needs dual branch CFG.
        if runtime_ctx is not None:
            calibrate_mode = runtime_ctx.get("calibrate_mode", "none")
            dump_rho_cfg = runtime_ctx.get("dump_residual_cfg", None)
            dump_gt_cfg = runtime_ctx.get("dump_gt_cfg", None)

            rho_enabled = (
                calibrate_mode == "rho"
                or (isinstance(dump_rho_cfg, dict) and dump_rho_cfg.get("enable", False))
            )
            gt_enabled = (
                calibrate_mode == "gt"
                or (isinstance(dump_gt_cfg, dict) and dump_gt_cfg.get("enable", False))
            )

            if (rho_enabled or gt_enabled) and not do_classifier_free_guidance:
                raise ValueError(
                    f"calibrate_mode={calibrate_mode!r} requires classifier-free guidance. "
                    "Please set guidance_scale > 1.0."
                )

        # 3. Encode input prompt.
        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt=prompt,
            negative_prompt=negative_prompt,
            do_classifier_free_guidance=do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        transformer_dtype = self.transformer.dtype if self.transformer is not None else self.transformer_2.dtype
        prompt_embeds = prompt_embeds.to(transformer_dtype)

        if negative_prompt_embeds is not None:
            negative_prompt_embeds = negative_prompt_embeds.to(transformer_dtype)

        # 4. Prepare timesteps.
        self.scheduler.set_timesteps(num_inference_steps, device=device)
        timesteps = self.scheduler.timesteps

        # 5. Prepare latent variables.
        num_channels_latents = (
            self.transformer.config.in_channels
            if self.transformer is not None
            else self.transformer_2.config.in_channels
        )
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            num_channels_latents,
            height,
            width,
            num_frames,
            torch.float32,
            device,
            generator,
            latents,
        )

        mask = torch.ones(latents.shape, dtype=torch.float32, device=device)

        # 6. Denoising loop.
        num_warmup_steps = len(timesteps) - num_inference_steps * self.scheduler.order
        self._num_timesteps = len(timesteps)

        # Remove DtoH sync, helpful especially during compilation.
        self.scheduler.set_begin_index(0)

        if self.config.boundary_ratio is not None:
            boundary_timestep = self.config.boundary_ratio * self.scheduler.config.num_train_timesteps
        else:
            boundary_timestep = None

        gt_state = _resolve_gt_run(
            runtime_ctx=runtime_ctx,
            cfg_scale=base_cfg_scale,
            batch_size=latents.shape[0],
            num_denoise_steps=len(timesteps),
        )

        with self.progress_bar(total=num_inference_steps) as progress_bar:
            prev_cond = None
            prev_uncond = None
            perturb_done = False

            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                if boundary_timestep is None or t >= boundary_timestep:
                    # Wan2.1 or high-noise stage in Wan2.2.
                    current_model = self.transformer
                    current_guidance_scale = guidance_scale
                    current_stage = "high"
                else:
                    # Low-noise stage in Wan2.2.
                    current_model = self.transformer_2
                    current_guidance_scale = guidance_scale_2
                    current_stage = "low"

                if current_model is None:
                    raise ValueError(
                        f"WanPipeline selected current_model=None at step {i}. "
                        "Please check transformer / transformer_2 / boundary_ratio config."
                    )

                if current_guidance_scale is None:
                    current_guidance_scale = guidance_scale

                latent_model_input = latents.to(transformer_dtype)

                if self.config.expand_timesteps:
                    # seq_len: num_latent_frames * latent_height//2 * latent_width//2
                    temp_ts = (mask[0][0][:, ::2, ::2] * t).flatten()
                    timestep = temp_ts.unsqueeze(0).expand(latents.shape[0], -1)
                else:
                    timestep = t.expand(latents.shape[0])

                if runtime_ctx is not None:
                    runtime_ctx["step"] = int(i)
                    runtime_ctx["num_steps"] = int(len(timesteps))
                    runtime_ctx["timestep"] = int(t.item()) if hasattr(t, "item") else t
                    runtime_ctx["model"] = "cond"
                    runtime_ctx["wan_stage"] = current_stage

                with _model_cache_context(current_model, "cond"):
                    noise_pred_text = current_model(
                        hidden_states=latent_model_input,
                        timestep=timestep,
                        encoder_hidden_states=prompt_embeds,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]

                noise_pred_text = noise_pred_text.float()

                if do_classifier_free_guidance:
                    if runtime_ctx is not None:
                        runtime_ctx["model"] = "uncond"

                    with _model_cache_context(current_model, "uncond"):
                        noise_pred_uncond = current_model(
                            hidden_states=latent_model_input,
                            timestep=timestep,
                            encoder_hidden_states=negative_prompt_embeds,
                            attention_kwargs=attention_kwargs,
                            return_dict=False,
                        )[0]

                    noise_pred_uncond = noise_pred_uncond.float()

                    # rho and GT are separated.
                    # GT must not dump xt/cond/uncond to avoid mixing trace semantics.
                    if runtime_ctx is not None and runtime_ctx.get("calibrate_mode", "none") == "rho":
                        self._dump_rho_calibration_step(
                            runtime_ctx=runtime_ctx,
                            step_idx=i,
                            latents=latents,
                            noise_pred_text=noise_pred_text,
                            noise_pred_uncond=noise_pred_uncond,
                            guidance_scale=base_cfg_scale,
                        )

                    guided_ref = noise_pred_uncond + float(current_guidance_scale) * (
                        noise_pred_text - noise_pred_uncond
                    )
                    noise_pred = guided_ref

                    if (
                        gt_state["enabled"]
                        and gt_state["role"] == "perturb"
                        and int(i) == int(gt_state["perturb_step"])
                    ):
                        if prev_cond is None or prev_uncond is None:
                            raise RuntimeError(
                                f"Cannot perturb step {i}: previous cond/uncond predictions are None."
                            )

                        guided_old = prev_uncond + float(current_guidance_scale) * (prev_cond - prev_uncond)

                        out_dirs = gt_state["out_dirs"]

                        _save_tensor_per_prompt(
                            tensor=guided_ref,
                            out_dirs=out_dirs,
                            filename="guided_ref.pt",
                        )
                        _save_tensor_per_prompt(
                            tensor=guided_old,
                            out_dirs=out_dirs,
                            filename="guided_old.pt",
                        )

                        noise_pred = guided_old
                        perturb_done = True

                    prev_cond = noise_pred_text.detach()
                    prev_uncond = noise_pred_uncond.detach()

                else:
                    noise_pred = noise_pred_text

                if runtime_ctx is not None:
                    runtime_ctx["model"] = "guided"

                # compute the previous noisy sample x_t -> x_t-1
                latents = self.scheduler.step(noise_pred, t, latents, return_dict=False)[0]

                if callback_on_step_end is not None:
                    callback_kwargs = {}
                    for k in callback_on_step_end_tensor_inputs:
                        callback_kwargs[k] = locals()[k]

                    callback_outputs = callback_on_step_end(self, i, t, callback_kwargs)

                    latents = callback_outputs.pop("latents", latents)
                    prompt_embeds = callback_outputs.pop("prompt_embeds", prompt_embeds)
                    negative_prompt_embeds = callback_outputs.pop("negative_prompt_embeds", negative_prompt_embeds)

                if i == len(timesteps) - 1 or ((i + 1) > num_warmup_steps and (i + 1) % self.scheduler.order == 0):
                    progress_bar.update()

                if XLA_AVAILABLE:
                    xm.mark_step()

            if gt_state["enabled"]:
                if gt_state["role"] == "perturb" and not perturb_done:
                    raise RuntimeError(
                        f"GT perturb_step={gt_state['perturb_step']} was not executed. "
                        f"num_denoise_steps={len(timesteps)}."
                    )

                out_dirs = gt_state["out_dirs"]

                _save_tensor_per_prompt(
                    tensor=latents,
                    out_dirs=out_dirs,
                    filename="final_latents.pt",
                )

                saved_tensors = ["final_latents.pt"]
                if gt_state["role"] == "perturb":
                    saved_tensors = ["guided_ref.pt", "guided_old.pt", "final_latents.pt"]

                for prompt_id, out_dir in zip(gt_state["prompt_indices"], out_dirs):
                    _write_json(
                        out_dir / "run_meta.json",
                        {
                            "model": "wan",
                            "prompt_id": int(prompt_id),
                            "calibrate_mode": "gt",
                            "gt_role": gt_state["role"],
                            "gt_run_name": gt_state["run_name"],
                            "perturb_step": None
                            if gt_state["perturb_step"] is None
                            else int(gt_state["perturb_step"]),
                            "num_denoise_steps": int(len(timesteps)),
                            "guidance_scale": float(guidance_scale),
                            "guidance_scale_2": None
                            if guidance_scale_2 is None
                            else float(guidance_scale_2),
                            "trace_cfg_scale": float(base_cfg_scale),
                            "height": int(height),
                            "width": int(width),
                            "num_frames": int(num_frames),
                            "saved_tensors": saved_tensors,
                        },
                    )

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents.to(self.vae.dtype)

            latents_mean = (
                torch.tensor(self.vae.config.latents_mean)
                .view(1, self.vae.config.z_dim, 1, 1, 1)
                .to(latents.device, latents.dtype)
            )

            latents_std = 1.0 / torch.tensor(self.vae.config.latents_std).view(
                1, self.vae.config.z_dim, 1, 1, 1
            ).to(latents.device, latents.dtype)

            latents = latents / latents_std + latents_mean
            video = self.vae.decode(latents, return_dict=False)[0]
            video = self.video_processor.postprocess_video(video, output_type=output_type)
        else:
            video = latents

        # Offload all models.
        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return WanPipelineOutput(frames=video)
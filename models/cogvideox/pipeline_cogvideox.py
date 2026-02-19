import inspect
import json
from pathlib import Path
import math
from typing import Any, Callable

import torch
from transformers import T5EncoderModel, T5Tokenizer

from diffusers.callbacks import MultiPipelineCallbacks, PipelineCallback
from diffusers.loaders import CogVideoXLoraLoaderMixin
from diffusers.models import AutoencoderKLCogVideoX
from diffusers.models.embeddings import get_3d_rotary_pos_embed
from diffusers.pipelines.pipeline_utils import DiffusionPipeline
from diffusers.schedulers import CogVideoXDDIMScheduler, CogVideoXDPMScheduler
from diffusers.utils import is_torch_xla_available, logging, replace_example_docstring
from diffusers.utils.torch_utils import randn_tensor
from diffusers.video_processor import VideoProcessor
from diffusers.pipelines.cogvideo.pipeline_output import CogVideoXPipelineOutput

from diffusers import CogVideoXTransformer3DModel


if is_torch_xla_available():
    import torch_xla.core.xla_model as xm

    XLA_AVAILABLE = True
else:
    XLA_AVAILABLE = False

logger = logging.get_logger(__name__)  # pylint: disable=invalid-name


EXAMPLE_DOC_STRING = """
    Examples:
        ```python
        >>> import torch
        >>> from diffusers import CogVideoXPipeline
        >>> from diffusers.utils import export_to_video

        >>> # Models: "THUDM/CogVideoX-2b" or "THUDM/CogVideoX-5b"
        >>> pipe = CogVideoXPipeline.from_pretrained("THUDM/CogVideoX-2b", torch_dtype=torch.float16).to("cuda")
        >>> prompt = (
        ...     "A panda, dressed in a small, red jacket and a tiny hat, sits on a wooden stool in a serene bamboo forest. "
        ...     "The panda's fluffy paws strum a miniature acoustic guitar, producing soft, melodic tunes. Nearby, a few other "
        ...     "pandas gather, watching curiously and some clapping in rhythm. Sunlight filters through the tall bamboo, "
        ...     "casting a gentle glow on the scene. The panda's face is expressive, showing concentration and joy as it plays. "
        ...     "The background includes a small, flowing stream and vibrant green foliage, enhancing the peaceful and magical "
        ...     "atmosphere of this unique musical performance."
        ... )
        >>> video = pipe(prompt=prompt, guidance_scale=6, num_inference_steps=50).frames[0]
        >>> export_to_video(video, "output.mp4", fps=8)
        ```
"""


# Similar to diffusers.pipelines.hunyuandit.pipeline_hunyuandit.get_resize_crop_region_for_grid
def get_resize_crop_region_for_grid(src, tgt_width, tgt_height):
    tw = tgt_width
    th = tgt_height
    h, w = src
    r = h / w
    if r > (th / tw):
        resize_height = th
        resize_width = int(round(th / h * w))
    else:
        resize_width = tw
        resize_height = int(round(tw / w * h))

    crop_top = int(round((th - resize_height) / 2.0))
    crop_left = int(round((tw - resize_width) / 2.0))

    return (crop_top, crop_left), (crop_top + resize_height, crop_left + resize_width)


# Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.retrieve_timesteps
def retrieve_timesteps(
    scheduler,
    num_inference_steps: int | None = None,
    device: str | torch.device | None = None,
    timesteps: list[int] | None = None,
    sigmas: list[float] | None = None,
    **kwargs,
):
    r"""
    Calls the scheduler's `set_timesteps` method and retrieves timesteps from the scheduler after the call. Handles
    custom timesteps. Any kwargs will be supplied to `scheduler.set_timesteps`.

    Args:
        scheduler (`SchedulerMixin`):
            The scheduler to get timesteps from.
        num_inference_steps (`int`):
            The number of diffusion steps used when generating samples with a pre-trained model. If used, `timesteps`
            must be `None`.
        device (`str` or `torch.device`, *optional*):
            The device to which the timesteps should be moved to. If `None`, the timesteps are not moved.
        timesteps (`list[int]`, *optional*):
            Custom timesteps used to override the timestep spacing strategy of the scheduler. If `timesteps` is passed,
            `num_inference_steps` and `sigmas` must be `None`.
        sigmas (`list[float]`, *optional*):
            Custom sigmas used to override the timestep spacing strategy of the scheduler. If `sigmas` is passed,
            `num_inference_steps` and `timesteps` must be `None`.

    Returns:
        `tuple[torch.Tensor, int]`: A tuple where the first element is the timestep schedule from the scheduler and the
        second element is the number of inference steps.
    """
    if timesteps is not None and sigmas is not None:
        raise ValueError("Only one of `timesteps` or `sigmas` can be passed. Please choose one to set custom values")
    if timesteps is not None:
        accepts_timesteps = "timesteps" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accepts_timesteps:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" timestep schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(timesteps=timesteps, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    elif sigmas is not None:
        accept_sigmas = "sigmas" in set(inspect.signature(scheduler.set_timesteps).parameters.keys())
        if not accept_sigmas:
            raise ValueError(
                f"The current scheduler class {scheduler.__class__}'s `set_timesteps` does not support custom"
                f" sigmas schedules. Please check whether you are using the correct scheduler."
            )
        scheduler.set_timesteps(sigmas=sigmas, device=device, **kwargs)
        timesteps = scheduler.timesteps
        num_inference_steps = len(timesteps)
    else:
        scheduler.set_timesteps(num_inference_steps, device=device, **kwargs)
        timesteps = scheduler.timesteps
    return timesteps, num_inference_steps


def _is_cfg_trace_root(path: Path) -> bool:
    name = path.name
    return name.startswith("cfg") and name.endswith("_traces")


def _resolve_cfg_trace_root(root: str | Path, cfg_scale: float) -> Path:
    root = Path(root).expanduser()
    if _is_cfg_trace_root(root):
        return root
    return root / f"cfg{float(cfg_scale):.1f}_traces"


def _normalize_prompt_indices(prompt_indices, batch_size: int) -> list[int]:
    """Make prompt/sample ids match the latent batch dimension."""
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
    """Resolve CogVideoX GT calibration metadata from runtime_ctx."""
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
class CogVideoXPipeline(DiffusionPipeline, CogVideoXLoraLoaderMixin):
    r"""
    Pipeline for text-to-video generation using CogVideoX.

    This model inherits from [`DiffusionPipeline`]. Check the superclass documentation for the generic methods the
    library implements for all the pipelines (such as downloading or saving, running on a particular device, etc.)

    Args:
        vae ([`AutoencoderKL`]):
            Variational Auto-Encoder (VAE) Model to encode and decode videos to and from latent representations.
        text_encoder ([`T5EncoderModel`]):
            Frozen text-encoder. CogVideoX uses
            [T5](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5EncoderModel); specifically the
            [t5-v1_1-xxl](https://huggingface.co/PixArt-alpha/PixArt-alpha/tree/main/t5-v1_1-xxl) variant.
        tokenizer (`T5Tokenizer`):
            Tokenizer of class
            [T5Tokenizer](https://huggingface.co/docs/transformers/model_doc/t5#transformers.T5Tokenizer).
        transformer ([`CogVideoXTransformer3DModel`]):
            A text conditioned `CogVideoXTransformer3DModel` to denoise the encoded video latents.
        scheduler ([`SchedulerMixin`]):
            A scheduler to be used in combination with `transformer` to denoise the encoded video latents.
    """

    _optional_components = []
    model_cpu_offload_seq = "text_encoder->transformer->vae"

    _callback_tensor_inputs = [
        "latents",
        "prompt_embeds",
        "negative_prompt_embeds",
    ]

    def __init__(
        self,
        tokenizer: T5Tokenizer,
        text_encoder: T5EncoderModel,
        vae: AutoencoderKLCogVideoX,
        transformer: CogVideoXTransformer3DModel,
        scheduler: CogVideoXDDIMScheduler | CogVideoXDPMScheduler,
    ):
        super().__init__()

        self.register_modules(
            tokenizer=tokenizer, text_encoder=text_encoder, vae=vae, transformer=transformer, scheduler=scheduler
        )
        self.vae_scale_factor_spatial = (
            2 ** (len(self.vae.config.block_out_channels) - 1) if getattr(self, "vae", None) else 8
        )
        self.vae_scale_factor_temporal = (
            self.vae.config.temporal_compression_ratio if getattr(self, "vae", None) else 4
        )
        self.vae_scaling_factor_image = self.vae.config.scaling_factor if getattr(self, "vae", None) else 0.7

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
        batch_size = len(prompt)

        text_inputs = self.tokenizer(
            prompt,
            padding="max_length",
            max_length=max_sequence_length,
            truncation=True,
            add_special_tokens=True,
            return_tensors="pt",
        )
        text_input_ids = text_inputs.input_ids
        untruncated_ids = self.tokenizer(prompt, padding="longest", return_tensors="pt").input_ids

        if untruncated_ids.shape[-1] >= text_input_ids.shape[-1] and not torch.equal(text_input_ids, untruncated_ids):
            removed_text = self.tokenizer.batch_decode(untruncated_ids[:, max_sequence_length - 1 : -1])
            logger.warning(
                "The following part of your input was truncated because `max_sequence_length` is set to "
                f" {max_sequence_length} tokens: {removed_text}"
            )

        prompt_embeds = self.text_encoder(text_input_ids.to(device))[0]
        prompt_embeds = prompt_embeds.to(dtype=dtype, device=device)

        # duplicate text embeddings for each generation per prompt, using mps friendly method
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

        Args:
            prompt (`str` or `list[str]`, *optional*):
                prompt to be encoded
            negative_prompt (`str` or `list[str]`, *optional*):
                The prompt or prompts not to guide the image generation. If not defined, one has to pass
                `negative_prompt_embeds` instead. Ignored when not using guidance (i.e., ignored if `guidance_scale` is
                less than `1`).
            do_classifier_free_guidance (`bool`, *optional*, defaults to `True`):
                Whether to use classifier free guidance or not.
            num_videos_per_prompt (`int`, *optional*, defaults to 1):
                Number of videos that should be generated per prompt. torch device to place the resulting embeddings on
            prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt weighting. If not
                provided, text embeddings will be generated from `prompt` input argument.
            negative_prompt_embeds (`torch.Tensor`, *optional*):
                Pre-generated negative text embeddings. Can be used to easily tweak text inputs, *e.g.* prompt
                weighting. If not provided, negative_prompt_embeds will be generated from `negative_prompt` input
                argument.
            device: (`torch.device`, *optional*):
                torch device
            dtype: (`torch.dtype`, *optional*):
                torch dtype
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

    def prepare_latents(
        self, batch_size, num_channels_latents, num_frames, height, width, dtype, device, generator, latents=None
    ):
        if isinstance(generator, list) and len(generator) != batch_size:
            raise ValueError(
                f"You have passed a list of generators of length {len(generator)}, but requested an effective batch"
                f" size of {batch_size}. Make sure the batch size matches the length of the generators."
            )

        shape = (
            batch_size,
            (num_frames - 1) // self.vae_scale_factor_temporal + 1,
            num_channels_latents,
            height // self.vae_scale_factor_spatial,
            width // self.vae_scale_factor_spatial,
        )

        if latents is None:
            latents = randn_tensor(shape, generator=generator, device=device, dtype=dtype)
        else:
            latents = latents.to(device)

        # scale the initial noise by the standard deviation required by the scheduler
        latents = latents * self.scheduler.init_noise_sigma
        return latents

    def decode_latents(self, latents: torch.Tensor) -> torch.Tensor:
        latents = latents.permute(0, 2, 1, 3, 4)  # [batch_size, num_channels, num_frames, height, width]
        latents = 1 / self.vae_scaling_factor_image * latents

        frames = self.vae.decode(latents).sample
        return frames

    # Copied from diffusers.pipelines.stable_diffusion.pipeline_stable_diffusion.StableDiffusionPipeline.prepare_extra_step_kwargs
    def prepare_extra_step_kwargs(self, generator, eta):
        # prepare extra kwargs for the scheduler step, since not all schedulers have the same signature
        # eta (η) is only used with the DDIMScheduler, it will be ignored for other schedulers.
        # eta corresponds to η in DDIM paper: https://huggingface.co/papers/2010.02502
        # and should be between [0, 1]

        accepts_eta = "eta" in set(inspect.signature(self.scheduler.step).parameters.keys())
        extra_step_kwargs = {}
        if accepts_eta:
            extra_step_kwargs["eta"] = eta

        # check if the scheduler accepts generator
        accepts_generator = "generator" in set(inspect.signature(self.scheduler.step).parameters.keys())
        if accepts_generator:
            extra_step_kwargs["generator"] = generator
        return extra_step_kwargs

    # Copied from diffusers.pipelines.latte.pipeline_latte.LattePipeline.check_inputs
    def check_inputs(
        self,
        prompt,
        height,
        width,
        negative_prompt,
        callback_on_step_end_tensor_inputs,
        prompt_embeds=None,
        negative_prompt_embeds=None,
    ):
        if height % 8 != 0 or width % 8 != 0:
            raise ValueError(f"`height` and `width` have to be divisible by 8 but are {height} and {width}.")

        if callback_on_step_end_tensor_inputs is not None and not all(
            k in self._callback_tensor_inputs for k in callback_on_step_end_tensor_inputs
        ):
            raise ValueError(
                f"`callback_on_step_end_tensor_inputs` has to be in {self._callback_tensor_inputs}, but found {[k for k in callback_on_step_end_tensor_inputs if k not in self._callback_tensor_inputs]}"
            )
        if prompt is not None and prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `prompt_embeds`: {prompt_embeds}. Please make sure to"
                " only forward one of the two."
            )
        elif prompt is None and prompt_embeds is None:
            raise ValueError(
                "Provide either `prompt` or `prompt_embeds`. Cannot leave both `prompt` and `prompt_embeds` undefined."
            )
        elif prompt is not None and (not isinstance(prompt, str) and not isinstance(prompt, list)):
            raise ValueError(f"`prompt` has to be of type `str` or `list` but is {type(prompt)}")

        if prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `prompt`: {prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if negative_prompt is not None and negative_prompt_embeds is not None:
            raise ValueError(
                f"Cannot forward both `negative_prompt`: {negative_prompt} and `negative_prompt_embeds`:"
                f" {negative_prompt_embeds}. Please make sure to only forward one of the two."
            )

        if prompt_embeds is not None and negative_prompt_embeds is not None:
            if prompt_embeds.shape != negative_prompt_embeds.shape:
                raise ValueError(
                    "`prompt_embeds` and `negative_prompt_embeds` must have the same shape when passed directly, but"
                    f" got: `prompt_embeds` {prompt_embeds.shape} != `negative_prompt_embeds`"
                    f" {negative_prompt_embeds.shape}."
                )

    def fuse_qkv_projections(self) -> None:
        r"""Enables fused QKV projections."""
        self.fusing_transformer = True
        self.transformer.fuse_qkv_projections()

    def unfuse_qkv_projections(self) -> None:
        r"""Disable QKV projection fusion if enabled."""
        if not self.fusing_transformer:
            logger.warning("The Transformer was not initially fused for QKV projections. Doing nothing.")
        else:
            self.transformer.unfuse_qkv_projections()
            self.fusing_transformer = False

    def _prepare_rotary_positional_embeddings(
        self,
        height: int,
        width: int,
        num_frames: int,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        grid_height = height // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)
        grid_width = width // (self.vae_scale_factor_spatial * self.transformer.config.patch_size)

        p = self.transformer.config.patch_size
        p_t = self.transformer.config.patch_size_t

        base_size_width = self.transformer.config.sample_width // p
        base_size_height = self.transformer.config.sample_height // p

        if p_t is None:
            # CogVideoX 1.0
            grid_crops_coords = get_resize_crop_region_for_grid(
                (grid_height, grid_width), base_size_width, base_size_height
            )
            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=grid_crops_coords,
                grid_size=(grid_height, grid_width),
                temporal_size=num_frames,
                device=device,
            )
        else:
            # CogVideoX 1.5
            base_num_frames = (num_frames + p_t - 1) // p_t

            freqs_cos, freqs_sin = get_3d_rotary_pos_embed(
                embed_dim=self.transformer.config.attention_head_dim,
                crops_coords=None,
                grid_size=(grid_height, grid_width),
                temporal_size=base_num_frames,
                grid_type="slice",
                max_size=(base_size_height, base_size_width),
                device=device,
            )

        return freqs_cos, freqs_sin

    @property
    def guidance_scale(self):
        return self._guidance_scale

    @property
    def num_timesteps(self):
        return self._num_timesteps

    @property
    def attention_kwargs(self):
        return self._attention_kwargs

    @property
    def current_timestep(self):
        return self._current_timestep

    @property
    def interrupt(self):
        return self._interrupt
    
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

        Saved before scheduler.step. GT mode does not call this function.
        """
        if runtime_ctx is None:
            return

        dump_cfg = runtime_ctx.get("dump_residual_cfg", None)
        if not isinstance(dump_cfg, dict) or not dump_cfg.get("enable", False):
            return

        root = dump_cfg.get("root", None) or runtime_ctx.get("calibrate_root", None)
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
    def __call__(
        self,
        prompt: str | list[str] | None = None,
        negative_prompt: str | list[str] | None = None,
        height: int | None = None,
        width: int | None = None,
        num_frames: int | None = None,
        num_inference_steps: int = 50,
        timesteps: list[int] | None = None,
        guidance_scale: float = 6,
        true_cfg_scale: float | None = None,
        use_dynamic_cfg: bool = False,
        num_videos_per_prompt: int = 1,
        eta: float = 0.0,
        generator: torch.Generator | list[torch.Generator] | None = None,
        latents: torch.FloatTensor | None = None,
        prompt_embeds: torch.FloatTensor | None = None,
        negative_prompt_embeds: torch.FloatTensor | None = None,
        output_type: str = "pil",
        return_dict: bool = True,
        attention_kwargs: dict[str, Any] | None = None,
        callback_on_step_end: Callable[[int, int], None] | PipelineCallback | MultiPipelineCallbacks | None = None,
        callback_on_step_end_tensor_inputs: list[str] = ["latents"],
        max_sequence_length: int = 226,
        runtime_ctx: dict[str, Any] | None = None,
    ) -> CogVideoXPipelineOutput | tuple:
        """
        CogVideoX full inference with optional calibration.

        - No cache logic.
        - CFG cond/uncond branches are forwarded serially.
        - rho mode saves xt/cond/uncond per step.
        - gt mode saves only guided_ref.pt, guided_old.pt, final_latents.pt.

        `true_cfg_scale` is added for consistency with Qwen/FLUX wrappers. If it is
        provided and > 1, it is used as the CFG composition scale and the cfgX.X
        trace-directory tag. Otherwise, `guidance_scale` is used.
        """
        if isinstance(callback_on_step_end, (PipelineCallback, MultiPipelineCallbacks)):
            callback_on_step_end_tensor_inputs = callback_on_step_end.tensor_inputs

        height = height or self.transformer.config.sample_height * self.vae_scale_factor_spatial
        width = width or self.transformer.config.sample_width * self.vae_scale_factor_spatial
        num_frames = num_frames or self.transformer.config.sample_frames

        num_videos_per_prompt = 1

        self.check_inputs(
            prompt,
            height,
            width,
            negative_prompt,
            callback_on_step_end_tensor_inputs,
            prompt_embeds,
            negative_prompt_embeds,
        )

        base_cfg_scale = (
            float(true_cfg_scale)
            if true_cfg_scale is not None and true_cfg_scale > 1.0
            else float(guidance_scale)
        )

        self._guidance_scale = base_cfg_scale
        self._attention_kwargs = attention_kwargs
        self._current_timestep = None
        self._interrupt = False

        if prompt is not None and isinstance(prompt, str):
            batch_size = 1
        elif prompt is not None and isinstance(prompt, list):
            batch_size = len(prompt)
        else:
            batch_size = prompt_embeds.shape[0]

        device = self._execution_device
        do_classifier_free_guidance = base_cfg_scale > 1.0

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
                    "Please set --true_cfg_scale > 1.0, or use guidance_scale > 1.0."
                )

        prompt_embeds, negative_prompt_embeds = self.encode_prompt(
            prompt,
            negative_prompt,
            do_classifier_free_guidance,
            num_videos_per_prompt=num_videos_per_prompt,
            prompt_embeds=prompt_embeds,
            negative_prompt_embeds=negative_prompt_embeds,
            max_sequence_length=max_sequence_length,
            device=device,
        )

        if XLA_AVAILABLE:
            timestep_device = "cpu"
        else:
            timestep_device = device
        timesteps, num_inference_steps = retrieve_timesteps(
            self.scheduler, num_inference_steps, timestep_device, timesteps
        )
        self._num_timesteps = len(timesteps)

        latent_frames = (num_frames - 1) // self.vae_scale_factor_temporal + 1

        patch_size_t = self.transformer.config.patch_size_t
        additional_frames = 0
        if patch_size_t is not None and latent_frames % patch_size_t != 0:
            additional_frames = patch_size_t - latent_frames % patch_size_t
            num_frames += additional_frames * self.vae_scale_factor_temporal

        latent_channels = self.transformer.config.in_channels
        latents = self.prepare_latents(
            batch_size * num_videos_per_prompt,
            latent_channels,
            num_frames,
            height,
            width,
            prompt_embeds.dtype,
            device,
            generator,
            latents,
        )

        extra_step_kwargs = self.prepare_extra_step_kwargs(generator, eta)

        image_rotary_emb = (
            self._prepare_rotary_positional_embeddings(height, width, latents.size(1), device)
            if self.transformer.config.use_rotary_positional_embeddings
            else None
        )

        gt_state = _resolve_gt_run(
            runtime_ctx=runtime_ctx,
            cfg_scale=base_cfg_scale,
            batch_size=latents.shape[0],
            num_denoise_steps=len(timesteps),
        )

        num_warmup_steps = max(len(timesteps) - num_inference_steps * self.scheduler.order, 0)
        with self.progress_bar(total=num_inference_steps) as progress_bar:
            old_pred_original_sample = None
            prev_cond = None
            prev_uncond = None
            perturb_done = False

            for i, t in enumerate(timesteps):
                if self.interrupt:
                    continue

                self._current_timestep = t

                latent_model_input = self.scheduler.scale_model_input(latents, t)
                timestep = t.expand(latent_model_input.shape[0])

                step_cfg_scale = base_cfg_scale
                if true_cfg_scale is None and do_classifier_free_guidance and use_dynamic_cfg:
                    step_cfg_scale = 1 + guidance_scale * (
                        (1 - math.cos(math.pi * ((num_inference_steps - t.item()) / num_inference_steps) ** 5.0)) / 2
                    )
                self._guidance_scale = float(step_cfg_scale)

                if runtime_ctx is not None:
                    runtime_ctx["step"] = i
                    runtime_ctx["num_steps"] = num_inference_steps - 1
                    runtime_ctx["timestep"] = int(t.item()) if hasattr(t, "item") else t
                    runtime_ctx["model"] = "cond"

                noise_pred_text = self.transformer(
                    hidden_states=latent_model_input,
                    encoder_hidden_states=prompt_embeds,
                    timestep=timestep,
                    image_rotary_emb=image_rotary_emb,
                    attention_kwargs=attention_kwargs,
                    return_dict=False,
                )[0]
                noise_pred_text = noise_pred_text.float()

                if do_classifier_free_guidance:
                    if runtime_ctx is not None:
                        runtime_ctx["model"] = "uncond"

                    noise_pred_uncond = self.transformer(
                        hidden_states=latent_model_input,
                        encoder_hidden_states=negative_prompt_embeds,
                        timestep=timestep,
                        image_rotary_emb=image_rotary_emb,
                        attention_kwargs=attention_kwargs,
                        return_dict=False,
                    )[0]
                    noise_pred_uncond = noise_pred_uncond.float()

                    # rho and GT are separated. GT must not dump xt/cond/uncond.
                    if runtime_ctx is not None and runtime_ctx.get("calibrate_mode", "none") == "rho":
                        self._dump_rho_calibration_step(
                            runtime_ctx=runtime_ctx,
                            step_idx=i,
                            latents=latents,
                            noise_pred_text=noise_pred_text,
                            noise_pred_uncond=noise_pred_uncond,
                            guidance_scale=step_cfg_scale,
                        )

                    guided_ref = noise_pred_uncond + float(step_cfg_scale) * (
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

                        guided_old = prev_uncond + float(step_cfg_scale) * (prev_cond - prev_uncond)

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

                if not isinstance(self.scheduler, CogVideoXDPMScheduler):
                    latents = self.scheduler.step(noise_pred, t, latents, **extra_step_kwargs, return_dict=False)[0]
                else:
                    latents, old_pred_original_sample = self.scheduler.step(
                        noise_pred,
                        old_pred_original_sample,
                        t,
                        timesteps[i - 1] if i > 0 else None,
                        latents,
                        **extra_step_kwargs,
                        return_dict=False,
                    )
                latents = latents.to(prompt_embeds.dtype)

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
                            "model": "cogvideox",
                            "prompt_id": int(prompt_id),
                            "calibrate_mode": "gt",
                            "gt_role": gt_state["role"],
                            "gt_run_name": gt_state["run_name"],
                            "perturb_step": None if gt_state["perturb_step"] is None else int(gt_state["perturb_step"]),
                            "num_denoise_steps": int(len(timesteps)),
                            "true_cfg_scale": float(base_cfg_scale),
                            "guidance_scale": float(guidance_scale),
                            "height": int(height),
                            "width": int(width),
                            "num_frames": int(num_frames),
                            "saved_tensors": saved_tensors,
                        },
                    )

        self._current_timestep = None

        if not output_type == "latent":
            latents = latents[:, additional_frames:]
            video = self.decode_latents(latents)
            video = self.video_processor.postprocess_video(video=video, output_type=output_type)
        else:
            video = latents

        self.maybe_free_model_hooks()

        if not return_dict:
            return (video,)

        return CogVideoXPipelineOutput(frames=video)

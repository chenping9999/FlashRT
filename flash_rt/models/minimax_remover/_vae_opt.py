"""FlashRT -- MiniMax-Remover VAE optimization: fp16-native fused kernels.

Replaces diffusers WanRMS_norm.forward (4 full-tensor fp32 passes,
~0.45 ms each at [1,384,1,240,432]) with the FlashRT
``fp16_rms_norm_ncdhw`` CUDA kernel (single-pass, fp16 in/out, fp32
internal statistics, ~0.07 ms -- a ~6x speed-up per call).

Additionally fuses RMS_norm + SiLU in every WanResidualBlock via
``fp16_rms_silu_ncdhw`` (one pass instead of norm->write->silu->write),
and eliminates the redundant fp32 cast in WanUpsample (nearest-exact
upsample is index-only, so fp16 == fp32 bit-for-bit).

Key design decision: **no dtype cast**. The VAE stays in fp16 (cuDNN
already dispatches fp16 tensorop conv kernels). Only the norm/activation
ops are replaced. This preserves fp16's 10-bit mantissa end-to-end,
keeping PSNR at ~40 dB vs the fp16 reference (vs ~15 dB for the
bf16-cast path which loses 3 bits of mantissa across 52 RMS_norm
layers).
"""
from __future__ import annotations

import logging
from typing import Dict

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

# The fp16 fused kernels live in a standalone pybind module
# (flash_rt_minimax_remover) that is opt-in:
#   cmake -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON -DGPU_ARCH=120 ...
_fvk = None
try:
    from flash_rt import flash_rt_minimax_remover as _fvk
except ImportError:
    try:
        import flash_rt_minimax_remover as _fvk
    except ImportError:
        pass

_FP16 = torch.float16
_EPS = 1e-6


def _shape_ncdhw(x: torch.Tensor):
    """Return (B, C, T, H, W) for 4D/5D NCDHW tensors, else None."""
    if x.dim() == 4:
        B, C, H, W = x.shape
        return B, C, 1, H, W
    if x.dim() == 5:
        B, C, T, H, W = x.shape
        return B, C, T, H, W
    return None


def _prep_gamma_bias(gamma, bias):
    """Return contiguous fp16 (gamma_flat, bias_ptr) for the kernels."""
    if gamma.dtype != _FP16:
        gamma = gamma.to(_FP16)
    gamma_flat = gamma.contiguous().view(-1)
    if isinstance(bias, torch.Tensor):
        bias_flat = bias.contiguous().view(-1).to(_FP16)
        return gamma_flat, bias_flat.data_ptr()
    return gamma_flat, 0


def _ref_rms_norm(gamma, bias, x):
    """Reference fallback (fp32 stats, fp16 out) -- WanRMS_norm semantics."""
    C = x.shape[1]
    scale = C ** 0.5
    out = torch.nn.functional.normalize(x.float(), dim=1).to(x.dtype)
    return out * scale * gamma + (bias if isinstance(bias, torch.Tensor) else 0.0)


def _flashrt_fp16_rms_norm_forward(self, x: torch.Tensor) -> torch.Tensor:
    """FlashRT fp16-native RMS_norm replacement for WanRMS_norm.forward.

    Computes: y = (x / rms(x)) * gamma + bias  (fp16 in/out, fp32 stats)
    which equals WanRMS_norm's F.normalize(x, dim=1) * sqrt(C) * gamma.
    """
    shp = _shape_ncdhw(x)
    if shp is None:
        return self._orig_forward(x)
    B, C, T, H, W = shp

    if x.dtype != _FP16:
        x = x.to(_FP16)
    if not x.is_contiguous():
        x = x.contiguous()

    gamma_flat, bias_ptr = _prep_gamma_bias(self.gamma, self.bias)
    out = torch.empty_like(x)
    stream = torch.cuda.current_stream().cuda_stream
    rc = _fvk.fp16_rms_norm_ncdhw(
        x.data_ptr(), gamma_flat.data_ptr(), bias_ptr,
        out.data_ptr(), B, C, T, H, W, _EPS, stream)
    if rc != 0:
        return _ref_rms_norm(self.gamma, self.bias, x)
    return out


class _FusedRmsSilu(nn.Module):
    """Drop-in for WanRMS_norm that outputs silu(rms_norm(x)) in one kernel.

    Installed as WanResidualBlock.norm1/norm2 while the block's
    ``nonlinearity`` is swapped to ``Identity`` -- so the existing
    ``forward`` (norm1 -> nonlinearity -> conv1 -> norm2 -> nonlinearity
    -> conv2) silently becomes a fused norm+silu path with no rewrite of
    the complex causal-cache logic.
    """

    def __init__(self, gamma, bias):
        super().__init__()
        self.gamma = gamma
        self.bias = bias

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shp = _shape_ncdhw(x)
        if shp is None:
            return torch.nn.functional.silu(_ref_rms_norm(self.gamma, self.bias, x))
        B, C, T, H, W = shp

        if x.dtype != _FP16:
            x = x.to(_FP16)
        if not x.is_contiguous():
            x = x.contiguous()

        gamma_flat, bias_ptr = _prep_gamma_bias(self.gamma, self.bias)
        out = torch.empty_like(x)
        stream = torch.cuda.current_stream().cuda_stream
        rc = _fvk.fp16_rms_silu_ncdhw(
            x.data_ptr(), gamma_flat.data_ptr(), bias_ptr,
            out.data_ptr(), B, C, T, H, W, _EPS, stream)
        if rc != 0:
            return torch.nn.functional.silu(_ref_rms_norm(self.gamma, self.bias, x))
        return out


def install_flashrt_fp16_rms_norm(vae) -> int:
    """Replace WanRMS_norm.forward (attention sites) with the FlashRT fp16
    kernel, and fuse norm+silu inside every WanResidualBlock.

    Attention-block norms (WanAttentionBlock) keep the plain rms_norm
    kernel (no SiLU follows them).  Residual-block norms (norm1/norm2)
    are swapped to the fused ``fp16_rms_silu_ncdhw`` kernel and the
    block's SiLU is set to Identity, so the existing ``forward`` runs the
    fused path without touching the causal-cache logic.

    Returns the count of patched modules.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import (
        WanRMS_norm, WanResidualBlock)

    n_fused = 0
    for blk in vae.modules():
        if isinstance(blk, WanResidualBlock):
            blk.norm1 = _FusedRmsSilu(blk.norm1.gamma, blk.norm1.bias)
            blk.norm2 = _FusedRmsSilu(blk.norm2.gamma, blk.norm2.bias)
            blk.nonlinearity = nn.Identity()
            n_fused += 1

    if not getattr(WanRMS_norm, "_flashrt_fp16_patched", False):
        WanRMS_norm._orig_forward = WanRMS_norm.forward
        WanRMS_norm.forward = _flashrt_fp16_rms_norm_forward
        WanRMS_norm._flashrt_fp16_patched = True
        logger.info("[minimax-vae] patched WanRMS_norm.forward -> FlashRT "
                    "fp16_rms_norm_ncdhw (fp16-native, no cast, ~6x faster)")

    logger.info("[minimax-vae] %d WanResidualBlock(s) now use fused "
                "fp16_rms_silu_ncdhw (norm+silu in one pass)", n_fused)
    return n_fused


def _install_wan_upsample_no_cast(vae) -> int:
    """Eliminate the redundant fp32 cast in WanUpsample.

    WanUpsample.forward does ``super().forward(x.float()).type_as(x)``.
    For ``nearest-exact`` mode the upsample is pure index selection (no
    arithmetic), so fp16 and fp32 give bit-identical results -- the cast
    is wasted bandwidth.  This swaps it to a fp16-native forward.
    """
    from diffusers.models.autoencoders.autoencoder_kl_wan import WanUpsample

    if not getattr(WanUpsample, "_flashrt_nocast", False):
        _orig_upsample_forward = WanUpsample.forward

        def _no_cast_forward(self, x):
            if self.mode == "nearest-exact":
                return nn.Upsample.forward(self, x)
            return _orig_upsample_forward(self, x)

        WanUpsample.forward = _no_cast_forward
        WanUpsample._flashrt_nocast = True
        logger.info("[minimax-vae] patched WanUpsample.forward -> "
                    "fp16-native (nearest-exact cast eliminated)")

    return sum(1 for m in vae.modules() if isinstance(m, WanUpsample))


def install_vae_optimizations(vae, dtype=None) -> Dict:
    """Apply VAE optimizations: fp16-native fused RMS_norm + RMS_SiLU kernels
    + WanUpsample cast elimination.

    No dtype cast is applied -- the VAE stays fp16. Only norm/activation
    ops are replaced with FlashRT fp16 CUDA kernels.

    Requires the standalone ``flash_rt_minimax_remover`` module, built with:
        cmake -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON -DGPU_ARCH=120 ...

    Args:
        vae: loaded ``diffusers.AutoencoderKLWan`` instance.
        dtype: ignored (kept for API compat with the old bf16 interface).

    Returns:
        stats dict.

    Raises:
        ImportError if flash_rt_minimax_remover is not built.
    """
    if _fvk is None:
        raise ImportError(
            "flash_rt_minimax_remover not found; rebuild FlashRT with: "
            "cmake -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON -DGPU_ARCH=120 ...")
    n_fused_blocks = install_flashrt_fp16_rms_norm(vae)
    n_upsample = _install_wan_upsample_no_cast(vae)
    return {
        "n_fused_res_blocks": n_fused_blocks,
        "n_upsample_nocast": n_upsample,
        "vae_dtype": str(next(vae.parameters()).dtype),
    }


@torch.no_grad()
def profile_vae(pipe, images_tensor, masks_infer, height, width, num_frames,
                iterations=6, num_inference_steps=12, seed=42,
                device=torch.device("cuda:0")) -> Dict[str, float]:
    """Time VAE encode + transformer denoise + VAE decode separately."""
    from einops import rearrange
    from diffusers.utils.torch_utils import randn_tensor

    vae = pipe.vae
    transformer = pipe.transformer
    scheduler = pipe.scheduler

    scheduler.set_timesteps(num_inference_steps, device=device)
    timesteps = scheduler.timesteps
    num_channels_latents = 16
    vae_scale_factor_temporal = pipe.vae_scale_factor_temporal
    vae_scale_factor_spatial = pipe.vae_scale_factor_spatial
    num_latent_frames = (num_frames - 1) // vae_scale_factor_temporal + 1

    shape = (1, num_channels_latents, num_latent_frames,
             height // vae_scale_factor_spatial,
             width // vae_scale_factor_spatial)
    generator = torch.Generator(device=device).manual_seed(seed)
    latents = randn_tensor(shape, generator=generator, device=device,
                           dtype=torch.float16)

    masks = pipe.expand_masks(masks_infer, iterations)
    masks = pipe.resize(masks, height, width).to(device).half()
    masks[masks > 0] = 1
    images = rearrange(images_tensor, "f h w c -> c f h w")
    images = pipe.resize(images[None, ...], height, width).to(device).half()
    masked_images = images * (1 - masks)

    latents_mean = (torch.tensor(vae.config.latents_mean)
                    .view(1, vae.config.z_dim, 1, 1, 1)
                    .to(vae.device, torch.float16))
    latents_std = 1.0 / torch.tensor(vae.config.latents_std).view(
        1, vae.config.z_dim, 1, 1, 1).to(vae.device, torch.float16)

    vae_dtype = next(vae.parameters()).dtype

    torch.cuda.synchronize()
    ev0 = torch.cuda.Event(enable_timing=True)
    ev_enc = torch.cuda.Event(enable_timing=True)
    ev0.record()
    masked_latents = vae.encode(masked_images.to(vae_dtype)).latent_dist.mode()
    masks_latents = vae.encode((2 * masks - 1.0).to(vae_dtype)).latent_dist.mode()
    masked_latents = (masked_latents - latents_mean) * latents_std
    masks_latents = (masks_latents - latents_mean) * latents_std
    ev_enc.record()
    torch.cuda.synchronize()
    vae_encode_ms = ev0.elapsed_time(ev_enc)

    ev_denoise_start = torch.cuda.Event(enable_timing=True)
    ev_denoise_end = torch.cuda.Event(enable_timing=True)
    ev_denoise_start.record()
    for i, t in enumerate(timesteps):
        latent_model_input = latents.to(torch.float16)
        latent_model_input = torch.cat(
            [latent_model_input, masked_latents, masks_latents], dim=1)
        timestep = t.expand(latents.shape[0])
        noise_pred = transformer(
            hidden_states=latent_model_input.half(), timestep=timestep)[0]
        latents = scheduler.step(noise_pred, t, latents,
                                 return_dict=False)[0]
    ev_denoise_end.record()
    torch.cuda.synchronize()
    denoise_ms = ev_denoise_start.elapsed_time(ev_denoise_end)

    latents = latents.half() / latents_std + latents_mean
    ev_dec_start = torch.cuda.Event(enable_timing=True)
    ev_dec_end = torch.cuda.Event(enable_timing=True)
    ev_dec_start.record()
    video = vae.decode(latents.to(vae_dtype), return_dict=False)[0]
    ev_dec_end.record()
    torch.cuda.synchronize()
    vae_decode_ms = ev_dec_start.elapsed_time(ev_dec_end)

    return {
        "vae_encode_ms": vae_encode_ms,
        "denoise_ms": denoise_ms,
        "vae_decode_ms": vae_decode_ms,
        "total_ms": vae_encode_ms + denoise_ms + vae_decode_ms,
    }

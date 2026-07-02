"""FlashRT -- MiniMax-Remover FP8 kernelized inference pipeline.

FP8 (W8A8) version for full-frame inpainting. Unlike NVFP4 (W4A4) which
produces black/drift outputs on full-frame large latents, FP8 stays close
to the fp16 reference: end-to-end cosine >= 0.999 and PSNR ~35-41 dB vs
fp16 on full-frame clips.

Uses static calibration: the first inference call runs in dynamic-FP8
calibration mode (accumulating activation amax on GPU), then freezes to a
static act_scale for all subsequent calls (zero CPU sync overhead in the
steady state).
"""

import logging
import os

import torch

logger = logging.getLogger(__name__)

from flash_rt.models.minimax_remover._utils import load_fp8_kernels


def _import_runtime_fp8():
    """Lazy import FP8 runtime dependencies."""
    missing = []
    for dep in ("diffusers", "einops", "triton"):
        try:
            __import__(dep)
        except ImportError:
            missing.append(dep)
    if missing:
        raise RuntimeError(
            f"MiniMax-Remover FP8 requires {', '.join(missing)}. "
            "Install: pip install -e '.[minimax-remover]'"
        )
    from ._fp8_linear import install_flashrt_fp8, set_calibration, freeze_calibration
    from ._kern_block import install_fused_blocks, install_fa2_attention
    return install_flashrt_fp8, set_calibration, freeze_calibration, \
           install_fused_blocks, install_fa2_attention


class MiniMaxRemoverPipelineFP8:
    """FP8 (W8A8) kernelized inference pipeline for full-frame inpainting.

    Unlike NVFP4 which is calibrated only for small cropped regions, FP8
    works on full-frame large latents: end-to-end cosine >= 0.999 and PSNR
    ~35-41 dB vs the fp16 reference on full-frame clips.

    The first ``__call__`` runs in calibration mode (dynamic FP8 + amax
    accumulation). At the end of that call the static act_scale is frozen
    and all subsequent calls use the frozen scale (zero CPU sync, suitable
    for CUDA Graph capture).

    Args:
        pipe: loaded diffusers pipeline
        num_inference_steps: denoise steps (12)
        fp8_target: "all" or "ffn_only"
        use_bf16: run transformer in bf16 (default False, keeps fp16)
        calib_margin: act_scale margin multiplier (1.1)
    """

    def __init__(self, pipe, num_inference_steps=12, fp8_target="all",
                 use_bf16=False, calib_margin=1.1):
        self.fvk = load_fp8_kernels()
        (install_flashrt_fp8, set_calibration, freeze_calibration,
         install_fused_blocks, install_fa2_attention) = _import_runtime_fp8()

        self.pipe = pipe
        self.transformer = pipe.transformer
        self.num_inference_steps = num_inference_steps
        self.calib_margin = calib_margin
        self._calibrated = False

        self._set_calibration = lambda on: set_calibration(self.transformer, on)
        self._freeze_calibration = lambda: freeze_calibration(
            self.transformer, margin=self.calib_margin)

        fp8_target_env = os.environ.get("FLASHRT_FP8_TARGET", fp8_target)
        n_lin = install_flashrt_fp8(self.transformer,
                                    verbose=True, target=fp8_target_env)
        logger.info("MiniMax-Remover FP8: target=%r, %d Linears -> FP8 W8A8 GEMM",
                    fp8_target_env, n_lin)

        if use_bf16:
            self.transformer.to(torch.bfloat16)
            logger.info("MiniMax-Remover FP8: transformer -> bf16")

        n_block = install_fused_blocks(self.transformer)
        logger.info("MiniMax-Remover FP8: %d blocks -> fused norm/gate/gelu kernels",
                    n_block)

        n_attn = install_fa2_attention(self.transformer)
        logger.info("MiniMax-Remover FP8: %d attention blocks -> kernel backend",
                    n_attn)

        self._install_calibrated_call()

    def _install_calibrated_call(self):
        """Wrap pipe.__call__ so the first invocation calibrates FP8 scales.

        On the first call: enable calibration mode, run the full diffusers
        __call__ (which runs num_inference_steps transformer forwards
        accumulating amax on GPU), then freeze the static act_scale.
        Subsequent calls use the frozen scales directly.
        """
        _orig_call = self.pipe.__class__.__call__

        pipeline_fp8 = self

        @torch.no_grad()
        def _calibrated_call(pipe_self, *args, **kwargs):
            if not pipeline_fp8._calibrated:
                logger.info("MiniMax-Remover FP8: calibration mode "
                            "(first call, dynamic FP8 + amax accumulation)")
                pipeline_fp8._set_calibration(True)

            result = _orig_call(pipe_self, *args, **kwargs)

            if not pipeline_fp8._calibrated:
                n = pipeline_fp8._freeze_calibration()
                pipeline_fp8._calibrated = True
                logger.info("MiniMax-Remover FP8: calibration done, "
                            "froze %d static act_scales (margin=%.2f)",
                            n, pipeline_fp8.calib_margin)
            return result

        self.pipe.__class__.__call__ = _calibrated_call

    def __call__(self, *args, **kwargs):
        return self.pipe(*args, **kwargs)

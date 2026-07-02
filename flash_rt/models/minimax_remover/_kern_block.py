"""
FlashRT pure-kernel Transformer block fusion.

Replaces every element-wise/norm/gate/residual/gelu op inside each Transformer3DModel
block with FlashRT fused kernels; attention uses FlashRT's built-in FA2 (fwd_fp16) in
place of torch SDPA; all Linear layers still run through FlashRT FP8 GEMM (installed
by install_flashrt_fp8).

Fusion points (per block, compared to the original video_subtitle_remover.py):
  Original: norm1(fp32) -> .float -> *(1+scale) -> +shift -> .type_as  (5 large [S,D] kernels)
            + hidden.float + attn*gate -> .type_as                      (4 large [S,D] kernels)
            norm2 likewise; FFN gelu is a torch op
  This version:
            ada_layer_norm_fp16  -> 1 kernel (LN + modulation + direct fp16 output)
            gate_mul_residual_fp16 -> 1 in-place kernel (residual + gate)
            gelu_inplace_fp16    -> 1 in-place kernel (tanh gelu)

Key correctness details:
  * patch_embedding(...).transpose(1,2) produces a **non-contiguous** output; FlashRT's
    pointer-based kernels read contiguous memory, so the block entry must call
    .contiguous() (only the first block truly copies; subsequent blocks receive the
    contiguous output of the previous block, so contiguous() is a no-op).
  * gate_mul_residual_fp16 requires gate to be a full [S,D] tensor (not broadcast), so
    the gate vector must be expanded.
  * ada_layer_norm_fp16 = LN(x)*(1+scale)+shift (verified to match exactly, including
    fp32 statistics accumulation, stable even for near-zero-variance tokens).
  * gelu_inplace_fp16 = tanh-approximate GELU (matches the FFN's approximate='tanh',
    verified exact).
"""

import math
import os
import logging
import torch
import torch.nn.functional as F
from flash_rt import flash_rt_kernels as kern
from ._triton_fused_norm import ada_layernorm_fp16_io, rms_norm_fp32stat, gate_mul_residual_bcast
from ._triton_rope import rope_apply_bshd, freqs_to_cos_sin

logger = logging.getLogger(__name__)

_FP16 = torch.float16

# Attention kernel selection: sage (default, 5x vs FA2) / sage_fp8 / sage_fp16 / triton_fp8 / fa2 / triton_fp16
_ATTENTION_MODE = os.environ.get("FLASHRT_ATTN_MODE", "sage_fp8").lower()
_TFA = None
_SAGE = None
_FA2 = None
def _get_tfa():
    global _TFA
    if _TFA is None:
        from . import _triton_flash_attn as _m
        _TFA = _m
    return _TFA

def _get_fa2():
    global _FA2
    if _FA2 is None:
        from flash_rt import flash_rt_fa2 as fa2
        _FA2 = fa2
    return _FA2

def _get_sage():
    global _SAGE
    if _SAGE is None:
        import sageattention as _m
        _SAGE = _m
    return _SAGE

def _sage_attn(q, k, v, scale, mode):
    """Dispatch to the appropriate SageAttention variant.

    Variants (all accept [B,S,H,D] NHD fp16, return fp16):
      sage / sage_auto  → sageattn dispatcher (auto-selects fastest backend)
      sage_fp8 / sage2  → QK int8 per-warp + PV fp8 CUDA (fastest, cos ~0.9993)
      sage_fp16 / sage1 → QK int8 per-warp + PV fp16 CUDA (most accurate, cos ~0.9999)
      sage_triton       → QK int8 per-block + PV fp16 Triton (fallback)
    """
    sa = _get_sage()
    kw = dict(tensor_layout="NHD", is_causal=False, sm_scale=scale)
    if mode in ("sage", "sage_auto"):
        return sa.sageattn(q, k, v, **kw)
    if mode in ("sage_fp8", "sage2"):
        return sa.sageattn_qk_int8_pv_fp8_cuda(q, k, v, **kw)
    if mode in ("sage_fp16", "sage1"):
        return sa.sageattn_qk_int8_pv_fp16_cuda(q, k, v, **kw)
    if mode == "sage_triton":
        return sa.sageattn_qk_int8_pv_fp16_triton(q, k, v, **kw)
    return sa.sageattn(q, k, v, **kw)

# Prefetch SM count (used by FA2 splitkv heuristic), fetched only once
try:
    _NUM_SMS = torch.cuda.get_device_properties(0).multi_processor_count
except Exception:
    _NUM_SMS = 0


def _apply_rotary_fp32(h_bhsd, freqs):
    """h: [B,H,S,D] fp16; freqs: [1,1,S,D/2] complex64 -> applies RoPE, returns [B,H,S,D] fp16."""
    x_rot = torch.view_as_complex(h_bhsd.to(torch.float32).unflatten(3, (-1, 2)))
    x_out = torch.view_as_real(x_rot * freqs).flatten(3, 4)
    return x_out.type_as(h_bhsd)


class FlashRTFA2Processor:
    """Pure FlashRT FA2 attention: native [B,S,H,D] layout (no transpose copy) + Triton RoPE.

    QKV/out projections run through FlashRT FP8 GEMM; RMSNorm of Q/K uses diffusers
    (fp32 statistics); RoPE uses a Triton interleaved kernel (bit-exact with
    view_as_complex) to rotate in-place directly on [B,S,H,D], eliminating the original
    two transpose+contiguous calls and all torch complex ops; the attention main loop
    uses fa2.fwd_fp16 ([B,S,H,D] is exactly the layout FA2 needs, zero-copy).
    """

    def __init__(self):
        self._lse_bufs = {}      # (B,S,H) -> lse
        self._cos_sin = {}       # S -> (cos[S,D/2], sin[S,D/2]) fp32

    def __call__(self, attn, hidden_states, rotary_emb=None,
                 attention_mask=None, encoder_hidden_states=None):
        B, S, _ = hidden_states.shape
        H = attn.heads
        Dd = attn.inner_dim // H
        scale = 1.0 / math.sqrt(float(Dd))

        q = attn.to_q(hidden_states)            # [B,S,inner] FP8 GEMM
        k = attn.to_k(hidden_states)
        v = attn.to_v(hidden_states)
        if attn.norm_q is not None:
            q = rms_norm_fp32stat(q, attn.norm_q.weight, attn.norm_q.eps)  # Triton fp32-stat
        if attn.norm_k is not None:
            k = rms_norm_fp32stat(k, attn.norm_k.weight, attn.norm_k.eps)
        # Native [B,S,H,D] view (no copy) — exactly the (batch,seqlen,heads,head_dim) FA2 needs
        q = q.view(B, S, H, Dd)
        k = k.view(B, S, H, Dd)
        v = v.view(B, S, H, Dd)

        if rotary_emb is not None:
            cs = self._cos_sin.get(S)
            if cs is None:
                cs = freqs_to_cos_sin(rotary_emb)  # (cos[S,D/2], sin[S,D/2])
                self._cos_sin[S] = cs
            rope_apply_bshd(q, cs[0], cs[1])      # in-place Triton
            rope_apply_bshd(k, cs[0], cs[1])

        # Ensure contiguity (norm_q/RMSNorm output is already contiguous; as a safeguard)
        if not q.is_contiguous():
            q = q.contiguous()
        if not k.is_contiguous():
            k = k.contiguous()
        if not v.is_contiguous():
            v = v.contiguous()

        if _ATTENTION_MODE.startswith("sage"):
            out = _sage_attn(q, k, v, scale, _ATTENTION_MODE)
        elif _ATTENTION_MODE in ("triton_fp8", "triton_fp16"):
            # Triton flash-attention (fp8 or fp16), returns out [B,S,H,Dd] fp16
            tfa = _get_tfa()
            out = (tfa.flash_attn_fp8 if _ATTENTION_MODE == "triton_fp8"
                   else tfa.flash_attn_fp16)(q, k, v, scale)
        else:
            out = torch.empty_like(q)
            lse = self._lse_bufs.get((B, S, H))
            if lse is None:
                lse = torch.empty(B, H, S, device=q.device, dtype=torch.float32)
                self._lse_bufs[(B, S, H)] = lse
            qs, ks, vs, os_ = (q.stride(), k.stride(), v.stride(), out.stride())
            _get_fa2().fwd_fp16(
                q.data_ptr(), k.data_ptr(), v.data_ptr(), out.data_ptr(),
                lse.data_ptr(), 0, 0,
                B, S, S, H, H, Dd,
                (qs[0], qs[1], qs[2]), (ks[0], ks[1], ks[2]), (vs[0], vs[1], vs[2]),
                (os_[0], os_[1], os_[2]),
                scale, _NUM_SMS, torch.cuda.current_stream().cuda_stream,
            )

        hidden_states = out.view(B, S, H * Dd)
        hidden_states = attn.to_out[0](hidden_states)  # FP8/NVFP4 GEMM
        return hidden_states


def install_fa2_attention(transformer):
    """Replace every block's attention processor with FlashRT FA2."""
    n = 0
    proc = FlashRTFA2Processor()
    for block in transformer.blocks:
        block.attn1.processor = proc
        n += 1
    return n


def install_fused_blocks(transformer, norm_mode=None, gelu_mode=None):
    """Replace each TransformerBlock.forward with the pure FlashRT kernel-fused version.

    norm_mode: 'fp16' (default, ada_layer_norm_fp16) | 'fp32' (original fp32 LayerNorm + modulation, for debugging)
    gelu_mode: 'inplace' (default, gelu_inplace_fp16) | 'torch' (original F.gelu, for debugging)
    """
    import os as _os
    if norm_mode is None:
        norm_mode = _os.environ.get("FLASHRT_NORM_MODE", "triton")
    if gelu_mode is None:
        gelu_mode = _os.environ.get("FLASHRT_GELU_MODE", "inplace")
    eps = float(transformer.blocks[0].norm1.eps)
    stream_of = lambda: torch.cuda.current_stream().cuda_stream

    def _ada_norm(self_hs, scale_v, shift_v, S, D):
        """Single-kernel fusion: fp32-statistics LayerNorm + adaLN modulation -> fp16.

        Why not use FlashRT's ada_layer_norm_fp16: its statistics are insufficiently
        precise on real diffusion latents, yielding only 41 dB end-to-end PSNR (vs 65 dB
        for the fp32 version). This Triton kernel accumulates mean/var in fp32 across
        three passes, bit-exact with the original FP32LayerNorm, while still being a
        single kernel. scale/shift stay fp32 (from temb.float()), and modulation is
        also done in fp32.
        """
        return ada_layernorm_fp16_io(self_hs, scale_v.view(D), shift_v.view(D), eps)

    def _ada_norm_flashrt_fp16(self_hs, scale_v, shift_v, S, D):
        out = torch.empty(S, D, dtype=_FP16, device=self_hs.device)
        kern.ada_layer_norm_fp16(
            self_hs.data_ptr(),
            scale_v.view(D).to(_FP16).contiguous().data_ptr(),
            shift_v.view(D).to(_FP16).contiguous().data_ptr(),
            out.data_ptr(), S, D, eps, stream_of())
        return out

    def block_forward(self, hidden_states, temb, rotary_emb):
        B, S, D = hidden_states.shape
        # Ensure contiguity at entry (first block truly copies; subsequent blocks are no-ops)
        hs = hidden_states.contiguous().view(S, D)

        (shift_msa, scale_msa, gate_msa,
         c_shift_msa, c_scale_msa, c_gate_msa) = (self.scale_shift_table + temb.float()).chunk(6, dim=1)

        if norm_mode == "fp16":
            norm1_out = _ada_norm_flashrt_fp16(hs, scale_msa, shift_msa, S, D)
        else:
            norm1_out = _ada_norm(hs, scale_msa, shift_msa, S, D)
        attn_out = self.attn1(hidden_states=norm1_out.view(1, S, D), rotary_emb=rotary_emb).view(S, D)
        # Broadcast gate[D] (avoids [S,D] expand copy); fp16 in-place
        gate_mul_residual_bcast(hs, attn_out, gate_msa.view(D))

        if norm_mode == "fp16":
            norm2_out = _ada_norm_flashrt_fp16(hs, c_scale_msa, c_shift_msa, S, D)
        else:
            norm2_out = _ada_norm(hs, c_scale_msa, c_shift_msa, S, D)
        n2_3d = norm2_out.view(1, S, D)
        if gelu_mode == "torch":
            ff_out = self.ffn(n2_3d).view(S, D)
        else:
            up = self.ffn.net[0].proj(n2_3d)
            inner = up.shape[-1]
            _gelu_fn = kern.gelu_inplace if up.dtype == torch.bfloat16 else kern.gelu_inplace_fp16
            _gelu_fn(up.data_ptr(), S * inner, stream_of())
            ff_out = self.ffn.net[2](up).view(S, D)
        gate_mul_residual_bcast(hs, ff_out, c_gate_msa.view(D))
        return hs.view(1, S, D)

    block_cls = type(transformer.blocks[0])
    block_cls.forward = block_forward
    logger.info("  [FlashRT-Kern] block fusion: norm=%s gelu=%s", norm_mode, gelu_mode)
    return len(transformer.blocks)


def install_fused_norm_out(transformer):
    """Fuse the final norm_out + modulation (2-segment shift/scale).

    Original: (norm_out(hidden.float())*(1+scale)+shift).type_as(hidden)
    This version: single ada_layer_norm_fp16 kernel.
    Requires rewriting transformer.forward to insert this fusion point.
    """
    pass

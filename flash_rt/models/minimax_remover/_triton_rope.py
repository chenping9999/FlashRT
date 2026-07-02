"""
Triton fused interleaved RoPE (native [B,S,H,D] layout, in-place)

The original version (including the previous fp32 optimization) converts Q/K from
[B,S,H*D] to [B,H,S,D] inside attention, performs complex rotation using
torch view_as_complex/view_as_real, then converts back to [B,S,H*D] — this produces
multiple large copies plus several elementwise kernels (profiling shows they account
for ~28% of the torch elementwise overhead).

This kernel performs the interleaved complex rotation directly on the [B,S,H*D] layout
required by FA2 (bit-for-bit identical to MiniMax's view_as_complex(unflatten(-1,2))),
as a single in-place kernel, eliminating two transpose+contiguous operations and all
torch complex ops.

Rotation definition (bit-for-bit identical to the original; cos/sin taken from
freqs = cos + i*sin):
  out[..., 2i]   = x[...,2i]*cos[s,i] - x[...,2i+1]*sin[s,i]
  out[..., 2i+1] = x[...,2i]*sin[s,i] + x[...,2i+1]*cos[s,i]
freqs depends only on S (broadcast over B/H).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _rope_bshd_kernel(X, COS, SIN, Nrows, S, H, Dhalf,
                      BLOCK_D: tl.constexpr, IO_DTYPE: tl.constexpr):
    r = tl.program_id(0)  # row index (b*S*H + h)
    s = (r // H) % S
    x_ptr = X + r * (2 * Dhalf)
    cos_ptr = COS + s * Dhalf
    sin_ptr = SIN + s * Dhalf
    offs = tl.arange(0, BLOCK_D)
    mask = offs < Dhalf
    cos = tl.load(cos_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    sin = tl.load(sin_ptr + offs, mask=mask, other=0.0).to(tl.float32)
    x0 = tl.load(x_ptr + 2 * offs, mask=mask, other=0.0).to(tl.float32)
    x1 = tl.load(x_ptr + 2 * offs + 1, mask=mask, other=0.0).to(tl.float32)
    o0 = x0 * cos - x1 * sin
    o1 = x0 * sin + x1 * cos
    tl.store(x_ptr + 2 * offs, o0.to(IO_DTYPE), mask=mask)
    tl.store(x_ptr + 2 * offs + 1, o1.to(IO_DTYPE), mask=mask)


def rope_apply_bshd(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """x: [B,S,H,D] fp16/bf16 contiguous (FA2 native layout), rotated in-place.
    cos/sin: [S, D//2] fp32. Returns x (modified in-place).
    """
    assert x.is_contiguous()
    B, S, H, D = x.shape
    Dhalf = D // 2
    Nrows = B * S * H
    BLOCK_D = triton.next_power_of_2(Dhalf) if hasattr(triton, "next_power_of_2") else (1 << (Dhalf - 1).bit_length())
    if BLOCK_D < 16:
        BLOCK_D = 16
    IO_DTYPE = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
    _rope_bshd_kernel[(Nrows,)](
        x, cos.contiguous().to(torch.float32), sin.contiguous().to(torch.float32),
        Nrows, S, H, Dhalf, BLOCK_D=BLOCK_D, num_warps=4, IO_DTYPE=IO_DTYPE,
    )
    return x


def freqs_to_cos_sin(freqs: torch.Tensor):
    """freqs: complex [1,1,S,D//2] -> (cos[S,D//2] fp32, sin[S,D2] fp32) on freqs.device."""
    f = freqs.squeeze().to(torch.complex64)  # [S, D//2]
    return f.real.contiguous().to(torch.float32), f.imag.contiguous().to(torch.float32)

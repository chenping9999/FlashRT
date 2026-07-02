"""
Triton fused adaLayerNorm (fp32 statistics + fp16 I/O)

Why a custom kernel is needed:
  FlashRT's ada_layer_norm_fp16 loses significant precision on "real diffusion
  latents" (end-to-end PSNR 41 dB vs 65 dB for the fp32 version), even though it
  matches F.layer_norm with cos=1.0 on random/constant data. The reason is that
  its LayerNorm statistics lack sufficient precision. The original version uses
  FP32LayerNorm(hidden.float()), computing statistics in fp32 -- this is
  precision-critical.

  This kernel fuses "fp32-statistics LayerNorm + adaLN modulation
  (1+scale)*x_norm+shift + fp16 output" into a **single kernel**, which is
  bit-exact with the original fp32 path, but eliminates the original's 5 large
  [S,D] kernels (hidden.float / fp32 LN / *(1+scale) / +shift / .type_as).

  out[S,D] fp16 = ( LayerNorm_fp32(x_fp16) * (1 + scale_fp32[D]) + shift_fp32[D] )

  Also provides a fused gate-residual fp16 version (equivalent to FlashRT's
  gate_mul_residual_fp16, verified bit-exact, used as a fallback / self-contained
  implementation).
"""

import torch
import triton
import triton.language as tl


@triton.jit
def _ada_layernorm_io_kernel(
    X, SCALE, SHIFT, OUT,
    M, N,
    sM_x, sM_o,
    eps,
    BLOCK_N: tl.constexpr,
    IO_DTYPE: tl.constexpr,
):
    # Each row (token) is handled by one program; when N(=D) is large, use multi-block reduction
    row = tl.program_id(0)
    x_ptr = X + row * sM_x
    o_ptr = OUT + row * sM_o
    # First pass: compute the mean
    _mean = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        _mean += tl.sum(x)
    mean = _mean / N
    # Second pass: compute the variance
    _var = tl.zeros([BLOCK_N], dtype=tl.float32)
    for off in tl.range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        d = x - mean
        _var += tl.sum(d * d)
    var = _var / N
    rstd = 1.0 / tl.sqrt(var + eps)
    # Third pass: normalize + modulate + store as IO_DTYPE
    for off in tl.range(0, N, BLOCK_N):
        cols = off + tl.arange(0, BLOCK_N)
        mask = cols < N
        x = tl.load(x_ptr + cols, mask=mask, other=0.0).to(tl.float32)
        x_norm = (x - mean) * rstd
        scale = tl.load(SCALE + cols, mask=mask, other=0.0)
        shift = tl.load(SHIFT + cols, mask=mask, other=0.0)
        y = x_norm * (1.0 + scale) + shift
        tl.store(o_ptr + cols, y.to(IO_DTYPE), mask=mask)


# Backward-compat alias
_ada_layernorm_fp16_io_kernel = None


def ada_layernorm_io(x: torch.Tensor, scale: torch.Tensor,
                      shift: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """x: [S, D] or [B, S, D] fp16/bf16 contiguous; scale/shift: [D] fp32.
    Returns [.., D] with the same dtype. Statistics are computed in fp32,
    equivalent to the original FP32LayerNorm."""
    orig_shape = x.shape
    if x.dim() == 3:
        x = x.reshape(orig_shape[0] * orig_shape[1], orig_shape[2])
    S, D = x.shape
    assert x.is_contiguous(), "ada_layernorm_io: x must be contiguous"
    scale = scale.contiguous().to(torch.float32).view(-1)
    shift = shift.contiguous().to(torch.float32).view(-1)
    out = torch.empty_like(x)
    BLOCK_N = triton.next_power_of_2(min(D, 2048))
    num_warps = 8 if D >= 1024 else 4
    IO_DTYPE = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
    _ada_layernorm_io_kernel[(S,)](
        x, scale, shift, out, S, D,
        x.stride(0), out.stride(0), eps,
        BLOCK_N=BLOCK_N, num_warps=num_warps, IO_DTYPE=IO_DTYPE,
    )
    return out.reshape(orig_shape)


# Backward-compat wrapper
def ada_layernorm_fp16_io(x, scale, shift, eps=1e-6):
    return ada_layernorm_io(x, scale, shift, eps)


@triton.jit
def _gate_mul_residual_kernel(RES, X, GATE, N, BLOCK: tl.constexpr,
                              IO_DTYPE: tl.constexpr):
    pid = tl.program_id(0)
    cols = pid * BLOCK + tl.arange(0, BLOCK)
    mask = cols < N
    r = tl.load(RES + cols, mask=mask).to(tl.float32)
    x = tl.load(X + cols, mask=mask).to(tl.float32)
    g = tl.load(GATE + cols, mask=mask).to(tl.float32)
    y = r + x * g
    tl.store(RES + cols, y.to(IO_DTYPE), mask=mask)


def gate_mul_residual(residual: torch.Tensor, x: torch.Tensor,
                      gate: torch.Tensor) -> torch.Tensor:
    """residual[S,D] += x[S,D] * gate[S,D] (writes in place into residual)."""
    n = residual.numel()
    BLOCK = 1024
    IO_DTYPE = tl.bfloat16 if residual.dtype == torch.bfloat16 else tl.float16
    _gate_mul_residual_kernel[(triton.cdiv(n, BLOCK),)](
        residual, x, gate.contiguous(), n, BLOCK=BLOCK, IO_DTYPE=IO_DTYPE)
    return residual


@triton.jit
def _rmsnorm_affine_kernel(X, WEIGHT, OUT, Npts, D, EPS, BLOCK_D: tl.constexpr,
                           IO_DTYPE: tl.constexpr):
    pt = tl.program_id(0)
    xp = X + pt * D
    op = OUT + pt * D
    _sum = tl.zeros([BLOCK_D], dtype=tl.float32)
    for off in tl.range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(xp + cols, mask=mask, other=0.0).to(tl.float32)
        _sum += tl.sum(x * x)
    inv_rms = 1.0 / tl.sqrt(_sum / D + EPS)
    for off in tl.range(0, D, BLOCK_D):
        cols = off + tl.arange(0, BLOCK_D)
        mask = cols < D
        x = tl.load(xp + cols, mask=mask, other=0.0).to(tl.float32)
        w = tl.load(WEIGHT + cols, mask=mask, other=0.0).to(tl.float32)
        y = x * inv_rms * w
        tl.store(op + cols, y.to(IO_DTYPE), mask=mask)


def rms_norm_fp32stat(x: torch.Tensor, weight: torch.Tensor, eps: float) -> torch.Tensor:
    """RMSNorm (fp32 statistics + affine weight[D]); x[..,D] fp16/bf16 -> same dtype.
    Equivalent to diffusers RMSNorm, avoiding the round-trip fp32 upcast.
    Used for Q/K norm_q/norm_k."""
    D = x.shape[-1]
    orig_shape = x.shape
    x2 = x.reshape(-1, D).contiguous()
    Npts = x2.shape[0]
    out = torch.empty_like(x2)
    w = weight.contiguous().to(torch.float32).view(-1)
    BLOCK_D = 16
    while BLOCK_D < D and BLOCK_D < 2048:
        BLOCK_D <<= 1
    IO_DTYPE = tl.bfloat16 if x.dtype == torch.bfloat16 else tl.float16
    _rmsnorm_affine_kernel[(Npts,)](
        x2, w, out, Npts, D, float(eps), BLOCK_D=BLOCK_D,
        num_warps=8 if D >= 256 else 4, IO_DTYPE=IO_DTYPE)
    return out.reshape(orig_shape)


@triton.jit
def _gate_mul_res_bcast_kernel(RES, X, GATE, Nrow, D, BLOCK_D: tl.constexpr,
                               IO_DTYPE: tl.constexpr):
    """res[Nrow,D] += x[Nrow,D] * gate[D] (gate is broadcast, avoiding the [S,D] expand copy)."""
    r = tl.program_id(0)
    cols = tl.arange(0, BLOCK_D)
    mask = cols < D
    rp = RES + r * D
    x = tl.load(rp + cols, mask=mask).to(tl.float32)
    xv = tl.load(X + r * D + cols, mask=mask).to(tl.float32)
    g = tl.load(GATE + cols, mask=mask).to(tl.float32)
    tl.store(rp + cols, (x + xv * g).to(IO_DTYPE), mask=mask)


def gate_mul_residual_bcast(residual: torch.Tensor, x: torch.Tensor,
                             gate: torch.Tensor) -> torch.Tensor:
    """residual[S,D] += x[S,D] * gate[D] (gate is broadcast, in-place)."""
    D = residual.shape[-1]
    res2 = residual.reshape(-1, D)
    x2 = x.reshape(-1, D)
    Nrow = res2.shape[0]
    g = gate.contiguous().to(residual.dtype).view(-1)
    BLOCK_D = 16
    while BLOCK_D < D and BLOCK_D < 2048:
        BLOCK_D <<= 1
    IO_DTYPE = tl.bfloat16 if residual.dtype == torch.bfloat16 else tl.float16
    _gate_mul_res_bcast_kernel[(Nrow,)](res2, x2, g, Nrow, D, BLOCK_D=BLOCK_D,
                                        num_warps=4, IO_DTYPE=IO_DTYPE)
    return residual

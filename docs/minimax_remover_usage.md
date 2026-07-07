# MiniMax-Remover — FlashRT Inference Pipeline

MiniMax-Remover video inpainting (subtitle / object removal) with FlashRT
kernelized inference on Blackwell SM120. Two precision paths ship under
`flash_rt.models.minimax_remover`:

| Entrypoint | Precision | Recommended use |
|------------|-----------|-----------------|
| `MiniMaxRemoverPipelineFP8` | FP8 (W8A8) | **Default.** Full-frame inpainting; end-to-end cosine >= 0.999, PSNR ~35-41 dB vs fp16 (see Performance). |
| `MiniMaxRemoverPipeline` | NVFP4 (W4A4) | Cropped small regions only. Full-frame large latents drift/blacken due to FP4 error accumulation. |

Both reuse the **generic** FlashRT kernels for the transformer denoise path
(quantized GEMMs, fused norm/gate/residual/gelu ops, kernel attention via
FA2 / SageAttention). The VAE encode/decode is additionally accelerated by
**model-specific fp16 fused CUDA kernels** in the standalone
`flash_rt_minimax_remover` module (opt-in build, see [Build](#build)):
`fp16_rms_norm_ncdhw` (single-pass RMSNorm, fp16-native),
`fp16_rms_silu_ncdhw` (fused RMSNorm + SiLU), and their channels-last
(NDHWC) variants `fp16_rms_norm_ndhwc` / `fp16_rms_silu_ndhwc` which
keep the entire VAE pipeline in channels-last 3D memory format —
eliminating cuDNN's per-conv `nchw↔nhwc` conversion kernels. The NVFP4 path additionally
captures the N-step flow-matching loop as a single CUDA Graph; the FP8 path
is graph-compatible (stable static scales, no host sync in steady state) but
does not itself capture a graph.

## Build

```bash
cd FlashRT
cmake -S . -B build -DGPU_ARCH=120 -DCMAKE_BUILD_TYPE=Release \
      -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON
cmake --build build -j --target flash_rt_kernels flash_rt_minimax_remover
pip install -e ".[torch,minimax-remover]"
```

`GPU_ARCH=120` (RTX 5090 / 5060 Ti) or `121` selects the Blackwell target;
the NVFP4 surface is compiled in automatically (internally gated by
`ENABLE_CUTLASS_SM120_NVFP4_W4A16`, which is set from `GPU_ARCH`, not a flag
users pass). The FP8 symbols are part of the default build. Then install the
runtime extras:

```bash
pip install -e ".[minimax-remover]"   # diffusers + einops + scipy + sageattention
```

### VAE fused kernels (standalone module, opt-in)

The fp16-native fused VAE kernels ship in a **separate** pybind module
`flash_rt_minimax_remover`, following the same opt-in pattern as
`flash_rt_omnivoice`. They are **not** compiled into the default
`flash_rt_kernels` target — enable explicitly:

```bash
cmake -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON -DGPU_ARCH=120 ...
cmake --build build -j --target flash_rt_minimax_remover
```

| Kernel | Module | Replaces | Effect |
|--------|--------|----------|--------|
| `fp16_rms_norm_ncdhw` | `flash_rt_minimax_remover` | `WanRMS_norm.forward` (4 full-tensor fp32 passes) in attention blocks | ~6x per-call; fp16 in/out, fp32 stats, **no dtype cast** |
| `fp16_rms_silu_ncdhw` | `flash_rt_minimax_remover` | `WanRMS_norm` + `F.silu` two-pass in every `WanResidualBlock` | fused single-pass; eliminates intermediate tensor R/W |
| `fp16_rms_norm_ndhwc` | `flash_rt_minimax_remover` | channels-last (NDHWC) variant of the above for attention-block norm | keeps pipeline in CL → eliminates nchw↔nhwc conversion |
| `fp16_rms_silu_ndhwc` | `flash_rt_minimax_remover` | channels-last fused norm+silu for residual blocks | CL-native, contiguous C reads → faster than NCDHW variant |
| `fp16_rms_silu_amax_ndhwc` | `flash_rt_minimax_remover` | channels-last fused norm+silu+**amax** | fuses amax into the norm+silu pass; saves 1 full read of the activation tensor per conv layer |
| `fp16_rms_silu_quant_fp8_ndhwc` | `flash_rt_minimax_remover` | channels-last fused norm+silu→**FP8 e4m3** quantize | eliminates the fp16 intermediate between norm and conv (reads pre-computed amax) |
| `fp16_rms_silu_amax_quant_fp8_ndhwc` | `flash_rt_minimax_remover` | 2-pass norm+silu+amax+quant→FP8 | combined launcher: pass 1 computes amax (no write), pass 2 quantizes; produces ONLY fp8 + scale |
| `fp8_conv3d_mm_ndhwc_fp16out` | `flash_rt_minimax_remover` | cuDNN fp16 conv3d for applicable 3×3×3 causal convs | FP8 e4m3 implicit-GEMM (no im2col materialization); per-channel weight dequant; fp16 in/out |
| `amax_fp16` + `quantize_fp16_fp8_with_amax` | `flash_rt_minimax_remover` | PyTorch multi-pass `abs().max()` + scale + cast for activation quantization | fused 2-pass device-side amax + quantize; no host sync; shared scale across cache+new frames |
| `quantize_fp16_fp8_with_amax_dual` | `flash_rt_minimax_remover` | two separate `quantize_fp16_fp8_with_amax` calls (cache + new) | single kernel launch quantizes two buffers with shared amax; saves 1 launch per conv layer |
| `bias_gelu_quant_fp16_fp8` | `flash_rt_minimax_remover` | transformer FFN `add_bias_fp16` + `gelu_inplace_fp16` + `quantize_fp8_static_fp16` (3 kernels) | fused single-pass: fp16 GEMM-out + bias → tanh-gelu → fp8 e4m3; output is the pre-quantised input of the next FP8 Linear (skips its activation quantise); eliminates 3 full-tensor fp16 round-trips per FFN block |
| `bias_quant_fp16_fp8` | `flash_rt_minimax_remover` | `add_bias_fp16` + `quantize_fp8_static_fp16` (identity activation variant) | fused bias + quant for Linear→Linear chains with no activation |
| channels-last pipeline | Python (`_vae_opt.py`) | Conv3d weight → CL, WanCausalConv3d → CL-preserving forward | eliminates ~97% of nchw↔nhwc conversion kernels (~280 ms / decode) |
| Running-max amax | Python (`_vae_opt.py`) | separate `amax_fp16` calls over cache + new each iteration | norm fuses amax via atomicMax into a persistent buffer shared with the sister conv; cache amax is skipped entirely (covered by running max) |
| `WanUpsample` patch | Python (no kernel) | `x.float().type_as(x)` for nearest-exact upsample | eliminates redundant fp32 cast (index-only op, fp16 == fp32) |

The VAE stays fp16 at the interface. Norm/activation ops use fp16-native
kernels (zero precision loss). Applicable 3×3×3 causal conv3d layers
(Ci % 32 == 0) use the FP8 implicit-GEMM kernel — fp16 activations are
quantized to FP8 e4m3 on-the-fly, the MMA runs on tensor cores in FP8,
and the result is dequantized back to fp16 via per-output-channel scales
(PSNR ~39 dB vs fp16 reference). Non-applicable convs (1×1×1, 3×1×1,
Ci not divisible by 32) fall back to cuDNN fp16. If the module is not
built, `install_vae_optimizations()` raises a clear `ImportError` with
the rebuild command.

#### Channels-last 3D pipeline

cuDNN's fp16 conv3d kernel (`sm80_xmma_fprop_implicit_gemm`) internally
operates in NHWC. When the VAE feeds NCDHW tensors, cuDNN inserts
`nchwToNhwcKernel` / `nhwcToNchwKernel` conversion kernels before and
after **every** conv3d call — totalling ~287 ms for a 18-frame decode
(~11% of wall time).

The `install_vae_optimizations()` call now enables a **channels-last
3D (NDHWC) pipeline** end-to-end:

1. All 61 `WanCausalConv3d` weight tensors are converted to
   `memory_format=torch.channels_last_3d` (done once at install time).
2. `WanCausalConv3d.forward` is patched to preserve the CL format
   (cat + pad + conv3d all preserve CL natively).
3. The FlashRT norm kernels are swapped to NDHWC variants
   (`fp16_rms_norm_ndhwc`, `fp16_rms_silu_ndhwc`) so the norm output
   stays in CL — no format break between norm and conv.

Result: **97% of format-conversion kernels eliminated** (287 ms → 9 ms),
plus a ~1.3× per-conv speedup from cuDNN's preferred CL algorithm.
Zero precision loss (PSNR 40.8 dB vs fp16 reference, identical to the
NCDHW path within fp16 rounding).

#### FP8 implicit-GEMM conv3d pipeline

The dominant VAE cost is cuDNN's fp16 conv3d (~1549 ms / decode, 64% of
wall time). The `fp8_conv3d_mm_ndhwc_fp16out` kernel replaces cuDNN for
applicable 3×3×3 causal conv3d layers (Ci % 32 == 0, Co >= 8) with a
**hand-rolled FP8 e4m3 implicit-GEMM** that runs entirely on tensor cores.

Key design:

- **No im2col materialization.** The kernel computes im2col indices
  on-the-fly inside the MMA loop, reading activations directly from the
  original NDHWC tensor via `cp.async`. This avoids the ~268 MB
  intermediate matrix that made the naive FP8 im2col + `_scaled_mm`
  approach 3.5× slower than cuDNN.
- **Virtual cache concat.** Two input pointers (`cache_x_fp8` +
  `new_x_fp8`) replace `torch.cat`, saving the concat kernel. The
  temporal addressing `d_in = t_out + kt` reads from cache for
  `d_in < T_cache`, else from new — zero-copy causal sliding window.
- **Direct causal output.** Output is `T_new` frames (not
  `T_cache + T_new`), avoiding the slice + wasted output write.
- **Per-output-channel weight quantization.** Weights are quantized
  once at install time with per-Co amax scales, maximizing FP8 dynamic
  range utilization. The dequant alpha = `act_scale × w_scale[co]` is
  applied in the bias-fused epilogue.
- **Fused activation quantization.** `amax_fp16` + atomicMax computes
  a shared per-tensor scale over cache+new (2-pass, no host sync), then
  `quantize_fp16_fp8_with_amax` scales and casts to FP8 e4m3.
- **Running-max amax.** The norm module fuses amax into its
  norm+silu pass via `fp16_rms_silu_amax_ndhwc` and accumulates it
  into a persistent device-side buffer (atomicMax). Because cache_x
  was a previous output of the same norm, its amax is already
  covered — so the conv **skips the cache amax pass entirely**
  (saves one full read of the 2-frame cache tensor per layer).
- **Dual-quantize launch.** `quantize_fp16_fp8_with_amax_dual`
  quantizes the cache and new tensors in a single kernel launch,
  reducing launch overhead by ~49 calls per decode.
- **Tile geometry.** BLOCK_M=128, BLOCK_N=128, BLOCK_K=32, 8 warps,
  2-stage cp.async pipeline, persistent Y-major CTA raster. The MMA
  instruction is
  `mma.sync.aligned.kind::f8f6f4.m16n8k32.row.col.f32.e4m3.e4m3.f32`.

Result: **1.7% additional decode speedup** over the previous FP8
conv3d path (8.75 s → 8.58 s) and **0.7 dB PSNR improvement**
(39.3 → 39.9 dB median) thanks to the temporally consistent
running-max scaling. Combined with the channels-last pipeline:
**2.02× vs baseline**. PSNR 39.9 dB (median).

#### Fused transformer FFN epilogue

The transformer denoise loop spends ~877 ms / 12-step run on three
elementwise "glue" kernels sandwiched between the FP8 GEMMs of each
FeedForward block:

  `add_bias_fp16` (607 ms) + `gelu_inplace_fp16` (233 ms) + `quantize_fp8` (270 ms)

Each is a full read+write of the `[S, inner_dim]` fp16 FFN-up output
(inner_dim = 13824). The `bias_gelu_quant_fp16_fp8` kernel collapses
all three into a **single pass** that reads the GEMM's raw fp16 output
once, applies bias + tanh-GELU in fp32, and writes fp8 e4m3 directly.
The fp8 tensor becomes the pre-quantised input of the FFN-down Linear,
which skips its own activation quantise step.

All arithmetic (bias + GELU) is done in fp32 before the fp8 cast, so
the result is actually **more accurate** than the original path (which
rounds to fp16 twice along the way) — end-to-end PSNR improves from
39.9 → 40.0 dB.

The FP8 pipeline freezes calibration after the **first denoise step**
(via a one-shot transformer forward hook), so steps 2..N run with
static scales and the fused epilogue active — a single-call invocation
benefits without needing a separate warm-up pass.

Result: **denoise GPU time 3.73 s → 3.16 s** (-15%), end-to-end
**8.58 s → 7.56 s** (**2.29× vs baseline**), PSNR **40.0 dB** (median).

Importing `flash_rt.models.minimax_remover` always succeeds — it needs
**none** of `diffusers` / `einops` / `scipy` / `triton` / `sageattention`. The kernel
surface is validated lazily in each pipeline's `__init__` via
`load_nvfp4_kernels()` / `load_fp8_kernels()` (`flash_rt/models/minimax_remover/_utils.py`),
and the runtime deps are resolved at construction via `_import_runtime()`. If
a required symbol or dep is missing the constructor raises a clear
`RuntimeError` with the rebuild/install hint, so a non-matching build or a
bare environment fails fast instead of crashing mid-run.

### Required kernel symbols

NVFP4 path (Blackwell-only, auto-enabled by `GPU_ARCH=120/121`):

| Symbol | Role |
|--------|------|
| `nvfp4_sf_swizzled_bytes` | block-scale-factor byte layout helper |
| `bf16_weight_to_nvfp4_swizzled` | one-shot weight -> NVFP4 quantise |
| `quantize_bf16_to_nvfp4_swizzled` | per-call dynamic activation quantise |
| `fp4_w4a16_gemm_sm120_bf16out_pingpong` | SM120-native W4A4 MMA -> bf16 |
| `add_bias_bf16` | in-place bias add on bf16 GEMM output |
| `fp4_w4a16_gemm_bias_gelu_fp4out_sm120` | fused FFN-up GEMM + bias + GELU -> FP4 |

FP8 path (default build):

| Symbol | Role |
|--------|------|
| `quantize_fp8_static_fp16` | weight + static activation FP8 quantise |
| `fp8_gemm_descale_fp16` | FP8 W8A8 MMA -> fp16 with scale descale |
| `add_bias_fp16` | in-place bias add on fp16 GEMM output |

Shared block-fusion symbols (default `gelu_mode="inplace"`, default build, required by both paths):

| Symbol | Role |
|--------|------|
| `gelu_inplace` | in-place tanh-approximate GELU on bf16 FFN-up output |
| `gelu_inplace_fp16` | in-place tanh-approximate GELU on fp16 FFN-up output |

The attention backend (`FLASHRT_ATTN_MODE`) optionally pulls in
`sageattention` (Sage); `fa2` uses the vendored `flash_rt_fa2.so` and is the
dependency-light fallback. The fused norm / RoPE / Euler-step elementwise
kernels are self-contained Triton JIT kernels shipped in the package
(`_kernels.py`) and need no build step.

## Pipelines

### FP8 — `MiniMaxRemoverPipelineFP8` (default, full-frame)

`flash_rt/models/minimax_remover/_fp8_pipeline.py`. Uses static calibration:
the first inference call runs in dynamic-FP8 calibration mode (accumulating
activation amax on GPU). A one-shot transformer forward hook **freezes the
calibration after the first denoise step**, so steps 2..N (and the fused FFN
epilogue kernel) run with static scales — a single-call invocation benefits
without needing a separate warm-up pass. **The frozen scale is calibrated to
the first call's input; if the input resolution/shape changes, construct a
new pipeline so the scale is re-calibrated.**

- every eligible transformer Linear -> FP8 W8A8 GEMM (weight quantised once
  at load time; activation quantised with a calibrated static scale);
- per-block LayerNorm + adaLN modulation + gate-residual fused into Triton
  kernels (fp32 statistics);
- `torch.nn.functional.scaled_dot_product_attention` -> FA2 / SageAttention.

### NVFP4 — `MiniMaxRemoverPipeline` (small-region only)

`flash_rt/models/minimax_remover/pipeline.py`. It wraps a loaded diffusers
MiniMax-Remover `pipe` and consumes it in place:

- every eligible transformer Linear -> NVFP4 W4A4 GEMM (weight quantised
  once at load time; activation quantised **dynamically** per call with
  per-16-element UE4M3 block scales computed on-GPU — no offline
  calibration, no CPU sync);
- transformer switched to bf16 (NVFP4-native, eliminates the fp16<->bf16
  cast pair);
- RoPE freqs cached as complex<float>;
- per-block LayerNorm + adaLN modulation + gate-residual fused into a
  single fp32-stat Triton kernel;
- the N-step flow-matching denoise loop replaced by a manual, graph-
  capturable pointer-based loop (`ManualRemoverPipeline`). QKV quantises
  the norm output **once** and reuses it for all three projections; the
  FFN-up GEMM fuses bias + GELU straight to FP4 output so the FFN-down
  projection skips re-quantisation. With `FLASHRT_MANUAL_GRAPH=1` the
  whole N-step x N-block loop is captured as a single CUDA Graph;
  inside the captured graph there are **zero** torch elementwise ops —
  every operation is a kernel launch.

Both paths run the VAE encode / decode from the loaded diffusers model
(one-shot per segment, outside the graph). With `--vae-opt` (default), the
VAE's `WanRMS_norm` / `WanResidualBlock` norm+silu sites are replaced by the
FlashRT fp16 fused kernels, all `WanCausalConv3d` weights are converted
to channels-last 3D, the norm kernels are swapped to NDHWC variants so
the entire norm→conv pipeline stays in channels-last, and applicable 3×3×3
causal conv3d layers are replaced by the FP8 implicit-GEMM kernel
(see [VAE fused kernels](#vae-fused-kernels-standalone-module-opt-in)).
No MiniMax-Remover source is imported; the `pipe` is duck-typed through
`.transformer` / `.vae` / `.scheduler` / `.video_processor` and the
`expand_masks` / `resize` helpers.

## Performance (RTX 5060 Ti, SM120, CUDA 13)

All numbers below are reproducible with the quickstart:

```bash
python3 examples/minimax_remover_quickstart.py \
    --model-dir ./minimax-remover \
    --frames-dir ./object_removal_data/<frames> \
    --masks-dir  ./object_removal_data/<masks> \
    --output-dir ./out                          # FP8 + VAE opt (default)
python3 examples/minimax_remover_quickstart.py ... --no-vae-opt   # FP8, no VAE kernels
python3 examples/minimax_remover_quickstart.py ... --use-fp4      # NVFP4
python3 examples/minimax_remover_quickstart.py ... --no-flashrt   # fp16 reference
```

Wall time is a single end-to-end segment (load -> encode -> denoise loop ->
decode -> save). FP8 numbers include the one-time calibration pass on the
first call. Correctness (PSNR / cosine) is the FP8/NVFP4 output compared
against the `--no-flashrt` fp16 reference output on the same input.

### End-to-end, full-frame (single segment, all frames at once)

All rows compare against the non-FlashRT `--no-flashrt` fp16 reference on the
**same** input clip (same seed, same frames, same masks).

| Clip (frames, resolution) | Stack | Wall time | Speedup vs fp16 ref | PSNR mean / worst vs fp16 ref | cosine mean |
|--------------------------|-------|-----------|---------------------|-------------------------------|-------------|
| tennis (70 frames, 432x240) | fp16 reference (`--no-flashrt`) | 17.33 s | 1.0x | — | — |
| tennis (70 frames, 432x240) | FlashRT FP8 + VAE opt + CL (`--no-fp8-conv`) | 10.01 s | 1.73x | 40.8 / 37.0 dB | 0.99981 |
| tennis (70 frames, 432x240) | FlashRT FP8 + VAE opt + CL + FP8 conv3d | 8.58 s | 2.02x | 39.9 / 36.4 dB | 0.99981 |
| tennis (70 frames, 432x240) | **FlashRT FP8 + VAE opt + CL + FP8 conv3d + fused FFN epilogue (default)** | **7.56 s** | **2.29x** | **40.0 / 36.4 dB** | 0.99981 |
| tennis (70 frames, 432x240) | FlashRT NVFP4 (`--use-fp4`) | 9.52 s | 1.82x | 7.0 / 6.2 dB | 0.00000 (broken) |
| bmx-trees (80 frames, 432x240) | fp16 reference (`--no-flashrt`) | 19.76 s | 1.0x | — | — |
| bmx-trees (80 frames, 432x240) | FlashRT FP8 (default) | 13.24 s | **1.49x** | 35.1 / 32.0 dB | 0.99912 |
| bmx-trees (80 frames, 432x240) | FlashRT NVFP4 (`--use-fp4`) | 10.72 s | 1.84x | 7.3 / 7.0 dB | 0.00000 (broken) |

> Tennis numbers are from a same-session A/B (`--no-flashrt` vs `--vae-opt`
> with/without `--fp8-conv`) measured on RTX 5060 Ti, CUDA 13.0, cuDNN 9.2,
> PyTorch 2.12. Peak VRAM: fp16 ref 3.67 GB, FlashRT FP8 conv3d 2.51 GB.

Takeaways:

- **FP8 + VAE opt + CL + FP8 conv3d + fused FFN epilogue is the recommended default**: 2.29x
  faster than the fp16 reference with PSNR 40.0 dB (median) on full-frame
  tennis clip, peak VRAM 2.51 GB. The fused FFN epilogue kernel
  (`bias_gelu_quant_fp16_fp8`) collapses bias-add + GELU + activation
  quantise into one pass, cutting denoise GPU time by 15% (3.73 → 3.16 s)
  and end-to-end from 8.58 → 7.56 s.
- **FP8 conv3d vs channels-last-only cuDNN**: the hand-rolled implicit-GEMM
  kernel (no im2col materialization, virtual cache concat, per-channel
  weight dequant, fused amax, running-max scale) beats cuDNN's fp16 conv3d
  while staying in fp16 at the interface. Disable with `--no-fp8-conv` to
  recover ~1 dB PSNR if absolute precision is preferred over speed.
- **NVFP4 is faster but unusable on full-frame latents**: cosine collapses to
  ~0.0 and PSNR to ~7 dB (median per-pixel deviation ~85/255). The FP4
  quantisation error accumulates over the large full-frame activations and the
  output drifts to black — exactly why FP8 is the default. NVFP4 is only
  appropriate for small cropped regions, where its per-block error stays
  bounded.

### Transformer GEMM (NVFP4 vs fp16 matmul, single layer)

| Linear | fp16 matmul | NVFP4 W4A4 | per-layer speedup |
|--------|-------------|------------|-------------------|
| FFN up [5120 -> 13824] | 1.095 ms | 0.840 ms | 1.30× |
| FFN down [13824 -> 5120] | 1.020 ms | 0.864 ms | 1.18× |
| QKV / out [5120 -> 5120] | 0.409 ms | 0.359 ms | 1.14× |

Including cast + quantise overhead, the isolated FP4 GEMM is 4–9× faster
than the fp16 matmul on the large FFN projections (e.g. ffn_up
3.95 ms -> 0.47 ms). NVFP4 is also 1.14–1.30× faster per layer than the
static-quant FP8 GEMM.

### Precision specification

The pipelines keep the math reference-equivalent on the precision-critical
path (fp32-stat LayerNorm / RMSNorm, interleaved RoPE) and confine the loss
to the quantised GEMMs and the attention backend.

| Component | Metric | Value |
|-----------|--------|-------|
| VAE `fp16_rms_norm_ncdhw` kernel | cosine vs fp32 reference | >= 0.9999999 |
| VAE `fp16_rms_silu_ncdhw` kernel | cosine vs fp32 reference | >= 0.9999999 |
| VAE `fp16_rms_norm_ndhwc` (CL) kernel | cosine vs fp32 reference | >= 0.9999999 |
| VAE `fp16_rms_silu_ndhwc` (CL) kernel | cosine vs fp32 reference | >= 0.9999999 |
| VAE `fp16_rms_silu_amax_ndhwc` kernel | amax vs fp32 reference | exact (atomicMax on non-negative floats) |
| VAE `fp8_conv3d_mm` kernel | cosine vs fp16 F.conv3d | >= 0.9993 (per-layer) |
| End-to-end FP8 + VAE opt + CL + FP8 conv3d + fused FFN epilogue (full-frame) | PSNR vs fp16 ref | 40.0 dB (median) / >= 36.4 dB (worst frame) |
| End-to-end FP8 + VAE opt + CL + FP8 conv3d (full-frame) | PSNR vs fp16 ref | 39.9 dB (median) / >= 36.4 dB (worst frame) |
| End-to-end FP8 + VAE opt + CL only (no FP8 conv3d) | PSNR vs fp16 ref | 40.8 dB (median) / >= 37.0 dB (worst frame) |
| Attention — SageAttention QK-int8 PV-fp8 (`sage_fp8`, default) | cosine vs SDPA | 0.9993 |
| Attention — SageAttention QK-int8 PV-fp16 (`sage_fp16`) | cosine vs SDPA | 0.9999 |
| NVFP4 W4A4 GEMM | cosine vs fp16 matmul | >= 0.999 |
| FP8 W8A8 GEMM | cosine vs fp16 matmul | >= 0.999 |
| End-to-end FP8 (full-frame) | PSNR vs fp16 ref | 35-41 dB (mean) / >= 32 dB (worst frame) |
| End-to-end FP8 (full-frame) | cosine vs fp16 ref | >= 0.999 |
| End-to-end NVFP4 (full-frame) | cosine vs fp16 ref | ~0.0 — **broken**, output drifts to black (median per-pixel deviation ~85 / 255) |
| End-to-end NVFP4 (small cropped region only) | PSNR vs fp16 ref | ~52 dB (mean) / ~45 dB (worst frame); per-block FP4 error stays bounded only when activations are small |

The default `sage_fp8` attention gives the best latency at cosine 0.9993;
switch to `FLASHRT_ATTN_MODE=sage_fp16` for cosine 0.9999 at a small
latency cost. NVFP4 needs no calibration, so the first call is already in
the steady state; the FP8 path calibrates on the first call then freezes.

## CLI flags (quickstart)

| Flag | Default | Effect |
|------|---------|--------|
| `--vae-opt` / `--no-vae-opt` | **enabled** | Install FlashRT fp16 fused VAE kernels (`fp16_rms_norm_ncdhw`, `fp16_rms_silu_ncdhw`, NDHWC variants, fused norm+silu+amax) + channels-last 3D pipeline + running-max amax sharing between norm and conv + FP8 implicit-GEMM conv3d + `WanUpsample` cast elimination. Requires the `flash_rt_minimax_remover` module. |
| `--fp8-conv` / `--no-fp8-conv` | **enabled** | Use FP8 implicit-GEMM conv3d kernel for applicable 3×3×3 causal convs (requires `--vae-opt`). Trades ~1 dB PSNR for ~14% decode speedup over channels-last-only cuDNN. |
| `--use-fp4` | off | Use NVFP4 (W4A4) instead of the default FP8 (W8A8). Small-region only. |
| `--no-flashrt` | off | Run the reference diffusers fp16 path (no FlashRT). |

## Environment variables

| Variable | Default | Effect |
|----------|---------|--------|
| `FLASHRT_ATTN_MODE` | `sage_fp8` | attention backend (`sage_fp8`/`sage_fp16`/`sage`/`sage_triton`/`triton_fp8`/`triton_fp16`/`fa2`) |
| `FLASHRT_FP4_GEMM` | `pingpong` | GEMM kernel variant (`pingpong`/`plain`/`widen`) (NVFP4 path) |
| `FLASHRT_FUSED_BLOCK` | `1` | fused QKV-quant-once + fused FFN-up GEMM+bias+gelu block (`0` = per-projection re-quant debug path) (NVFP4 path) |
| `FLASHRT_MANUAL_GRAPH` | `0` | capture the whole denoise loop as one CUDA Graph (NVFP4 path) |
| `FLASHRT_NUM_STEPS` | unset | override the denoise step count (default 12) |
| `FLASHRT_FP8_TARGET` | `all` | FP8 Linear scope (`all` / `ffn_only`) (FP8 path) |
| `FLASHRT_NORM_MODE` | `triton` | per-block LayerNorm kernel: `triton` (fp32-stat Triton, bit-exact) / `fp16` (FlashRT `ada_layer_norm_fp16`, lower precision, debug only) |
| `FLASHRT_GELU_MODE` | `inplace` | FFN GELU kernel: `inplace` (FlashRT fused `gelu_inplace*`) / `torch` (original `F.gelu`, debug only) |

## Usage

### FP8 (recommended default — full-frame)

```python
from flash_rt.models.minimax_remover import MiniMaxRemoverPipelineFP8

# `pipe` is a loaded diffusers Minimax_Remover_Pipeline (transformer + vae +
# scheduler). The FlashRT pipeline consumes it in place. The first call
# calibrates the FP8 act_scale; subsequent calls reuse the frozen scale.
pipeline = MiniMaxRemoverPipelineFP8(pipe)
output = pipeline(
    images=frames,        # [F, H, W, 3] uint8/np, 0..255
    masks=masks,          # [F, H, W, 1] np, 0/1
    num_frames=len(frames),
    height=720, width=1280,
    num_inference_steps=12,
)
video = output.frames
```

### NVFP4 (small cropped regions only)

```python
from flash_rt.models.minimax_remover import MiniMaxRemoverPipeline

pipeline = MiniMaxRemoverPipeline(pipe)
output = pipeline(
    images=frames, masks=masks, num_frames=len(frames),
    height=720, width=1280, num_inference_steps=12,
)
video = output.frames
```

## Model weights

MiniMax-Remover checkpoint + the `Transformer3DModel` / `AutoencoderKLWan`
definitions are loaded by the reference project (unmodified, via the loaded
diffusers `pipe`). This FlashRT pipeline module imports no MiniMax-Remover
source.

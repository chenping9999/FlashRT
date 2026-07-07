// ================================================================
// flash_rt_minimax_remover — standalone pybind module for
// MiniMax-Remover VAE-specific fused fp16 kernels.
//
// Kept separate from flash_rt_kernels so they can be added/rebuilt
// independently without touching the main bindings.  Build with:
//   cmake -DFLASHRT_ENABLE_MINIMAX_REMOVER=ON -DGPU_ARCH=120 ...
//
// Kernels: fp16_rms_norm_ncdhw, fp16_rms_silu_ncdhw
// ================================================================
#include <pybind11/pybind11.h>
#include <cstdint>
#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "kernels/minimax_remover/fp16_rms_norm_ncdhw.cuh"
#include "kernels/minimax_remover/fp16_rms_silu_ncdhw.cuh"
#include "kernels/minimax_remover/fp16_rms_norm_ndhwc.cuh"
#include "kernels/minimax_remover/fp8_conv3d_mm_ndhwc_fp16out.cuh"
#include "kernels/minimax_remover/fp16_quant_fp8_per_tensor.cuh"
#include "kernels/minimax_remover/fp16_rms_silu_fp8_ndhwc.cuh"
#include "kernels/minimax_remover/fp16_bias_gelu_quant_fp8.cuh"

namespace py = pybind11;

static inline void* to_ptr(uintptr_t p) { return reinterpret_cast<void*>(p); }
static inline cudaStream_t to_stream(uintptr_t s) {
    return reinterpret_cast<cudaStream_t>(s);
}

PYBIND11_MODULE(flash_rt_minimax_remover, m) {
    m.doc() = "MiniMax-Remover VAE fused fp16 kernels";

    m.def("fp16_rms_norm_ncdhw",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp16, int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_rms_norm_ncdhw(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp16), B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 NCDHW RMSNorm (fp16 in/out, fp32 stats, no cast). "
        "Replaces WanRMS_norm.forward (4 full-tensor fp32 passes).");

    m.def("fp16_rms_silu_ncdhw",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp16, int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_rms_silu_ncdhw(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp16), B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 NCDHW RMSNorm + SiLU (fp16 in/out, fp32 stats+act, "
        "no cast). Replaces norm->silu two-pass in WanResidualBlock.");

    // ── Channels-last (NDHWC) norm kernels ──
    m.def("fp16_rms_norm_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp16, int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_rms_norm_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp16), B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 channels-last (NDHWC) RMSNorm. C values contiguous, "
        "eliminates nchw<->nhwc format conversion for cuDNN conv3d.");

    m.def("fp16_rms_silu_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp16, int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::fp16_rms_silu_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp16), B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp16"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 channels-last (NDHWC) RMSNorm + SiLU.");

    // ── FP8 implicit-GEMM conv3d (3×3×3 causal, NDHWC, fp16 output) ──
    m.def("fp8_conv3d_mm_ndhwc_fp16out",
        [](uintptr_t cache_x_fp8, uintptr_t new_x_fp8,
           uintptr_t w_fp8, uintptr_t y_fp16,
           uintptr_t bias_fp16, uintptr_t alpha_vec,
           int N, int T_cache, int T_new, int H, int W, int Ci, int Co,
           uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp8_conv3d_mm_ndhwc_fp16out(
                to_ptr(cache_x_fp8), to_ptr(new_x_fp8),
                to_ptr(w_fp8), to_ptr(y_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                alpha_vec ? to_ptr(alpha_vec) : nullptr,
                N, T_cache, T_new, H, W, Ci, Co,
                to_stream(stream));
        },
        py::arg("cache_x_fp8"), py::arg("new_x_fp8"),
        py::arg("w_fp8"), py::arg("y_fp16"),
        py::arg("bias_fp16"), py::arg("alpha_vec"),
        py::arg("N"), py::arg("T_cache"), py::arg("T_new"),
        py::arg("H"), py::arg("W"), py::arg("Ci"), py::arg("Co"),
        py::arg("stream") = 0,
        "FP8 e4m3 implicit-GEMM conv3d fprop (3x3x3 causal, NDHWC, "
        "fp16 output). Per-channel alpha vector [Co] float and fp16 "
        "bias [Co]. No im2col materialization.");

    // ── Fused fp16→fp8 per-tensor quantize (2-pass, no host sync) ──
    m.def("fp16_quant_fp8_per_tensor",
        [](uintptr_t x_fp16, uintptr_t y_fp8,
           uintptr_t scale_out, uintptr_t amax_buf,
           int n, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_quant_fp8_per_tensor(
                to_ptr(x_fp16), to_ptr(y_fp8),
                to_ptr(scale_out), to_ptr(amax_buf),
                n, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("y_fp8"),
        py::arg("scale_out"), py::arg("amax_buf"),
        py::arg("n"), py::arg("stream") = 0,
        "Fused per-tensor fp16→fp8 e4m3 quantize (amax + scale on "
        "device, no host sync). Writes float scale to scale_out.");

    m.def("amax_fp16",
        [](uintptr_t x_fp16, uintptr_t amax_buf,
           int n, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::amax_fp16(
                to_ptr(x_fp16), to_ptr(amax_buf), n, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("amax_buf"),
        py::arg("n"), py::arg("stream") = 0,
        "Grid-stride amax reduction into amax_buf via atomicMax. "
        "Caller must zero amax_buf before first call. Multiple calls "
        "accumulate (for multi-tensor shared-scale quantization).");

    m.def("quantize_fp16_fp8_with_amax",
        [](uintptr_t x_fp16, uintptr_t y_fp8,
           uintptr_t amax_buf, uintptr_t scale_out,
           int n, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                quantize_fp16_fp8_with_amax(
                to_ptr(x_fp16), to_ptr(y_fp8),
                to_ptr(amax_buf), to_ptr(scale_out),
                n, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("y_fp8"),
        py::arg("amax_buf"), py::arg("scale_out"),
        py::arg("n"), py::arg("stream") = 0,
        "Quantize fp16→fp8 using pre-computed amax in amax_buf. "
        "Writes float scale to scale_out.");

    // ── Dual quantize: two buffers, one shared amax, one launch ──
    m.def("quantize_fp16_fp8_with_amax_dual",
        [](uintptr_t x1_fp16, uintptr_t y1_fp8, int n1,
           uintptr_t x2_fp16, uintptr_t y2_fp8, int n2,
           uintptr_t amax_buf, uintptr_t scale_out,
           uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                quantize_fp16_fp8_with_amax_dual(
                to_ptr(x1_fp16), to_ptr(y1_fp8), n1,
                to_ptr(x2_fp16), to_ptr(y2_fp8), n2,
                to_ptr(amax_buf),
                scale_out ? to_ptr(scale_out) : nullptr,
                to_stream(stream));
        },
        py::arg("x1_fp16"), py::arg("y1_fp8"), py::arg("n1"),
        py::arg("x2_fp16"), py::arg("y2_fp8"), py::arg("n2"),
        py::arg("amax_buf"), py::arg("scale_out") = 0,
        py::arg("stream") = 0,
        "Dual quantize: two fp16 buffers → fp8 with shared amax in "
        "one kernel launch. Saves one launch vs two separate calls.");

    // ── Fused norm+silu+amax / norm+silu+quant_fp8 (NDHWC) ──
    m.def("fp16_rms_silu_amax_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp16, uintptr_t amax_buf,
           int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_rms_silu_amax_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp16), to_ptr(amax_buf),
                B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp16"), py::arg("amax_buf"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 NDHWC RMSNorm+SiLU+amax. Writes fp16 output and "
        "accumulates |output| into amax_buf via atomicMax (caller must "
        "zero amax_buf before first call). Saves one full read of y "
        "vs separate norm+silu then amax.");

    m.def("fp16_rms_silu_quant_fp8_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp8, uintptr_t amax_buf, uintptr_t scale_out,
           int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_rms_silu_quant_fp8_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp8), to_ptr(amax_buf),
                scale_out ? to_ptr(scale_out) : nullptr,
                B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp8"), py::arg("amax_buf"), py::arg("scale_out") = 0,
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "Fused FP16 NDHWC RMSNorm+SiLU → FP8 e4m3 quantize. Reads "
        "pre-computed amax from device. Does NOT write fp16 output — "
        "eliminates the fp16 intermediate between norm and conv.");

    m.def("fp16_rms_silu_amax_quant_fp8_ndhwc",
        [](uintptr_t x_fp16, uintptr_t gamma_fp16, uintptr_t bias_fp16,
           uintptr_t y_fp8, uintptr_t scale_out, uintptr_t amax_buf,
           int B, int C, int T, int H, int W,
           float eps, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                fp16_rms_silu_amax_quant_fp8_ndhwc(
                to_ptr(x_fp16), to_ptr(gamma_fp16),
                bias_fp16 ? to_ptr(bias_fp16) : nullptr,
                to_ptr(y_fp8), to_ptr(scale_out), to_ptr(amax_buf),
                B, C, T, H, W, eps, to_stream(stream));
        },
        py::arg("x_fp16"), py::arg("gamma_fp16"), py::arg("bias_fp16"),
        py::arg("y_fp8"), py::arg("scale_out"), py::arg("amax_buf"),
        py::arg("B"), py::arg("C"), py::arg("T"), py::arg("H"), py::arg("W"),
        py::arg("eps") = 1e-6f, py::arg("stream") = 0,
        "2-pass fused norm+silu+amax+quant → FP8. Pass 1 computes amax "
        "(no write); pass 2 re-reads x and quantizes. Produces ONLY fp8 "
        "output + scale, no fp16 intermediate.");

    // ── Fused FFN epilogue: bias + gelu + quant → fp8 (transformer) ──
    m.def("bias_gelu_quant_fp16_fp8",
        [](uintptr_t gemm_out, uintptr_t bias, uintptr_t out,
           uintptr_t d_scale, int M, int N, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                bias_gelu_quant_fp16_fp8(
                to_ptr(gemm_out), to_ptr(bias), to_ptr(out),
                reinterpret_cast<const float*>(to_ptr(d_scale)),
                M, N, to_stream(stream));
        },
        py::arg("gemm_out"), py::arg("bias"), py::arg("out"),
        py::arg("d_scale"), py::arg("M"), py::arg("N"),
        py::arg("stream") = 0,
        "Fused FFN epilogue: fp16 GEMM-out + bias → tanh-gelu → fp8 e4m3. "
        "Replaces add_bias_fp16 + gelu_inplace_fp16 + quantize_fp8 (3 "
        "kernels → 1). Output is the pre-quantised input of the next FP8 "
        "Linear, which skips its own activation quantise.");

    m.def("bias_quant_fp16_fp8",
        [](uintptr_t gemm_out, uintptr_t bias, uintptr_t out,
           uintptr_t d_scale, int M, int N, uintptr_t stream) {
            return flash_rt::kernels::minimax_remover::
                bias_quant_fp16_fp8(
                to_ptr(gemm_out), to_ptr(bias), to_ptr(out),
                reinterpret_cast<const float*>(to_ptr(d_scale)),
                M, N, to_stream(stream));
        },
        py::arg("gemm_out"), py::arg("bias"), py::arg("out"),
        py::arg("d_scale"), py::arg("M"), py::arg("N"),
        py::arg("stream") = 0,
        "Fused: fp16 GEMM-out + bias → fp8 e4m3 (identity activation). "
        "For Linear→Linear chains with no activation in between.");
}

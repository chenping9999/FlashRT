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
}

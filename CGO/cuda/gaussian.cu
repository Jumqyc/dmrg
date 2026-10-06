// CUDA implementation of the patch projection of ``CGO.gaussian.GaussianCGO``: one
// Majorana label of a string is expanded over the n normal modes of the patch,
// and every mode (its even and its odd Majorana together) is applied to the
// whole batched amplitude state at once.
//
// The state is a tensor (pure_count, size, size): axis 0 is the pure register
// (the pure modes, pinned to their ground state), the two matrix axes are the
// mixed register.  Each mode maps an output element back to a single source
// element, so the kernel gathers and accumulates with no atomics and no host
// round trip, and the even and the odd Majorana of a mode share one read.
//
// Build and load through CGO/cuda/gaussian.py (torch.utils.cpp_extension.load).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>
#include <algorithm>

namespace {

constexpr int BLOCK = 256;   // threads per block

__device__ __forceinline__ void accumulate(cuDoubleComplex& acc, double er, double ei,
                                           cuDoubleComplex a) {
    acc.x += er * a.x - ei * a.y;
    acc.y += er * a.y + ei * a.x;
}

// Apply one label to the batched amplitude state.
//
// For the output element (pure_out, row_out, column) mode k contributes from the
// source  pure = pure_out ^ subset_flip[k],  row = row_out ^ flip_mask[k],
// with the Jordan-Wigner sign of the occupied modes below it.  The even Majorana
// of the mode is a real factor; the odd one carries an extra i, with the sign of
// the flipped row bit when the mode is mixed.
__global__ void apply_label_kernel(
        const cuDoubleComplex* __restrict__ amps,
        cuDoubleComplex* __restrict__ out,
        const cuDoubleComplex* __restrict__ elem_even,
        const cuDoubleComplex* __restrict__ elem_odd,
        const int32_t* __restrict__ below,
        const int32_t* __restrict__ flip_mask,
        const int32_t* __restrict__ subset_flip,
        const int32_t* __restrict__ pure_lo,
        const int32_t* __restrict__ pure_hi,
        const int8_t* __restrict__ mixed,
        const int64_t num_modes,
        const int64_t pure_count,
        const int64_t size,
        const int64_t size_bits) {
    // size is a power of two, so the (pure, row, column) split of the flat index
    // is shifts and masks; integer division would be a long-latency instruction.
    const int64_t plane = size * size;
    const int64_t total = pure_count * plane;
    const int64_t stride = (int64_t)gridDim.x * BLOCK;
    for (int64_t index = (int64_t)blockIdx.x * BLOCK + threadIdx.x;
         index < total; index += stride) {
        const int64_t pure_out = index >> (2 * size_bits);
        const int64_t rem = index & (plane - 1);
        const int64_t row_out = rem >> size_bits;
        const int64_t column = rem & (size - 1);
        cuDoubleComplex acc = make_cuDoubleComplex(0.0, 0.0);
        for (int64_t k = 0; k < num_modes; ++k) {
            const cuDoubleComplex ee = elem_even[k];
            const cuDoubleComplex eo = elem_odd[k];
            const bool has_even = !(ee.x == 0.0 && ee.y == 0.0);
            const bool has_odd = !(eo.x == 0.0 && eo.y == 0.0);
            if (!has_even && !has_odd) {
                continue;
            }
            const int64_t pure = pure_out ^ subset_flip[k];
            const int64_t row = row_out ^ flip_mask[k];
            // Jordan-Wigner sign of the occupied modes below this one, in the
            // mixed register (the row bits) and the pure register (the subset).
            const double row_sign = ((__popc((unsigned)(row & below[k])) & 1) ? -1.0 : 1.0);
            const cuDoubleComplex a = amps[(pure * size + row) * size + column];
            if (has_even) {
                const double sign = row_sign
                    * ((__popc((unsigned)(pure & pure_lo[k])) & 1) ? -1.0 : 1.0);
                accumulate(acc, ee.x * sign, ee.y * sign, a);
            }
            if (has_odd) {
                // the odd Majorana is the even one rotated by +i, with an extra
                // -1 when it flips a row bit of a mixed mode
                const double sign = row_sign
                    * ((__popc((unsigned)(pure & pure_hi[k])) & 1) ? -1.0 : 1.0)
                    * (mixed[k] && (row & flip_mask[k]) ? -1.0 : 1.0);
                accumulate(acc, -eo.y * sign, eo.x * sign, a);
            }
        }
        out[index] = acc;
    }
}

torch::Tensor apply_label(torch::Tensor amps,
                          torch::Tensor elem_even,
                          torch::Tensor elem_odd,
                          torch::Tensor below,
                          torch::Tensor flip_mask,
                          torch::Tensor subset_flip,
                          torch::Tensor pure_lo,
                          torch::Tensor pure_hi,
                          torch::Tensor mixed) {
    TORCH_CHECK(amps.is_cuda(), "amps has to be a CUDA tensor");
    TORCH_CHECK(amps.scalar_type() == at::kComplexDouble, "amps has to be complex128");
    TORCH_CHECK(amps.dim() == 3, "amps has to be (pure_count, size, size)");
    TORCH_CHECK(amps.is_contiguous(), "amps has to be contiguous");
    for (const auto* elem : {&elem_even, &elem_odd}) {
        TORCH_CHECK(elem->is_cuda() && elem->scalar_type() == at::kComplexDouble
                    && elem->is_contiguous(),
                    "the coefficient tables have to be contiguous complex128 CUDA tensors");
    }
    TORCH_CHECK(elem_even.numel() == elem_odd.numel(),
                "the even and odd coefficient tables have to have the same length");
    const int64_t num_modes = elem_even.numel();
    for (const auto* index_tensor : {&below, &flip_mask, &subset_flip, &pure_lo, &pure_hi}) {
        TORCH_CHECK(index_tensor->is_cuda() && index_tensor->scalar_type() == at::kInt
                    && index_tensor->is_contiguous() && index_tensor->numel() == num_modes,
                    "the index tables have to be contiguous int32 CUDA tensors of length num_modes");
    }
    TORCH_CHECK(mixed.is_cuda() && mixed.scalar_type() == at::kChar && mixed.is_contiguous()
                && mixed.numel() == num_modes,
                "mixed has to be a contiguous int8 CUDA tensor of length num_modes");

    const int64_t pure_count = amps.size(0);
    const int64_t size = amps.size(1);
    TORCH_CHECK(size > 0 && (size & (size - 1)) == 0, "size has to be a power of two");
    int64_t size_bits = 0;
    while ((int64_t(1) << size_bits) < size) {
        ++size_bits;
    }
    auto out = torch::empty_like(amps);
    if (num_modes == 0 || size == 0) {
        return out.zero_();
    }
    const int64_t total = pure_count * size * size;
    const int64_t blocks = std::min<int64_t>((total + BLOCK - 1) / BLOCK, 65535);
    apply_label_kernel<<<blocks, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
        reinterpret_cast<const cuDoubleComplex*>(amps.data_ptr<c10::complex<double>>()),
        reinterpret_cast<cuDoubleComplex*>(out.data_ptr<c10::complex<double>>()),
        reinterpret_cast<const cuDoubleComplex*>(elem_even.data_ptr<c10::complex<double>>()),
        reinterpret_cast<const cuDoubleComplex*>(elem_odd.data_ptr<c10::complex<double>>()),
        below.data_ptr<int32_t>(),
        flip_mask.data_ptr<int32_t>(),
        subset_flip.data_ptr<int32_t>(),
        pure_lo.data_ptr<int32_t>(),
        pure_hi.data_ptr<int32_t>(),
        mixed.data_ptr<int8_t>(),
        num_modes, pure_count, size, size_bits);
    return out;
}

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("apply_label", &apply_label,
          "Apply one normal-mode expansion of a Majorana label to the batched patch state");
}

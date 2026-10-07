// CUDA implementation of the patch projection of ``CGO.gaussian.GaussianCGO``.
//
// ``PatchProjector`` owns everything the kernel needs and nothing the algorithm
// needs: from the Williamson rotation of the patch block and the classification
// of its modes into pure and mixed, it derives the register masks, the
// Jordan-Wigner signs and the coefficient columns, and it hands out the vacuum
// amplitude state.  The Python side only sees this object, so the bookkeeping of
// the registers never leaves the extension.
//
// The amplitude state is a tensor (pure_count, size, size): axis 0 is the pure
// register (the pure modes, pinned to their ground state), the two matrix axes
// are the mixed register.  Applying one Majorana label expands it over the n
// normal modes of the patch; each mode maps an output element back to a single
// source element, so the kernel gathers and accumulates with no atomics, and the
// even and the odd Majorana of a mode share one read.
//
// Build and load through CGO/cuda/gaussian.py (torch.utils.cpp_extension.load).
#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda_runtime.h>
#include <cuComplex.h>
#include <cstdint>
#include <algorithm>
#include <vector>

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
        const int64_t coefficient_offset,
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
            const cuDoubleComplex ee = elem_even[coefficient_offset + k];
            const cuDoubleComplex eo = elem_odd[coefficient_offset + k];
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

// The registers of one patch: the geometry the kernel is written against, derived
// once from the rotation and the pure/mixed classification of the modes.
class PatchProjector {
public:
    PatchProjector(torch::Tensor rotation, torch::Tensor is_mixed) {
        TORCH_CHECK(rotation.is_cuda(), "rotation has to be a CUDA tensor");
        TORCH_CHECK(rotation.dim() == 2 && rotation.size(0) == rotation.size(1)
                    && rotation.size(0) % 2 == 0,
                    "rotation has to be a square (2 * num_modes) x (2 * num_modes) tensor");
        const int64_t num_modes = rotation.size(0) / 2;
        TORCH_CHECK(is_mixed.numel() == num_modes, "is_mixed has to have one entry per mode");
        num_modes_ = num_modes;
        device_ = rotation.device();

        const auto flags = is_mixed.to(torch::kCPU).to(torch::kLong).contiguous();
        const auto* flag_ptr = flags.data_ptr<int64_t>();
        num_mixed_ = static_cast<int64_t>(std::count(flag_ptr, flag_ptr + num_modes, 1));
        num_pure_ = num_modes_ - num_mixed_;
        size_ = int64_t(1) << num_mixed_;
        pure_count_ = int64_t(1) << num_pure_;
        size_bits_ = num_mixed_;

        // Per-mode tables.  Mode k of the mixed register occupies bit
        // num_mixed - 1 - (its position among the mixed modes), big-endian, and
        // the modes below it are the ones the Jordan-Wigner sign counts.
        std::vector<int32_t> below(num_modes), flip_mask(num_modes), subset_flip(num_modes);
        std::vector<int32_t> pure_lo(num_modes), pure_hi(num_modes);
        std::vector<int8_t> mixed(num_modes, 0);
        int64_t mixed_position = 0, pure_position = 0;
        for (int64_t k = 0; k < num_modes; ++k) {
            pure_lo[k] = static_cast<int32_t>((int64_t(1) << pure_position) - 1);
            pure_hi[k] = pure_lo[k];
            below[k] = static_cast<int32_t>(((int64_t(1) << mixed_position) - 1)
                                            << (num_mixed_ - mixed_position));
            if (flag_ptr[k] == 1) {
                mixed[k] = 1;
                flip_mask[k] = static_cast<int32_t>(int64_t(1) << (num_mixed_ - 1 - mixed_position));
                ++mixed_position;
            } else {
                subset_flip[k] = static_cast<int32_t>(int64_t(1) << pure_position);
                pure_hi[k] |= subset_flip[k];
                ++pure_position;
            }
        }
        // the derived tables are built on the host and copied to the device
        const auto index_options = torch::TensorOptions().dtype(torch::kInt32);
        const auto flag_options = torch::TensorOptions().dtype(torch::kInt8);
        below_ = torch::from_blob(below.data(), {num_modes}, index_options).to(device_);
        flip_mask_ = torch::from_blob(flip_mask.data(), {num_modes}, index_options).to(device_);
        subset_flip_ = torch::from_blob(subset_flip.data(), {num_modes}, index_options).to(device_);
        pure_lo_ = torch::from_blob(pure_lo.data(), {num_modes}, index_options).to(device_);
        pure_hi_ = torch::from_blob(pure_hi.data(), {num_modes}, index_options).to(device_);
        mixed_ = torch::from_blob(mixed.data(), {num_modes}, flag_options).to(device_);

        // The even and the odd Majorana of every mode, as one column per patch
        // Majorana: rotation rows 0, 2, ... and 1, 3, ... transposed.
        const auto complex_rotation = rotation.to(torch::kComplexDouble);
        const auto complex_options = torch::TensorOptions().dtype(torch::kComplexDouble)
                                                         .device(device_);
        elem_even_ = complex_rotation.slice(0, 0, complex_rotation.size(0), 2)
                                    .transpose(0, 1).contiguous();
        elem_odd_ = complex_rotation.slice(0, 1, complex_rotation.size(0), 2)
                                   .transpose(0, 1).contiguous();

        // The pure vacuum: the identity on the mixed register, pure register at 0.
        vacuum_ = torch::zeros({pure_count_, size_, size_}, complex_options);
        vacuum_.select(0, 0).copy_(torch::eye(size_, complex_options));
    }

    // The amplitude state a string starts from, shape (pure_count, size, size).
    // The same buffer is returned every time; ``apply`` never writes into it.
    torch::Tensor vacuum() const {
        return vacuum_;
    }

    // Apply the normal-mode expansion of patch Majorana `patch_index` to the
    // batched amplitude state `amps`, shape (pure_count, size, size).
    torch::Tensor apply(torch::Tensor amps, int64_t patch_index) const {
        TORCH_CHECK(amps.is_cuda() && amps.scalar_type() == at::kComplexDouble
                    && amps.is_contiguous() && amps.dim() == 3,
                    "amps has to be a contiguous complex128 (pure_count, size, size) CUDA tensor");
        TORCH_CHECK(amps.size(0) == pure_count_ && amps.size(1) == size_ && amps.size(2) == size_,
                    "amps has the wrong shape for this patch");
        TORCH_CHECK(patch_index >= 0 && patch_index < 2 * num_modes_,
                    "patch_index is outside the Majoranas of the patch");
        auto out = torch::empty_like(amps);
        if (num_modes_ == 0 || size_ == 0) {
            return out.zero_();
        }
        const int64_t total = pure_count_ * size_ * size_;
        const int64_t blocks = std::min<int64_t>((total + BLOCK - 1) / BLOCK, 65535);
        apply_label_kernel<<<blocks, BLOCK, 0, at::cuda::getCurrentCUDAStream()>>>(
            reinterpret_cast<const cuDoubleComplex*>(amps.data_ptr<c10::complex<double>>()),
            reinterpret_cast<cuDoubleComplex*>(out.data_ptr<c10::complex<double>>()),
            reinterpret_cast<const cuDoubleComplex*>(elem_even_.data_ptr<c10::complex<double>>()),
            reinterpret_cast<const cuDoubleComplex*>(elem_odd_.data_ptr<c10::complex<double>>()),
            patch_index * num_modes_,
            below_.data_ptr<int32_t>(),
            flip_mask_.data_ptr<int32_t>(),
            subset_flip_.data_ptr<int32_t>(),
            pure_lo_.data_ptr<int32_t>(),
            pure_hi_.data_ptr<int32_t>(),
            mixed_.data_ptr<int8_t>(),
            num_modes_, pure_count_, size_, size_bits_);
        return out;
    }

private:
    int64_t num_modes_ = 0, num_mixed_ = 0, num_pure_ = 0;
    int64_t size_ = 1, pure_count_ = 1, size_bits_ = 0;
    torch::Device device_ = torch::kCUDA;
    torch::Tensor elem_even_, elem_odd_;
    torch::Tensor below_, flip_mask_, subset_flip_, pure_lo_, pure_hi_, mixed_;
    torch::Tensor vacuum_;
};

}  // namespace

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    pybind11::class_<PatchProjector>(m, "PatchProjector",
                                     "The registers of one Gaussian patch: the kernel geometry\n"
                                     "derived from the rotation and the pure/mixed modes.")
        .def(pybind11::init<torch::Tensor, torch::Tensor>(),
             pybind11::arg("rotation"), pybind11::arg("is_mixed"))
        .def("vacuum", &PatchProjector::vacuum,
             "The amplitude state a Majorana string starts from, (pure_count, size, size)")
        .def("apply", &PatchProjector::apply,
             pybind11::arg("amps"), pybind11::arg("patch_index"),
             "Apply one patch Majorana to the batched amplitude state");
}

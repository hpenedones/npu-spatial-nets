// SPDX-License-Identifier: Apache-2.0
//
// Custom AIE2P kernels for the recurrent MLP.
//
// These kernels supplement IRON's built-in matmul (mm.cc) with the
// activation and buffer-management operations our recurrent loop needs:
//
//   relu_inplace_bf16  — In-place ReLU: max(x, 0)
//   copy_bf16          — Vectorized buffer copy (ping-pong output staging)
//
// Both operate on bf16 buffers using 512-bit SIMD (v32bfloat16 = 32 × 16-bit).
// Buffers MUST be aligned to 64 bytes (32 bf16 elements) and sizes MUST be
// multiples of 32 elements — these are guaranteed by IRON's ObjectFIFO
// allocation for our tile sizes (B×H where B=16, H=128 → 2048 elements).
//
// event0()/event1() are AIE profiling markers for cycle-level tracing.

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>

// v32bfloat16 = 32 bf16 values = 512 bits = one AIE SIMD register width.
static constexpr int SIMD_WIDTH = 32;

extern "C" {

// Apply max(x, 0) element-wise, in-place.
// Requires: num_elements % SIMD_WIDTH == 0, buf 64-byte aligned.
void relu_inplace_bf16(bfloat16 *__restrict buf, int32_t num_elements)
{
    event0();

    v32bfloat16 zeroes = broadcast_zero_to_v32bfloat16();
    for (int i = 0; i < num_elements; i += SIMD_WIDTH) {
        v32bfloat16 val = *(v32bfloat16 *)(buf + i);
        *(v32bfloat16 *)(buf + i) = max(val, zeroes);
    }

    event1();
}

// Copy num_elements bf16 values from src to dst.
// Used to stage the final loop result into the output ObjectFIFO buffer.
// Requires: num_elements % SIMD_WIDTH == 0, both pointers 64-byte aligned.
void copy_bf16(bfloat16 *__restrict src, bfloat16 *__restrict dst,
               int32_t num_elements)
{
    event0();

    for (int i = 0; i < num_elements; i += SIMD_WIDTH) {
        v32bfloat16 v = *(v32bfloat16 *)(src + i);
        *(v32bfloat16 *)(dst + i) = v;
    }

    event1();
}

} // extern "C"

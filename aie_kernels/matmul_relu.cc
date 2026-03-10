// SPDX-License-Identifier: Apache-2.0
//
// Fused matmul + ReLU kernel for the recurrent MLP: C = ReLU(A × B)
//
// This replaces THREE separate kernel calls (zero_bf16 + matmul_bf16_bf16 +
// relu_inplace_bf16) with ONE fused call:
//   1. Accumulators are zero-initialized in registers (no separate zero call)
//   2. ReLU is applied during the store phase (no separate relu call)
//
// Based on the same 2×2 mmul expansion strategy as IRON's mm.cc for high
// SIMD utilization. The key difference is that mm.cc accumulates (C += A*B)
// by loading existing C values, while this kernel starts fresh (C = A*B).
//
// Compile with: -DDIM_M=B -DDIM_K=H -DDIM_N=H
//               -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16  (for 8×8×8 tiles)

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 16
#endif
#ifndef DIM_K
#define DIM_K 128
#endif
#ifndef DIM_N
#define DIM_N 128
#endif

// Core fused matmul+ReLU with 2×2 mmul tile expansion.
//
// Template params rowA/colA/colB are tile-level loop bounds (matrix dims
// divided by the mmul block factors r/s/t). The 2×2 expansion processes
// four output blocks per inner iteration for better register utilization.
template <typename T_in, typename T_out,
          unsigned rowA, unsigned colA, unsigned colB,
          unsigned r, unsigned s, unsigned t>
static inline void
matmul_relu_2x2(const T_in *__restrict pA,
                const T_in *__restrict pB,
                T_out *__restrict pC)
{
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    const aie::vector<T_out, MMUL::size_C> zeros =
        aie::zeros<T_out, MMUL::size_C>();

    event0();

    for (unsigned z = 0; z < rowA; z += 2)
        chess_prepare_for_pipelining chess_loop_range(4, )
        {
            T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;
            T_out *__restrict pC2 = pC + ((z + 1) * colB) * MMUL::size_C;

            for (unsigned j = 0; j < colB; j += 2)
                chess_flatten_loop
                {
                const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
                const T_in *__restrict pA2 =
                    pA + ((z + 1) * colA) * MMUL::size_A;
                const T_in *__restrict pB1 = pB + j * MMUL::size_B;
                const T_in *__restrict pB2 = pB + (j + 1) * MMUL::size_B;

                // Zero-init accumulators (replaces separate zero_bf16 call)
                MMUL C00(zeros);
                MMUL C01(zeros);
                MMUL C10(zeros);
                MMUL C11(zeros);

                aie::vector<T_in, MMUL::size_A> A0, A1;
                aie::vector<T_in, MMUL::size_B> B0, B1;

                for (unsigned i = 0; i < colA; ++i)
                    chess_flatten_loop
                    {
                    A0 = aie::load_v<MMUL::size_A>(pA1);
                    pA1 += MMUL::size_A;
                    A1 = aie::load_v<MMUL::size_A>(pA2);
                    pA2 += MMUL::size_A;
                    B0 = aie::load_v<MMUL::size_B>(pB1);
                    pB1 += MMUL::size_B * colB;
                    B1 = aie::load_v<MMUL::size_B>(pB2);
                    pB2 += MMUL::size_B * colB;

                    C00.mac(A0, B0);
                    C01.mac(A0, B1);
                    C10.mac(A1, B0);
                    C11.mac(A1, B1);
                }

                // Fused ReLU + store (replaces separate relu_inplace_bf16 call)
                aie::store_v(pC1,
                    aie::max(C00.template to_vector<T_out>(), zeros));
                pC1 += MMUL::size_C;
                aie::store_v(pC1,
                    aie::max(C01.template to_vector<T_out>(), zeros));
                pC1 += MMUL::size_C;
                aie::store_v(pC2,
                    aie::max(C10.template to_vector<T_out>(), zeros));
                pC2 += MMUL::size_C;
                aie::store_v(pC2,
                    aie::max(C11.template to_vector<T_out>(), zeros));
                pC2 += MMUL::size_C;
            }
        }

    event1();
}

extern "C" {

// Fused C = ReLU(A × B) for bfloat16.
// BFP16 emulation uses 8×8×8 mmul blocks; native bf16 uses 4×8×8.
#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
void matmul_relu_bf16_bf16(bfloat16 *a, bfloat16 *b, bfloat16 *c)
{
    constexpr int r = 8, s = 8, t = 8;
    static_assert(DIM_M % (2 * r) == 0);
    static_assert(DIM_K % s == 0);
    static_assert(DIM_N % (2 * t) == 0);
    ::aie::set_rounding(aie::rounding_mode::floor);
    matmul_relu_2x2<bfloat16, bfloat16,
                     (DIM_M / r), (DIM_K / s), (DIM_N / t),
                     r, s, t>(a, b, c);
}
#else
void matmul_relu_bf16_bf16(bfloat16 *a, bfloat16 *b, bfloat16 *c)
{
    constexpr int r = 4, s = 8, t = 8;
    static_assert(DIM_M % (2 * r) == 0);
    static_assert(DIM_K % s == 0);
    static_assert(DIM_N % (2 * t) == 0);
    ::aie::set_rounding(aie::rounding_mode::floor);
    matmul_relu_2x2<bfloat16, bfloat16,
                     (DIM_M / r), (DIM_K / s), (DIM_N / t),
                     r, s, t>(a, b, c);
}
#endif

} // extern "C"

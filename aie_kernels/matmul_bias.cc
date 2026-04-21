// SPDX-License-Identifier: Apache-2.0
//
// Fused matmul + bias kernel:  c = a @ w + b_broadcast
//
// Used by the embed and head stages of the full-NPU residual pipeline.
// The residual body uses the separate matmul_relu_skip kernel; this file
// covers the two bare Linear layers that bracket it.
//
// Memory layout (tiled 8x8 blocks, bf16):
//   a:    [M, K]  tiled as [M/8, K/8, 8, 8]
//   w:    [K, N]  tiled as [K/8, N/8, 8, 8]   (resident in tile SRAM)
//   bias: [N/8, 64]                           (resident in tile SRAM)
//         Row j contains the 8-element bias slice bias[j*8:(j+1)*8]
//         replicated across 8 rows of a 64-element MMUL output tile,
//         so the kernel adds it with a single aie::add.
//   c:    [M, N]  tiled as [M/8, N/8, 8, 8]
//
// Loop structure mirrors matmul_relu_skip.cc: z-outer, j-middle with
// ColBStep accumulators kept live across the K-reduction, i-innermost.
// A is loaded once per i and MAC'd into all live output accumulators.
// See matmul_relu_skip.cc for the reload-amortization rationale.
//
// Compile with:
//   -DDIM_M=<B>  -DDIM_K=<input dim>  -DDIM_N=<output dim>
//   -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_K
#define DIM_K 32
#endif
#ifndef DIM_N
#define DIM_N 64
#endif
#ifndef MATMUL_BIAS_ENTRY_NAME
#define MATMUL_BIAS_ENTRY_NAME matmul_bias_bf16
#endif

template <typename T, typename MMUL, typename Vec>
static inline void
store_with_bias(const MMUL &acc,
                const T *__restrict pBias,
                T *__restrict pC,
                const Vec & /*zeros*/)
{
    auto vec_c = acc.template to_vector<T>();
    auto bias_v = aie::load_v<MMUL::size_C>(pBias);
    aie::store_v(pC, aie::add(vec_c, bias_v));
}

template <typename T, unsigned rowA, unsigned colA, unsigned colB, unsigned ColBStep>
static inline void
matmul_bias_blocked(const T *__restrict pA,
                    const T *__restrict pB,
                    const T *__restrict pBias,
                    T *__restrict pC)
{
    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, T, T, accauto>;
    static_assert(ColBStep == 1 || ColBStep == 2 || ColBStep == 4, "Unsupported ColBStep");
    const auto zeros = aie::zeros<T, MMUL::size_C>();

    event0();

    for (unsigned z = 0; z < rowA; z += 1) {
        T *__restrict pC_row = pC + z * colB * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += ColBStep)
        {
            const T *__restrict pA1 = pA + z * colA * MMUL::size_A;
            const T *__restrict pB0 = pB + (j + 0) * MMUL::size_B;

            if constexpr (ColBStep == 1) {
                MMUL C00(zeros);

                for (unsigned i = 0; i < colA; ++i)
                    chess_prepare_for_pipelining chess_loop_range(3, )
                {
                    auto A0 = aie::load_v<MMUL::size_A>(pA1);
                    pA1 += MMUL::size_A;
                    auto B0 = aie::load_v<MMUL::size_B>(pB0);
                    pB0 += MMUL::size_B * colB;
                    C00.mac(A0, B0);
                }

                store_with_bias<T, MMUL>(
                    C00,
                    pBias + (j + 0) * MMUL::size_C,
                    pC_row + (j + 0) * MMUL::size_C,
                    zeros
                );
            } else if constexpr (ColBStep == 2) {
                MMUL C00(zeros), C01(zeros);
                const T *__restrict pB1 = pB + (j + 1) * MMUL::size_B;

                for (unsigned i = 0; i < colA; ++i)
                    chess_prepare_for_pipelining chess_loop_range(3, )
                {
                    auto A0 = aie::load_v<MMUL::size_A>(pA1);
                    pA1 += MMUL::size_A;

                    auto B0 = aie::load_v<MMUL::size_B>(pB0);
                    pB0 += MMUL::size_B * colB;
                    auto B1 = aie::load_v<MMUL::size_B>(pB1);
                    pB1 += MMUL::size_B * colB;

                    C00.mac(A0, B0);
                    C01.mac(A0, B1);
                }

                store_with_bias<T, MMUL>(
                    C00,
                    pBias + (j + 0) * MMUL::size_C,
                    pC_row + (j + 0) * MMUL::size_C,
                    zeros
                );
                store_with_bias<T, MMUL>(
                    C01,
                    pBias + (j + 1) * MMUL::size_C,
                    pC_row + (j + 1) * MMUL::size_C,
                    zeros
                );
            } else {
                MMUL C00(zeros), C01(zeros), C02(zeros), C03(zeros);
                const T *__restrict pB1 = pB + (j + 1) * MMUL::size_B;
                const T *__restrict pB2 = pB + (j + 2) * MMUL::size_B;
                const T *__restrict pB3 = pB + (j + 3) * MMUL::size_B;

                for (unsigned i = 0; i < colA; ++i)
                    chess_prepare_for_pipelining chess_loop_range(3, )
                {
                    auto A0 = aie::load_v<MMUL::size_A>(pA1);
                    pA1 += MMUL::size_A;

                    auto B0 = aie::load_v<MMUL::size_B>(pB0);
                    pB0 += MMUL::size_B * colB;
                    auto B1 = aie::load_v<MMUL::size_B>(pB1);
                    pB1 += MMUL::size_B * colB;
                    auto B2 = aie::load_v<MMUL::size_B>(pB2);
                    pB2 += MMUL::size_B * colB;
                    auto B3 = aie::load_v<MMUL::size_B>(pB3);
                    pB3 += MMUL::size_B * colB;

                    C00.mac(A0, B0);
                    C01.mac(A0, B1);
                    C02.mac(A0, B2);
                    C03.mac(A0, B3);
                }

                store_with_bias<T, MMUL>(C00, pBias + (j + 0) * MMUL::size_C, pC_row + (j + 0) * MMUL::size_C, zeros);
                store_with_bias<T, MMUL>(C01, pBias + (j + 1) * MMUL::size_C, pC_row + (j + 1) * MMUL::size_C, zeros);
                store_with_bias<T, MMUL>(C02, pBias + (j + 2) * MMUL::size_C, pC_row + (j + 2) * MMUL::size_C, zeros);
                store_with_bias<T, MMUL>(C03, pBias + (j + 3) * MMUL::size_C, pC_row + (j + 3) * MMUL::size_C, zeros);
            }
        }
    }

    event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned colB>
static inline void
matmul_bias(const T *__restrict pA,
            const T *__restrict pB,
            const T *__restrict pBias,
            T *__restrict pC)
{
    if constexpr (colB >= 4 && (colB % 4) == 0) {
        matmul_bias_blocked<T, rowA, colA, colB, 4>(pA, pB, pBias, pC);
    } else if constexpr (colB >= 2 && (colB % 2) == 0) {
        matmul_bias_blocked<T, rowA, colA, colB, 2>(pA, pB, pBias, pC);
    } else {
        matmul_bias_blocked<T, rowA, colA, colB, 1>(pA, pB, pBias, pC);
    }
}

extern "C" {

// Single entry point used by both the embed and head stages. The caller
// compiles this file twice with different (DIM_K, DIM_N) to get two archives.
void MATMUL_BIAS_ENTRY_NAME(bfloat16 *a, bfloat16 *w, bfloat16 *bias, bfloat16 *c)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_N % 8 == 0, "Output dim must be multiple of 8");

    matmul_bias<bfloat16, (DIM_M / 8), (DIM_K / 8), (DIM_N / 8)>(a, w, bias, c);
}

} // extern "C"

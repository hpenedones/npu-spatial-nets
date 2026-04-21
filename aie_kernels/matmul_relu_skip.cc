// SPDX-License-Identifier: Apache-2.0
//
// Residual MLP kernel:  c = relu(a @ w) + a
//
// Each compute tile in the snake pipeline applies this operation once.
// The weight matrix (w) lives permanently in tile SRAM; only the small
// activation vector (a → c) flows between tiles.
//
// Memory layout:
//   a: [B × H] bf16, tiled as [B/8, H/8, 8, 8]
//   w: [H × H] bf16, tiled as [H/8, H/8, 8, 8]
//   c: [B × H] bf16, tiled as [B/8, H/8, 8, 8]
//
// Compile with:
//   -DDIM_M=<B>  -DDIM_K=<H>  -DDIM_N=<H>
//   -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_K
#define DIM_K 160
#endif
#ifndef DIM_N
#define DIM_N 160
#endif

// ── Fused matmul + ReLU + skip ────────────────────────────────────────
//
// Computes  c = max(a × w, 0) + a  using the 8×8×8 BFP16 mmul intrinsic.
//
// Template parameters:
//   rowA = DIM_M / 8   (number of 8-row blocks in A)
//   colA = DIM_K / 8   (reduction dimension in 8-col blocks)
//   colB = DIM_N / 8   (number of 8-col blocks in output)
//
// The key throughput optimization is to accumulate several output-column blocks
// at once. The old kernel iterated j outside i, which reloaded the same A block
// once per output tile. Here we keep a small bank of MMUL accumulators across
// the j dimension, load each A tile once per i step, and MAC it into multiple
// outputs before advancing. The residual add is fused into the final store so
// there is no separate pass over C afterwards.

template <typename T, typename MMUL, bool StoreMask, typename Vec>
static inline void
store_relu_skip(const MMUL &acc,
                const T *__restrict pSkip,
                T *__restrict pC,
                T *__restrict pMask,
                const Vec &zeros,
                const Vec &ones)
{
    auto vec_c = acc.template to_vector<T>();
    auto relu_c = aie::max(vec_c, zeros);
    auto skip_v = aie::load_v<MMUL::size_C>(pSkip);
    aie::store_v(pC, aie::add(relu_c, skip_v));

    if constexpr (StoreMask) {
        auto gt_mask = aie::gt(vec_c, zeros);
        auto mask_v = aie::select(zeros, ones, gt_mask);
        aie::store_v(pMask, mask_v);
    }
}

template <typename T, unsigned rowA, unsigned colA, unsigned colB, bool StoreMask, unsigned ColBStep>
static inline void
matmul_relu_blocked(const T *__restrict pA,
                    const T *__restrict pB,
                    T *__restrict pC,
                    T *__restrict pMask = nullptr)
{
    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, T, T, accauto>;
    static_assert(ColBStep == 1 || ColBStep == 2 || ColBStep == 4, "Unsupported ColBStep");
    static_assert(colA == colB, "Residual skip requires square tiled activations");
    const auto zeros = aie::zeros<T, MMUL::size_C>();
    const auto ones = aie::broadcast<T, MMUL::size_C>(1.0f);

    event0();

    for (unsigned z = 0; z < rowA; z += 1) {
        T *__restrict pC_row = pC + z * colB * MMUL::size_C;
        T *__restrict pM_row = nullptr;
        if constexpr (StoreMask) {
            pM_row = pMask + z * colB * MMUL::size_C;
        }

        for (unsigned j = 0; j < colB; j += ColBStep)
        {
            const T *__restrict pA1 = pA + z * colA * MMUL::size_A;
            MMUL C00(zeros);
            const T *__restrict pB0 = pB + (j + 0) * MMUL::size_B;

            if constexpr (ColBStep == 1) {
                for (unsigned i = 0; i < colA; ++i)
                    chess_prepare_for_pipelining chess_loop_range(3, )
                {
                    auto A0 = aie::load_v<MMUL::size_A>(pA1);
                    pA1 += MMUL::size_A;
                    auto B0 = aie::load_v<MMUL::size_B>(pB0);
                    pB0 += MMUL::size_B * colB;
                    C00.mac(A0, B0);
                }

                const T *__restrict pSkip = pA + (z * colA + j) * MMUL::size_C;
                T *__restrict pC_block = pC_row + j * MMUL::size_C;
                T *__restrict pM_block = nullptr;
                if constexpr (StoreMask) {
                    pM_block = pM_row + j * MMUL::size_C;
                }

                store_relu_skip<T, MMUL, StoreMask>(C00, pSkip, pC_block, pM_block, zeros, ones);
            } else if constexpr (ColBStep == 2) {
                MMUL C01(zeros);
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

                const T *__restrict pSkip = pA + (z * colA + j) * MMUL::size_C;
                T *__restrict pC_block = pC_row + j * MMUL::size_C;
                T *__restrict pM_block = nullptr;
                if constexpr (StoreMask) {
                    pM_block = pM_row + j * MMUL::size_C;
                }

                store_relu_skip<T, MMUL, StoreMask>(C00, pSkip, pC_block, pM_block, zeros, ones);
                store_relu_skip<T, MMUL, StoreMask>(
                    C01,
                    pSkip + MMUL::size_C,
                    pC_block + MMUL::size_C,
                    pM_block ? pM_block + MMUL::size_C : nullptr,
                    zeros,
                    ones
                );
            } else {
                MMUL C01(zeros), C02(zeros), C03(zeros);
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

                const T *__restrict pSkip = pA + (z * colA + j) * MMUL::size_C;
                T *__restrict pC_block = pC_row + j * MMUL::size_C;
                T *__restrict pM_block = nullptr;
                if constexpr (StoreMask) {
                    pM_block = pM_row + j * MMUL::size_C;
                }

                store_relu_skip<T, MMUL, StoreMask>(C00, pSkip, pC_block, pM_block, zeros, ones);
                store_relu_skip<T, MMUL, StoreMask>(
                    C01,
                    pSkip + MMUL::size_C,
                    pC_block + MMUL::size_C,
                    pM_block ? pM_block + MMUL::size_C : nullptr,
                    zeros,
                    ones
                );
                store_relu_skip<T, MMUL, StoreMask>(
                    C02,
                    pSkip + 2 * MMUL::size_C,
                    pC_block + 2 * MMUL::size_C,
                    pM_block ? pM_block + 2 * MMUL::size_C : nullptr,
                    zeros,
                    ones
                );
                store_relu_skip<T, MMUL, StoreMask>(
                    C03,
                    pSkip + 3 * MMUL::size_C,
                    pC_block + 3 * MMUL::size_C,
                    pM_block ? pM_block + 3 * MMUL::size_C : nullptr,
                    zeros,
                    ones
                );
            }
        }
    }

    event1();
}

template <typename T, unsigned rowA, unsigned colA, unsigned colB, bool StoreMask>
static inline void
matmul_relu(const T *__restrict pA,
            const T *__restrict pB,
            T *__restrict pC,
            T *__restrict pMask = nullptr)
{
    if constexpr (colB >= 4 && (colB % 4) == 0) {
        matmul_relu_blocked<T, rowA, colA, colB, StoreMask, 4>(pA, pB, pC, pMask);
    } else if constexpr (colB >= 2 && (colB % 2) == 0) {
        matmul_relu_blocked<T, rowA, colA, colB, StoreMask, 2>(pA, pB, pC, pMask);
    } else {
        matmul_relu_blocked<T, rowA, colA, colB, StoreMask, 1>(pA, pB, pC, pMask);
    }
}

// ── Entry point ────────────────────────────────────────────────────────

extern "C" {

void matmul_relu_skip_bf16(bfloat16 *a, bfloat16 *w, bfloat16 *c, bfloat16 *mask_out, int mask_offset)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_N % 8 == 0, "Output dim must be multiple of 8");

    bfloat16 *mask = mask_out + mask_offset;

    matmul_relu<bfloat16, (DIM_M / 8), (DIM_K / 8), (DIM_N / 8), true>(a, w, c, mask);
}

void matmul_relu_skip_infer_bf16(bfloat16 *a, bfloat16 *w, bfloat16 *c)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_N % 8 == 0, "Output dim must be multiple of 8");

    matmul_relu<bfloat16, (DIM_M / 8), (DIM_K / 8), (DIM_N / 8), false>(a, w, c);
}

} // extern "C"

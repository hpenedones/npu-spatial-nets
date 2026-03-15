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

// ── Fused matmul + ReLU ────────────────────────────────────────────────
//
// Computes  c = max(a × w, 0)  using the 8×8×8 BFP16 mmul intrinsic.
//
// Template parameters:
//   rowA = DIM_M / 8   (number of 8-row blocks in A)
//   colA = DIM_K / 8   (reduction dimension in 8-col blocks)
//   colB = DIM_N / 8   (number of 8-col blocks in output)
//
// The outer z-loop walks row blocks one at a time (1×1 expansion).
// The middle j-loop walks output column blocks — with ≥3 iterations
// for effective chess pipelining (H ≥ 24).
// The inner i-loop accumulates the K-dimension reduction.

template <typename T, unsigned rowA, unsigned colA, unsigned colB, bool StoreMask>
static inline void
matmul_relu(const T *__restrict pA,
            const T *__restrict pB,
            T *__restrict pC,
            T *__restrict pMask = nullptr)
{
    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, T, T, accauto>;
    const auto zeros = aie::zeros<T, MMUL::size_C>();
    const auto ones = aie::broadcast<T, MMUL::size_C>(1.0f);

    event0();

    for (unsigned z = 0; z < rowA; z += 1) {
        T *__restrict pC1 = pC + z * colB * MMUL::size_C;
        T *__restrict pM1 = nullptr;
        if constexpr (StoreMask) {
            pM1 = pMask + z * colB * MMUL::size_C;
        }

        for (unsigned j = 0; j < colB; j += 1)
            chess_prepare_for_pipelining chess_loop_range(3, )
        {
            const T *__restrict pA1 = pA + z * colA * MMUL::size_A;
            const T *__restrict pB1 = pB + j * MMUL::size_B;

            MMUL C00(zeros);

            for (unsigned i = 0; i < colA; ++i)
                chess_flatten_loop
            {
                auto A0 = aie::load_v<MMUL::size_A>(pA1);
                pA1 += MMUL::size_A;
                auto B0 = aie::load_v<MMUL::size_B>(pB1);
                pB1 += MMUL::size_B * colB;
                C00.mac(A0, B0);
            }

            auto vec_c = C00.template to_vector<T>();
            auto relu_c = aie::max(vec_c, zeros);

            aie::store_v(pC1, relu_c);
            pC1 += MMUL::size_C;
            if constexpr (StoreMask) {
                auto gt_mask = aie::gt(vec_c, zeros);
                auto mask_v = aie::select(zeros, ones, gt_mask);
                aie::store_v(pM1, mask_v);
                pM1 += MMUL::size_C;
            }
        }
    }

    event1();
}

// ── Residual (skip) connection ─────────────────────────────────────────
//
// Element-wise:  c[i] += a[i]
// Both a and c are in tiled layout, but element-wise add is order-agnostic.

static inline void
residual_add(const bfloat16 *__restrict a, bfloat16 *__restrict c)
{
    constexpr int total = DIM_M * DIM_N;
    // Process 32 elements per iteration to match the vector add used below.
    static_assert(total % 32 == 0, "Total elements must be divisible by 32");

    for (int i = 0; i < total; i += 32) {
        auto vc = aie::load_v<32>(c + i);
        auto va = aie::load_v<32>(a + i);
        aie::store_v(c + i, aie::add(vc, va));
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

    // Step 1: c = relu(a @ w)
    matmul_relu<bfloat16, (DIM_M / 8), (DIM_K / 8), (DIM_N / 8), true>(a, w, c, mask);

    // Step 2: c += a  (residual skip connection)
    residual_add(a, c);
}

void matmul_relu_skip_infer_bf16(bfloat16 *a, bfloat16 *w, bfloat16 *c)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_N % 8 == 0, "Output dim must be multiple of 8");

    matmul_relu<bfloat16, (DIM_M / 8), (DIM_K / 8), (DIM_N / 8), false>(a, w, c);
    residual_add(a, c);
}

} // extern "C"

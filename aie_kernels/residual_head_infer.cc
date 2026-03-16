// SPDX-License-Identifier: Apache-2.0
//
// Fused tail kernel for streaming residual inference:
//   hidden = relu(a @ w_res) + a
//   logits = hidden @ w_head + b_head
//
// The residual input/output use tiled 8x8 layout. The final logits are written
// row-major so the host can drain compact class scores directly.

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_H
#define DIM_H 160
#endif
#ifndef DIM_N_CLS
#define DIM_N_CLS 16
#endif

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
}

template <typename T, unsigned rowA, unsigned colA, unsigned colB>
static inline void
matmul_plain(const T *__restrict pA,
             const T *__restrict pB,
             T *__restrict pC)
{
    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, T, T, accauto>;
    const auto zeros = aie::zeros<accfloat, MMUL::size_C>();

    for (unsigned z = 0; z < rowA; ++z) {
        T *__restrict pC1 = pC + z * colB * MMUL::size_C;

        for (unsigned j = 0; j < colB; ++j)
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

            aie::store_v(pC1, C00.template to_vector<T>());
            pC1 += MMUL::size_C;
        }
    }
}

static inline void
residual_add(const bfloat16 *__restrict a, bfloat16 *__restrict c)
{
    constexpr int total = DIM_M * DIM_H;
    static_assert(total % 32 == 0, "Total elements must be divisible by 32");

    for (int i = 0; i < total; i += 32) {
        auto vc = aie::load_v<32>(c + i);
        auto va = aie::load_v<32>(a + i);
        aie::store_v(c + i, aie::add(vc, va));
    }
}

extern "C" {

void residual_head_infer_bf16(
    bfloat16 *a,
    bfloat16 *w_res,
    bfloat16 *w_head,
    bfloat16 *b_head,
    bfloat16 *logits_out)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_H % 8 == 0, "Hidden dim must be multiple of 8");
    static_assert(DIM_N_CLS % 8 == 0, "Class dim must be multiple of 8");

    alignas(32) bfloat16 hidden[DIM_M * DIM_H];

    matmul_relu<bfloat16, (DIM_M / 8), (DIM_H / 8), (DIM_H / 8), false>(a, w_res, hidden);
    residual_add(a, hidden);

    matmul_plain<bfloat16, (DIM_M / 8), (DIM_H / 8), (DIM_N_CLS / 8)>(
        hidden, w_head, logits_out);

    constexpr int br = 8, bc = 8;
    for (int bi = 0; bi < DIM_M / br; ++bi) {
        for (int bj = 0; bj < DIM_N_CLS / bc; ++bj) {
            bfloat16 *block = logits_out + (bi * (DIM_N_CLS / bc) + bj) * br * bc;
            bfloat16 *bias_block = b_head + bj * bc;
            for (int r = 0; r < br; ++r) {
                for (int c = 0; c < bc; ++c) {
                    int idx = r * bc + c;
                    block[idx] = (bfloat16)((float)block[idx] + (float)bias_block[c]);
                }
            }
        }
    }
}

} // extern "C"

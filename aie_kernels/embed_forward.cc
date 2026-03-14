// SPDX-License-Identifier: Apache-2.0
//
// Embed forward kernel:  y = x @ W_chunk, accumulated over streamed K chunks.

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_K_EMBED
#define DIM_K_EMBED 56
#endif
#ifndef DIM_H
#define DIM_H 32
#endif

template <typename T, unsigned rowA, unsigned colA, unsigned colB>
static inline void
matmul_plain(const T *__restrict pA, const T *__restrict pB, T *__restrict pC)
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

extern "C" {

void embed_forward_bf16(bfloat16 *x, bfloat16 *w, bfloat16 *y, int32_t clear_y)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K_EMBED % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_H % 8 == 0, "Hidden dim must be multiple of 8");

    alignas(32) bfloat16 partial[DIM_M * DIM_H];
    static_assert((DIM_M * DIM_H) % 32 == 0, "Output elements must be divisible by 32");

    matmul_plain<bfloat16, (DIM_M / 8), (DIM_K_EMBED / 8), (DIM_H / 8)>(x, w, partial);

    if (clear_y) {
        for (int i = 0; i < DIM_M * DIM_H; i += 32) {
            auto v = aie::load_v<32>(partial + i);
            aie::store_v(y + i, v);
        }
        return;
    }

    for (int i = 0; i < DIM_M * DIM_H; i += 32) {
        auto vy = aie::load_v<32>(y + i);
        auto vp = aie::load_v<32>(partial + i);
        aie::store_v(y + i, aie::add(vy, vp));
    }
}

} // extern "C"

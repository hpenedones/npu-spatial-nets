// SPDX-License-Identifier: Apache-2.0
//
// Embed backward kernel:  W_chunk = W_chunk - lr * (x_chunk^T @ dy)

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_K_EMBED
#define DIM_K_EMBED 56
#endif
#ifndef DIM_H
#define DIM_H 32
#endif

template <typename T>
static inline void
transpose_tile_8x8(const T *__restrict src, T *__restrict dst)
{
    for (unsigned r = 0; r < 8; ++r) {
        for (unsigned c = 0; c < 8; ++c) {
            dst[c * 8 + r] = src[r * 8 + c];
        }
    }
}

extern "C" {

void embed_backward_bf16(bfloat16 *x, bfloat16 *w, bfloat16 *dy)
{
    float lr = 0.01f;

    static_assert(DIM_M % 8 == 0);
    static_assert(DIM_K_EMBED % 8 == 0);
    static_assert(DIM_H % 8 == 0);

    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, bfloat16, bfloat16, accauto>;
    const auto zeros = aie::zeros<accfloat, MMUL::size_C>();
    alignas(32) bfloat16 a_tile_t[MMUL::size_A];

    constexpr unsigned rowsK = DIM_K_EMBED / 8;
    constexpr unsigned rowsM = DIM_M / 8;
    constexpr unsigned colsH = DIM_H / 8;

    for (unsigned z = 0; z < rowsK; ++z) {
        for (unsigned j = 0; j < colsH; ++j) {
            bfloat16 *w_block = w + (z * colsH + j) * MMUL::size_C;
            MMUL C00(zeros);

            for (unsigned i = 0; i < rowsM; ++i)
                chess_flatten_loop
            {
                const bfloat16 *pA_block = x + (i * rowsK + z) * MMUL::size_A;
                const bfloat16 *pB_block = dy + (i * colsH + j) * MMUL::size_B;

                transpose_tile_8x8<bfloat16>(pA_block, a_tile_t);

                auto A0 = aie::load_v<MMUL::size_A>(a_tile_t);
                auto B0 = aie::load_v<MMUL::size_B>(pB_block);
                C00.mac(A0, B0);
            }

            auto w_vec = aie::load_v<MMUL::size_C>(w_block);
            auto lr_vec = aie::broadcast<bfloat16, 64>(lr);
            auto dw_vec = C00.template to_vector<bfloat16>();
            auto step_vec = aie::mul(lr_vec, dw_vec).template to_vector<bfloat16>();
            auto w_new = aie::sub(w_vec, step_vec);
            aie::store_v(w_block, w_new);
        }
    }
}

} // extern "C"

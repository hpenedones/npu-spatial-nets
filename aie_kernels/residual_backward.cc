// SPDX-License-Identifier: Apache-2.0
//
// Backward kernels for one residual layer:
//
//   y  = relu(x @ W) + x
//   gz = gy * relu_mask
//   dW = x^T @ gz
//   gx = gy + gz @ W^T
//
// Inputs use the same tiled bf16 layout as the forward kernel.  The backward
// path is intentionally split in two:
//
//   1. residual_grad_input_bf16  : computes gx from [gy | mask] and W^T
//   2. residual_weight_grad_bf16 : computes dW from [x | gy | mask]
//
// This keeps each phase within the tile SRAM budget during phase-0 validation.

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

template <typename T, unsigned rowA, unsigned colA, unsigned colB>
static inline void
matmul_plain(const T *__restrict pA,
             const T *__restrict pB,
             T *__restrict pC)
{
    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, T, T, accauto>;
    const auto zeros = aie::zeros<T, MMUL::size_C>();

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

template <typename T, unsigned rowA, unsigned colA, unsigned colB>
static inline void
matmul_transpose_a(const T *__restrict pA,
                   const T *__restrict pB,
                   T *__restrict pC)
{
    constexpr unsigned r = 8, s = 8, t = 8;
    using MMUL = aie::mmul<r, s, t, T, T, accauto>;
    const auto zeros = aie::zeros<T, MMUL::size_C>();
    alignas(32) T a_tile_t[MMUL::size_A];

    for (unsigned z = 0; z < rowA; ++z) {
        T *__restrict pC1 = pC + z * colB * MMUL::size_C;

        for (unsigned j = 0; j < colB; ++j)
            chess_prepare_for_pipelining chess_loop_range(3, )
        {
            MMUL C00(zeros);

            for (unsigned i = 0; i < colA; ++i)
                chess_flatten_loop
            {
                const T *__restrict pA_block =
                    pA + (i * rowA + z) * MMUL::size_A;
                const T *__restrict pB_block =
                    pB + (i * colB + j) * MMUL::size_B;

                transpose_tile_8x8(pA_block, a_tile_t);
                auto A0 = aie::load_v<MMUL::size_A>(a_tile_t);
                auto B0 = aie::load_v<MMUL::size_B>(pB_block);
                C00.mac(A0, B0);
            }

            aie::store_v(pC1, C00.template to_vector<T>());
            pC1 += MMUL::size_C;
        }
    }
}

static inline void
elementwise_mul(const bfloat16 *__restrict a,
                const bfloat16 *__restrict b,
                bfloat16 *__restrict c)
{
    constexpr int total = DIM_M * DIM_N;
    static_assert(total % 32 == 0, "Total elements must be divisible by 32");

    for (int i = 0; i < total; i += 32) {
        auto va = aie::load_v<32>(a + i);
        auto vb = aie::load_v<32>(b + i);
        auto vc = aie::mul(va, vb).template to_vector<bfloat16>();
        aie::store_v(c + i, vc);
    }
}

static inline void
residual_add(const bfloat16 *__restrict a, bfloat16 *__restrict c)
{
    constexpr int total = DIM_M * DIM_N;
    static_assert(total % 32 == 0, "Total elements must be divisible by 32");

    for (int i = 0; i < total; i += 32) {
        auto vc = aie::load_v<32>(c + i);
        auto va = aie::load_v<32>(a + i);
        aie::store_v(c + i, aie::add(vc, va));
    }
}

extern "C" {

void residual_grad_input_bf16(bfloat16 *state,
                              bfloat16 *w_t,
                              bfloat16 *gx)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_N % 8 == 0, "Output dim must be multiple of 8");

    bfloat16 *gy = state;
    bfloat16 *mask = gy + DIM_M * DIM_N;

    alignas(32) bfloat16 gz[DIM_M * DIM_N];

    ::aie::set_rounding(aie::rounding_mode::floor);

    // Step 1: gz = gy * relu_mask
    elementwise_mul(gy, mask, gz);

    // Step 2: gx = gz @ W^T
    matmul_plain<bfloat16, (DIM_M / 8), (DIM_K / 8), (DIM_N / 8)>(
        gz, w_t, gx
    );

    // Step 3: gx += gy   (skip-connection gradient)
    residual_add(gy, gx);
}

void residual_weight_grad_bf16(bfloat16 *state, bfloat16 *dw)
{
    static_assert(DIM_M % 8 == 0, "Batch must be multiple of 8");
    static_assert(DIM_K % 8 == 0, "Input dim must be multiple of 8");
    static_assert(DIM_N % 8 == 0, "Output dim must be multiple of 8");

    bfloat16 *x = state;
    bfloat16 *gy = state + DIM_M * DIM_K;
    bfloat16 *mask = gy + DIM_M * DIM_N;

    alignas(32) bfloat16 gz[DIM_M * DIM_N];

    ::aie::set_rounding(aie::rounding_mode::floor);

    // Step 1: gz = gy * relu_mask
    elementwise_mul(gy, mask, gz);

    // Step 2: dw = x^T @ gz
    matmul_transpose_a<bfloat16, (DIM_K / 8), (DIM_M / 8), (DIM_N / 8)>(
        x, gz, dw
    );
}

} // extern "C"

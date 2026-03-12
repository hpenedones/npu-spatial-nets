// SPDX-License-Identifier: Apache-2.0
//
// Fused RMSNorm + matmul + ReLU kernel: C = ReLU(RMSNorm(A, scale) × W)
//
// Each pipeline stage normalises its input before the matrix multiply,
// preventing activation explosion through the 4-layer chain.  This is
// the key kernel that closes the quality gap between the "blocked" model
// (pure matmul+ReLU per stage, val loss 2.42) and the per-layer-norm
// model (val loss 2.03).
//
// Weight buffer layout:  [W (DIM_K × DIM_N bf16, tiled), scale (DIM_K bf16)]
// The matmul reads the first DIM_K*DIM_N elements; the norm reads the
// last DIM_K elements as the RMSNorm learned scale vector.
//
// Compile with: -DDIM_M=B -DDIM_K=H -DDIM_N=H
//               -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

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

// ── RMSNorm (in-place) ─────────────────────────────────────────────────
//
// For each of the M rows of length K:
//   rms = sqrt(mean(x^2) + eps)
//   x[k] = x[k] / rms * scale[k]
//
// Uses scalar float32 for the reduction (sum of squares) to maintain
// numerical stability.  The normalise-and-scale pass writes bf16 back
// in-place so the subsequent matmul sees normalised input.
//
// Cost: ~650 scalar float ops per row × DIM_M rows.  At 1.25 GHz this
// is roughly 25 µs for M=48, K=128 — modest compared to the ~120 µs
// NPU call overhead that dominates end-to-end latency.

static inline void
rms_norm_inplace(bfloat16 *__restrict input,
                 const bfloat16 *__restrict scale)
{
    constexpr float eps = 1e-6f;
    constexpr float inv_K = 1.0f / (float)DIM_K;

    for (int row = 0; row < DIM_M; row++)
        chess_prepare_for_pipelining
    {
        bfloat16 *__restrict row_ptr = input + row * DIM_K;

        // Accumulate sum of squares in float32
        float sum_sq = 0.0f;
        for (int k = 0; k < DIM_K; k++) {
            float v = (float)row_ptr[k];
            sum_sq += v * v;
        }

        // Inverse RMS via Babylonian sqrt + reciprocal
        // (avoids dependence on sqrtf library function)
        float mean_sq = sum_sq * inv_K + eps;
        float y = 1.0f;
        // 8 Babylonian iterations: converges across 4 orders of magnitude
        for (int i = 0; i < 8; i++)
            chess_flatten_loop
        {
            y = 0.5f * (y + mean_sq / y);
        }
        float inv_rms = 1.0f / y;

        // Normalise and apply learned scale, write back as bf16
        for (int k = 0; k < DIM_K; k++) {
            float v = (float)row_ptr[k];
            row_ptr[k] = (bfloat16)(v * inv_rms * (float)scale[k]);
        }
    }
}

// ── Matmul + ReLU (1×1 tile expansion) ─────────────────────────────────
//
// Processes one output tile block at a time.  Works for any DIM_M
// divisible by r (=8).  The outer loop over column blocks (colB)
// provides pipelining opportunities when colB ≥ 3 (H ≥ 24).
//
// Used when DIM_M < 2*r (e.g. B=8) — the 2×2 variant would read
// out-of-bounds on the second row block.

template <typename T_in, typename T_out,
          unsigned rowA, unsigned colA, unsigned colB,
          unsigned r, unsigned s, unsigned t>
static inline void
matmul_relu_1x1(const T_in *__restrict pA,
                const T_in *__restrict pB,
                T_out *__restrict pC)
{
    using MMUL = aie::mmul<r, s, t, T_in, T_in, accauto>;
    const aie::vector<T_out, MMUL::size_C> zeros =
        aie::zeros<T_out, MMUL::size_C>();

    event0();

    for (unsigned z = 0; z < rowA; z += 1) {
        T_out *__restrict pC1 = pC + (z * colB) * MMUL::size_C;

        for (unsigned j = 0; j < colB; j += 1)
            chess_prepare_for_pipelining chess_loop_range(3, )
        {
            const T_in *__restrict pA1 = pA + (z * colA) * MMUL::size_A;
            const T_in *__restrict pB1 = pB + j * MMUL::size_B;

            MMUL C00(zeros);

            for (unsigned i = 0; i < colA; ++i)
                chess_flatten_loop
            {
                aie::vector<T_in, MMUL::size_A> A0 =
                    aie::load_v<MMUL::size_A>(pA1);
                pA1 += MMUL::size_A;
                aie::vector<T_in, MMUL::size_B> B0 =
                    aie::load_v<MMUL::size_B>(pB1);
                pB1 += MMUL::size_B * colB;

                C00.mac(A0, B0);
            }

            aie::store_v(pC1,
                aie::max(C00.template to_vector<T_out>(), zeros));
            pC1 += MMUL::size_C;
        }
    }

    event1();
}

// ── Matmul + ReLU (2×2 tile expansion) ─────────────────────────────────
//
// Processes 2×2 output tile blocks per inner-loop iteration for high
// register utilisation.  Requires DIM_M >= 2*r (i.e. B >= 16).

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

// ── Entry point ─────────────────────────────────────────────────────────

extern "C" {

#ifdef AIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16
void norm_matmul_relu_bf16_bf16(bfloat16 *a, bfloat16 *w_and_scale,
                                bfloat16 *c)
{
    // Scale vector sits right after the H×H weight matrix
    const bfloat16 *scale = w_and_scale + DIM_K * DIM_N;

    // Step 1: normalise input rows in-place
    rms_norm_inplace(a, scale);

    // Step 2: fused matmul + ReLU on the normalised input
    constexpr int r = 8, s = 8, t = 8;
    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);
    ::aie::set_rounding(aie::rounding_mode::floor);

#if (DIM_M / 8) >= 2
    // 2×2 expansion — better register utilization (B >= 16)
    static_assert(DIM_N % (2 * t) == 0);
    matmul_relu_2x2<bfloat16, bfloat16,
                     (DIM_M / r), (DIM_K / s), (DIM_N / t),
                     r, s, t>(a, w_and_scale, c);
#else
    // 1×1 expansion — works for any B divisible by 8
    static_assert(DIM_N % t == 0);
    matmul_relu_1x1<bfloat16, bfloat16,
                     (DIM_M / r), (DIM_K / s), (DIM_N / t),
                     r, s, t>(a, w_and_scale, c);
#endif
}
#else
void norm_matmul_relu_bf16_bf16(bfloat16 *a, bfloat16 *w_and_scale,
                                bfloat16 *c)
{
    const bfloat16 *scale = w_and_scale + DIM_K * DIM_N;
    rms_norm_inplace(a, scale);

    constexpr int r = 4, s = 8, t = 8;
    static_assert(DIM_M % r == 0);
    static_assert(DIM_K % s == 0);
    ::aie::set_rounding(aie::rounding_mode::floor);

#if (DIM_M / 4) >= 2
    static_assert(DIM_N % (2 * t) == 0);
    matmul_relu_2x2<bfloat16, bfloat16,
                     (DIM_M / r), (DIM_K / s), (DIM_N / t),
                     r, s, t>(a, w_and_scale, c);
#else
    static_assert(DIM_N % t == 0);
    matmul_relu_1x1<bfloat16, bfloat16,
                     (DIM_M / r), (DIM_K / s), (DIM_N / t),
                     r, s, t>(a, w_and_scale, c);
#endif
}
#endif

} // extern "C"

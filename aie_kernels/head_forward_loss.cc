// SPDX-License-Identifier: Apache-2.0
//
// Head tile kernel: matmul → softmax → cross-entropy → d_logits
//
// Fuses the classification head forward pass and loss/gradient computation
// into a single kernel so no intermediate buffers leave the tile.
//
// Forward:   logits[B×N_CLS] = y_hidden[B×H] @ w_head[H×N_CLS]
//            probs = softmax(logits, dim=1)
//            loss  = -mean(log(probs[labels]))
//
// Backward:  d_logits = probs - one_hot(labels)
//            (This is the gradient of CE loss w.r.t. logits, divided by B)
//
// Memory layout (tiled 8×8 blocks, N_CLS padded to multiple of 8):
//   y_hidden:  [B × H] bf16
//   w_head:    [H × N_CLS] bf16
//   labels:    [B] int32
//   d_logits:  [B × N_CLS] bf16 (output)
//   loss_out:  [1] float (output, for host logging)
//
// Compile with:
//   -DDIM_M=<B>  -DDIM_H=<hidden_dim>  -DDIM_N_CLS=<padded_num_classes>
//   -DNUM_CLASSES=<actual_classes>
//   -DAIE_API_EMULATE_BFLOAT16_MMUL_WITH_BFP16

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_H
#define DIM_H 32
#endif
#ifndef DIM_N_CLS
#define DIM_N_CLS 16    // padded to multiple of 8
#endif
#ifndef LOSS_GRAD_SCALE
#define LOSS_GRAD_SCALE 1.0f
#endif
#ifndef NUM_CLASSES
#define NUM_CLASSES 10  // actual number of classes
#endif

// ── Tiled matmul (same as matmul_plain) ───────────────────────────────

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

// ── Un-tile a small [B×N] tiled buffer into row-major ─────────────────
// tiled layout: [B/8, N/8, 8, 8] → row-major [B, N]

static inline void
untile(const bfloat16 *tiled, float *row_major, int B, int N)
{
    int br = 8, bc = 8;
    for (int bi = 0; bi < B / br; ++bi) {
        for (int bj = 0; bj < N / bc; ++bj) {
            for (int r = 0; r < br; ++r) {
                for (int c = 0; c < bc; ++c) {
                    int tiled_idx = (bi * (N / bc) + bj) * br * bc + r * bc + c;
                    int rm_idx = (bi * br + r) * N + (bj * bc + c);
                    row_major[rm_idx] = (float)tiled[tiled_idx];
                }
            }
        }
    }
}

// ── Re-tile a small [B×N] row-major buffer back to tiled layout ──────

static inline void
retile(const float *row_major, bfloat16 *tiled, int B, int N)
{
    int br = 8, bc = 8;
    for (int bi = 0; bi < B / br; ++bi) {
        for (int bj = 0; bj < N / bc; ++bj) {
            for (int r = 0; r < br; ++r) {
                for (int c = 0; c < bc; ++c) {
                    int tiled_idx = (bi * (N / bc) + bj) * br * bc + r * bc + c;
                    int rm_idx = (bi * br + r) * N + (bj * bc + c);
                    tiled[tiled_idx] = (bfloat16)row_major[rm_idx];
                }
            }
        }
    }
}

static inline float
fast_exp_approx(float x)
{
    if (x < -10.0f) return 0.0f;
    if (x > 10.0f) x = 10.0f;

    const float inv_ln2 = 1.44269504089f;
    const float ln2 = 0.69314718056f;
    float scaled = x * inv_ln2;
    int exp2_i = (int)scaled;
    if (scaled < (float)exp2_i) {
        exp2_i -= 1;
    }
    float frac = scaled - (float)exp2_i;
    float z = frac * ln2;
    float z2 = z * z;
    float z3 = z2 * z;
    float mant = 1.0f + z + 0.5f * z2 + (1.0f / 6.0f) * z3;

    if (exp2_i < -126) return 0.0f;
    if (exp2_i > 127) exp2_i = 127;

    union {
        uint32_t bits;
        float value;
    } exp2_scale = {(uint32_t)(exp2_i + 127) << 23};
    return exp2_scale.value * mant;
}

static inline float
fast_log_approx(float x)
{
    if (x < 1e-7f) x = 1e-7f;

    union {
        float value;
        uint32_t bits;
    } repr = {x};

    int exp2_i = (int)((repr.bits >> 23) & 0xff) - 127;
    repr.bits = (repr.bits & 0x007fffff) | 0x3f800000;

    float y = repr.value - 1.0f;
    float y2 = y * y;
    float y3 = y2 * y;
    float y4 = y3 * y;
    float log_mant = y - 0.5f * y2 + (1.0f / 3.0f) * y3 - 0.25f * y4;
    return (float)exp2_i * 0.69314718056f + log_mant;
}

extern "C" {

void head_forward_loss_bf16(bfloat16 *y_hidden, bfloat16 *w_head,
                             int32_t *labels, bfloat16 *d_logits,
                             int32_t *preds_out)
{
    static_assert(DIM_M % 8 == 0);
    static_assert(DIM_H % 8 == 0);
    static_assert(DIM_N_CLS % 8 == 0);

    // Scratch space for logits in tiled layout, then row-major probs
    // logits: B × N_CLS bf16 tiled
    alignas(32) bfloat16 logits_tiled[DIM_M * DIM_N_CLS];

    // Step 1: logits = y_hidden @ w_head
    matmul_plain<bfloat16, (DIM_M / 8), (DIM_H / 8), (DIM_N_CLS / 8)>(
        y_hidden, w_head, logits_tiled);

    // Step 2: Un-tile logits for softmax (scalar math, needs row-major)
    float logits_rm[DIM_M * DIM_N_CLS];
    untile(logits_tiled, logits_rm, DIM_M, DIM_N_CLS);

    // Step 3: Softmax + cross-entropy loss + d_logits (all in row-major)
    float d_logits_rm[DIM_M * DIM_N_CLS];
    float total_loss = 0.0f;

    for (int b = 0; b < DIM_M; ++b) {
        float *row = logits_rm + b * DIM_N_CLS;
        int pred_class = 0;

        // Find max for numerical stability (only over actual classes)
        float max_val = row[0];
        for (int c = 1; c < NUM_CLASSES; ++c) {
            if (row[c] > max_val) {
                max_val = row[c];
                pred_class = c;
            }
        }
        preds_out[b] = pred_class;

        // Exp and sum
        float exp_vals[DIM_N_CLS];
        float sum_exp = 0.0f;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            exp_vals[c] = fast_exp_approx(row[c] - max_val);
            sum_exp += exp_vals[c];
        }
        // Padded classes get zero probability
        for (int c = NUM_CLASSES; c < DIM_N_CLS; ++c) {
            exp_vals[c] = 0.0f;
        }

        // Softmax probabilities
        float inv_sum = 1.0f / sum_exp;
        for (int c = 0; c < DIM_N_CLS; ++c) {
            exp_vals[c] *= inv_sum;
        }

        int label = labels[b];
        bool training_mode = label >= 0 && label < NUM_CLASSES;
        if (training_mode) {
            float prob_label = exp_vals[label];
            total_loss += -fast_log_approx(prob_label);
        }

        // Negative labels are used as a prediction-only mode for evaluation.
        float scale = LOSS_GRAD_SCALE / (float)DIM_M;
        for (int c = 0; c < DIM_N_CLS; ++c) {
            if (training_mode) {
                float one_hot = (c == label) ? 1.0f : 0.0f;
                d_logits_rm[b * DIM_N_CLS + c] = (exp_vals[c] - one_hot) * scale;
            } else {
                d_logits_rm[b * DIM_N_CLS + c] = 0.0f;
            }
        }
    }

    // Step 4: Re-tile d_logits back to bf16 tiled layout
    retile(d_logits_rm, d_logits, DIM_M, DIM_N_CLS);
}

} // extern "C"

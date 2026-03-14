// SPDX-License-Identifier: Apache-2.0
//
// Head backward kernel:
//   dy_hidden = d_logits @ W_head^T
//   W_head   -= lr * y_hidden^T @ d_logits

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_H
#define DIM_H 32
#endif
#ifndef DIM_N_CLS
#define DIM_N_CLS 16
#endif
#ifndef SGD_LR
#define SGD_LR 0.01f
#endif

static inline void untile(const bfloat16 *tiled, float *row_major, int M, int K)
{
    constexpr int br = 8;
    constexpr int bc = 8;
    for (int bi = 0; bi < M / br; ++bi) {
        for (int bj = 0; bj < K / bc; ++bj) {
            for (int r = 0; r < br; ++r) {
                for (int c = 0; c < bc; ++c) {
                    int tiled_idx = (bi * (K / bc) + bj) * br * bc + r * bc + c;
                    int rm_idx = (bi * br + r) * K + (bj * bc + c);
                    row_major[rm_idx] = (float)tiled[tiled_idx];
                }
            }
        }
    }
}

static inline void retile(const float *row_major, bfloat16 *tiled, int M, int K)
{
    constexpr int br = 8;
    constexpr int bc = 8;
    for (int bi = 0; bi < M / br; ++bi) {
        for (int bj = 0; bj < K / bc; ++bj) {
            for (int r = 0; r < br; ++r) {
                for (int c = 0; c < bc; ++c) {
                    int tiled_idx = (bi * (K / bc) + bj) * br * bc + r * bc + c;
                    int rm_idx = (bi * br + r) * K + (bj * bc + c);
                    tiled[tiled_idx] = (bfloat16)row_major[rm_idx];
                }
            }
        }
    }
}

extern "C" {

void head_backward_bf16(bfloat16 *y_hidden, bfloat16 *w_head,
                        bfloat16 *d_logits, bfloat16 *dy_hidden)
{
    alignas(32) float y_rm[DIM_M * DIM_H];
    alignas(32) float w_rm[DIM_H * DIM_N_CLS];
    alignas(32) float w_old_rm[DIM_H * DIM_N_CLS];
    alignas(32) float d_rm[DIM_M * DIM_N_CLS];
    alignas(32) float dy_rm[DIM_M * DIM_H];

    untile(y_hidden, y_rm, DIM_M, DIM_H);
    untile(w_head, w_rm, DIM_H, DIM_N_CLS);
    untile(d_logits, d_rm, DIM_M, DIM_N_CLS);

    for (int i = 0; i < DIM_H * DIM_N_CLS; ++i) {
        w_old_rm[i] = w_rm[i];
    }

    for (int b = 0; b < DIM_M; ++b) {
        for (int h = 0; h < DIM_H; ++h) {
            float acc = 0.0f;
            const float *w_row = w_old_rm + h * DIM_N_CLS;
            const float *d_row = d_rm + b * DIM_N_CLS;
            for (int c = 0; c < DIM_N_CLS; ++c) {
                acc += d_row[c] * w_row[c];
            }
            dy_rm[b * DIM_H + h] = acc;
        }
    }

    for (int h = 0; h < DIM_H; ++h) {
        float *w_row = w_rm + h * DIM_N_CLS;
        for (int c = 0; c < DIM_N_CLS; ++c) {
            float grad = 0.0f;
            for (int b = 0; b < DIM_M; ++b) {
                grad += y_rm[b * DIM_H + h] * d_rm[b * DIM_N_CLS + c];
            }
            w_row[c] -= SGD_LR * grad;
        }
    }

    retile(dy_rm, dy_hidden, DIM_M, DIM_H);
    retile(w_rm, w_head, DIM_H, DIM_N_CLS);
}

} // extern "C"

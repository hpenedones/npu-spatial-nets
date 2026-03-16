// SPDX-License-Identifier: Apache-2.0
//
// Tiny row-major classifier kernels for the one-column simple convnet.

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>

#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif
#ifndef DIM_H
#define DIM_H 16
#endif
#ifndef NUM_CLASSES
#define NUM_CLASSES 10
#endif
#ifndef SGD_LR
#define SGD_LR 0.0005f
#endif

extern "C" {

void simple_head_forward_loss_bf16(
    bfloat16 *pooled,
    bfloat16 *w_head,
    int32_t *labels,
    bfloat16 *d_logits,
    int32_t *preds_out)
{
    for (int b = 0; b < BATCH_SIZE; ++b) {
        int label = labels[b];
        int pred_class = 0;
        int best_wrong_class = 0;
        float max_val = 0.0f;
        float correct_logit = 0.0f;
        float best_wrong_logit = 0.0f;
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float acc = 0.0f;
            for (int h = 0; h < DIM_H; ++h) {
                acc += (float)pooled[b * DIM_H + h] * (float)w_head[h * NUM_CLASSES + c];
            }
            d_logits[b * NUM_CLASSES + c] = (bfloat16)0.0f;
            if (c == 0 || acc > max_val) {
                max_val = acc;
                pred_class = c;
            }
            if (c == label) {
                correct_logit = acc;
            } else if (c == 0 || best_wrong_class == label || acc > best_wrong_logit) {
                best_wrong_logit = acc;
                best_wrong_class = c;
            }
        }
        preds_out[b] = pred_class;

        if (best_wrong_logit - correct_logit + 1.0f > 0.0f) {
            d_logits[b * NUM_CLASSES + best_wrong_class] = (bfloat16)(1.0f / (float)BATCH_SIZE);
            d_logits[b * NUM_CLASSES + label] = (bfloat16)(-1.0f / (float)BATCH_SIZE);
        }
    }
}

void simple_head_infer_bf16(
    bfloat16 *pooled,
    bfloat16 *w_head,
    bfloat16 *logits_out)
{
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float acc = 0.0f;
            for (int h = 0; h < DIM_H; ++h) {
                acc += (float)pooled[b * DIM_H + h] * (float)w_head[h * NUM_CLASSES + c];
            }
            logits_out[b * NUM_CLASSES + c] = (bfloat16)acc;
        }
    }
}

void simple_head_backward_bf16(
    bfloat16 *pooled,
    bfloat16 *w_head,
    bfloat16 *d_logits,
    bfloat16 *d_pooled)
{
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int h = 0; h < DIM_H; ++h) {
            float acc = 0.0f;
            for (int c = 0; c < NUM_CLASSES; ++c) {
                acc += (float)d_logits[b * NUM_CLASSES + c] * (float)w_head[h * NUM_CLASSES + c];
            }
            d_pooled[b * DIM_H + h] = (bfloat16)acc;
        }
    }

    for (int h = 0; h < DIM_H; ++h) {
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float grad = 0.0f;
            for (int b = 0; b < BATCH_SIZE; ++b) {
                grad += (float)pooled[b * DIM_H + h] * (float)d_logits[b * NUM_CLASSES + c];
            }
            w_head[h * NUM_CLASSES + c] = (bfloat16)((float)w_head[h * NUM_CLASSES + c] - SGD_LR * grad);
        }
    }

}

} // extern "C"

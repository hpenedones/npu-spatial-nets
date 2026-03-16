// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif
#ifndef IN_DIM
#define IN_DIM 256
#endif
#ifndef NUM_CLASSES
#define NUM_CLASSES 10
#endif

extern "C" {

void flatten_head_infer_bf16(bfloat16 *x, bfloat16 *w_head, bfloat16 *logits_out)
{
    for (int b = 0; b < BATCH_SIZE; ++b) {
        for (int c = 0; c < NUM_CLASSES; ++c) {
            float acc = 0.0f;
            for (int i = 0; i < IN_DIM; ++i) {
                acc += (float)x[b * IN_DIM + i] * (float)w_head[i * NUM_CLASSES + c];
            }
            logits_out[b * NUM_CLASSES + c] = (bfloat16)acc;
        }
    }
}

} // extern "C"

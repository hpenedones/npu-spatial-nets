// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif
#ifndef IN_H
#define IN_H 28
#endif
#ifndef IN_W
#define IN_W 28
#endif
#ifndef IN_C
#define IN_C 1
#endif
#ifndef OUT_H
#define OUT_H 14
#endif
#ifndef OUT_W
#define OUT_W 14
#endif
#ifndef OUT_C
#define OUT_C 4
#endif

extern "C" {

void conv1_infer_relu_bf16(bfloat16 *x, bfloat16 *w, bfloat16 *y)
{
    for (int b = 0; b < BATCH_SIZE; ++b) {
        int x_batch = b * IN_H * IN_W * IN_C;
        int y_batch = b * OUT_H * OUT_W * OUT_C;
        for (int oy = 0; oy < OUT_H; ++oy) {
            int iy0 = oy * 2 - 1;
            for (int ox = 0; ox < OUT_W; ++ox) {
                int ix0 = ox * 2 - 1;
                int out_base = y_batch + (oy * OUT_W + ox) * OUT_C;
                for (int oc = 0; oc < OUT_C; ++oc) {
                    const bfloat16 *w_oc = w + oc * (IN_C * 9);
                    float acc = 0.0f;
                    for (int ky = 0; ky < 3; ++ky) {
                        int iy = iy0 + ky;
                        if (iy < 0 || iy >= IN_H) {
                            continue;
                        }
                        int row_base = x_batch + iy * IN_W * IN_C;
                        for (int kx = 0; kx < 3; ++kx) {
                            int ix = ix0 + kx;
                            if (ix < 0 || ix >= IN_W) {
                                continue;
                            }
                            int pix_base = row_base + ix * IN_C;
                            const bfloat16 *w_kernel = w_oc + (ky * 3 + kx) * IN_C;
                            for (int ic = 0; ic < IN_C; ++ic) {
                                acc += (float)x[pix_base + ic] * (float)w_kernel[ic];
                            }
                        }
                    }
                    y[out_base + oc] = (bfloat16)(acc > 0.0f ? acc : 0.0f);
                }
            }
        }
    }
}

} // extern "C"

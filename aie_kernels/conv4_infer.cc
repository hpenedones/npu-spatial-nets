// SPDX-License-Identifier: Apache-2.0

#define NOCPP

#include <aie_api/aie.hpp>

namespace {

#ifndef BATCH_SIZE
#define BATCH_SIZE 8
#endif

constexpr int BATCH = BATCH_SIZE;
constexpr int IN_H = 4;
constexpr int IN_W = 4;
constexpr int IN_C = 16;
constexpr int OUT_H = 4;
constexpr int OUT_W = 4;
constexpr int OUT_C = 16;

} // namespace

extern "C" {

void conv4_infer_relu_bf16(bfloat16 *x, bfloat16 *w, bfloat16 *y)
{
    for (int b = 0; b < BATCH; ++b) {
        int x_batch = b * IN_H * IN_W * IN_C;
        int y_batch = b * OUT_H * OUT_W * OUT_C;
        for (int oy = 0; oy < OUT_H; ++oy) {
            for (int ox = 0; ox < OUT_W; ++ox) {
                int out_base = y_batch + (oy * OUT_W + ox) * OUT_C;
                int skip_base = x_batch + (oy * IN_W + ox) * IN_C;
                for (int oc = 0; oc < OUT_C; ++oc) {
                    const bfloat16 *w_oc = w + oc * (IN_C * 9);
                    float acc = 0.0f;
                    for (int ky = 0; ky < 3; ++ky) {
                        int iy = oy + ky - 1;
                        if (iy < 0 || iy >= IN_H) {
                            continue;
                        }
                        int row_base = x_batch + iy * IN_W * IN_C;
                        for (int kx = 0; kx < 3; ++kx) {
                            int ix = ox + kx - 1;
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
                    float relu = acc > 0.0f ? acc : 0.0f;
                    y[out_base + oc] = (bfloat16)(relu + (float)x[skip_base + oc]);
                }
            }
        }
    }
}

} // extern "C"

// SPDX-License-Identifier: Apache-2.0
//
// Custom kernels for spatial MLP pipelines on AIE2P.
//
// 1. relu_inplace_bf16: in-place max(x, 0)
// 2. copy_bf16: vectorized buffer copy (for ping-pong output staging)

#define NOCPP

#include <aie_api/aie.hpp>
#include <stdint.h>

extern "C" {

void relu_inplace_bf16(bfloat16 *__restrict buf, int32_t size)
{
    event0();

    const int v_factor = 32;
    v32bfloat16 zeroes = broadcast_zero_to_v32bfloat16();

    for (int i = 0; i < size; i += v_factor) {
        v32bfloat16 val = *(v32bfloat16 *)(buf + i);
        *(v32bfloat16 *)(buf + i) = max(val, zeroes);
    }

    event1();
}

void copy_bf16(bfloat16 *__restrict src, bfloat16 *__restrict dst,
               int32_t count)
{
    event0();

    const int v_factor = 32;
    for (int i = 0; i < count; i += v_factor) {
        v32bfloat16 v = *(v32bfloat16 *)(src + i);
        *(v32bfloat16 *)(dst + i) = v;
    }

    event1();
}

} // extern "C"


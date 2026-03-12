// SPDX-License-Identifier: Apache-2.0
//
// Copy one BxH activation tile in tiled bf16 layout.

#define NOCPP

#include <aie_api/aie.hpp>

#ifndef DIM_M
#define DIM_M 8
#endif
#ifndef DIM_K
#define DIM_K 160
#endif

extern "C" {

void copy_activation_bf16(bfloat16 *in, bfloat16 *out)
{
    constexpr int total = DIM_M * DIM_K;
    static_assert(total % 32 == 0, "Total elements must be divisible by 32");

    for (int i = 0; i < total; i += 32) {
        auto v = aie::load_v<32>(in + i);
        aie::store_v(out + i, v);
    }
}

} // extern "C"

// Shared by the block-scaled projection kernels (q2.cu, nemotron.cu): the table of signed trits and the loads of the regrouped weight layout.
//
// Layout of a row (built on the host by `group_blocks` in q2.rs from the planar file layout): the blocks are taken in groups of four. In a full group the four words of lane j of the four blocks are adjacent, `[lane j][block u]` at word `32 g + 4 j + u`, so that a
// thread reads them with one 16-byte load and the eight lanes of a row read 128 contiguous bytes. The last group of `tail` (1 to 3) blocks is stored as
// `[lane j][block u]` at word `32 g + tail * j + u`. The matrices of q2.cu pad every row to whole groups (a row is `32 * groups` words, the padding is not read, and every
// row starts on a 128-byte line); the experts of nemotron.cu are not padded (a row is 8 words per block).
#pragma once
#include <cstdint>

// The trits of the byte i (a code of five base-3 digits) in plane order, one trit per byte (-1 = 0xff); the five codes 251..255 are not valid and give digit 0, that
// is -1, in every plane, like the digit table of the CPU decoder.
__device__ __forceinline__ uint32_t digits_word_q2(int i) {
    if (i > 250) return 0xffffffffu;
    const int d4 = i / 84, r4 = i - 84 * d4;
    const int d3 = r4 / 28, r3 = r4 - 28 * d3;
    const int d2 = r3 / 9, r2 = r3 - 9 * d2;
    const int d1 = r2 / 3, d0 = r2 - 3 * d1;
    return __vsub4(d0 | (d1 << 8) | (d2 << 16) | (d3 << 24), 0x01010101u);
}

// The words of lane j of group g of the row that starts at `row`, one per block of the group; the blocks the row does not have read as 0 without a memory access.
__device__ __forceinline__ void load_group(const uint32_t* __restrict__ row, int g, int full_groups, int tail, int j, uint32_t (&words)[4]) {
    if (g < full_groups) {
        const uint4 v = __ldg(reinterpret_cast<const uint4*>(row + g * 32 + 4 * j));
        words[0] = v.x; words[1] = v.y; words[2] = v.z; words[3] = v.w;
    } else {
#pragma unroll
        for (int u = 0; u < 4; ++u) words[u] = u < tail ? __ldg(row + g * 32 + tail * j + u) : 0u;
    }
}

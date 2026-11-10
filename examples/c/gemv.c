/*
 * Kernel level in C: pack a ternary matrix, multiply it with int8 activations and check the result against a plain
 * integer product.
 *
 *     gemv [rows] [cols]          (defaults: 512 x 2048)
 */
#include "quintrit.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

/* Every call returns a status code; the message of the last failure is kept per thread. */
#define CHECK(call)                                                                         \
    do {                                                                                    \
        int status_ = (call);                                                               \
        if (status_ != QUINTRIT_OK) {                                                       \
            fprintf(stderr, "%s failed: %s (%s)\n", #call, quintrit_status_string(status_), \
                    quintrit_last_error_message());                                         \
            return 1;                                                                       \
        }                                                                                   \
    } while (0)

/* A small deterministic generator keeps the example free of platform-specific randomness. */
static uint64_t state = 0x243F6A8885A308D3ULL;
static uint32_t next_u32(void) {
    state ^= state >> 12;
    state ^= state << 25;
    state ^= state >> 27;
    return (uint32_t)((state * 0x2545F4914F6CDD1DULL) >> 32);
}

int main(int argc, char** argv) {
    size_t rows = argc > 1 ? (size_t)strtoul(argv[1], NULL, 10) : 512;
    size_t cols = argc > 2 ? (size_t)strtoul(argv[2], NULL, 10) : 2048;
    int32_t kernel = 0;
    size_t bytes = 0, i, r, c;
    uint32_t threads = 1;
    int mismatches = 0;

    int8_t* trits = (int8_t*)malloc(rows * cols);
    int8_t* acts = (int8_t*)malloc(cols);
    int32_t* out = (int32_t*)malloc(rows * sizeof(int32_t));
    uint8_t* packed;
    if (!trits || !acts || !out) return 1;

    printf("Quintrit %s\n", quintrit_version());

    /* QUINTRIT_KERNEL_AUTO picks the best kernel of this CPU; resolve it to see which one runs. */
    CHECK(quintrit_kernel_resolve(QUINTRIT_KERNEL_AUTO, &kernel));
    CHECK(quintrit_default_threads(&threads));
    printf("kernel: %s, threads: %u\n", quintrit_kernel_name(kernel), (unsigned)threads);

    /* Ternary weights in {-1, 0, +1} (row-major) and int8 activations over the whole range. */
    for (i = 0; i < rows * cols; i++) trits[i] = (int8_t)((int)(next_u32() % 3) - 1);
    for (i = 0; i < cols; i++) acts[i] = (int8_t)(next_u32() & 0xFF);

    /* Pack into the tiled layout of the kernel: 5 trits per byte. */
    CHECK(quintrit_tiled_size(kernel, rows, cols, &bytes));
    packed = (uint8_t*)malloc(bytes);
    if (!packed) return 1;
    CHECK(quintrit_pack_matrix(kernel, trits, rows * cols, rows, cols, packed, bytes));
    printf("%lu x %lu trits: %lu bytes as int8, %lu bytes packed (%.2f bits per weight)\n", (unsigned long)rows,
           (unsigned long)cols, (unsigned long)(rows * cols), (unsigned long)bytes,
           8.0 * (double)bytes / (double)(rows * cols));

    /* The product is exact: out[r] = sum over c of trits[r][c] * acts[c]. */
    CHECK(quintrit_gemv_i32(kernel, packed, bytes, rows, cols, acts, cols, out, rows, threads));
    for (r = 0; r < rows; r++) {
        int64_t sum = 0;
        for (c = 0; c < cols; c++) sum += (int64_t)trits[r * cols + c] * acts[c];
        if (sum != (int64_t)out[r]) mismatches++;
    }
    printf("first rows: %d %d %d %d\n", out[0], out[1], out[2], out[3]);
    printf("%s (%d mismatching rows)\n", mismatches == 0 ? "OK: matches the integer product" : "MISMATCH",
           mismatches);

    free(trits);
    free(acts);
    free(out);
    free(packed);
    return mismatches == 0 ? 0 : 1;
}

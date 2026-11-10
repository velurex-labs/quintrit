/*
 * C test of the kernel level of the Quintrit C API: version and CPU queries, pack / unpack, GEMV against a brute-force
 * product, and the error paths. Exit code 0 means success.
 */
#include "quintrit.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

#define CHECK(cond)                                                                           \
    do {                                                                                      \
        if (!(cond)) {                                                                        \
            fprintf(stderr, "%s:%d: CHECK failed: %s (last error: %s)\n", __FILE__, __LINE__, \
                    #cond, quintrit_last_error_message());                                    \
            failures++;                                                                       \
        }                                                                                     \
    } while (0)

#define CHECK_STATUS(call, expected)                                                          \
    do {                                                                                      \
        int st_ = (call);                                                                     \
        if (st_ != (expected)) {                                                              \
            fprintf(stderr, "%s:%d: %s returned %d (%s), expected %d (last error: %s)\n",     \
                    __FILE__, __LINE__, #call, st_, quintrit_status_string(st_), (expected),  \
                    quintrit_last_error_message());                                           \
            failures++;                                                                       \
        }                                                                                     \
    } while (0)

/* Small deterministic generator, so the test needs no libc randomness. */
static uint64_t rng_state = 0x243F6A8885A308D3ULL;
static uint32_t next_u32(void) {
    rng_state ^= rng_state >> 12;
    rng_state ^= rng_state << 25;
    rng_state ^= rng_state >> 27;
    return (uint32_t)((rng_state * 0x2545F4914F6CDD1DULL) >> 32);
}

static void test_basics(void) {
    uint64_t features = 0;
    size_t count = 0;
    int32_t kernels[8];
    int32_t unknown = 0;

    CHECK(quintrit_abi_version() == QUINTRIT_ABI_VERSION);
    CHECK(strcmp(quintrit_version(), QUINTRIT_VERSION_STRING) == 0);
    CHECK(strlen(quintrit_status_string(QUINTRIT_ERR_BUSY)) > 0);

    CHECK_STATUS(quintrit_cpu_features(&features), QUINTRIT_OK);
    CHECK_STATUS(quintrit_cpu_features(NULL), QUINTRIT_ERR_NULL_POINTER);
    CHECK(quintrit_last_error_code() == QUINTRIT_ERR_NULL_POINTER);
    CHECK(strlen(quintrit_last_error_message()) > 0);

    CHECK_STATUS(quintrit_supported_kernels(NULL, 0, &count), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    CHECK(count >= 1);
    CHECK(count <= 8);
    CHECK_STATUS(quintrit_supported_kernels(kernels, 8, &count), QUINTRIT_OK);
    CHECK(kernels[count - 1] == QUINTRIT_KERNEL_SCALAR);
    printf("version %s, cpu features 0x%llx, best kernel %s\n", quintrit_version(),
           (unsigned long long)features, quintrit_kernel_name(kernels[0]));

    CHECK_STATUS(quintrit_kernel_supported(1234, &unknown), QUINTRIT_ERR_INVALID_ARGUMENT);
    CHECK(quintrit_kernel_name(1234) == NULL);
}

/* Brute-force W x with 64-bit accumulation. */
static void reference_gemv(const int8_t* trits, const int8_t* acts, size_t rows, size_t cols, int64_t* out) {
    size_t r, c;
    for (r = 0; r < rows; r++) {
        int64_t sum = 0;
        for (c = 0; c < cols; c++) sum += (int64_t)trits[r * cols + c] * acts[c];
        out[r] = sum;
    }
}

static void test_pack_and_gemv(int32_t kernel, size_t rows, size_t cols, uint32_t threads) {
    size_t i, r, bytes = 0;
    int mismatches = 0;
    int8_t* trits = (int8_t*)malloc(rows * cols);
    int8_t* acts = (int8_t*)malloc(cols);
    int8_t* back = (int8_t*)malloc(rows * cols);
    int64_t* want = (int64_t*)malloc(rows * sizeof(int64_t));
    int32_t* got = (int32_t*)malloc(rows * sizeof(int32_t));
    float* gotf = (float*)malloc(rows * sizeof(float));
    uint8_t* tiled;
    quintrit_tiled_layout_t layout;

    CHECK(trits && acts && back && want && got && gotf);
    for (i = 0; i < rows * cols; i++) trits[i] = (int8_t)((int)(next_u32() % 3) - 1);
    for (i = 0; i < cols; i++) acts[i] = (int8_t)(next_u32() & 0xFF);
    acts[0] = -128;
    if (cols > 1) acts[1] = 127;

    CHECK_STATUS(quintrit_tiled_size(kernel, rows, cols, &bytes), QUINTRIT_OK);
    tiled = (uint8_t*)malloc(bytes);
    CHECK(tiled != NULL);

    memset(&layout, 0, sizeof layout);
    layout.struct_size = sizeof layout;
    CHECK_STATUS(quintrit_tiled_layout(kernel, rows, cols, &layout), QUINTRIT_OK);
    CHECK(layout.total_bytes == bytes);
    CHECK(layout.padded_rows % layout.tile_rows == 0 && layout.padded_rows >= rows);

    CHECK_STATUS(quintrit_pack_matrix(kernel, trits, rows * cols, rows, cols, tiled, bytes), QUINTRIT_OK);
    CHECK_STATUS(quintrit_unpack_matrix(kernel, tiled, bytes, rows, cols, back, rows * cols), QUINTRIT_OK);
    CHECK(memcmp(trits, back, rows * cols) == 0);

    reference_gemv(trits, acts, rows, cols, want);
    CHECK_STATUS(quintrit_gemv_i32(kernel, tiled, bytes, rows, cols, acts, cols, got, rows, threads), QUINTRIT_OK);
    CHECK_STATUS(quintrit_gemv_f32(kernel, tiled, bytes, rows, cols, acts, cols, 0.5f, gotf, rows, threads), QUINTRIT_OK);
    for (r = 0; r < rows; r++) {
        if ((int64_t)got[r] != want[r] || gotf[r] != (float)want[r] * 0.5f) mismatches++;
    }
    if (mismatches != 0) {
        fprintf(stderr, "kernel %s %lux%lu threads %u: %d mismatching rows\n", quintrit_kernel_name(kernel),
                (unsigned long)rows, (unsigned long)cols, (unsigned)threads, mismatches);
        failures++;
    }
    free(trits); free(acts); free(back); free(want); free(got); free(gotf); free(tiled);
}

static void test_errors(void) {
    static int8_t trits[64 * 64];
    static uint8_t out[64 * 64];
    size_t bytes = 0;
    quintrit_tiled_layout_t layout;
    quintrit_gguf_t* gguf = NULL;
    quintrit_gguf_info_t info;
    quintrit_model_t* model = NULL;

    memset(trits, 0, sizeof trits);
    CHECK_STATUS(quintrit_tiled_size(QUINTRIT_KERNEL_SIMD32, 64, 64, &bytes), QUINTRIT_OK);
    CHECK(bytes <= sizeof out);

    CHECK_STATUS(quintrit_pack_matrix(QUINTRIT_KERNEL_SIMD32, NULL, 64 * 64, 64, 64, out, bytes), QUINTRIT_ERR_NULL_POINTER);
    CHECK_STATUS(quintrit_pack_matrix(QUINTRIT_KERNEL_SIMD32, trits, 64 * 64, 64, 64, NULL, bytes), QUINTRIT_ERR_NULL_POINTER);
    CHECK_STATUS(quintrit_pack_matrix(QUINTRIT_KERNEL_SIMD32, trits, 64 * 64, 0, 64, out, bytes), QUINTRIT_ERR_INVALID_SHAPE);
    CHECK_STATUS(quintrit_pack_matrix(QUINTRIT_KERNEL_SIMD32, trits, 64 * 64, 64, 64, out, bytes - 1), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    CHECK_STATUS(quintrit_pack_matrix(QUINTRIT_KERNEL_SIMD32, trits, 10, 64, 64, out, bytes), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    CHECK_STATUS(quintrit_pack_matrix(77, trits, 64 * 64, 64, 64, out, bytes), QUINTRIT_ERR_INVALID_ARGUMENT);
    trits[100] = 2;
    CHECK_STATUS(quintrit_pack_matrix(QUINTRIT_KERNEL_SIMD32, trits, 64 * 64, 64, 64, out, bytes), QUINTRIT_ERR_INVALID_VALUE);
    CHECK(strstr(quintrit_last_error_message(), "row 1, column 36") != NULL);

    /* A layout struct with a wrong struct_size is rejected. */
    memset(&layout, 0, sizeof layout);
    CHECK_STATUS(quintrit_tiled_layout(QUINTRIT_KERNEL_SIMD32, 4, 4, &layout), QUINTRIT_ERR_INVALID_ARGUMENT);

    /* Handles: NULL is "invalid handle" for use and fine for free. */
    CHECK_STATUS(quintrit_gguf_open("this-file-does-not-exist.gguf", &gguf), QUINTRIT_ERR_IO);
    CHECK(gguf == NULL);
    CHECK_STATUS(quintrit_gguf_close(NULL), QUINTRIT_OK);
    info.struct_size = sizeof info;
    CHECK_STATUS(quintrit_gguf_info(NULL, &info), QUINTRIT_ERR_INVALID_HANDLE);
    CHECK_STATUS(quintrit_model_open("this-file-does-not-exist.gguf", NULL, &model), QUINTRIT_ERR_IO);
    CHECK(model == NULL);
    CHECK_STATUS(quintrit_model_free(NULL), QUINTRIT_OK);
    CHECK_STATUS(quintrit_session_free(NULL), QUINTRIT_OK);
}

int main(void) {
    size_t count = 0, k, s;
    int32_t kernels[8];
    static const size_t shapes[][2] = {{1, 1}, {5, 7}, {32, 160}, {33, 161}, {64, 320}, {65, 250}, {96, 3072}, {130, 640}};

    test_basics();

    /* Every kernel this CPU supports, with full tiles and with remainder rows / columns. */
    CHECK_STATUS(quintrit_supported_kernels(kernels, 8, &count), QUINTRIT_OK);
    for (k = 0; k < count; k++) {
        for (s = 0; s < sizeof shapes / sizeof shapes[0]; s++) {
            test_pack_and_gemv(kernels[k], shapes[s][0], shapes[s][1], 1);
            test_pack_and_gemv(kernels[k], shapes[s][0], shapes[s][1], 3);
        }
    }
    test_pack_and_gemv(QUINTRIT_KERNEL_AUTO, 100, 1000, 0);

    test_errors();

    if (failures != 0) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("C test passed\n");
    return 0;
}

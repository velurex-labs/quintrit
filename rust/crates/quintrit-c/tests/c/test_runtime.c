/*
 * C test of the format and runtime levels. argv[1] is the path of a (tiny, synthetic) GGUF model. The generated text is
 * printed as "TEXT_HEX=<hex>" so that the harness can compare it with the output of the Rust API. Exit code 0 = success.
 */
#include "quintrit.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int failures = 0;

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

#define CHECK(cond)                                                                           \
    do {                                                                                      \
        if (!(cond)) {                                                                        \
            fprintf(stderr, "%s:%d: CHECK failed: %s\n", __FILE__, __LINE__, #cond);          \
            failures++;                                                                       \
        }                                                                                     \
    } while (0)

typedef struct {
    char text[4096];
    size_t len;
    int pieces;
} collector_t;

static int collect(const char* text, size_t len, void* user_data) {
    collector_t* c = (collector_t*)user_data;
    if (c->len + len < sizeof c->text) {
        memcpy(c->text + c->len, text, len);
        c->len += len;
    }
    c->pieces++;
    return 0;
}

static int stop_after_two(const char* text, size_t len, void* user_data) {
    int* pieces = (int*)user_data;
    (void)text;
    (void)len;
    return ++(*pieces) >= 2;
}

static void print_hex(const char* label, const char* data, size_t len) {
    size_t i;
    printf("%s=", label);
    for (i = 0; i < len; i++) printf("%02x", (unsigned)(unsigned char)data[i]);
    printf("\n");
}

static void test_gguf(const char* path) {
    quintrit_gguf_t* gguf = NULL;
    quintrit_gguf_info_t info;
    quintrit_tensor_info_t t;
    quintrit_tiled_layout_t layout;
    size_t i, ternary = 0, n, rows, cols;
    float scale = 0;
    uint8_t* tiled;
    int8_t* trits;
    int8_t* acts;
    int32_t* got;
    char name[256];

    CHECK_STATUS(quintrit_gguf_open(path, &gguf), QUINTRIT_OK);
    memset(&info, 0, sizeof info);
    info.struct_size = sizeof info;
    CHECK_STATUS(quintrit_gguf_info(gguf, &info), QUINTRIT_OK);
    CHECK(info.version == 3 && info.tensor_count > 10);

    memset(&t, 0, sizeof t);
    t.struct_size = sizeof t;
    name[0] = '\0';
    for (i = 0; i < info.tensor_count; i++) {
        CHECK_STATUS(quintrit_gguf_tensor_at(gguf, i, &t), QUINTRIT_OK);
        if (t.is_ternary) {
            if (ternary == 0) {
                size_t len = strlen(t.name);
                if (len >= sizeof name) len = sizeof name - 1;
                memcpy(name, t.name, len);
                name[len] = '\0';
            }
            ternary++;
        }
    }
    CHECK(ternary > 0 && name[0] != '\0');
    CHECK_STATUS(quintrit_gguf_tensor_find(gguf, "does.not.exist", &t), QUINTRIT_ERR_NOT_FOUND);
    CHECK_STATUS(quintrit_gguf_tensor_find(gguf, name, &t), QUINTRIT_OK);
    CHECK(t.n_dims == 2);
    cols = (size_t)t.shape[0];
    rows = (size_t)t.shape[1];

    /* Convert the tensor, unpack it again and check one GEMV against a brute-force product. */
    memset(&layout, 0, sizeof layout);
    layout.struct_size = sizeof layout;
    CHECK_STATUS(quintrit_gguf_tensor_layout(gguf, name, QUINTRIT_KERNEL_AUTO, &layout, &scale), QUINTRIT_OK);
    CHECK(layout.rows == rows && layout.cols == cols);
    tiled = (uint8_t*)malloc((size_t)layout.total_bytes);
    trits = (int8_t*)malloc(rows * cols);
    acts = (int8_t*)malloc(cols);
    got = (int32_t*)malloc(rows * sizeof(int32_t));
    CHECK(tiled && trits && acts && got);
    CHECK_STATUS(quintrit_gguf_convert_tensor(gguf, name, layout.kernel, tiled, (size_t)layout.total_bytes, &layout, &scale), QUINTRIT_OK);
    CHECK_STATUS(quintrit_unpack_matrix(layout.kernel, tiled, (size_t)layout.total_bytes, rows, cols, trits, rows * cols), QUINTRIT_OK);
    for (i = 0; i < cols; i++) acts[i] = (int8_t)((int)(i * 37 % 255) - 127);
    CHECK_STATUS(quintrit_gemv_i32(layout.kernel, tiled, (size_t)layout.total_bytes, rows, cols, acts, cols, got, rows, 2), QUINTRIT_OK);
    for (n = 0; n < rows; n++) {
        int64_t sum = 0;
        for (i = 0; i < cols; i++) sum += (int64_t)trits[n * cols + i] * acts[i];
        CHECK(sum == got[n]);
    }
    printf("converted '%s' (%lux%lu, scale %g)\n", name, (unsigned long)rows, (unsigned long)cols, scale);
    free(tiled); free(trits); free(acts); free(got);

    {
        char arch[64];
        size_t len = 0;
        CHECK_STATUS(quintrit_gguf_metadata_string(gguf, "general.architecture", arch, sizeof arch, &len), QUINTRIT_OK);
        CHECK(strcmp(arch, "llama") == 0 && len == 5);
    }
    CHECK_STATUS(quintrit_gguf_close(gguf), QUINTRIT_OK);
}

static void test_runtime(const char* path) {
    static const char prompt[] = "<|user|>\nHello there\n<|assistant|>\n";
    quintrit_model_options_t opts;
    quintrit_model_t* model = NULL;
    quintrit_model_info_t minfo;
    quintrit_session_options_t sopts;
    quintrit_session_t* session = NULL;
    quintrit_session_info_t sinfo;
    quintrit_generate_params_t params;
    quintrit_generate_result_t result;
    collector_t col;
    char out[1024], again[1024], tiny[4];
    size_t out_len = 0, again_len = 0, count = 0, text_len = 0;
    uint32_t ids[64];
    char text[64];
    int pieces = 0;

    memset(&opts, 0, sizeof opts);
    opts.struct_size = sizeof opts;
    opts.num_threads = 2;
    opts.kernel = QUINTRIT_KERNEL_AUTO;
    CHECK_STATUS(quintrit_model_open(path, &opts, &model), QUINTRIT_OK);
    if (model == NULL) return;

    memset(&minfo, 0, sizeof minfo);
    minfo.struct_size = sizeof minfo;
    CHECK_STATUS(quintrit_model_info(model, &minfo), QUINTRIT_OK);
    CHECK(minfo.num_layers == 2 && minfo.hidden_size == 128 && strcmp(minfo.architecture, "llama") == 0);
    CHECK(minfo.num_threads == 2);

    /* Tokenizer: size query, fill, round trip. */
    CHECK_STATUS(quintrit_model_tokenize(model, "Hello", 5, 0, NULL, 0, &count), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    CHECK(count == 5);
    CHECK_STATUS(quintrit_model_tokenize(model, "Hello", 5, 0, ids, 64, &count), QUINTRIT_OK);
    CHECK_STATUS(quintrit_model_detokenize(model, ids, count, text, sizeof text, &text_len), QUINTRIT_OK);
    CHECK(text_len == 5 && strcmp(text, "Hello") == 0);
    CHECK_STATUS(quintrit_model_detokenize(model, ids, count, tiny, sizeof tiny, &text_len), QUINTRIT_ERR_BUFFER_TOO_SMALL);

    memset(&sopts, 0, sizeof sopts);
    sopts.struct_size = sizeof sopts;
    CHECK_STATUS(quintrit_session_create(model, &sopts, &session), QUINTRIT_OK);
    if (session == NULL) {
        quintrit_model_free(model);
        return;
    }
    /* The thread count of the model is frozen while a session exists. */
    CHECK_STATUS(quintrit_model_set_threads(model, 1), QUINTRIT_ERR_BUSY);

    CHECK_STATUS(quintrit_generate_params_default(&params), QUINTRIT_OK);
    params.max_tokens = 8;

    /* Streaming and buffer output agree. */
    memset(&col, 0, sizeof col);
    memset(&result, 0, sizeof result);
    result.struct_size = sizeof result;
    CHECK_STATUS(quintrit_session_generate(session, prompt, strlen(prompt), &params, collect, &col, out, sizeof out, &out_len, &result), QUINTRIT_OK);
    CHECK(out_len == col.len && memcmp(out, col.text, out_len) == 0);
    CHECK(result.generated_tokens > 0 && result.prompt_tokens == strlen(prompt) + 1);
    print_hex("TEXT_HEX", out, out_len);

    /* A repeated call restarts from an empty cache and gives the same text. */
    CHECK_STATUS(quintrit_session_generate(session, prompt, strlen(prompt), &params, NULL, NULL, again, sizeof again, &again_len, NULL), QUINTRIT_OK);
    CHECK(again_len == out_len && memcmp(out, again, out_len) == 0);

    /* The callback can stop the generation. A token that ends inside a UTF-8 character yields no piece of its own, so
     * two pieces can take more than two tokens. */
    params.max_tokens = 8;
    pieces = 0;
    CHECK_STATUS(quintrit_session_generate(session, prompt, strlen(prompt), &params, stop_after_two, &pieces, NULL, 0, NULL, &result), QUINTRIT_OK);
    CHECK(result.finish_reason == QUINTRIT_FINISH_CALLBACK && pieces == 2);
    CHECK(result.generated_tokens >= 2 && result.generated_tokens < params.max_tokens);

    /* The session reports its position; reset empties it. */
    memset(&sinfo, 0, sizeof sinfo);
    sinfo.struct_size = sizeof sinfo;
    CHECK_STATUS(quintrit_session_info(session, &sinfo), QUINTRIT_OK);
    CHECK(sinfo.position > strlen(prompt));
    CHECK_STATUS(quintrit_session_reset(session), QUINTRIT_OK);
    CHECK_STATUS(quintrit_session_info(session, &sinfo), QUINTRIT_OK);
    CHECK(sinfo.position == 0);

    /* A prompt longer than the context window is an error, not a crash. */
    {
        char* huge = (char*)malloc(10000);
        memset(huge, 'x', 10000);
        CHECK_STATUS(quintrit_session_generate(session, huge, 10000, &params, NULL, NULL, NULL, 0, NULL, NULL), QUINTRIT_ERR_CONTEXT_OVERFLOW);
        free(huge);
    }

    CHECK_STATUS(quintrit_session_free(session), QUINTRIT_OK);
    CHECK_STATUS(quintrit_model_free(model), QUINTRIT_OK);
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "usage: %s <model.gguf>\n", argv[0]);
        return 2;
    }
    test_gguf(argv[1]);
    test_runtime(argv[1]);
    if (failures != 0) {
        fprintf(stderr, "%d check(s) failed\n", failures);
        return 1;
    }
    printf("C runtime test passed\n");
    return 0;
}

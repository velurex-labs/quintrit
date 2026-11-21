/*
 * Runtime level in C on an NVIDIA GPU: list the GPUs, load a model onto the first supported one and stream generated text.
 *
 *     generate_gpu model.gguf ["prompt"]
 *
 * The GPU functions exist in every build of the library: a build without GPU support (or a machine without an NVIDIA driver)
 * reports it, and this program says so instead of failing.
 */
#include "quintrit.h"

#include <stdio.h>
#include <string.h>

#define CHECK(call)                                                                         \
    do {                                                                                    \
        int status_ = (call);                                                               \
        if (status_ != QUINTRIT_OK) {                                                       \
            fprintf(stderr, "%s failed: %s (%s)\n", #call, quintrit_status_string(status_), \
                    quintrit_last_error_message());                                         \
            return 1;                                                                       \
        }                                                                                   \
    } while (0)

static int on_text(const char* text, size_t len, void* user_data) {
    (void)user_data;
    fwrite(text, 1, len, stdout);
    fflush(stdout);
    return 0;
}

int main(int argc, char** argv) {
    const char* prompt = argc > 2 ? argv[2] : "<|user|>\nWhat is ternary computing?\n<|assistant|>\n";
    uint32_t count = 0, i, chosen = 0;
    int found = 0;
    quintrit_model_t* model = NULL;
    quintrit_session_t* session = NULL;
    quintrit_gpu_options_t options;
    quintrit_generate_params_t params;
    quintrit_generate_result_t result;

    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [\"prompt\"]\n", argv[0]);
        return 2;
    }

    if (quintrit_cuda_device_count(&count) == QUINTRIT_ERR_UNSUPPORTED) {
        printf("this build of the library has no GPU support (build quintrit-c with the cargo feature `cuda`)\n");
        return 0;
    }
    for (i = 0; i < count; ++i) {
        quintrit_cuda_device_info_t info;
        info.struct_size = sizeof info;
        CHECK(quintrit_cuda_device_info(i, &info));
        printf("GPU %u: %s, compute capability %d.%d, %u SMs, %llu MiB%s\n", (unsigned)info.index, info.name,
               (int)info.compute_major, (int)info.compute_minor, (unsigned)info.sm_count,
               (unsigned long long)(info.total_memory >> 20), info.supported ? "" : " (not supported: needs 8.0 or newer)");
        if (info.supported && !found) {
            found = 1;
            chosen = i;
        }
    }
    if (!found) {
        printf("no supported NVIDIA GPU found\n");
        return 0;
    }

    /* The weights are transcoded into the GPU layout while the file is read; sessions keep their KV cache on the GPU. */
    memset(&options, 0, sizeof options);
    options.struct_size = sizeof options;
    options.device_index = chosen;
    CHECK(quintrit_model_open_gpu(argv[1], &options, &model));
    CHECK(quintrit_session_create(model, NULL, &session));
    CHECK(quintrit_generate_params_default(&params));
    params.max_tokens = 64;
    result.struct_size = sizeof result;
    CHECK(quintrit_session_generate(session, prompt, strlen(prompt), &params, on_text, NULL, NULL, 0, NULL, &result));
    printf("\n[%u prompt tokens, %u generated, finish reason %d]\n", (unsigned)result.prompt_tokens,
           (unsigned)result.generated_tokens, (int)result.finish_reason);

    quintrit_session_free(session);
    quintrit_model_free(model);
    return 0;
}

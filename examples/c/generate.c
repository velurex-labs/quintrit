/*
 * Runtime level in C: load a ternary GGUF model and stream generated text to the terminal.
 *
 *     generate model.gguf ["prompt"]
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

/* Receives pieces of UTF-8 text; return 0 to continue or non-zero to stop the generation. */
static int on_text(const char* text, size_t len, void* user_data) {
    (void)user_data;
    fwrite(text, 1, len, stdout);
    fflush(stdout);
    return 0;
}

int main(int argc, char** argv) {
    const char* prompt = argc > 2 ? argv[2] : "<|user|>\nWhat is ternary computing?\n<|assistant|>\n";
    quintrit_model_t* model = NULL;
    quintrit_session_t* session = NULL;
    quintrit_model_info_t info;
    quintrit_generate_params_t params;
    quintrit_generate_result_t result;

    if (argc < 2) {
        fprintf(stderr, "usage: %s model.gguf [\"prompt\"]\n", argv[0]);
        return 2;
    }

    /* Loading transcodes the weights into RAM; it is silent unless QUINTRIT_MODEL_OPEN_VERBOSE is set. */
    CHECK(quintrit_model_open(argv[1], NULL, &model));
    info.struct_size = sizeof info;
    CHECK(quintrit_model_info(model, &info));
    printf("%s model, %u layers, kernel %s, %u threads\n", info.architecture, (unsigned)info.num_layers,
           quintrit_kernel_name(info.kernel), (unsigned)info.num_threads);

    /* A session holds the conversation (KV cache). Greedy decoding is the default and deterministic. */
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

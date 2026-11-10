/*
 * quintrit.h - C API of the Quintrit ternary inference library (libquintrit_c / quintrit_c.dll).
 *
 * Quintrit stores ternary weights {-1, 0, +1} five to a byte (velurex encoding, 1.60 bits per weight) and multiplies
 * them with int8 activations using SIMD lookup-table kernels. This header exposes three layers:
 *
 *   1. kernel level   pack / unpack a ternary matrix into the tiled layout and run GEMV on int8 activations;
 *   2. format level   open a GGUF file, list its tensors and convert ternary tensors to the tiled layout;
 *   3. runtime level  load a model, tokenize, and generate text.
 *
 * Conventions
 *   - Every function that can fail returns an int status: QUINTRIT_OK (0) or one of QUINTRIT_ERR_*. Results are
 *     returned through out-parameters. On failure quintrit_last_error_message() describes what went wrong (per thread).
 *   - Functions that cannot fail return their value directly: quintrit_version, quintrit_abi_version,
 *     quintrit_status_string, quintrit_kernel_name, quintrit_last_error_code, quintrit_last_error_message.
 *   - Every pointer argument is validated; a NULL pointer, an undersized buffer, a bad shape or an unknown enumeration
 *     value yields an error code, never a crash. Every buffer comes with an explicit length.
 *   - Strings are UTF-8. Paths are UTF-8 on every platform (also on Windows).
 *   - Sizes are size_t, element counts are never narrowed to int.
 *   - Versioned structs start with a uint32_t struct_size that the caller sets to sizeof(the struct).
 *   - Size queries: a function that fills a variable-size buffer reports the required size through an out-parameter
 *     even when it fails with QUINTRIT_ERR_BUFFER_TOO_SMALL, so call it once with a NULL buffer to size the allocation.
 *
 * Thread safety
 *   - All functions are thread-safe on distinct handles. A quintrit_gguf_t and a quintrit_model_t may be shared by
 *     threads for reading; a quintrit_session_t is used by one thread at a time (a concurrent call returns
 *     QUINTRIT_ERR_BUSY). Freeing a handle while another thread uses it is a bug of the caller.
 *   - The error message is thread-local.
 *
 * Panics and the release profile: see the README. With the default release profile (panic = "abort") an internal
 * defect terminates the process; build with the `release-unwind` profile to receive QUINTRIT_ERR_INTERNAL instead.
 *
 * The header is valid C99 and C++11. A header-only C++ wrapper is available in quintrit.hpp.
 */
#ifndef QUINTRIT_H
#define QUINTRIT_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

/* ----------------------------------------------------------------------------------------------------------------- */
/* Version                                                                                                           */
/* ----------------------------------------------------------------------------------------------------------------- */

#define QUINTRIT_VERSION_MAJOR 0
#define QUINTRIT_VERSION_MINOR 1
#define QUINTRIT_VERSION_PATCH 0
#define QUINTRIT_VERSION_STRING "0.1.0"

/* ABI version of this header. Compare it with quintrit_abi_version() at start-up. */
#define QUINTRIT_ABI_VERSION 1

/*
 * QUINTRIT_API marks exported functions. Define QUINTRIT_DLL_IMPORT when linking the Windows DLL through its import
 * library to let the compiler emit direct imports; it is optional.
 */
#ifndef QUINTRIT_API
#  if defined(_WIN32) && defined(QUINTRIT_DLL_IMPORT)
#    define QUINTRIT_API __declspec(dllimport)
#  else
#    define QUINTRIT_API
#  endif
#endif

/* ----------------------------------------------------------------------------------------------------------------- */
/* Status codes                                                                                                      */
/* ----------------------------------------------------------------------------------------------------------------- */

#define QUINTRIT_OK                      0  /* success */
#define QUINTRIT_ERR_NULL_POINTER        1  /* a required pointer argument is NULL */
#define QUINTRIT_ERR_INVALID_ARGUMENT    2  /* bad enumeration, flag, struct_size, alignment, overlapping buffers, ... */
#define QUINTRIT_ERR_INVALID_SHAPE       3  /* a dimension is zero or too large */
#define QUINTRIT_ERR_BUFFER_TOO_SMALL    4  /* a caller buffer is too small; the needed size is reported if possible */
#define QUINTRIT_ERR_INVALID_VALUE       5  /* data holds an invalid value (trit outside {-1,0,1}, packed byte > 250) */
#define QUINTRIT_ERR_UNSUPPORTED         6  /* not available on this CPU or in this build */
#define QUINTRIT_ERR_IO                  7  /* a file could not be opened or read */
#define QUINTRIT_ERR_FORMAT              8  /* not a valid GGUF file, or a tensor cannot be decoded */
#define QUINTRIT_ERR_NOT_FOUND           9  /* a tensor or metadata key does not exist */
#define QUINTRIT_ERR_MODEL              10  /* a model file is well formed but cannot be run */
#define QUINTRIT_ERR_CONTEXT_OVERFLOW   11  /* the prompt does not fit into the context window */
#define QUINTRIT_ERR_BUSY               12  /* the handle is used by another thread, or the operation needs it idle */
#define QUINTRIT_ERR_INTERNAL           13  /* a caught internal panic */
#define QUINTRIT_ERR_INVALID_HANDLE     14  /* a handle is NULL, of the wrong type, or freed */

/* Returns a static, NUL-terminated description of a status code. Never NULL. */
QUINTRIT_API const char* quintrit_status_string(int status);

/* Status code of the last failed call on the calling thread; 0 if the last call succeeded. */
QUINTRIT_API int quintrit_last_error_code(void);

/*
 * Message of the last failed call on the calling thread; an empty string if the last call succeeded. Never NULL. The
 * pointer is valid until the next quintrit_* call on the same thread.
 */
QUINTRIT_API const char* quintrit_last_error_message(void);

/* ----------------------------------------------------------------------------------------------------------------- */
/* Library, CPU and kernel queries                                                                                    */
/* ----------------------------------------------------------------------------------------------------------------- */

/* Library version as a static string, for example "0.1.0". */
QUINTRIT_API const char* quintrit_version(void);

/* ABI version of the loaded library; it must equal QUINTRIT_ABI_VERSION. */
QUINTRIT_API uint32_t quintrit_abi_version(void);

/* CPU feature bits reported by quintrit_cpu_features (runtime detection). */
#define QUINTRIT_CPU_AVX2          ((uint64_t)1 << 0)
#define QUINTRIT_CPU_AVX512F       ((uint64_t)1 << 1)
#define QUINTRIT_CPU_AVX512BW      ((uint64_t)1 << 2)
#define QUINTRIT_CPU_AVX512VBMI    ((uint64_t)1 << 3)
#define QUINTRIT_CPU_AVX512VNNI    ((uint64_t)1 << 4)
#define QUINTRIT_CPU_NEON          ((uint64_t)1 << 8)
#define QUINTRIT_CPU_NEON_DOTPROD  ((uint64_t)1 << 9)
#define QUINTRIT_CPU_NEON_I8MM     ((uint64_t)1 << 10)

QUINTRIT_API int quintrit_cpu_features(uint64_t* out_mask);

/*
 * Kernels. A kernel fixes the instruction set and with it the tile height of the weight layout:
 *   QUINTRIT_KERNEL_AUTO    best kernel usable on this CPU (resolved at run time)
 *   QUINTRIT_KERNEL_SCALAR  portable reference, 32-row tiles, always available, slow (runs the NEON code on AArch64)
 *   QUINTRIT_KERNEL_SIMD32  AVX2 (x86-64) or NEON (AArch64), 32-row tiles
 *   QUINTRIT_KERNEL_AVX512  AVX-512 BW + VBMI (x86-64), 64-row tiles
 * SCALAR and SIMD32 read the same layout; AVX512 needs its own (pack with the kernel you will run, or with
 * QUINTRIT_KERNEL_AUTO on the machine that runs it; persist the resolved kernel id, not AUTO, if the buffer is stored).
 * All kernels return identical results.
 */
#define QUINTRIT_KERNEL_AUTO    0
#define QUINTRIT_KERNEL_SCALAR  1
#define QUINTRIT_KERNEL_SIMD32  2
#define QUINTRIT_KERNEL_AVX512  3

/* Largest accepted thread count (a thread count of 0 means "automatic"). */
#define QUINTRIT_MAX_THREADS 256

/* Sets *out_supported to 1 if the kernel can run on this CPU, else 0. AUTO and SCALAR are always supported. */
QUINTRIT_API int quintrit_kernel_supported(int32_t kernel, int32_t* out_supported);

/* Resolves AUTO to the concrete kernel that would run; other ids are returned unchanged. */
QUINTRIT_API int quintrit_kernel_resolve(int32_t kernel, int32_t* out_kernel);

/* Name of a kernel ("auto", "scalar", "simd32", "avx512"); NULL for an unknown id. */
QUINTRIT_API const char* quintrit_kernel_name(int32_t kernel);

/* Rows per weight tile of the layout of a kernel (32 or 64; AUTO is resolved first). */
QUINTRIT_API int quintrit_kernel_tile_rows(int32_t kernel, size_t* out_tile_rows);

/*
 * Lists the kernels usable on this CPU, best first (SCALAR is always the last). *out_count always receives the count;
 * QUINTRIT_ERR_BUFFER_TOO_SMALL is returned if capacity is smaller (call with out = NULL, capacity = 0 to size it).
 */
QUINTRIT_API int quintrit_supported_kernels(int32_t* out, size_t capacity, size_t* out_count);

/* Number of worker threads a thread count of 0 stands for (logical CPUs, at most QUINTRIT_MAX_THREADS). */
QUINTRIT_API int quintrit_default_threads(uint32_t* out_threads);

/* ----------------------------------------------------------------------------------------------------------------- */
/* Kernel level: tiled layout, pack / unpack, GEMV                                                                    */
/* ----------------------------------------------------------------------------------------------------------------- */

/*
 * Layout of a rows x cols ternary matrix for one kernel. Weights are stored five per byte; the matrix is cut into tiles
 * of tile_rows rows, and each tile stores, for every byte column, the tile_rows bytes of its rows contiguously. Rows are
 * padded to a multiple of tile_rows with zero weights (padded_rows); columns need no padding.
 */
typedef struct quintrit_tiled_layout {
    uint32_t struct_size;     /* set to sizeof(quintrit_tiled_layout_t) before the call */
    int32_t  kernel;          /* concrete kernel id (AUTO resolved) */
    uint64_t tile_rows;       /* 32 or 64 */
    uint64_t rows;
    uint64_t cols;
    uint64_t padded_rows;     /* rows rounded up to a multiple of tile_rows */
    uint64_t bytes_per_row;   /* bytes of one row inside a tile */
    uint64_t total_bytes;     /* padded_rows * bytes_per_row: the size of the tiled buffer */
} quintrit_tiled_layout_t;

/* Fills *out_layout for a rows x cols matrix. rows and cols must be in [1, 2^31]. */
QUINTRIT_API int quintrit_tiled_layout(int32_t kernel, size_t rows, size_t cols, quintrit_tiled_layout_t* out_layout);

/* Size in bytes of the tiled buffer of a rows x cols matrix. */
QUINTRIT_API int quintrit_tiled_size(int32_t kernel, size_t rows, size_t cols, size_t* out_bytes);

/*
 * Packs a row-major rows x cols matrix of trits (int8_t values -1, 0, +1) into the tiled layout of `kernel`.
 * trits_len is the number of readable int8_t (at least rows * cols); out_capacity the number of writable bytes (at
 * least quintrit_tiled_size). Any rows and cols are accepted (remainder rows are padded). A trit outside {-1, 0, 1}
 * yields QUINTRIT_ERR_INVALID_VALUE. The buffers must not overlap.
 */
QUINTRIT_API int quintrit_pack_matrix(int32_t kernel, const int8_t* trits, size_t trits_len, size_t rows, size_t cols,
                                      uint8_t* out, size_t out_capacity);

/* Inverse of quintrit_pack_matrix: writes rows * cols trits (padding rows are dropped). */
QUINTRIT_API int quintrit_unpack_matrix(int32_t kernel, const uint8_t* tiled, size_t tiled_len, size_t rows, size_t cols,
                                        int8_t* out, size_t out_capacity);

/*
 * Integer GEMV: out[i] = sum_j W[i][j] * activations[j] for i in [0, rows), with W a tiled matrix (weights/weights_len,
 * packed for the same `kernel`) and int8 activations (the full range -128..127 is exact). cols must not exceed 131072.
 * num_threads: 0 = automatic, 1 = single thread (the scalar kernel ignores it); at most QUINTRIT_MAX_THREADS. The kernel
 * must be supported on this CPU (QUINTRIT_ERR_UNSUPPORTED otherwise). out_capacity >= rows.
 */
QUINTRIT_API int quintrit_gemv_i32(int32_t kernel, const uint8_t* weights, size_t weights_len, size_t rows, size_t cols,
                                   const int8_t* activations, size_t activations_len, int32_t* out, size_t out_capacity,
                                   uint32_t num_threads);

/*
 * Float GEMV: out[i] = scale * (sum_j W[i][j] * activations[j]). The sum is exact; pass scale = weight_scale /
 * activation_scale (one division per matrix, not per element) to dequantize, or 1.0f for the raw accumulators.
 */
QUINTRIT_API int quintrit_gemv_f32(int32_t kernel, const uint8_t* weights, size_t weights_len, size_t rows, size_t cols,
                                   const int8_t* activations, size_t activations_len, float scale, float* out,
                                   size_t out_capacity, uint32_t num_threads);

/* ----------------------------------------------------------------------------------------------------------------- */
/* Format level: GGUF files                                                                                          */
/* ----------------------------------------------------------------------------------------------------------------- */

/* An open, memory-mapped GGUF file. Tensor names and raw tensor data stay valid until quintrit_gguf_close. */
typedef struct quintrit_gguf quintrit_gguf_t;

#define QUINTRIT_MAX_DIMS 8

/* GGML type ids this library knows by name (others are listed as "UNKNOWN"). */
#define QUINTRIT_GGML_F32    0
#define QUINTRIT_GGML_F16    1
#define QUINTRIT_GGML_Q8_0   8
#define QUINTRIT_GGML_BF16   30
#define QUINTRIT_GGML_TQ1_0  34
#define QUINTRIT_GGML_TQ2_0  35
#define QUINTRIT_GGML_I2_S   36

typedef struct quintrit_gguf_info {
    uint32_t struct_size;     /* set to sizeof(quintrit_gguf_info_t) before the call */
    uint32_t version;         /* GGUF version (2 or 3) */
    uint64_t tensor_count;
    uint64_t metadata_count;
    uint64_t data_offset;     /* absolute file offset of the tensor data section */
} quintrit_gguf_info_t;

typedef struct quintrit_tensor_info {
    uint32_t    struct_size;  /* set to sizeof(quintrit_tensor_info_t) before the call */
    int32_t     ggml_type;    /* QUINTRIT_GGML_* */
    uint32_t    n_dims;
    int32_t     is_ternary;   /* 1 for I2_S, TQ1_0 and TQ2_0 tensors, which quintrit_gguf_convert_tensor can convert */
    uint64_t    shape[QUINTRIT_MAX_DIMS]; /* as stored: shape[0] is the innermost dimension (a weight matrix is [cols, rows]) */
    uint64_t    num_elements;
    uint64_t    offset;       /* offset inside the data section */
    uint64_t    data_bytes;   /* raw payload size in bytes; 0 for types that cannot be read */
    const char* name;         /* NUL-terminated, valid until quintrit_gguf_close */
    const char* type_name;    /* "F32", "F16", "Q8_0", "BF16", "TQ1_0", "TQ2_0", "I2_S" or "UNKNOWN"; static */
} quintrit_tensor_info_t;

/*
 * Opens a GGUF v2/v3 file. The file is scanned with bounds checks first (truncated or inconsistent files give
 * QUINTRIT_ERR_FORMAT) and then memory-mapped: do not modify or truncate the file while it is open. On failure
 * *out_gguf is set to NULL.
 */
QUINTRIT_API int quintrit_gguf_open(const char* path, quintrit_gguf_t** out_gguf);

/* Closes the file. NULL is allowed. Invalidates tensor names and raw data pointers obtained from the handle. */
QUINTRIT_API int quintrit_gguf_close(quintrit_gguf_t* gguf);

QUINTRIT_API int quintrit_gguf_info(const quintrit_gguf_t* gguf, quintrit_gguf_info_t* out_info);

/* Describes tensor `index` in [0, tensor_count), ordered by position in the file. */
QUINTRIT_API int quintrit_gguf_tensor_at(const quintrit_gguf_t* gguf, size_t index, quintrit_tensor_info_t* out_info);

/* Describes the tensor called `name`; QUINTRIT_ERR_NOT_FOUND if there is none. */
QUINTRIT_API int quintrit_gguf_tensor_find(const quintrit_gguf_t* gguf, const char* name, quintrit_tensor_info_t* out_info);

/*
 * Reads a string metadata value. *out_len receives its length without the NUL. If capacity <= *out_len nothing is
 * written and QUINTRIT_ERR_BUFFER_TOO_SMALL is returned (use buffer = NULL, capacity = 0 to size it).
 */
QUINTRIT_API int quintrit_gguf_metadata_string(const quintrit_gguf_t* gguf, const char* key, char* buffer, size_t capacity,
                                               size_t* out_len);

/* Reads an unsigned (or non-negative signed) integer metadata value, for example "llama.context_length". */
QUINTRIT_API int quintrit_gguf_metadata_u64(const quintrit_gguf_t* gguf, const char* key, uint64_t* out_value);

/* Pointer to the raw tensor payload inside the mapped file (valid until quintrit_gguf_close). */
QUINTRIT_API int quintrit_gguf_tensor_data(const quintrit_gguf_t* gguf, const char* name, const uint8_t** out_data,
                                           size_t* out_len);

/*
 * Tiled layout and weight scale of a ternary tensor for a kernel, without converting it. The matrix shape is taken from
 * the tensor ([cols, rows]). out_layout and out_scale may be NULL.
 */
QUINTRIT_API int quintrit_gguf_tensor_layout(const quintrit_gguf_t* gguf, const char* name, int32_t kernel,
                                             quintrit_tiled_layout_t* out_layout, float* out_scale);

/*
 * Converts a ternary tensor (I2_S, TQ1_0 or TQ2_0) into the tiled layout of `kernel` in the caller's buffer, the same
 * layout quintrit_pack_matrix produces and quintrit_gemv_* reads. out_capacity must be at least layout.total_bytes.
 * out_layout and out_scale (the per-tensor weight scale) may be NULL.
 */
QUINTRIT_API int quintrit_gguf_convert_tensor(const quintrit_gguf_t* gguf, const char* name, int32_t kernel, uint8_t* out,
                                              size_t out_capacity, quintrit_tiled_layout_t* out_layout, float* out_scale);

/* ----------------------------------------------------------------------------------------------------------------- */
/* Runtime level: models and sessions                                                                                 */
/* ----------------------------------------------------------------------------------------------------------------- */

/* A loaded model: transcoded weights plus tokenizer. Immutable while sessions exist; may be shared by threads. */
typedef struct quintrit_model quintrit_model_t;

/* A conversation: KV cache and scratch buffers of one model. Used by one thread at a time. */
typedef struct quintrit_session quintrit_session_t;

typedef struct quintrit_model_options {
    uint32_t struct_size;     /* set to sizeof(quintrit_model_options_t) */
    uint32_t num_threads;     /* 0 = automatic */
    int32_t  kernel;          /* QUINTRIT_KERNEL_AUTO, _SIMD32 or _AVX512 (the scalar kernel is not available for models) */
    uint32_t flags;           /* QUINTRIT_MODEL_OPEN_VERBOSE or 0 (no other bits are defined) */
} quintrit_model_options_t;

typedef struct quintrit_model_info {
    uint32_t struct_size;     /* set to sizeof(quintrit_model_info_t) before the call */
    int32_t  kernel;          /* concrete kernel in use */
    uint32_t num_threads;
    uint32_t num_layers;
    uint32_t hidden_size;
    uint32_t num_heads;
    uint32_t num_kv_heads;
    uint32_t head_dim;
    uint64_t vocab_size;
    uint64_t context_length;  /* maximum context window of the model, in tokens */
    int64_t  bos_token_id;    /* -1 if the model has none */
    int64_t  eos_token_id;    /* -1 if the model has none */
    char     architecture[16];/* "bitnet", "falcon3" or "llama" */
} quintrit_model_info_t;

typedef struct quintrit_session_options {
    uint32_t struct_size;     /* set to sizeof(quintrit_session_options_t) */
    uint32_t context_length;  /* tokens; 0 = the model's maximum. Smaller values save KV-cache memory. */
} quintrit_session_options_t;

typedef struct quintrit_session_info {
    uint32_t struct_size;     /* set to sizeof(quintrit_session_info_t) before the call */
    uint32_t reserved;
    uint64_t context_length;
    uint64_t position;        /* tokens currently held in the KV cache */
} quintrit_session_info_t;

/* quintrit_model_options_t flags: print loading progress to standard output. Without it, loading is silent. */
#define QUINTRIT_MODEL_OPEN_VERBOSE 1

/* quintrit_model_tokenize flags */
#define QUINTRIT_TOKENIZE_ADD_BOS 1

/* quintrit_generate_params_t::flags */
#define QUINTRIT_GENERATE_KEEP_CONTEXT 1  /* continue after the tokens already in the KV cache (no reset, no BOS) */

/* quintrit_generate_result_t::finish_reason */
#define QUINTRIT_FINISH_LENGTH         0  /* max_tokens reached */
#define QUINTRIT_FINISH_EOS            1  /* the model produced its end-of-sequence token */
#define QUINTRIT_FINISH_STOP_SEQUENCE  2  /* a stop sequence matched (the sequence is not part of the text) */
#define QUINTRIT_FINISH_CALLBACK       3  /* the callback asked to stop */
#define QUINTRIT_FINISH_CONTEXT_FULL   4  /* the context window is full */

typedef struct quintrit_generate_params {
    uint32_t           struct_size;    /* set to sizeof(quintrit_generate_params_t) */
    uint32_t           flags;          /* QUINTRIT_GENERATE_* */
    uint32_t           max_tokens;     /* new tokens to generate at most */
    uint32_t           top_k;          /* 0 = off, 1 = greedy */
    float              temperature;    /* 0 = greedy argmax; otherwise finite and > 0 */
    float              top_p;          /* nucleus threshold in (0, 1]; 1 = off */
    uint64_t           seed;           /* seed of the sampler; the same seed gives the same text */
    const char* const* stop_sequences; /* stop_count NUL-terminated UTF-8 strings, or NULL */
    size_t             stop_count;     /* at most 64 */
} quintrit_generate_params_t;

typedef struct quintrit_generate_result {
    uint32_t struct_size;     /* set to sizeof(quintrit_generate_result_t) before the call */
    int32_t  finish_reason;   /* QUINTRIT_FINISH_* */
    uint32_t prompt_tokens;   /* tokens ingested by this call */
    uint32_t generated_tokens;
    uint64_t text_bytes;      /* length of the generated text without the NUL */
} quintrit_generate_result_t;

typedef struct quintrit_chat_message {
    const char* role;         /* "system", "user" or "assistant" */
    const char* content;
} quintrit_chat_message_t;

/*
 * Streaming callback: receives a piece of UTF-8 text (text, len; NOT NUL-terminated, valid only during the call) and
 * returns 0 to continue or non-zero to stop the generation. A piece never ends inside a multi-byte character, so a
 * token that ends inside one yields no piece of its own and one piece can hold the text of several tokens. It runs on the calling thread; it must not call into the
 * same session and must not throw a C++ exception or longjmp across the library.
 */
typedef int (*quintrit_token_callback)(const char* text, size_t len, void* user_data);

/*
 * Loads a GGUF model (BitNet b1.58, Falcon3 and LLaMA-style files with I2_S / TQ1_0 / TQ2_0 ternary tensors). The ternary
 * weights are transcoded into the tiled layout in RAM, which takes from seconds to minutes for large models. Loading is
 * silent unless options->flags contains QUINTRIT_MODEL_OPEN_VERBOSE, which prints progress to standard output. options
 * may be NULL (automatic threads, best kernel, silent). The file is checked first; a model
 * whose hyper-parameters or tensors are inconsistent returns QUINTRIT_ERR_MODEL. On failure *out_model is NULL.
 */
QUINTRIT_API int quintrit_model_open(const char* path, const quintrit_model_options_t* options, quintrit_model_t** out_model);

/* Frees a model. NULL is allowed. Sessions keep the model alive until they are freed too. */
QUINTRIT_API int quintrit_model_free(quintrit_model_t* model);

QUINTRIT_API int quintrit_model_info(const quintrit_model_t* model, quintrit_model_info_t* out_info);

/*
 * Sets the worker thread count (0 = automatic). Returns QUINTRIT_ERR_BUSY while sessions of the model exist: set the
 * count before creating sessions (or free them first).
 */
QUINTRIT_API int quintrit_model_set_threads(quintrit_model_t* model, uint32_t num_threads);

/*
 * Converts UTF-8 text (text_len bytes, not necessarily NUL-terminated) into token ids. *out_count always receives the
 * token count; QUINTRIT_ERR_BUFFER_TOO_SMALL is returned if capacity is smaller (use out = NULL, capacity = 0 to size
 * the buffer). flags: QUINTRIT_TOKENIZE_ADD_BOS or 0.
 */
QUINTRIT_API int quintrit_model_tokenize(const quintrit_model_t* model, const char* text, size_t text_len, uint32_t flags,
                                         uint32_t* out, size_t capacity, size_t* out_count);

/*
 * Converts token ids into NUL-terminated UTF-8 text. *out_len receives the length without the NUL;
 * QUINTRIT_ERR_BUFFER_TOO_SMALL (nothing written) if capacity <= *out_len. Token ids >= vocab_size are rejected.
 */
QUINTRIT_API int quintrit_model_detokenize(const quintrit_model_t* model, const uint32_t* tokens, size_t count, char* out,
                                           size_t capacity, size_t* out_len);

/*
 * Formats a conversation into the prompt format of the model's architecture, ending with the assistant prefix. Output
 * as quintrit_model_detokenize. Roles other than "system" / "user" / "assistant" are treated as "user".
 */
QUINTRIT_API int quintrit_model_format_chat(const quintrit_model_t* model, const quintrit_chat_message_t* messages,
                                            size_t count, char* out, size_t capacity, size_t* out_len);

/* Creates a session. options may be NULL (full context window). On failure *out_session is NULL. */
QUINTRIT_API int quintrit_session_create(const quintrit_model_t* model, const quintrit_session_options_t* options,
                                         quintrit_session_t** out_session);

/* Frees a session. NULL is allowed. */
QUINTRIT_API int quintrit_session_free(quintrit_session_t* session);

/* Forgets the conversation (empties the KV cache). */
QUINTRIT_API int quintrit_session_reset(quintrit_session_t* session);

QUINTRIT_API int quintrit_session_info(const quintrit_session_t* session, quintrit_session_info_t* out_info);

/* Fills *params with the defaults: greedy decoding, max_tokens 128, no stop sequences, flags 0. */
QUINTRIT_API int quintrit_generate_params_default(quintrit_generate_params_t* params);

/*
 * Generates text for a prompt (prompt_len bytes of UTF-8).
 *
 * Unless QUINTRIT_GENERATE_KEEP_CONTEXT is set, the KV cache is reset first and the BOS token is prepended. The text is
 * streamed to `callback` (may be NULL) and also stored in `out` (may be NULL with out_capacity 0), snprintf-style:
 * truncated at a character boundary and NUL-terminated whenever out_capacity > 0. *out_len (may be NULL) receives the
 * full length without the NUL, so out_len >= out_capacity means truncation. `params` may be NULL for the defaults;
 * `result` may be NULL.
 *
 * QUINTRIT_ERR_CONTEXT_OVERFLOW if the prompt (plus the tokens kept in the cache) does not fit into the context window.
 * QUINTRIT_ERR_BUSY if another thread is using the session. Decoding with temperature 0 is deterministic and
 * bit-identical for every kernel.
 */
QUINTRIT_API int quintrit_session_generate(quintrit_session_t* session, const char* prompt, size_t prompt_len,
                                           const quintrit_generate_params_t* params, quintrit_token_callback callback,
                                           void* user_data, char* out, size_t out_capacity, size_t* out_len,
                                           quintrit_generate_result_t* result);

#ifdef __cplusplus
}
#endif

#endif /* QUINTRIT_H */

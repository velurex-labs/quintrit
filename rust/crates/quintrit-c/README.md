# quintrit-c

A stable C ABI for Quintrit: ternary matrix kernels, a GGUF reader and an inference runtime behind plain `extern "C"` functions and opaque handles. It produces

| Platform | Shared library | Import library | Static library |
|---|---|---|---|
| Windows (MSVC) | `quintrit_c.dll` | `quintrit_c.dll.lib` | `quintrit_c.lib` |
| Linux | `libquintrit_c.so` | - | `libquintrit_c.a` |
| macOS | `libquintrit_c.dylib` | - | `libquintrit_c.a` |

(The library files are called `quintrit_c`, not `quintrit`, so that they cannot collide with the `quintrit` command-line
binary of `quintrit-runtime`. The C symbols, the headers and the C++ namespace keep the `quintrit` name.)

and two headers in [`include/`](./include):

- [`quintrit.h`](./include/quintrit.h): the C API (valid C99 and C++11, documented per function).
- [`quintrit.hpp`](./include/quintrit.hpp): an optional header-only C++11 wrapper with RAII classes and exceptions.

Quintrit stores five ternary weights `{-1, 0, +1}` in one byte (velurex encoding, 1.60 bits per weight) and multiplies them with int8 activations using SIMD activation-lookup-table kernels (AVX2, AVX-512, NEON, plus a portable scalar kernel). All kernels return identical integer results.

## Layers

| Layer | Functions | What it does |
|---|---|---|
| Kernel | `quintrit_cpu_features`, `quintrit_supported_kernels`, `quintrit_kernel_*`, `quintrit_tiled_layout`, `quintrit_tiled_size`, `quintrit_pack_matrix`, `quintrit_unpack_matrix`, `quintrit_gemv_i32`, `quintrit_gemv_f32` | Pack a row-major ternary matrix into the tiled layout and run `W x` on int8 activations. |
| Format | `quintrit_gguf_open`, `_close`, `_info`, `_tensor_at`, `_tensor_find`, `_metadata_string`, `_metadata_u64`, `_tensor_data`, `_tensor_layout`, `_convert_tensor` | Open a GGUF file, list tensors, read metadata, convert an `I2_S`, `TQ1_0` or `TQ2_0` tensor into the tiled layout. |
| Runtime | `quintrit_model_open`, `_free`, `_info`, `_set_threads`, `_tokenize`, `_detokenize`, `_format_chat`; `quintrit_session_create`, `_free`, `_reset`, `_info`, `_generate`; `quintrit_generate_params_default` | Load a model, tokenize, generate text (streaming callback and/or buffer). |

Every function is declared in `quintrit.h` with its contract; the sections below explain the rules that apply to all of them.

## Build

```bash
# from rust/
cargo build --release -p quintrit-c                     # target/release/{quintrit_c.dll, libquintrit_c.so, ...}
cargo build --profile release-unwind -p quintrit-c      # same, but panics unwind (see "Panics" below)
cargo test  -p quintrit-c                               # Rust, C and C++ tests (no model files, no network)
```

CMake can drive the same build and the C/C++ tests, and can be used from other projects with `add_subdirectory`:

```bash
cmake -S rust/crates/quintrit-c -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

```cmake
add_subdirectory(path/to/quintrit/rust/crates/quintrit-c quintrit_c)
target_link_libraries(my_app PRIVATE quintrit::quintrit)          # shared
# target_link_libraries(my_app PRIVATE quintrit::quintrit_static) # static
```

Linking by hand: add `include/` to the include path and link `quintrit_c.dll.lib` (Windows), `-lquintrit_c` (Unix); the DLL/shared object must be found at run time. The static library contains the Rust standard library; `rustc --print native-static-libs` lists the system libraries it needs on your target (on Windows MSVC: `kernel32 ntdll userenv ws2_32 dbghelp`, with the dynamic CRT `/MD`).

## Conventions

- **Status codes.** Every function that can fail returns an `int`: `QUINTRIT_OK` (0) or one of `QUINTRIT_ERR_*`. Results come back through out-parameters. A few functions cannot fail and return their value: `quintrit_version`, `quintrit_abi_version`, `quintrit_status_string`, `quintrit_kernel_name`, `quintrit_last_error_code`, `quintrit_last_error_message`.
- **Error messages.** A failed call stores a message in a thread-local slot: `quintrit_last_error_message()` returns it (never NULL; empty after a successful call), `quintrit_last_error_code()` the matching code. The pointer is valid until the next `quintrit_*` call on the same thread; copy it if you need it longer. Every `quintrit_*` call except those two clears the slot first.
- **Validation.** Every pointer, length, alignment, shape, enumeration, flag and `struct_size` is checked; a bad argument yields an error code, never undefined behavior. Buffers always come with explicit lengths (`trits_len`, `out_capacity`, ...) and are checked against the shape. Input and output buffers of a call must not overlap.
- **Sizes and queries.** Variable-size outputs follow one pattern: call with a NULL buffer and capacity 0 to learn the size from the `out_*` parameter (the call then returns `QUINTRIT_ERR_BUFFER_TOO_SMALL`), allocate, call again. `quintrit_session_generate` is the exception: it truncates `snprintf`-style and reports the full length.
- **Versioned structs.** Structs that cross the boundary start with `uint32_t struct_size`; set it to `sizeof` the struct. The library rejects a smaller value, so a header/library mismatch fails loudly. Compare `quintrit_abi_version()` with `QUINTRIT_ABI_VERSION` at start-up.
- **Strings** are UTF-8, also paths on Windows. Texts (prompts, tokenizer input) are passed as pointer + length and need no terminator; names and keys are NUL-terminated.
- **Kernels.** `QUINTRIT_KERNEL_AUTO`, `_SCALAR`, `_SIMD32` (AVX2 on x86-64, NEON on AArch64), `_AVX512` (BW + VBMI, x86-64). Asking for a kernel the CPU lacks fails with `QUINTRIT_ERR_UNSUPPORTED`; nothing falls back silently. `AUTO` picks the best usable kernel at run time.

## Tiled weight layout

A `rows x cols` matrix is cut into tiles of 32 rows (`SCALAR`, `SIMD32`) or 64 rows (`AVX512`); inside a tile the bytes of a column are stored contiguously across the rows, and each byte holds five trits. `rows` is padded to a whole number of tiles with zero weights; any `rows` and `cols` are accepted. `quintrit_tiled_layout` reports `tile_rows`, `padded_rows`, `bytes_per_row` and `total_bytes`.

- Pack with the kernel you will run. `SCALAR` and `SIMD32` share the 32-row layout; `AVX512` needs its own.
- A tiled buffer carries no header (the layout is fixed). If you store one, store the *resolved* kernel id (`quintrit_kernel_resolve`) and the shape next to it, never `AUTO`: `AUTO` means a different layout on a machine with and without AVX-512.
- `quintrit_gemv_*` accepts `cols <= 131072`: the kernels return float-exact integer sums, which holds while `127 * cols < 2^24`. The result is exact for the whole int8 activation range `-128..127`.

## Example: kernel level (C)

```c
#include "quintrit.h"
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    enum { ROWS = 100, COLS = 1000 };
    static int8_t trits[ROWS * COLS];      /* -1, 0 or +1 */
    static int8_t acts[COLS];              /* any int8 value */
    static int32_t out[ROWS];
    /* ... fill trits and acts ... */

    if (quintrit_abi_version() != QUINTRIT_ABI_VERSION) return 2;

    size_t bytes = 0;
    if (quintrit_tiled_size(QUINTRIT_KERNEL_AUTO, ROWS, COLS, &bytes) != QUINTRIT_OK) goto fail;
    uint8_t* tiled = malloc(bytes);

    if (quintrit_pack_matrix(QUINTRIT_KERNEL_AUTO, trits, sizeof trits, ROWS, COLS, tiled, bytes) != QUINTRIT_OK) goto fail;
    if (quintrit_gemv_i32(QUINTRIT_KERNEL_AUTO, tiled, bytes, ROWS, COLS, acts, COLS, out, ROWS, /*threads*/ 0) != QUINTRIT_OK) goto fail;

    printf("out[0] = %d\n", out[0]);
    free(tiled);
    return 0;
fail:
    fprintf(stderr, "quintrit: %s\n", quintrit_last_error_message());
    return 1;
}
```

`quintrit_gemv_f32` additionally takes a `scale`: pass `weight_scale / activation_scale` (one division per matrix, as the runtime does) to dequantize.

## Example: format and runtime level (C)

```c
#include "quintrit.h"
#include <stdio.h>
#include <string.h>

static int on_text(const char* text, size_t len, void* user) {
    (void)user;
    fwrite(text, 1, len, stdout);
    return 0;                                /* non-zero stops the generation */
}

int main(void) {
    quintrit_model_t* model = NULL;
    quintrit_session_t* session = NULL;
    /* NULL options: automatic threads and kernel, silent. Pass a quintrit_model_options_t with
       flags = QUINTRIT_MODEL_OPEN_VERBOSE to print the loading progress to standard output. */
    if (quintrit_model_open("model.gguf", NULL, &model) != QUINTRIT_OK) goto fail;
    if (quintrit_session_create(model, NULL, &session) != QUINTRIT_OK) goto fail;

    quintrit_generate_params_t params;
    quintrit_generate_params_default(&params);   /* greedy, 128 tokens */
    params.max_tokens = 64;

    const char* prompt = "<|user|>\nWhat is ternary computing?\n<|assistant|>\n";
    quintrit_generate_result_t result = { sizeof result };
    if (quintrit_session_generate(session, prompt, strlen(prompt), &params, on_text, NULL, NULL, 0, NULL, &result) != QUINTRIT_OK) goto fail;

    quintrit_session_free(session);
    quintrit_model_free(model);
    return 0;
fail:
    fprintf(stderr, "quintrit: %s\n", quintrit_last_error_message());
    quintrit_session_free(session);          /* NULL is fine */
    quintrit_model_free(model);
    return 1;
}
```

`quintrit_model_format_chat` builds the prompt of the model's architecture from `{role, content}` messages. With `QUINTRIT_GENERATE_KEEP_CONTEXT` a call continues after the tokens already in the KV cache (no reset, no BOS), so a chat does not re-process the history; `quintrit_session_reset` forgets it.

## Example: C++

```cpp
#include "quintrit.hpp"

quintrit::Model model("model.gguf", /*threads*/ 0);   // silent; the fourth argument `verbose = true` prints progress
quintrit::Session session = model.create_session();

quintrit::GenerateOptions options;
options.max_tokens = 64;
quintrit::Generation g = session.generate(
    model.format_chat({{"user", "What is ternary computing?"}}), options,
    [](const char* text, std::size_t len) { std::cout.write(text, len); return true; });  // false stops
```

The wrapper reports errors by exceptions: every failing C call throws `quintrit::Error`, which derives from `std::runtime_error`, carries the status code (`code()`) and the library's message. Classes (`Gguf`, `Model`, `Session`) own their handle, are movable and not copyable, and free it in the destructor. An exception thrown by a streaming callback is caught at the C boundary, stops the generation and is rethrown from `generate`.

## Error handling

| Code | Meaning |
|---|---|
| `QUINTRIT_ERR_NULL_POINTER` | a required pointer is NULL |
| `QUINTRIT_ERR_INVALID_ARGUMENT` | bad enumeration, flag, `struct_size`, alignment, thread count, overlapping buffers, invalid UTF-8 |
| `QUINTRIT_ERR_INVALID_SHAPE` | `rows` or `cols` is 0, above 2^31, or the matrix is too large |
| `QUINTRIT_ERR_BUFFER_TOO_SMALL` | a buffer is shorter than the shape or the result needs |
| `QUINTRIT_ERR_INVALID_VALUE` | a trit outside `{-1,0,1}`, a packed byte above 250, a bad sampling parameter |
| `QUINTRIT_ERR_UNSUPPORTED` | kernel or tensor type not available on this CPU or build |
| `QUINTRIT_ERR_IO` / `_FORMAT` / `_NOT_FOUND` | file cannot be read / is not a valid GGUF file or tensor / name or key does not exist |
| `QUINTRIT_ERR_MODEL` | the model file is well formed but inconsistent or unsupported |
| `QUINTRIT_ERR_CONTEXT_OVERFLOW` | the prompt does not fit into the context window |
| `QUINTRIT_ERR_BUSY` | the handle is used by another thread, or the operation needs it idle |
| `QUINTRIT_ERR_INTERNAL` | a Rust panic was caught (only possible when panics unwind, see below) |
| `QUINTRIT_ERR_INVALID_HANDLE` | a handle is NULL, of the wrong type or freed |

## Panics and `panic = "abort"`

Every entry point validates its arguments and runs inside `catch_unwind`, so that a panic never unwinds across the C boundary (which would be undefined behavior). The workspace `release` profile, however, sets `panic = "abort"`: there `catch_unwind` cannot catch anything, and a panic that slips through the validation (a bug, or a malformed model the pre-checks of `quintrit_model_open` did not foresee) terminates the host process. The `dev` and `test` profiles unwind and report `QUINTRIT_ERR_INTERNAL`.

If your application must survive such a defect, build the library with the `release-unwind` profile (same optimization level, panics unwind): `cargo build --profile release-unwind -p quintrit-c`. The cost is a slightly larger binary. Memory exhaustion aborts in either case.

## Thread safety

- All functions are safe to call concurrently on **distinct** handles, and the stateless functions (kernel level, queries) from any number of threads.
- A `quintrit_gguf_t` and a `quintrit_model_t` can be shared by threads for reading (tokenizing, creating sessions, listing tensors).
- A `quintrit_session_t` is used by one thread at a time; a concurrent call fails with `QUINTRIT_ERR_BUSY` instead of blocking. Use one session per concurrent conversation.
- `quintrit_model_set_threads` needs exclusive access: it fails with `QUINTRIT_ERR_BUSY` while sessions of the model exist. Set the thread count at `quintrit_model_open` (or before creating sessions).
- The `num_threads` arguments (0 = automatic, at most `QUINTRIT_MAX_THREADS`) select a persistent worker pool of that size; each distinct count creates one pool that lives until the process ends, so use few distinct values. Setting `QUINTRIT_POOL_PIN=1` pins the workers to cores.
- Freeing a handle while another thread uses it is a bug of the caller and is not detected.
- The streaming callback runs on the thread that called `quintrit_session_generate`. It must not call into the same session and must not throw an exception or `longjmp` across the library.

## Lifetime rules

- `*_open` / `*_create` set the out-pointer to NULL on failure. `*_close` / `*_free` accept NULL.
- Handles are tagged; passing the wrong kind of handle, NULL, or a handle that has been freed (usually) returns `QUINTRIT_ERR_INVALID_HANDLE`. This is a best-effort check, not protection against dangling pointers.
- A model stays alive as long as any of its sessions exists: `quintrit_model_free` only drops the model handle.
- Tensor names (`quintrit_tensor_info_t::name`) and raw tensor data (`quintrit_gguf_tensor_data`) point into the open GGUF handle and become invalid at `quintrit_gguf_close`. The file is memory-mapped: do not modify or truncate it while it is open.
- The pointer from `quintrit_last_error_message` is valid until the next `quintrit_*` call on the same thread. `quintrit_version`, `quintrit_status_string` and `quintrit_kernel_name` return static strings.

## Models and files

- `quintrit_gguf_*` and `quintrit_model_open` read GGUF v2/v3 files (BitNet b1.58, Falcon3 and LLaMA-style models with `I2_S`, `TQ1_0` or `TQ2_0` ternary tensors). There is no separate on-disk quintrit weight format yet: the tiled layout is produced in memory by `quintrit_gguf_convert_tensor` and `quintrit_model_open`.
- The GGUF reader of `quintrit-format` bounds-checks every count, length, shape, offset and the alignment against the file size before it allocates anything, and a model's hyper-parameters and tensor sizes are validated before loading, so truncated, inconsistent or hostile files are reported as `QUINTRIT_ERR_FORMAT` / `QUINTRIT_ERR_MODEL`. This is a sanity check, not a security boundary: load model files from sources you trust. A path that is not a regular file is reported as `QUINTRIT_ERR_IO`.
- `quintrit_model_open` is silent (it takes seconds to minutes to transcode the weights of a large model). Set `QUINTRIT_MODEL_OPEN_VERBOSE` in `quintrit_model_options_t::flags` to print progress lines to the standard output; unknown flag bits are rejected with `QUINTRIT_ERR_INVALID_ARGUMENT`.
- `shape[0]` of a tensor is its innermost dimension, so a weight matrix is `[cols, rows]`.
- Decoding with `temperature = 0` is deterministic and bit-identical for every kernel. Sampling with `temperature > 0` is reproducible for a fixed `seed`.
- A model runs on `SIMD32` or `AVX512` (`AUTO` picks); the scalar kernel exists at the kernel level only.

## Tests

`cargo test -p quintrit-c` needs no model files and no network:

| Test | What it checks |
|---|---|
| `tests/ffi_kernel.rs` | pack / unpack / GEMV through the `extern "C"` functions against brute-force `int64` ground truth, for every usable kernel, many shapes (full tiles, remainder rows and columns, partial 160-trit blocks), thread counts, the full int8 range, and every error path |
| `tests/ffi_gguf.rs` | a synthetic GGUF file (random ternary weights): listing, metadata, tensor conversion to every layout against the original trits and the format crate's transcoder, bad and randomly damaged files |
| `tests/ffi_runtime.rs` | the same synthetic model through the runtime API: tokenizer, greedy generation compared with `quintrit-runtime`'s own engine, streaming, stop sequences, context limits, sessions on several threads, `BUSY`, handle lifetimes, damaged models |
| `tests/quiet_load.rs` | loading a model prints nothing to the standard output unless `QUINTRIT_MODEL_OPEN_VERBOSE` is set (the load runs in a child process whose output is captured) |
| `tests/header_sync.rs` | `include/quintrit.h` against the Rust sources: exported functions with parameter and return types, `repr(C)` struct fields, constants, version; header structure (C99, include guard, `extern "C"`) |
| `tests/c_cpp.rs` | builds `tests/c/*.c` and `tests/cpp/*.cpp` with the system C / C++ compiler (found by the `cc` crate: MSVC, clang or gcc) against the real shared library and runs them; the runtime tests there load the synthetic model and compare the generated text with the Rust API |

Without a C or C++ compiler the `c_cpp` tests print `SKIPPED` and pass; set `QUINTRIT_REQUIRE_C_COMPILER=1` to make a missing compiler a failure, and `QUINTRIT_STRICT_WARNINGS=1` to compile the test programs with warnings as errors. AVX-512 and NEON kernels are exercised only on CPUs that have them; the tests report which kernels were usable.


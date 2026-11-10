# Using Quintrit from C and C++

The library exposes a C99 ABI (`include/quintrit.h`) and a header-only C++17 wrapper
(`include/quintrit.hpp`) over it. The reference of every function is the header; this guide shows how to build, link and
use it. Full details are in the [`quintrit-c` README](../../rust/crates/quintrit-c/README.md).

## Build and link with CMake

The project in [`rust/crates/quintrit-c`](../../rust/crates/quintrit-c/CMakeLists.txt) runs `cargo` and imports the
library as the target `quintrit::quintrit` (shared) or `quintrit::quintrit_static`. It needs a Rust toolchain (1.98 or
newer), CMake 3.14 or newer, and a C++17 compiler (for `quintrit.hpp`). The `cargo` executable must be available in `PATH` (e.g. via `source ~/.cargo/env` when using rustup), or passed via `-DQUINTRIT_CARGO=/path/to/cargo`.

```cmake
cmake_minimum_required(VERSION 3.14)
project(my_project CXX)

set(QUINTRIT_BUILD_TESTS OFF CACHE BOOL "" FORCE)
add_subdirectory(path/to/rust/crates/quintrit-c quintrit_c)

add_executable(app app.cpp)
target_compile_features(app PRIVATE cxx_std_17)
target_link_libraries(app PRIVATE quintrit::quintrit)  # or quintrit::quintrit_static
```

Use `quintrit::quintrit_static` to link statically and eliminate any need to deploy or locate the dynamic library (`.so` / `.dll`).

> [!NOTE]
> The first invocation of `cmake --build` compiles the underlying Rust crates from source via Cargo (typically taking 1–2 minutes). Subsequent builds are incremental and Cargo runs as a near-instant no-op when Rust sources have not changed.

Without CMake: `cargo build --release -p quintrit-c` in `rust/` produces `quintrit_c.dll` (Windows),
`libquintrit_c.so` (Linux) or `libquintrit_c.dylib` (macOS) and the static library, and `include/` holds the headers.

The release profile of the workspace aborts on a panic. If your application must survive an internal error (the call
then returns `QUINTRIT_ERR_INTERNAL`), build with `--profile release-unwind`.

## Conventions

- Every function returns a status code; `QUINTRIT_OK` is 0. After a failure, `quintrit_last_error_message()` returns a
  message for the calling thread. `quintrit_status_string(code)` names the code.
- Handles are opaque and freed with the matching `_free`/`_close` function; freeing `NULL` is allowed.
- Structs have a `struct_size` field that you set to `sizeof` before the call, so that later versions can add fields.
- Check `quintrit_abi_version() == QUINTRIT_ABI_VERSION` once at start-up when you load the library dynamically.
- Loading a model is silent unless you pass `QUINTRIT_MODEL_OPEN_VERBOSE`.
- A session is used by one thread at a time; a model can be shared by several sessions.

## Kernel level: a matrix-vector product

In C99 using `quintrit.h`:

```c
#include "quintrit.h"

int32_t kernel;
size_t bytes;
quintrit_kernel_resolve(QUINTRIT_KERNEL_AUTO, &kernel);              /* best kernel of this CPU */
quintrit_tiled_size(kernel, rows, cols, &bytes);
uint8_t* packed = malloc(bytes);
quintrit_pack_matrix(kernel, trits, rows * cols, rows, cols, packed, bytes);   /* trits: int8 in {-1, 0, +1} */
quintrit_gemv_i32(kernel, packed, bytes, rows, cols, acts, cols, out, rows, /* threads */ 0);   /* exact */
```

In C++17 using `quintrit.hpp`:

```cpp
#include "quintrit.hpp"

// trits: std::vector<int8_t> with values in {-1, 0, +1}; acts: std::vector<int8_t>
auto kernel = quintrit::resolve_kernel(quintrit::Kernel::Auto);
std::vector<uint8_t> packed = quintrit::pack_matrix(kernel, trits, rows, cols);
std::vector<int32_t> out = quintrit::gemv_i32(kernel, packed, rows, cols, acts);   // exact
```

`quintrit_gemv_f32` (and `quintrit::gemv_f32` in C++) returns `scale * (W·x)` as floats.
[`examples/c/gemv.c`](../../examples/c/gemv.c) and [`examples/cpp/gemv.cpp`](../../examples/cpp/gemv.cpp) are complete
programs with error handling and checks against a plain integer product.

## Format level: GGUF files

`quintrit_gguf_open` opens a file (bounds-checked); `quintrit_gguf_info`, `quintrit_gguf_tensor_at` and
`quintrit_gguf_tensor_find` describe it, `quintrit_gguf_metadata_string` and `_u64` read metadata, and
`quintrit_gguf_convert_tensor` converts a ternary tensor into the tiled layout of a kernel.

## Runtime level: generate text

```cpp
#include "quintrit.hpp"

quintrit::Model model("model.gguf");                  // throws quintrit::Error on failure
quintrit::Session session = model.create_session();
quintrit::GenerateOptions options;                    // greedy by default
options.max_tokens = 64;
auto result = session.generate("<|user|>\nHello\n<|assistant|>\n", options,
                               [](const char* text, std::size_t len) { std::cout.write(text, len); return true; });
```

The C equivalent uses `quintrit_model_open`, `quintrit_session_create` and `quintrit_session_generate` with a callback
that returns 0 to continue. See [`examples/c/generate.c`](../../examples/c/generate.c) and
[`examples/cpp/generate.cpp`](../../examples/cpp/generate.cpp). `quintrit_model_format_chat` builds a chat prompt in the
format of the model, and `quintrit_model_tokenize` and `quintrit_model_detokenize` convert between text and token ids.

## Building the examples

```bash
cd examples
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure
```

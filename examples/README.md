# Examples

**Start here if you only want to try it:** [`cli/`](cli/README.md) downloads the prebuilt `quintrit` program and a small model
and runs one answer, a chat and a benchmark, with no Rust toolchain, compiler or Python.

The programs below use Quintrit at the kernel level (pack a ternary matrix and multiply it with `int8` activations;
each checks its result against a plain integer product) and at the runtime level (load a GGUF model and generate
text). The runtime examples need a ternary GGUF model; the kernel examples need nothing.

| Language | Kernel level | Runtime level | Build and run |
|---|---|---|---|
| Rust | [`rust/src/bin/gemv.rs`](rust/src/bin/gemv.rs) | [`rust/src/bin/generate.rs`](rust/src/bin/generate.rs), on a GPU: [`rust/src/bin/generate_gpu.rs`](rust/src/bin/generate_gpu.rs) | `cd rust && cargo run --release --bin gemv` and `cargo run --release --bin generate -- model.gguf` |
| C | [`c/gemv.c`](c/gemv.c) | [`c/generate.c`](c/generate.c), on a GPU: [`c/generate_gpu.c`](c/generate_gpu.c) | CMake, see below |
| C++ | [`cpp/gemv.cpp`](cpp/gemv.cpp) | [`cpp/generate.cpp`](cpp/generate.cpp), on a GPU: [`cpp/generate_gpu.cpp`](cpp/generate_gpu.cpp) | CMake, see below |
| Python | [`python/gemv.py`](python/gemv.py) | [`python/generate.py`](python/generate.py), on a GPU: [`python/generate_gpu.py`](python/generate_gpu.py) | install `quintrit`, then `python python/gemv.py` (see below) |

## C and C++

The C and C++ examples are built by the CMake project in this directory, which builds the C library with cargo:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release
cmake --build build --config Release
ctest --test-dir build -C Release --output-on-failure     # runs the kernel examples
build/c_generate model.gguf                                # on Windows: build/Release/c_generate.exe
```

Rust 1.98 or newer and CMake 3.14 or newer are required.

## NVIDIA GPU

The `generate_gpu` examples run a model on a GPU of compute capability 8.0 or newer ([`../docs/guides/gpu.md`](../docs/guides/gpu.md)).
The Rust one is built with `cargo run --release --features cuda --bin generate_gpu -- model.gguf` in `rust/` (it needs the
CUDA Toolkit). The C and C++ ones are always built; configure CMake with `-DQUINTRIT_CUDA=ON` for a library with GPU support,
otherwise they report that the build has none. The Python one needs a package built with `maturin develop --release --features cuda`
in `rust/crates/quintrit-py`.

## Python

Install the package (or build from source with `maturin develop --release` in `rust/crates/quintrit-py`) and run:

```bash
python python/gemv.py
python python/generate.py model.gguf
```

An extended tutorial suite covering GGUF tensor inspection and sampling options is in
[`rust/crates/quintrit-py/examples/`](../rust/crates/quintrit-py/examples/).

The guides explain the code:
[Rust](../docs/guides/rust.md), [C and C++](../docs/guides/c-cpp.md), [Python](../docs/guides/python.md).

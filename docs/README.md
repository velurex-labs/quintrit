# Documentation

| Document | Content |
|---|---|
| [`velurex-encoding.md`](velurex-encoding.md) | The encoding: definition, why the radix 28 allows a division-free decoder, comparison with other ternary formats, citation |
| [`architecture.md`](architecture.md) | The crates, the path from a GGUF file to the first token, the tile layout |
| [`kernels.md`](kernels.md) | The kernels, their exactness and the instruction sets |
| [`benchmarks.md`](benchmarks.md) | Kernel comparison with BitNet on four machines, the memory ceilings, how to run the benchmarks |

## Guides

| Guide | For |
|---|---|
| [`guides/rust.md`](guides/rust.md) | Rust programs: kernels, GGUF files, the engine |
| [`guides/c-cpp.md`](guides/c-cpp.md) | C and C++ programs: the C ABI and the C++ wrapper, CMake |
| [`guides/python.md`](guides/python.md) | Python programs: the `quintrit` package |
| [`guides/cli.md`](guides/cli.md) | The `quintrit` command-line tool |

Runnable programs for each language are in [`../examples/`](../examples/README.md). The reference of each package is the
README of its crate in [`../rust/crates/`](../rust/crates/).

# Quintrit Rust workspace

The code of Quintrit. Overview, documentation and examples are in the [repository README](../README.md) and in
[`../docs/`](../docs/README.md).

| Crate | Content |
|---|---|
| [`quintrit-core`](crates/quintrit-core/README.md) | Kernels: velurex encoding, activation tables, tiled layouts, SIMD GEMV (AVX2, AVX-512, NEON, portable), threading |
| [`quintrit-format`](crates/quintrit-format/README.md) | GGUF reader, ternary tensor decoders (`I2_S`, `TQ1_0`, `TQ2_0`), transcoder into the tiled layout |
| [`quintrit-bitnet`](crates/quintrit-bitnet/README.md) | Rust transcription of the 4-trit BitNet kernels, the baseline for kernel comparisons (Rust only, not published) |
| [`quintrit-runtime`](crates/quintrit-runtime/README.md) | Transformer engine, KV cache, tokenizer, sampling and the `quintrit` command-line tool |
| [`quintrit-c`](crates/quintrit-c/README.md) | C ABI (`quintrit_c` library, `include/quintrit.h`) and the C++ wrapper `include/quintrit.hpp` |
| [`quintrit-py`](crates/quintrit-py/README.md) | Python package `quintrit` (PyO3 and maturin) |

`quintrit-py` is deliberately not a workspace member, so `cargo build` and `cargo test` of the workspace never need a
Python interpreter.

## Build and test

The minimum supported Rust version is 1.98 (edition 2024; the AArch64 NEON dot-product intrinsics are stable from that
release).

```bash
cargo test                      # all crates; needs no model files and no network
cargo test -p quintrit-core     # one crate
cargo build --release -p quintrit-runtime --bin quintrit
```

`cargo test` checks every kernel, packer, table builder, format decoder, numerical primitive, projection layer, the
attention block and a small synthetic decoder stack against an independent brute-force or `f64` ground truth. SIMD
variants run only on CPUs that advertise the required features, so the AVX-512 tests are skipped elsewhere.

The brute-force ground truth is plain scalar code that uses only standard CPU operations: no SIMD intrinsics and none
of the library's own packing, tiling or lookup tables. It multiplies the unpacked ternary weights (`-1`, `0`, `1`) by
the `int8` activations in a simple loop and accumulates in `i64`, so the result is exact and cannot overflow or round.
The tests pack the same unpacked matrix and demand that every kernel returns exactly these integers, which checks the
packer, the tile layout and the kernel in one step. The portable scalar kernel of the library is treated like any
other kernel and is checked against the same ground truth. Floating-point primitives of the runtime are compared with
an `f64` evaluation within a stated tolerance instead.

The numerical primitives of the runtime (half-precision dot product and multiply-add, SiLU and SwiGLU, softmax, RMSNorm)
perform the same arithmetic on AVX2 and on NEON; their unit tests print checksums that must be equal on x86-64 and
AArch64.

## Benchmarks

The commands, the method and the published results are in [`../docs/benchmarks.md`](../docs/benchmarks.md). The benchmark programs write
their reports to a `results/` directory next to the crate; it is ignored by git.

# quintrit-bitnet

A faithful Rust transcription of the 4-trit kernels of [Microsoft BitNet](https://github.com/microsoft/BitNet)
(2.00 bits per weight). It exists for one purpose: to compare the quintrit kernels (5 trits per byte, 1.60 bits per
weight) with the original encoding under the same compiler, the same machine and the same harness.

The crate is for Rust only. It is not part of the C library or the Python package, and it is not published to
crates.io.

## Kernels

| Kernel | Encoding | Instruction sets | What it is |
|---|---|---|---|
| `mad` (`gemv_bitnet_mad_*`) | 4 trits per byte (`I2_S`) | scalar, AVX2, NEON | BitNet's planar multiply-add kernel, the kernel BitNet runs for `I2_S` weights |
| `lut` (`gemv_bitnet_lut*`) | 4 trits per byte | scalar, AVX2, NEON | BitNet's 32-row tiled pair-table lookup kernel, with BitNet's original activation handling (63 levels) |
| `mad512` (`gemv_bitnet_mad_avx512_*`) | 4 trits per byte | AVX-512 | The `mad` algorithm widened to 512-bit registers. Microsoft ships no AVX-512 kernel; this widening lets the 5-trit AVX-512 kernel be compared on the same instruction set |

The kernels are transcribed, not tuned: data layouts, tiling and the order of the SIMD instructions follow the C++
kernels, so that a difference between a BitNet kernel and a quintrit kernel can be attributed to the encoding and not
to tuning effort. For that reason the sources are not optimized, extended or restructured.

The `lut` kernel is not Microsoft's TL1 or TL2 format; it applies a pair table to the same 4-trit weights.

## Tests

```bash
cargo test -p quintrit-bitnet
```

The tests compare every kernel with a brute-force product in plain scalar integer arithmetic. They need no model and
no network. Tests of a SIMD variant skip themselves when the CPU lacks the instruction set.

## License

Apache-2.0. The kernels reproduce algorithms and data layouts of Microsoft BitNet, which is MIT-licensed; see the
`NOTICE` file.

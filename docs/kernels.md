# Kernels

All kernels use velurex encoding ([`velurex-encoding.md`](velurex-encoding.md)): five trits per byte, 1.60 bits per
weight. They differ in the instruction set and with it in the tile height of the in-memory weight layout.

| Kernel | Instruction set | Tile rows | CLI `--kernel` | C and C++ | Python |
|---|---|---:|---|---|---|
| `quintrit` | AVX2 (x86-64) or NEON (AArch64), with the ARMv8.2-A dot product where present | 32 | `quintrit` (default) | `QUINTRIT_KERNEL_SIMD32` | `"quintrit"` |
| `quintrit_512` | AVX-512 (BW and VBMI) on x86-64 | 64 | `quintrit_512` | `QUINTRIT_KERNEL_AVX512` | `"quintrit_512"` |
| portable | none; runs the NEON code on AArch64 | 32 | – | `QUINTRIT_KERNEL_SCALAR` | – |

`auto` (`QUINTRIT_KERNEL_AUTO`, `kernel="auto"`) picks the best kernel the CPU can run. The CPU features are detected
at run time, so one build runs everywhere, and a kernel that needs a feature the CPU lacks is never selected. The
scalar kernel is slow and exists as a reference and as a fallback for other architectures.

A buffer packed for a 32-row kernel can be read by every 32-row kernel; the AVX-512 kernel needs its own 64-row layout.
When you store packed weights, store the resolved kernel, not `auto`.

## Exactness

The kernels compute the exact integer product `Σ w·x` of ternary weights and `int8` activations. They do not round
and do not approximate.

- The activation tables hold small partial sums, so an `int8` activation is split as `x = 16·hi + lo`, with `lo` in
  `[-8, 7]` and `hi` in `[-8, 8]`. The fused kernels decode a weight byte once, look up both tables and combine the
  results as `lo + 16·hi`. This is exact for every `int8` value, `-128` and `127` included.
- Every kernel and every instruction set returns the same integers for the same input. The tests compare the kernels
  with a brute-force product and with each other, and run on every CPU that supports the instruction set. The
  brute-force product is a plain scalar loop over the unpacked weights and activations with `i64` accumulation. It
  uses only standard CPU operations, no SIMD and none of the library's packing, tiling or lookup tables.
- The integer result is returned as `int32` (C, C++) or as `float32` (Rust kernel functions, Python). An `f32` holds
  integers up to 2^24 exactly, which covers products of up to about 131 000 columns with the largest activations.

## Instruction sets

| | Weight decode | Table lookup |
|---|---|---|
| AVX2 | shifts and byte averages, 32 rows per register | `vpshufb` |
| AVX-512 | shuffle-based decode of 64 rows per register | `vpermb` (VBMI) |
| NEON | shifts and byte averages | `vqtbl` |

The summing of the looked-up partial products differs by instruction set and by CPU class (for example, the NEON
kernel has a variant that accumulates with the `FEAT_DotProd` instruction `vdotq_s32`, and the AVX-512 kernel can use
VNNI). All variants return the same integers. The module documentation of `quintrit_core::simd` describes each one.

The runtime's floating-point routines (half-precision dot product and multiply-add, SiLU and SwiGLU, softmax,
RMSNorm) perform the same arithmetic on AVX2 and on NEON: same lanes, same fused operations, same reduction order.
Their unit tests print checksums that are equal on x86-64 and on AArch64. This is what keeps the generated text
the same across instruction sets (see below).

## Reproducible text

With temperature 0 (greedy decoding), the generated text and the logits are identical for every kernel on the same
operating system. The runtime reproduces the arithmetic of the reference runtimes it was checked against, including
their fused multiply-adds; results can differ between platforms whose math library rounds `expf`, `cosf`, `sinf` or
`powf` differently in the last bit, because a transformer amplifies such a difference after a few dozen tokens.

## Requirements

- x86-64: AVX2 for `quintrit`; AVX-512 BW and VBMI for `quintrit_512`.
- AArch64: NEON (part of the architecture); the dot product (`dotprod`) is used where present.
- Any other architecture: the portable kernel.

`quintrit.cpu_features()` in Python and `quintrit_cpu_features()` in C report what was detected.

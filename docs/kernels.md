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

## Block-scaled kernel (`Q2_0`)

Models stored with llama.cpp's `Q2_0` type (one scale per 64 weights) do not go through the kernels above, because a
table kernel with one scale per tensor cannot apply per-block scales. The loader chooses the block-scaled kernel
(`quintrit_core::block_scaled`) for these tensors: the weights are kept in velurex encoding plus one `f32` scale per
64 weights, the activations are quantized in blocks of 32 (`Q8_0`), and the result reproduces llama.cpp's
`ggml_vec_dot_q2_0_q8_0` bit for bit. It has an AVX-512 path (VL, VBMI and VNNI: the weights are decoded with byte
permutes and the products formed with `vpdpbusd`; a 512-bit register holds both of ggml's accumulator chains, so it
takes two activation blocks per step without changing the order of any sum), an AVX2 path, a NEON path (AArch64 with
the dot product) and a portable path, which all give the same bits. The other numerical routines these models need
(F32, BF16 and `Q8_0` dot products, the F16 x F32 attention dot product and value accumulation) have AVX-512 and NEON
paths with the same property, and the `Q8_0` quantization of the activations and the Walsh-Hadamard transform have
AVX-512 paths.

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
`powf` differently in the last bit, because a transformer amplifies such a difference after a few dozen tokens. Outside Windows the runtime
evaluates `log1pf` with its own copy of the x86-64 glibc routine, so x86-64 and AArch64 Linux agree.

## GPU kernels (NVIDIA)

The optional crate `quintrit-cuda` runs the same encoding on NVIDIA GPUs of compute capability 8.0 or newer
([`guides/gpu.md`](guides/gpu.md)). Its projection kernel reads the weights in the planar layout of
`quintrit_core::pack::pack_matrix` (a row is a sequence of 160-trit blocks of 32 bytes; byte `j` of a block holds the
trits of the columns `p·32 + j` for the five planes `p`), decodes a weight byte through a small table in shared memory and
multiplies with `dp4a` against the `int8` activations: one 32-bit weight word (four bytes) per lane and step, several rows
per warp so that the activations are read once for all of them. The digit table has four bytes per entry and the fifth digit
is computed as `(b · 781) >> 16`, which halves the shared-memory wavefronts. Like the CPU kernels it returns the exact
`int32` product `Σ w·x`, which the tests compare with a brute-force product for random, all `+1`, all `-1` and all `0` weights
and for extreme activations including `-128`.

- **Epilogues.** The projection kernels have fused epilogues: the gate and up projection (rows interleaved) applies SiLU or
  squared ReLU and collects the maximum for the next quantization, and the stacked query, key and value projection applies
  the rotary embedding and appends the key and value to the F16 KV cache.
- **Configuration by device resources.** Rows per warp, warps per block, how the first weight word and the activations are
  loaded: the best choice depends on what the GPU has, so the number of multiprocessors and the size of the L2 cache select
  the nearest of the measured profiles (one for a GPU with 24 multiprocessors and 24 MiB of L2, one for 46 multiprocessors and
  4 MiB). A row count that a configuration cannot tile falls back to a smaller one that is correct on every device. All
  configurations return the same integers.
- **Layer kernels.** Norm with quantization, residual add, attention, the final norm and the F16 output head reproduce the
  CPU runtime's arithmetic, so the whole forward pass is bit-identical to the CPU path. The attention follows the CPU
  runtime's flash attention, including the split of the padded key-value axis into four chunks beyond 256 positions.
  The four chunks of a split axis run in blocks of their own (one block per query head and chunk), so a long context
  keeps several multiprocessors busy; within a block four lanes compute the score of one position and the chain of the
  half-precision accumulator, which is sequential per element by definition, runs two elements per thread. The last block
  that finishes for a head combines the chunk results in the CPU's order.
- **Layout.** The weights are converted into this layout while a model is loaded; the layout is a run-time representation
  and not a file format.

## Requirements

- x86-64: AVX2 for `quintrit`; AVX-512 BW and VBMI for `quintrit_512`.
- AArch64: NEON (part of the architecture); the dot product (`dotprod`) is used where present.
- Any other architecture: the portable kernel.
- NVIDIA GPU (optional): compute capability 8.0 or newer, CUDA Toolkit 13.x to build.

`quintrit.cpu_features()` in Python and `quintrit_cpu_features()` in C report what was detected.

## Compiler flags

No compiler flags are needed. Every code path that depends on the instruction set is compiled for it explicitly and
chosen by CPU detection when the program runs: the kernels and also the plain code around them (attention,
normalization, activations, half-precision conversions, the recurrent layers of hybrid models). A build without
`RUSTFLAGS` gives the same results as a build with `-C target-cpu=native` and runs the same instructions on the hot
paths.

What a build for one CPU model still gains is the compiler's own tuning of the kernels (instruction scheduling,
unrolling, inlining across function boundaries). It is worth a few percent. Decoding with the same models and thread
count, a default build reaches about 97 % of the speed of a build with `-C target-cpu=x86-64-v3` on an Intel Core
i7-7700 (AVX2), 90 to 95 % of a build with `-C target-cpu=znver4` on an AMD EPYC 9V45 (AVX-512) and 100 to 104 % of a build with
`-C target-cpu=neoverse-n2` on an Arm Neoverse-N2. For the last few percent build with `-C target-cpu=native`, or with `x86-64-v3` (AVX2) or `x86-64-v4`
(AVX-512) for a binary that runs on other machines of the same class; such a binary needs that instruction set on every
machine it runs on, a default build does not.

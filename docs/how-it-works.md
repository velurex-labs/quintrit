# How Quintrit works

A short, non-mathematical tour of the ideas behind Quintrit: why ternary models are limited by memory, how five weights fit into one byte, and how the kernels read them without division. The exact definitions are in [`velurex-encoding.md`](velurex-encoding.md), the instruction sequences in [`simd-decode.md`](simd-decode.md), the code layout in [`architecture.md`](architecture.md).

## The memory wall of 1.58-bit models

In 1.58-bit language models (BitNet b1.58, Falcon3 and others) every weight is ternary: $w \in \{-1, 0, +1\}$. During text generation, the multiplication of an activation vector by a weight matrix (GEMV) simplifies from floating-point multiplications into additions and subtractions:

$$y_j = \sum_{i} w_{ji} \cdot x_i \quad \text{where} \quad w_{ji} \in \{-1, 0, +1\}$$

With so little arithmetic left, generating a token on a modern CPU is limited by **memory bandwidth**: every weight of every layer is read from RAM once per token. A model that stores its weights in fewer bytes is read faster.

## The storage gap: 2.00 vs. 1.60 bits per weight

A ternary value carries $\log_2(3) \approx 1.585$ bits. The common 2-bit formats (such as `I2_S`) pack only four ternary weights into a byte, that is 2.00 bits per weight, and leave most of the byte's states unused ($3^4 = 81$ of $2^8 = 256$).

Quintrit packs **five weights per byte**, because

$$3^5 = 243 \le 256$$

- **1.60 bits per weight**, which uses 99.1 % of what a byte can hold.
- **20 % less memory traffic**: 200 MB instead of 250 MB per billion weights and token.
- **Up to 25 % higher speed** where memory bandwidth is the limit, because 20 % less data is streamed.

This encoding is called **velurex encoding**. The models are not converted in advance: Quintrit reads standard GGUF files and transcodes the weights into its layout in RAM when a model is loaded.

## Decoding without division

Packing five ternary weights into a byte is traditionally done with the base-3 powers $(1, 3, 9, 27, 81)$. Read as a plain integer, such a byte needs integer division and remainder to separate the digits, and SIMD vector lanes have neither. llama.cpp's `TQ1_0` therefore stores the byte as a base-3 fixed-point fraction and extracts every weight on its own, with a byte multiplication by a power of 3, a multiplication by 3 and a shift.

Velurex encoding uses a different radix structure for vector hardware. It needs no division or remainder and splits each byte at once into two digit groups (two and three weights). The weights are decoded across vector lanes with only bitwise shifts, byte averages (`vpavgb`, `vrhadd`) or shuffles (`vpermb`), and the two groups index directly into fused look-up tables (LUT) that evaluate integer dot products against `int8` activations with no rounding error.

The complete definition and derivation are in [`velurex-encoding.md`](velurex-encoding.md).

## Exact arithmetic

The kernels work on the full `int8` activation range by splitting each activation as $x = 16 \cdot \text{hi} + \text{lo}$ and using fused tables. The results are identical, bit for bit, on AVX2, AVX-512 and ARM NEON, and on the GPU. The test suite compares every kernel with a brute-force product written in plain scalar code. See [`kernels.md`](kernels.md).

## Where the speed comes from, and where it does not

The kernels beat a 2-bit layout where the memory is the limit (several cores, a slow memory system) and are slower where one core is limited by computation. A whole token also runs the output projection and the attention, which are the same for every encoding, so the gain per token is smaller than the gain of the kernel alone. The measurements, the memory ceilings and the method are in [`benchmarks.md`](benchmarks.md); the decode speed of whole models is in [`../model_benchmarks/`](../model_benchmarks/README.md).

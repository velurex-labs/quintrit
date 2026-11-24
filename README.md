# Quintrit

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![Rust Version](https://img.shields.io/badge/rustc-1.98%2B-orange.svg)](https://blog.rust-lang.org/)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20AVX--512%20%7C%20NEON-brightgreen.svg)](#key-features)
[![Format](https://img.shields.io/badge/encoding-velurex%201.60%20bpw-blueviolet.svg)](docs/velurex-encoding.md)

**High-performance, memory-efficient inference for ternary Large Language Models (1.58-bit LLMs).**

`LLM Inference` &bull; `Ternary Computing` &bull; `1.58-bit Weights` &bull; `SIMD Kernels` &bull; `GGUF` &bull; `Rust / C++ / Python`

---

Quintrit implements **"velurex encoding"**, packing five ternary weights into a single byte (**1.60 bits per weight**) to achieve a **20% memory bandwidth reduction** compared to conventional 2-bit formats. Coupled with division-free SIMD decoding and exact Look-Up Table (LUT) kernels for AVX2, AVX-512, and ARM NEON, Quintrit delivers higher inference throughput while directly reading standard GGUF models.

---

## Ternary Computing & 5 Trits per Byte

### The Memory Wall of 1.58-bit LLMs
In 1.58-bit Large Language Models (such as BitNet b1.58 and Falcon3), neural network weights are restricted to ternary values: $w \in \{-1, 0, +1\}$. During autoregressive token generation, multiplying an activation vector by a weight projection matrix (GEMV) simplifies from expensive floating-point multiplications into pure additions and subtractions:

$$y_j = \sum_{i} w_{ji} \cdot x_i \quad \text{where} \quad w_{ji} \in \{-1, 0, +1\}$$

Because arithmetic computation is minimized, autoregressive token generation on modern CPUs is fundamentally bottlenecked by **memory bandwidth**: every weight parameter across all transformer layers must be streamed from RAM to CPU registers once per generated token.

### The Storage Gap: 2.00 bpw vs. 1.60 bpw
The theoretical information content of a ternary state is $\log_2(3) \approx 1.585$ bits. However, conventional 2-bit formats (such as Microsoft's `I2_S`) pack only 4 trits into an 8-bit byte ($2.00$ bits per weight). This leaves nearly 70% of available byte states unused ($3^4 = 81$ states out of $2^8 = 256$).

Quintrit bridges this gap with **velurex encoding**, packing **5 trits per byte**:

$$3^5 = 243 \le 256$$

- **1.60 bits per weight:** Utilizes 99.1% of theoretical byte capacity.
- **20% less memory traffic:** Reduces weight data transfer from 250 MB down to 200 MB per billion weights per token.
- **25% higher theoretical throughput ceiling:** Because memory bandwidth dictates decoding speed, streaming 20% less data allows up to 25% higher token generation rates under memory-bound conditions.

### Division-Free SIMD Decoding
Packing five ternary weights into a byte is traditionally done using standard base-3 powers $(1, 3, 9, 27, 81)$. Read as a plain integer, such a byte needs integer division and modulo arithmetic to separate the digits, and SIMD vector lanes have neither. llama.cpp's `TQ1_0` therefore stores the byte as a base-3 fixed-point fraction and extracts every weight on its own, with a byte multiplication by a power of 3, a multiplication by 3 and a shift.

Velurex encoding departs from the traditional $(1, 3, 9, 27, 81)$ scheme. By employing an algebraic radix structure specifically engineered for vector hardware, it eliminates integer division and remainder operations and splits each byte at once into two digit groups (two and three weights). Weights decode across vector lanes using only fast bitwise shifts, byte averages (`vpavgb`, `vrhadd`), or vector shuffles (`vpermb`), and the two groups index directly into fused Look-Up Table (LUT) kernels that evaluate integer dot products against `int8` activations with zero rounding error.

The complete mathematical definition and derivation are documented in [`docs/velurex-encoding.md`](docs/velurex-encoding.md).

---

## Key Features

- **Exact Integer Arithmetic:** Operates on the full activation range using a fused $x = 16 \cdot \text{hi} + \text{lo}$ split. Yields identical, bit-exact results across AVX2, AVX-512, and ARM NEON.
- **Native GGUF Compatibility:** Directly loads standard GGUF models containing `I2_S`, `TQ1_0`, `TQ2_0` or `Q2_0` tensors and transcodes them in memory into velurex-encoded SIMD layouts (`Q2_0`, which has a scale per 64 weights, keeps its scales next to the trits). Supported architectures: BitNet b1.58, LLaMA and Falcon3 (including Falcon-E), Qwen3, Qwen3.5 and Nemotron-H MoE. This guarantees instant, zero-conversion compatibility with existing models from the `llama.cpp` and BitNet ecosystems without requiring separate weight distribution.
- **NVIDIA GPUs (optional):** The same encoding runs on GPUs of compute capability 8.0 or newer with results bit-identical to the CPU path, from Rust, C, C++ and Python ([`docs/guides/gpu.md`](docs/guides/gpu.md)).
- **Multi-Platform SIMD:** Hand-optimized kernels for x86-64 (AVX2, AVX-512 with VBMI/VNNI) and AArch64 (ARM NEON with DotProd and I8MM).
- **Unified C / C++ / Python / Rust API:** Rust workspace (`quintrit-core`, `quintrit-format`, `quintrit-runtime`), C ABI (`libquintrit`), and Python bindings (`quintrit`).

---

## Benchmarks

The quintrit kernels are compared with faithful Rust transcriptions of Microsoft BitNet's kernels (`mad` on AVX2 and NEON, its AVX-512 widening `mad512`) in the same harness, on the same machines, over the weight-streaming pattern of one Falcon3-10B decode step. Before every measurement, the runtime's 256 greedy tokens and checked logits were bit-identical to the BitNet reference runtime.

### Excerpt: GEMV kernel throughput (Falcon3-10B shapes, weights from DRAM)

| Machine | Instruction set | Threads | BitNet kernel | Quintrit kernel | Relative speed |
|---|---|---:|---:|---:|---:|
| AMD EPYC 9V45 | AVX-512 (vs. `mad512`) | 4 | 164.2 Gweights/s | **221.6 Gweights/s** | **1.35×** |
| Intel Core i7-7700 | AVX2 | 4 | 97.9 Gweights/s | **146.9 Gweights/s** | **1.50×** |
| Raspberry Pi 5 | NEON | 4 | 40.7 Gweights/s | **55.1 Gweights/s** | **1.35×** |
| Raspberry Pi 5 | NEON | 1 | 24.5 Gweights/s | 17.3 Gweights/s | 0.71× |

The 5-trit kernels win where the BitNet kernel is close to the memory ceiling of the machine and lose where one core is compute-bound. These are kernel rates; a whole token also runs the F16 output projection and attention, which are the same for both encodings, so the gain per token is smaller. The full table, the method and the memory ceilings are in [`docs/benchmarks.md`](docs/benchmarks.md).

The decode speed of whole models is reported in [`model_benchmarks/`](model_benchmarks/README.md): 17 ternary language models on four CPUs and two NVIDIA GPUs, with one page per model that holds the thread scaling, the long-prompt figures, the measurement conditions and the command to reproduce the numbers.

---

## Documentation

Full documentation and guides are available in [`docs/`](docs/):

| Document | Description |
|---|---|
| [`docs/velurex-encoding.md`](docs/velurex-encoding.md) | Mathematical definition, radix structure, and division-free SIMD decoding |
| [`docs/simd-decode.md`](docs/simd-decode.md) | Per-ISA weight decode (AVX2, AVX-512, NEON): instruction sequences and bit tricks |
| [`docs/architecture.md`](docs/architecture.md) | System architecture, crate layout, tile formats, and memory pipeline |
| [`docs/kernels.md`](docs/kernels.md) | Kernel implementations, exact arithmetic, and supported instruction sets |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Benchmarking methodology, memory ceiling analysis, and hardware reports |
| [`model_benchmarks/`](model_benchmarks/README.md) | Decode speed of 17 ternary models on four CPUs and two NVIDIA GPUs, one page per model |
| [`docs/guides/`](docs/guides/) | Guides for Rust, C/C++, Python, the CLI and running on an NVIDIA GPU |

### Building & Testing

The Rust workspace is located in [`rust/`](rust/). To compile and execute tests without needing models or network access:

```bash
cd rust
cargo test --workspace
```

The kernel tests compare every SIMD kernel bit for bit with a brute-force product written in plain scalar code on standard CPU operations, without SIMD or any of the library's packing and lookup tables.

---

## Roadmap

Planned milestones and future developments:

- **Official Distribution:** Publish packages on [crates.io](https://crates.io) and [PyPI](https://pypi.org) with pre-built binary wheels for Linux, Windows, and macOS.
- **Pre-Tiled Native Format & Direct `mmap`:** Introduce a dedicated file format / GGUF tensor type extension with pre-tiled velurex weights, enabling direct zero-copy memory mapping (`mmap`) straight from disk without startup transcoding or RAM allocation.
- **Hardware Acceleration:** Further GPU backends, and faster kernels for the block-scaled `Q2_0` models on the NVIDIA GPU (they are bit-exact, but their projection kernels have not been tuned yet).
- **RISC-V Support:** Native kernels for RISC-V processors implementing the RVA23 profile, built on the mandatory RISC-V Vector Extension (RVV 1.0).
- **Extended Benchmarks:** Per-model benchmarks on additional CPU microarchitectures, including Apple processors (Apple Silicon M-series), alongside dedicated performance benchmarks for long context sequences.
- **Edge Efficiency:** Telemetry-backed energy consumption profiling (Joules per decoded token) across edge platforms.
- **Hardware-Level Implementation (FPGA & ASIC):** Explore dedicated digital architecture and hardware-level implementation possibilities (FPGA prototyping and ASIC designs) leveraging division-free velurex decoding for custom ternary accelerator cores.

---

## Contributing, Citation & License

- **Contributing:** Guidelines for reporting issues, contributing code, and running tests are in [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **License:** Quintrit is licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details.
- **Citation:** If you use Quintrit or the velurex encoding in your research or project, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).

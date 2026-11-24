# Quintrit

[![License](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![SIMD](https://img.shields.io/badge/SIMD-AVX2%20%7C%20AVX--512%20%7C%20NEON-brightgreen.svg)](docs/kernels.md)
[![Encoding](https://img.shields.io/badge/weights-1.60%20bits-blueviolet.svg)](docs/how-it-works.md)

**Run ternary (1.58-bit) language models fast on ordinary CPUs.** One small program for the command line, and libraries
for Python, C, C++ and Rust. From a Raspberry Pi to a server, no GPU needed.

`LLM Inference` &bull; `Ternary Computing` &bull; `1.58-bit Weights` &bull; `SIMD Kernels` &bull; `GGUF` &bull; `Rust / C++ / Python / CUDA`

## Try it

No Rust, no compiler, no Python: download the program and a small model, and ask a question.

```bash
# Linux x86-64 (ARM64 and Windows: see examples/cli)
curl -L https://github.com/velurex-labs/quintrit/releases/latest/download/quintrit-linux-x86_64.tar.gz | tar xz
curl -L -o model.gguf https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf/resolve/main/Ternary-Bonsai-1.7B-Q2_0.gguf

./quintrit-linux-x86_64/quintrit --model model.gguf --prompt "What is ternary computing?" --chat-template --max-tokens 100
```

The same program also chats (`--chat`) and measures the speed of your machine (`--bench`). The step-by-step walkthrough
for Linux, ARM and Windows is in [`examples/cli/`](examples/cli/README.md).

## Why Quintrit

- **Built for ordinary CPUs.** Quintrit targets conventional desktop, server and edge CPUs, where generating a token is
  limited by how fast memory delivers the weights, not by arithmetic: the cores keep up with the data. So the layout
  matters. Weights take 1.60 bits each instead of 2.00, which means up to 20 % less memory traffic per token. (On a
  single slow core the arithmetic can become the limit; the [benchmarks](docs/benchmarks.md) show where.) Kernels for
  AVX2, AVX-512 and ARM NEON are chosen when the program starts. [How it works](docs/how-it-works.md)
- **Reads the models you already have.** Standard GGUF files (`I2_S`, `TQ1_0`, `TQ2_0`, `Q2_0`) are loaded directly:
  BitNet b1.58, Falcon3 and Falcon-E, Llama 3 1.58, Qwen3-based Ternary Bonsai and Nemotron-H.
- **Exact.** The results are identical on AVX2, AVX-512, NEON and the GPU, and match the reference implementations
  token for token. A faster kernel never trades away correctness.
- **Easy to embed.** One engine with a command-line tool, a Python package, a C library with a C++ wrapper, and Rust
  crates. An optional NVIDIA GPU backend gives the same results.

## Use it in your code

Python (`pip install quintrit`):

```python
import quintrit

model = quintrit.Model.load("model.gguf")
print(model.chat([{"role": "user", "content": "What is ternary computing?"}], max_tokens=64))
```

C++ (header `quintrit.hpp` and the library from the `quintrit-c-<platform>` archive of a release):

```cpp
#include <iostream>
#include "quintrit.hpp"

int main() {
    quintrit::Model model("model.gguf");
    quintrit::Session session = model.create_session();
    quintrit::GenerateOptions options;
    options.max_tokens = 64;
    session.generate(model.format_chat({{"user", "What is ternary computing?"}}), options,
                     [](const char* text, std::size_t len) { std::cout.write(text, len); return true; });
}
```

Rust (crate `quintrit-runtime`):

```rust
use quintrit_runtime::{InferenceEngine, SamplingConfig, TernaryKernelType};

let mut engine = InferenceEngine::load_from_gguf("model.gguf", 4, Some(TernaryKernelType::QuintritLut))?;
engine.generate_streaming("Ternary computing is", 64, &SamplingConfig::greedy(), |piece| print!("{piece}"))?;
```

Guides: [Python](docs/guides/python.md), [C and C++](docs/guides/c-cpp.md), [Rust](docs/guides/rust.md),
[command line](docs/guides/cli.md). Runnable programs are in [`examples/`](examples/README.md).

## What runs, and how fast

Generated tokens per second on the CPU (median of 5 runs, 256 greedy tokens, all hardware threads), measured with the
`--bench` command above:

| Model | Core i7-7700<br>AVX2 | EPYC 9V45<br>AVX-512 | Raspberry Pi 5<br>NEON |
|---|---:|---:|---:|
| [Falcon-E 1B](model_benchmarks/falcon-e-1b-instruct.md) | 46.3 | 78.9 | 20.6 |
| [BitNet b1.58 2B4T](model_benchmarks/bitnet-b1.58-2b-4t.md) | 22.6 | 37.8 | 9.6 |
| [Falcon3 1.58bit 10B](model_benchmarks/falcon3-10b-instruct-1.58bit.md) | 9.0 | 14.6 | 3.6 |

Seventeen models on four CPUs and two NVIDIA GPUs, with the thread scaling and the measurement conditions of each
number, are in [`model_benchmarks/`](model_benchmarks/README.md).

| Platform | Program and C library | Python package |
|---|---|---|
| Linux x86-64 (AVX2, AVX-512) | yes | yes |
| Linux ARM64 (NEON; tested on Raspberry Pi 5 and Neoverse-N2) | yes | yes |
| Windows x86-64 | yes | yes |
| macOS (Apple silicon, Intel) | build from source; not tested yet | not tested yet |
| NVIDIA GPU (compute capability 8.0 or newer) | build from source with the CUDA toolkit ([guide](docs/guides/gpu.md)) | build from source |

Good to know:

- Quintrit runs ternary models. It does not run ordinary 4-bit or 8-bit models.
- A prompt is processed one token at a time (there is no batched prompt processing yet), so a very long prompt takes a
  while before the first word appears.
- Models are not part of this repository; take them from their publishers and read their terms.

## Documentation

| Document | Content |
|---|---|
| [`docs/how-it-works.md`](docs/how-it-works.md) | Why ternary models are limited by memory, and how five weights fit into one byte |
| [`docs/guides/`](docs/guides/) | Guides for the command-line tool, Python, C and C++, Rust, and the GPU |
| [`examples/`](examples/README.md) | Runnable programs, starting with the one that needs no installation |
| [`model_benchmarks/`](model_benchmarks/README.md) | Decode speed of 17 models on four CPUs and two GPUs |
| [`docs/benchmarks.md`](docs/benchmarks.md) | Kernel comparisons, memory ceilings and how to reproduce the measurements |
| [`docs/velurex-encoding.md`](docs/velurex-encoding.md) | The exact definition of the encoding |
| [`docs/architecture.md`](docs/architecture.md), [`docs/kernels.md`](docs/kernels.md), [`docs/simd-decode.md`](docs/simd-decode.md) | Crates, kernels and the instruction sequences |

To build from source (Rust 1.98 or newer): `cd rust && cargo test --workspace` runs the tests, which need no model and no
network, and `cargo build --release -p quintrit-runtime --bin quintrit` builds the program.

## Roadmap

Planned, in no particular order:

- macOS and Apple silicon: tests, wheels, program archives and benchmarks.
- A pre-tiled weight format with direct memory mapping, so that a model loads without conversion.
- RISC-V kernels for processors with the vector extension (RVV 1.0).
- Faster kernels for the block-scaled `Q2_0` models on the NVIDIA GPU, and further GPU backends.
- Energy per generated token on edge devices.
- Hardware-level implementations (FPGA prototypes, ASIC designs) that use the division-free decoding.

## Contributing, Citation & License

- **Contributing:** Guidelines for reporting issues, contributing code, and running tests are in [`CONTRIBUTING.md`](CONTRIBUTING.md).
- **Conduct and security:** Participation follows the [`Code of Conduct`](CODE_OF_CONDUCT.md); report a vulnerability as described in [`SECURITY.md`](SECURITY.md), not in a public issue.
- **License:** Quintrit is licensed under the Apache License, Version 2.0. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE) for details.
- **Citation:** If you use Quintrit or the velurex encoding in your research or project, please cite it using the metadata in [`CITATION.cff`](CITATION.cff).
- **Maintainer and contact:** Quintrit is developed by Velurex Labs. Contact: <quintrit@velurex-labs.com>.

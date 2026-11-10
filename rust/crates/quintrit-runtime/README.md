# Quintrit Runtime (`quintrit-runtime`)

Ultra-lean, high-performance local transformer inference engine and CLI in pure Rust without external C++ or GGML runtime dependencies.

---

## Architectural Principles

### 1. Pure Rust Inference
- **Zero C++ / Python Dependencies**: Compiles with standard `cargo build --release` on Linux, macOS, and Windows.
- **Hardware SIMD Acceleration**: Dispatches matrix-vector multiplication directly to [`quintrit-core`](../quintrit-core) (AVX2 on x86_64, NEON on AArch64 / Raspberry Pi 5, with portable scalar fallback).
- **Grouped-Query Attention (GQA)**: Full multi-head and grouped-query attention forward pass with rotary embeddings (RoPE), preallocated contiguous Key-Value cache, and sub-layer normalization.

### 2. Standard GGUF Ingress & In-Memory Transcoding
- **No Proprietary On-Disk Format**: Support for bespoke on-disk model archive formats is intentionally suspended to avoid format churn.
- **Frictionless Community Verification**: Ingests standard, community-verified GGUF models directly (e.g., Microsoft BitNet b1.58 from Hugging Face).
- **Dynamic In-Memory Transcoding**: Weight matrices in GGUF (`GGML_TYPE_I2_S`, 2.00 bpw planar) are transcoded on-the-fly during startup into 1.60 bpw **velurex encoding** 32-row tiled layouts in RAM via [`quintrit-format`](../quintrit-format).
- **20.0% Exact DRAM Bandwidth Reduction**: Once resident in RAM, weights occupy canonical 1.60 bpw (5 trits per byte), saving memory bandwidth on memory-bound autoregressive decoding passes.

---

## Library use (quiet by default)

The crate is a library first: loading a model and generating text print nothing. The progress lines of the loader
(`==> Loading GGUF model: ...`) appear only after `quintrit_runtime::set_verbose(true)`, which the `quintrit` CLI and
the benchmarks call at start-up. `VerboseScope` switches the flag for one scope and restores the previous value. The
flag is process-wide; set it once if you load several models from different threads.

```rust,no_run
use quintrit_runtime::{InferenceEngine, SamplingConfig, TernaryKernelType};

fn main() -> Result<(), String> {
    // quintrit_runtime::set_verbose(true); // uncomment to see the loading progress
    let mut engine = InferenceEngine::load_from_gguf(
        "models/BitNet-b1.58-2B-4T.gguf",
        /* threads */ 4,
        Some(TernaryKernelType::QuintritLut),
    )?;
    let text = engine.generate_streaming(
        "<|user|>\nWhat is ternary computing?\n<|assistant|>\n",
        64,
        &SamplingConfig::greedy(),
        |piece| print!("{piece}"),
    )?; // an error if the prompt does not fit into the context window
    println!("\n{} characters generated", text.len());
    Ok(())
}
```

---

## Kernels

`--kernel` selects the GEMV kernel: `quintrit` (default; AVX2 and NEON, 32-row tiles) or `quintrit_512` (AVX-512,
64-row tiles). Both use velurex encoding (5 trits per byte). See the workspace README, section
[Kernels](../../../docs/kernels.md).

The cargo feature `bitnet-baseline` (off by default) adds two comparison kernels from the crate `quintrit-bitnet`,
`bitnet_mad` (BitNet's planar multiply-add kernel, AVX2 and NEON) and `bitnet_mad_512` (its widening to AVX-512), both
with 4 trits per byte. They exist so that the end-to-end benchmarks can compare the 4-trit encoding with the 5-trit
encoding; the C library and the Python package do not include them. Build with
`cargo build --release -p quintrit-runtime --features bitnet-baseline`.

---

## Modules

- **`config`**: Architecture and hyperparameter detection from GGUF metadata (BitNet b1.58, Falcon3, LLaMA).
- **`ops`**: High-performance mathematical primitives (RMSNorm, RoPE, Int8 activation quantization, SiLU, Squared ReLU, SwiGLU, FP16 conversions).
- **`kv_cache`**: Contiguous preallocated Key-Value cache storing FP16 projections.
- **`attention`**: Grouped-Query Attention with causal masking and sub-layer normalization.
- **`layer`**: Unified ternary linear layer container dispatching to the 1.60 bpw Velurex tiled GEMV kernels.
- **`model`**: Full transformer forward execution graph.
- **`sampler`**: Autoregressive sampling strategies (greedy argmax, temperature, top-k, nucleus top-p).
- **`tokenizer`**: HuggingFace BPE tokenizer loaded directly from GGUF embedded metadata (control tokens such as `<|endoftext|>` keep their ids) or external `tokenizer.json`, and `StreamDecoder`, which decodes generated tokens one at a time without splitting multi-byte UTF-8 characters.
- **`engine`**: High-level engine coordinating model loading, prompt ingestion, autoregressive token streaming, and empirical benchmarking. A generation call reuses the KV cache of the previous call for the tokens the two prompts share, so a chat that re-sends its history computes only the new turn.
- **`verbosity`**: The process-wide switch for load-time progress output (`set_verbose`, `is_verbose`, `VerboseScope`); off by default.

---

## CLI Quickstart (`quintrit`)

The command-line tool prints loading progress and the generated text to the console. Run text generation from the command line:

```bash
# Autoregressive text generation
cargo run --release -p quintrit-runtime --bin quintrit -- \
  --model models/BitNet-b1.58-2B-4T.gguf \
  --prompt "What is ternary computing?" \
  --max-tokens 64

# Interactive multi-turn chat session
cargo run --release -p quintrit-runtime --bin quintrit -- \
  --model models/BitNet-b1.58-2B-4T.gguf \
  --chat

# Throughput and latency benchmark pass
cargo run --release -p quintrit-runtime --bin quintrit -- \
  --model models/BitNet-b1.58-2B-4T.gguf \
  --bench \
  --max-tokens 128
```

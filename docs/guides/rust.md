# Using Quintrit from Rust

Rust 1.98 or newer (edition 2024).

Three crates, three levels. Use the one you need; each depends only on the ones below it.

| Crate | Use it to |
|---|---|
| [`quintrit-core`](../../rust/crates/quintrit-core/README.md) | pack ternary matrices and multiply them with `int8` activations |
| [`quintrit-format`](../../rust/crates/quintrit-format/README.md) | read GGUF files and convert their ternary tensors into the tiled layout |
| [`quintrit-runtime`](../../rust/crates/quintrit-runtime/README.md) | load a model and generate text |

```toml
[dependencies]
quintrit-core = "0.1"        # kernels
quintrit-runtime = "0.1"     # engine and tokenizer (also pulls in quintrit-format)
```

## Kernel level: a matrix-vector product

The weights are a row-major matrix of `i8` values in `{-1, 0, +1}`; `rows` must be a multiple of 32. The steps are
pack, arrange in tiles, tabulate the activations, multiply.

```rust,ignore
use quintrit_core::lut::{build_activation_lut_wide, gemv_lut_wide_1xn, transform_weights_lut};
use quintrit_core::pack::pack_matrix;

let (rows, cols) = (512, 2048);
let trits: Vec<i8> = vec![1; rows * cols];          // your ternary weights
let activations: Vec<i8> = vec![-3; cols];          // any int8 values, -128 and 127 included

let packed = pack_matrix(&trits, rows, cols);                   // five trits per byte
let tiled = transform_weights_lut(&packed, rows, cols);         // 32-row tiles for AVX2 and NEON
let table = build_activation_lut_wide(&activations, cols);      // exact for the whole int8 range

let mut out = vec![0.0f32; rows];
gemv_lut_wide_1xn(&tiled, &table, rows, cols, &mut out);        // out[r] = sum_c trits[r][c] * activations[c]
```

The complete program, which also checks the result against a plain integer product, is
[`examples/rust/src/bin/gemv.rs`](../../examples/rust/src/bin/gemv.rs). The functions pick the best instruction set of
the CPU at run time. `gemv_lut_wide_parallel` runs the same product on several threads; the AVX-512 variants
(`gemv_quintrit_lut_64_*`) read 64-row tiles. See [`../kernels.md`](../kernels.md).

## Format level: a GGUF file

```rust,ignore
use quintrit_format::GgufFile;

let gguf = GgufFile::open("model.gguf")?;       // bounds-checked; a damaged file is an error, not a panic
println!("{:?}", gguf.architecture());
for (name, info) in &gguf.tensors {
    println!("{name}: type {}", info.tensor_type);
}
```

The decoders (`I2_S`, `TQ1_0`, `TQ2_0`) and the transcoder into the tiled layout are described in the
[crate README](../../rust/crates/quintrit-format/README.md).

## Runtime level: generate text

```rust,ignore
use quintrit_runtime::{InferenceEngine, SamplingConfig, TernaryKernelType};

let mut engine = InferenceEngine::load_from_gguf("model.gguf", /* threads */ 4, Some(TernaryKernelType::QuintritLut))?;
let text = engine.generate_streaming(
    "<|user|>\nWhat is ternary computing?\n<|assistant|>\n",
    64,                                   // new tokens at most
    &SamplingConfig::greedy(),            // deterministic
    |piece| print!("{piece}"),            // called with every piece of text
)?;                                       // an error if the prompt does not fit into the context window
```

Loading is silent. Call `quintrit_runtime::set_verbose(true)` first to see the progress lines. The complete program is
[`examples/rust/src/bin/generate.rs`](../../examples/rust/src/bin/generate.rs); build and run the examples with

```bash
cd examples/rust
cargo run --release --bin gemv
cargo run --release --bin generate -- path/to/model.gguf
```

For the AVX-512 kernel pass `Some(TernaryKernelType::QuintritLutAVX512)` on a CPU that has it.

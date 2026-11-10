# Architecture

## Crates

```
rust/
├── crates/
│   ├── quintrit-core      kernels: velurex encoding, activation tables, tiled layouts, SIMD GEMV, threading
│   ├── quintrit-format    GGUF reader, ternary tensor decoders, transcoder into the tiled layout
│   ├── quintrit-bitnet    Rust transcription of the 4-trit BitNet kernels, the comparison baseline (Rust only, not published)
│   ├── quintrit-runtime   transformer engine, KV cache, tokenizer, sampling, the `quintrit` command-line tool
│   ├── quintrit-c         C ABI (`quintrit_c` library, `quintrit.h`) and the header-only C++ wrapper `quintrit.hpp`
│   └── quintrit-py        Python package `quintrit` (PyO3 and maturin; a standalone package outside the workspace)
```

The dependencies point one way: `quintrit-core` depends on nothing but `rayon`; `quintrit-format` and
`quintrit-runtime` build on it; the C and Python packages wrap all three levels. `quintrit-bitnet` also builds on
`quintrit-core` and is used only for comparisons; the C and Python packages do not include it.

The bindings expose the same three levels:

| Level | Content | Rust | C and C++ | Python |
|---|---|---|---|---|
| Kernel | pack, unpack, matrix-vector product on `int8` activations | `quintrit-core` | `quintrit_pack_matrix`, `quintrit_gemv_*` | `quintrit.pack`, `quintrit.gemv` |
| Format | open a GGUF file, list tensors, read and convert ternary tensors | `quintrit-format` | `quintrit_gguf_*` | `quintrit.open`, `GgufFile` |
| Runtime | load a model, tokenize, generate, stream | `quintrit-runtime` | `quintrit_model_*`, `quintrit_session_*` | `quintrit.Model`, `Tokenizer` |

## From a GGUF file to the first token

```text
GGUF file ──► quintrit-format ──► tiles in RAM ──► quintrit-core kernels ──► quintrit-runtime
 (I2_S, TQ1_0,   reader and         velurex          activation tables        forward pass,
  TQ2_0)         decoders           encoding         and SIMD GEMV            sampling
```

1. **Read.** `quintrit-format` reads GGUF v2 and v3. The file is treated as untrusted input: counts, lengths, shapes,
   offsets and the alignment are checked before anything is allocated, and a damaged file produces an error and
   never a panic.
2. **Decode.** The ternary tensor types `I2_S` (2.00 bpw), `TQ1_0` and `TQ2_0` are decoded into one trit per byte.
3. **Transcode.** The trits are packed five per byte (velurex encoding, [`velurex-encoding.md`](velurex-encoding.md))
   and arranged in tiles: 32 rows per tile for AVX2 and NEON, 64 rows for AVX-512. This pass runs in parallel over
   the weight matrices when the model is loaded.
4. **Run.** For every token, the activations of a layer are quantized to `int8` and tabulated once; the kernels then
   stream the weight tiles from memory and produce the exact integer products (see [`kernels.md`](kernels.md)).

### Why weights are transcoded in memory (and the path to direct memory-mapping)

#### Current approach: Zero-friction ecosystem compatibility with `llama.cpp`
Quintrit intentionally loads standard, unmodified GGUF files (v2 and v3) distributed across the open-source ecosystem (e.g., Hugging Face repositories for BitNet b1.58, Falcon3, and LLaMA-derived models). These models serialize ternary weights in existing quantization types such as `I2_S` (2.00 bpw), `TQ1_0`, or `TQ2_0`.

By reading these files directly, users can run existing models immediately with zero prior offline conversion, zero risk of third-party weight corruption, and full drop-in compatibility with the `llama.cpp` and BitNet model distribution pipeline.

#### The limitation: Why direct `mmap` is not possible today
Standard GGUF files store ternary weights in planar rows using 2-bit representations (`I2_S`) or non-interleaved blocks. However, Quintrit's high-throughput SIMD kernels require weights in **tiled velurex encoding**:
1. Packed at **1.60 bits per weight** (5 trits per byte).
2. Grouped into cache-aligned 2D tiles: 32 rows per tile for AVX2 and ARM NEON, or 64 rows per tile for AVX-512, with packed partial tails.

Because the byte layout on disk does not match the tiled layout required by the hardware kernels, standard GGUF weights cannot be executed via direct file memory mapping (`mmap`). Consequently, `quintrit-format` memory-maps the GGUF file solely as the input source for an in-memory transcoding pass during model initialization, converting weights into RAM buffers where they occupy 1.60 bits per weight for the lifetime of the session.

#### Future path: Native pre-tiled format and direct `mmap`
For future releases, the roadmap includes introducing a dedicated native file container (`.quintrit`) or extending the GGUF specification with a first-class velurex tensor type:
- **Pre-tiled on-disk layout:** Weight matrices will be serialized on disk already encoded in velurex format and structured according to the exact 32-row or 64-row tile geometry.
- **Direct file memory-mapping (`mmap`):** The runtime will map the pre-tiled weight files directly into the virtual address space.
- **Benefits:**
  - **Instant startup:** Completely eliminates startup transcoding overhead—models begin generating tokens immediately.
  - **Zero RAM allocation for weights:** The operating system's unified page cache streams pages directly from disk/SSD on demand, avoiding duplicate weight copies in user-space RAM.
  - **Process sharing:** Multiple local inference processes or workers can map the identical physical file pages concurrently without multiplying system RAM usage.

## The tile layout

A kernel reads weights from tiles. A tile row holds `ceil(cols / 160)` groups; a full group of 160 trits takes 32
bytes. The partial last group of a row is packed five trits per byte as well, so for 3072 columns a row is 616 bytes
(`quintrit_core::pack::tile_row_bytes`) instead of the 640 that padding to full groups would need. The layout is
fixed: there is no switch and no padded variant. A change of the layout would be a format change with a new layout
version.

## Threading

`quintrit-core` runs a matrix-vector product on persistent `rayon` worker pools that can optionally be pinned to
cores. The C and Python packages use the same pools, and the number of threads can be set in every API.

## Quiet by default

The libraries never write to the standard output. Loading a model prints progress lines only after the application has
called `quintrit_runtime::set_verbose(true)` (the `quintrit` tool and the benchmarks do), after
`QUINTRIT_MODEL_OPEN_VERBOSE` was passed to the C API, or with `verbose=True` in Python. Errors are returned, never
printed.

## Where the design comes from

- GGUF and the `TQ1_0` and `TQ2_0` layouts are the formats of ggml and llama.cpp; `I2_S` is the format of BitNet. The
  runtime reproduces their floating-point arithmetic so that results are bit-identical to the reference runtimes. The
  credits are in [`NOTICE`](../NOTICE).
- The kernels are exact: every kernel and every instruction set returns the same integers for the same input.

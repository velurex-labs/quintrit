# Quintrit Format & In-Memory Transcoder (`quintrit-format`)

`quintrit-format` provides GGUF model parsing (memory-mapped, bounds-checked) and dynamic on-the-fly weight transcoding into **velurex encoding** (1.60 bpw) 32-row tiled matrix layouts for high-performance CPU inference. It depends on `quintrit-core`, `memmap2` and `rayon`.

---

## Architectural Strategy: Why GGUF Instead of a Custom Proprietary On-Disk Format?

Rather than publishing and enforcing a proprietary on-disk model format at this stage, Quintrit ingests standard GGUF model files (such as Microsoft BitNet b1.58 checkpoints and community ternary models with `GGML_TYPE_I2_S` tensors) directly and transcodes them dynamically into host RAM.

This architectural decision is driven by three foundational considerations:

### 1. Frictionless Verification for the Community
- AI researchers, engineers, and evaluators verifying Quintrit benchmarks should not be forced to download unknown, proprietary binary archives.
- Anyone can take pre-existing, verified GGUF weights directly from Hugging Face or local repositories and evaluate Quintrit immediately.

### 2. Preventing Premature On-Disk Format Churn
Finalizing an immutable on-disk file specification prematurely risks technical debt and repeated breaking changes. The final on-disk layout is intentionally postponed until three key R&D milestones are completed:
- **AVX-512 Kernels**: Evaluating 512-bit vector registers (`zmm`), ternary masking, and optimal block-interleaved bit-widths.
- **GPU Acceleration**: Investigating Compute Shader / Tensor Core layouts where warp coalescing, shared memory bank conflicts, and SIMT alignment dictate memory organization.
- **Alternative Radix Vector Research**: Evaluating alternative radix decomposition schemes beyond canonical Velurex $(1, 3, 9, 28, 84)$—specifically candidate radices such as $(1, 3, 10, ...)$ where an even number like $10$ in the third radix position opens unique arithmetic quotient and bit-shift advantages.

### 3. Operational Trade-Off: In-Memory Transcoding vs. Zero-Copy Mmap
Because weights on disk are stored in standard GGUF formats (e.g. 2.00 bpw planar `GGML_TYPE_I2_S` or unquantized trits), **direct zero-copy memory-mapping (`mmap`) straight into execution registers is not possible**:
- **Startup Phase**: The entire weight tensor must first be read from disk into host memory and transformed on-the-fly into the 1.60 bpw 32-row tiled Velurex representation. This pass is fully parallelized across CPU cores via Rayon.
- **Runtime Phase**: Once resident in RAM, weights occupy canonical 1.60 bpw (5 trits per byte), delivering the full **20.0% DRAM traffic and bandwidth reduction** and executing at maximum speed on SIMD GEMV kernels.

---

## Features

- **GGUF v2/v3 Parser**: Parses GGUF metadata, architecture parameters, and tensor catalogs using memory-mapped files. GGUF is the only model format read.
- **Hardened Against Malformed Files**: Every count, length, shape, offset and the alignment is validated against the real file size with checked arithmetic before anything is allocated (see "Malformed and hostile files" below).
- **Multi-Threaded Transcoding**: Transcodes planar `I2_S` 2.00 bpw tensors into Velurex 1.60 bpw 32-row tiled layouts in parallel across 32-row blocks.
- **100% Bit-for-Bit Parity Verification**: Includes programmatic verification (`verify_transcoded_parity`) proving that decoded weights in RAM match original GGUF trits bit-for-bit.
- **DRAM Footprint Telemetry**: Provides exact memory usage and bandwidth reduction metrics (`dram_savings_pct`).

---

## Usage Example

```rust,no_run
use quintrit_core::lut::{build_activation_lut_wide, gemv_lut_wide_1xn};
use quintrit_format::{load_and_transcode_gguf_tensor, GgufFile};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // 1. Open standard GGUF model file (e.g., BitNet-b1.58-2B)
    let gguf = GgufFile::open("models/bitnet_b1_58-2b.gguf")?;

    // 2. Transcode a 2.00 bpw planar tensor into 1.60 bpw 32-row tiled format in RAM
    let rows = 2048;
    let cols = 2048;
    let layer = load_and_transcode_gguf_tensor(&gguf, "blk.0.attn_q.weight", rows, cols)?;

    println!("Transcoded tensor DRAM footprint: {} bytes", layer.dram_bytes());
    println!("DRAM savings vs 2.00 bpw baseline: {:.2}%", layer.dram_savings_pct());

    // 3. Execute hardware-accelerated SIMD GEMV
    let activations = vec![0i8; cols];
    let qlut = build_activation_lut_wide(&activations, cols);
    let mut logits = vec![0.0f32; rows];

    gemv_lut_wide_1xn(&layer.packed_tiled, &qlut, rows, cols, &mut logits);
    Ok(())
}
```

---

## Malformed and hostile files

`GgufFile::open` / `GgufFile::from_mmap` never trust the file. They return an `std::io::Error` (`InvalidData`, or
`UnexpectedEof` for a truncated file) instead of panicking or allocating on the word of the file when:

- the magic bytes or the version (2 or 3) are wrong;
- the tensor or metadata count cannot fit into the remaining bytes, or a string, key or metadata array is longer than
  the rest of the file (keys and tensor names are limited to 65536 bytes, arrays to four levels of nesting);
- `general.alignment` is not an unsigned power of two, or the data section would start beyond the end of the file;
- a tensor has no dimension or more than eight, more than 2^40 elements, or dimensions whose product overflows;
- a tensor offset is not a multiple of the alignment or lies outside the data section, a payload of a known type
  (`F32`, `F16`, `BF16`, `Q8_0`, `I2_S`, `TQ1_0`, `TQ2_0`) extends beyond the end of the file, or two payloads overlap;
- a metadata key or tensor name occurs twice.

The accessors (`get_tensor_data`, `get_tensor_f32`, ...) return `None` rather than reading out of bounds. A valid file
can still describe a model that is larger than your memory, so load model files from sources you trust. The tests
with crafted files are in [`tests/hostile_files.rs`](tests/hostile_files.rs).

---

## Testing

Run unit and integration tests across all matrix shapes:

```bash
cargo test -p quintrit-format
```

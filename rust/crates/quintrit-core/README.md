# quintrit-core

![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)

High-performance 1.60 bpw ternary quantization microkernel using **velurex encoding** and hardware-accelerated SIMD GEMV.

---

## Highlights

- **Canonical 1.60 bpw Density**: Packs 5 ternary weights `{-1, 0, +1}` into a single byte (`0..=250`) via velurex encoding (`1, 3, 9, 28, 84`), achieving an **exact 20.0% reduction in DRAM bandwidth** compared to 2.00 bpw formats (such as the GGUF `I2_S` tensor type).
- **Zero-Modulo SIMD Decompression**: The radix separation at 28 isolates ternary digits algebraically, eliminating integer modulo and division instructions entirely.
- **Hardware-Accelerated GEMV**: activation lookup table (QLUT) execution on 32-row tiles with AVX2 on x86_64 and NEON on AArch64 (including native ARMv8.2-A `FEAT_DotProd` direct 32-bit `vdotq_s32` acceleration with runtime CPU detection), 64-row tiles with AVX-512 (BW + VBMI) on x86_64, and a portable scalar fallback.
- **One Dependency**: Pure mathematical core; the only third-party crate is `rayon`, which provides the persistent worker pools.
- **Thread-Safe & Embeddable**: Multi-threaded GEMV on persistent, optionally core-pinned Rayon thread pools. The library prints nothing.

---

## Module Layout

| Module | Contents |
|---|---|
| `simd` | Quintrit kernels for AVX2, ARM NEON, AVX-512 and a scalar fallback |
| `lut`, `pack`, `threading` | Velurex encoding, activation tables, tiled layouts, dispatchers and threading |

---

## Mathematical Formulation: Velurex Encoding

Each ternary weight $t_k \in \{-1, 0, +1\}$ is mapped to a balanced positive digit:

$$d_k = t_k + 1 \in \{0, 1, 2\}$$

Five consecutive digits $(d_0, d_1, d_2, d_3, d_4)$ are packed into a single byte $B$:

$$B = d_0 + 3 \cdot d_1 + 9 \cdot d_2 + 28 \cdot d_3 + 84 \cdot d_4$$

- **Value Range**: $0 \le B \le 250$ (Max value: $2 + 6 + 18 + 56 + 168 = 250$).
- **Sentinel Space**: Values $251 \dots 255$ never appear in valid data, providing deterministic error detection.
- **Separation Property**: Shifting the radix from 27 to 28 enables dual-hash table lookups for $(d_3, d_4)$ and $(d_0, d_1)$ alongside pure-ALU vector extraction of $d_2$ on modern SIMD units.

---

## Quickstart

Add `quintrit-core` to your `Cargo.toml`:

```toml
[dependencies]
quintrit-core = "0.1"
```

### Example Usage

```rust
use quintrit_core::lut::{build_activation_lut_wide, gemv_lut_wide_1xn, transform_weights_lut};
use quintrit_core::pack::pack_matrix;

fn main() {
    let rows = 64;
    let cols = 640;

    // 1. Prepare ternary weights {-1, 0, +1} and 8-bit activations [-128..127]
    let trits: Vec<i8> = vec![0; rows * cols];
    let activations: Vec<i8> = vec![1; cols];

    // 2. Pack weights into 5-planar layout (5 trits / byte = 1.60 bpw)
    let packed = pack_matrix(&trits, rows, cols);

    // 3. Transform into 32-row tiled format for cache-friendly GEMV
    let tiled_weights = transform_weights_lut(&packed, rows, cols);

    // 4. Precompute the activation lookup table (QLUT); the wide table is exact for every int8 value
    let qlut = build_activation_lut_wide(&activations, cols);

    // 5. Execute hardware-accelerated GEMV
    let mut output = vec![0.0f32; rows];
    gemv_lut_wide_1xn(&tiled_weights, &qlut, rows, cols, &mut output);
}
```

---

## Verification & Testing

Run unit tests and verification across all 243 states:

```bash
cargo test
```

Run the memory-bandwidth ceiling benchmark (read-only stream over a buffer much larger than the last-level cache,
thread sweep); a kernel can only be judged against what the memory system delivers:

```bash
cargo bench -p quintrit-core --bench memory_bandwidth -- --suffix i7-7700 --label "Intel Core i7-7700"
```

The report is written to `results/memory_bandwidth_<SUFFIX>.md` in the crate directory (ignored by git). The energy per decoded token is measured by the
`token_energy` benchmark of the `quintrit-runtime` crate (Raspberry Pi 5 PMIC rail telemetry).

## Tile layout (fixed)

The LUT kernels read weights from in-memory tiles (32 rows per tile for AVX2 and NEON, 64 rows for AVX-512) that are built from the packed
(planar) layout of `pack_matrix` when a model is loaded. A row of a tile is `pack::tile_row_bytes(cols)` bytes long: the full 160-trit blocks
are copied, the partial last block is re-encoded five trits per byte (rounded up to an even count; for 3072 columns a row is 616 instead of 640
bytes), and the activation-table builders map the tail columns to consecutive activations. Results are bit-identical to the padded layout.

**This tile layout is fixed and cannot be switched off.** If weights are ever stored as finished tiles (for example to memory-map a file and
read it without a transcoding step), the file must contain exactly this layout and carry a layout version in its header; the 32- versus 64-row
tiling is a second property of such a file. The packed layout of `pack_matrix` (what a converter writes) is independent of all this.

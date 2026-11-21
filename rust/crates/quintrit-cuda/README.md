# quintrit-cuda

![License: Apache-2.0](https://img.shields.io/badge/License-Apache--2.0-blue.svg)

Optional NVIDIA GPU backend of Quintrit: CUDA kernels for velurex-encoded ternary matrices (five trits per byte, 1.60 bits per weight) and a small safe layer over the CUDA driver API.

---

## Requirements

- **GPU:** NVIDIA, compute capability 8.0 or newer (Ampere: A100, RTX 30xx, RTX 40xx and newer). Turing (RTX 20xx, GTX 16xx, T4) is not supported; `Device::open` returns `Error::Unsupported` for it.
- **Driver:** any driver that supports your GPU. The driver library is loaded at run time; the crate links no NVIDIA library.
- **Build:** the CUDA Toolkit 13.x (`nvcc`) is needed to compile the kernels. `build.rs` looks for it in `NVCC`, `CUDA_PATH`, `CUDA_HOME`, `PATH` and `/usr/local/cuda*`. Without the toolkit the crate still builds, contains no kernels, and loading one returns `Error::NoKernels`.

The crate is outside the default workspace members. The runtime, the C library and the Python package enable it with their `cuda` feature.

## Build options

| Environment variable | Effect |
|---|---|
| `QUINTRIT_CUDA_SKIP_NVCC=1` | do not compile kernels even if `nvcc` exists |
| `QUINTRIT_CUDA_REQUIRE=1` | fail the build when the kernels cannot be compiled |
| `QUINTRIT_CUDA_GENCODE` | replaces the `-gencode` list (default: SASS for `sm_80`, `sm_86`, `sm_89` and PTX `compute_89`), for example `-gencode=arch=compute_86,code=sm_86` |
| `QUINTRIT_CUDA_PROFILE=<index>` | run time: use a given device profile instead of the nearest one (experiments) |
| `QUINTRIT_CUDA_NVCC_FLAGS` | extra flags for `nvcc`, split at spaces (timing experiments: `-DATTN_STOP=1`, `2` or `3` makes the attention kernel leave after a phase; the results are then wrong) |

## What the crate contains

| Module | Content |
|---|---|
| `device`, `driver` | device query, context, buffers, streams, events, kernel launches and CUDA graphs over the CUDA driver API |
| `pack` | host-side helpers for the weight layout the kernels read (the planar layout of `quintrit_core::pack::pack_matrix`), stacking and row interleaving, the permuted activation layout |
| `gemv` | exact int32 projection kernels with the fused epilogues: gated activation (SwiGLU or squared ReLU) with the block maximum, rotary embedding with the KV append |
| `layer` | normalization with quantization (and the fused residual add), standalone rotary embedding (interleaved or half-split pairing), attention (with the `AttentionWorkspace` that holds the partial results of its chunks), final norm, output head |
| `profile` | kernel configurations by device resources |

The results of the layer kernels are bit-identical to the CPU runtime. The attention follows the CPU runtime's flash attention exactly, including the split of the padded key-value axis into four chunks from a context of 257 tokens on. It supports head dimensions 32, 64, 128 and 256 and contexts up to 7936 positions.

The attention runs one block per query head and chunk of the key-value axis (a launch of `heads x 4` blocks; with a context of up to 256 tokens only the first block of a head works and the others return at once, so a launch or a captured graph serves every position). A block computes the scores of its positions with four lanes per position, the running maximum and the softmax terms of the chunk, and the chain of the half-precision accumulator over its positions; the chunk results go to the `AttentionWorkspace` and the block that finishes last for a head combines them in the CPU's order. All launches that share a workspace must run in order on one stream (one workspace per sequence).

## Kernel configuration

The launch configuration of the projection kernel is chosen by device resources, not by the card's name: the number of streaming multiprocessors and the size of the L2 cache select the nearest of the measured profiles. Every configuration gives the same exact integer results; they differ only in speed. Row counts that a configuration cannot tile fall back to a smaller one that is correct on every device but not tuned.

## Query a device

```rust,no_run
use quintrit_cuda::Device;

fn main() -> Result<(), quintrit_cuda::Error> {
    let device = Device::open(0)?;
    let info = device.info();
    println!(
        "{}: compute capability {}.{}, {} SMs, {} MiB L2",
        info.name, info.compute_capability.0, info.compute_capability.1, info.sm_count, info.l2_cache_bytes >> 20
    );
    Ok(())
}
```

## License

Apache-2.0. See `LICENSE` and `NOTICE`.

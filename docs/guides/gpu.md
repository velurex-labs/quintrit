# Running on an NVIDIA GPU

Quintrit can run the decoder of a ternary model on an NVIDIA GPU. The weights stay in velurex encoding (1.60 bits per
weight) in device memory; the projections are exact integer kernels, and everything around them (norms, rotary
embedding, attention, activation functions, output head) follows the CPU arithmetic operation by operation. The text
and the logits of a model are therefore **bit-identical** to what the CPU path produces for the same model and prompt,
and the GPU needs no tolerance in any test. The GPU backend is optional: the default build of every package contains
no GPU code and needs neither a toolkit nor a GPU.

## Requirements

The backend is tested on an A100 (8.0), an RTX 3070 (8.6) and an RTX 4060 (8.9); other GPUs of compute capability 8.0 or newer run the same kernels.

| | |
|---|---|
| GPU | NVIDIA, compute capability **8.0 or newer**: A100, RTX 30xx, RTX 40xx and later. Turing (RTX 20xx, GTX 16xx, T4) is not supported. |
| Driver | Any driver that supports the GPU. It is loaded at run time (`libcuda.so.1`, `nvcuda.dll`); nothing of NVIDIA is linked at build time. |
| Build | The CUDA Toolkit 13.x (`nvcc`; 12.8 also works). The kernels are compiled for compute capabilities 8.0, 8.6 and 8.9, and as PTX for newer GPUs. |
| Models | `I2_S` models: BitNet b1.58, Falcon3, Falcon-E and LLaMA-style files. Block-scaled `Q2_0`, mixture-of-experts and hybrid models run on the CPU only; the GPU path reports an error that says why. |
| Memory | The weights (about 1.6 bits per weight, plus the 16-bit embedding and output head) and, per session, the KV cache: `2 × layers × kv heads × context × head dim × 2` bytes. Falcon3-10B with a context of 4096 tokens needs about 4.7 GB. |

Head dimensions 32, 64, 128 and 256 are supported. The context window is at most 7936 tokens and defaults to the
model's maximum or 4096 tokens, whichever is smaller; it is rounded down to a multiple of 256.

## Build

The GPU backend is the crate `quintrit-cuda` (`rust/crates/quintrit-cuda`). The runtime, the C library and the Python
package enable it with the cargo feature `cuda`. `build.rs` compiles the kernels with `nvcc`, which it finds through
`NVCC`, `CUDA_PATH`, `CUDA_HOME`, `PATH` or `/usr/local/cuda*`. Without `nvcc` the crates still build, contain no GPU
kernels and report that when a GPU is requested, so `cargo test --workspace` never needs a toolkit.

| Environment variable | Effect |
|---|---|
| `QUINTRIT_CUDA_REQUIRE=1` | fail the build if the kernels cannot be compiled |
| `QUINTRIT_CUDA_SKIP_NVCC=1` | do not compile kernels even if `nvcc` exists |
| `QUINTRIT_CUDA_GENCODE` | replace the list of `-gencode` options, for example `-gencode=arch=compute_86,code=sm_86` for one GPU |

```bash
# from rust/
cargo build --release -p quintrit-runtime --features cuda --bin quintrit          # CLI
cargo build --release -p quintrit-c --features cuda                                # C library
# from rust/crates/quintrit-py/
maturin develop --release --features cuda                                          # Python package
```

## Command line

```bash
./target/release/quintrit --model model.gguf --device cuda --prompt "What is ternary computing?" --max-tokens 64
./target/release/quintrit --model model.gguf --device cuda:1 --context 2048 --chat
```

`--device cpu|cuda|cuda:<index>` chooses where the model runs (default `cpu`), and `--context <tokens>` bounds the
context window on a GPU. `--threads` and `--kernel` do not apply to a GPU. `--bench` works as on the CPU.

## Rust

```rust,no_run
use quintrit_runtime::{InferenceEngine, SamplingConfig};

fn main() -> Result<(), String> {
    // GPU 0; None keeps the default context window.
    let mut engine = InferenceEngine::load_from_gguf_on_gpu("model.gguf", 0, None)?;
    println!("running on {}", engine.gpu_device_name().unwrap_or_default());
    let text = engine.generate_streaming("What is ternary computing?", 64, &SamplingConfig::greedy(), |piece| print!("{piece}"))?;
    println!("\n[{} characters]", text.len());
    Ok(())
}
```

The engine's generation loop, sampling, stop sequences and prefix reuse are the CPU engine's. The lower layers are
public as well: `quintrit_runtime::gpu::GpuModel` holds the weights on a GPU and makes sessions
(`GpuModel::new_session`), each with its own KV cache; `quintrit-cuda` holds the device layer and the kernels.
[`examples/rust/src/bin/generate_gpu.rs`](../../examples/rust/src/bin/generate_gpu.rs) is a complete program.

## C and C++

The GPU functions exist in every build of the library. A build without GPU support returns `QUINTRIT_ERR_UNSUPPORTED`
from all of them, so one program can run against both builds and fall back to the CPU:

```c
uint32_t gpus = 0;
if (quintrit_cuda_device_count(&gpus) == QUINTRIT_OK && gpus > 0) {
    quintrit_gpu_options_t options = { sizeof options, /*device_index=*/0, /*context_length=*/0, /*flags=*/0 };
    quintrit_model_t* model = NULL;
    if (quintrit_model_open_gpu("model.gguf", &options, &model) == QUINTRIT_OK) {
        /* the same handle type as a CPU model: tokenize, sessions, generation are unchanged */
    }
}
```

`quintrit_cuda_device_info` describes a GPU (name, compute capability, multiprocessors, memory, whether it is supported),
and `quintrit_model_device` tells where a model runs. Sessions of a GPU model keep their KV cache on the GPU and share
the weights, so several conversations can run next to each other. In C++ use `quintrit::Model::open_gpu(path, device, context)`
and `model.on_gpu()`. Build the library with GPU support through CMake with `-DQUINTRIT_CUDA=ON`. The status code
`QUINTRIT_ERR_DEVICE` reports a failed GPU call (driver error, out of device memory). Full programs:
[`examples/c/generate_gpu.c`](../../examples/c/generate_gpu.c) and [`examples/cpp/generate_gpu.cpp`](../../examples/cpp/generate_gpu.cpp).

## Python

```python
import quintrit

print(quintrit.cuda_built(), quintrit.cuda_devices())   # was the package built with GPU support; the GPUs the driver reports

model = quintrit.Model.load("model.gguf", device="cuda")   # "cuda:1" for the second GPU; context=2048 bounds the window
print(model.device, model.context_window)
print(model.generate("What is ternary computing?", max_tokens=64))
```

`Model.logits`, `stream`, `chat` and `reset` work as on the CPU. `threads` and `kernel` are ignored on a GPU. Errors that
come from the GPU raise `quintrit.DeviceError`. A full program: [`examples/python/generate_gpu.py`](../../examples/python/generate_gpu.py).

## What runs where

One decoded token is one CUDA graph that is recorded on the first token and replayed afterwards; the position of the token
is read from device memory, so the graph does not change from token to token. The embedding lookup and the sampling run on
the host. Per layer the device runs:

1. norm and quantization of the activations to `int8` (with the residual add of the previous projection fused in),
2. the stacked query, key and value projection with the rotary embedding and the KV-cache append in the same kernel
   (models with half-split rotary pairing use a separate small kernel),
3. the attention, the output projection, the gate and up projection with the activation function, and the down projection.

The projection kernel is configured by what the GPU offers, not by its name: the number of streaming multiprocessors and
the size of the L2 cache select the nearest of the measured configurations (see [`../kernels.md`](../kernels.md)).

## Exactness

The projections are exact integer products. The other kernels reproduce the CPU routines: the single-precision
exponential of the attention softmax is an implementation of glibc's `expf` algorithm for the device, the SiLU uses ggml's
polynomial exponential, the half-precision attention dot products and value accumulators follow the CPU's lane order and
fused multiply-adds, and the key-value axis is split into four chunks beyond 256 positions exactly as the CPU runtime does
it. As on the CPU, the rotary table is built on the host with the platform's math library, so on a given operating system
GPU and CPU agree bit for bit; a platform whose math library rounds `expf`, `cosf`, `sinf` or `powf` differently in the last
bit gives a different text after a few dozen tokens on the CPU and the GPU alike.

## Limits

- Compute capability below 8.0 is not supported; the call fails with a message that names the GPU and the capability.
- One GPU per model. Sessions of one model share its weights.
- Contexts above 7936 tokens are not supported on the GPU.
- Speed at long contexts is limited by the attention, whose half-precision accumulator is a sequential chain over the cached positions (see [`../benchmarks.md`](../benchmarks.md)).

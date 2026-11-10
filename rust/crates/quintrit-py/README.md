# quintrit (Python)

Python bindings for the Quintrit SDK: ternary weights `{-1, 0, +1}` packed at **5 trits per byte (1.60 bits per
weight)** with SIMD GEMV kernels, a GGUF reader, and a small inference runtime. Built with
[PyO3](https://pyo3.rs) and [maturin](https://www.maturin.rs); numpy in, numpy out.

The package has three levels. Use as many as you need.

| Level | What it does | Entry points |
|---|---|---|
| Kernels | pack ternary matrices, multiply them with int8 activations | `pack`, `unpack`, `gemv`, `PackedMatrix` |
| Format | inspect GGUF files, read or transcode ternary tensors | `open`, `GgufFile`, `TensorInfo`, `read_tensor` |
| Runtime | load a ternary language model, tokenize, generate, stream | `Model`, `Tokenizer`, `TokenStream` |

## Install

From source (from the repository root; needs Rust 1.98 or newer (edition 2024; the NEON dot-product intrinsics are stable from 1.98) and Python 3.9 or newer):

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install maturin numpy pytest
cd rust/crates/quintrit-py
maturin develop --release        # builds the extension and installs it into the active environment
pytest                           # runs the test suite (no model needed)
```

The wheels are `abi3` wheels: one wheel per platform works for every CPython from 3.9 on.

## Quick start

### Kernels

```python
import numpy as np
import quintrit

weights = np.random.choice([-1, 0, 1], size=(2560, 2560)).astype(np.int8)
packed = quintrit.pack(weights)                      # PackedMatrix, 1.60 bits per weight
print(packed)                                        # PackedMatrix(rows=2560, cols=2560, kernel="quintrit", ...)

x = np.random.randint(-128, 128, size=2560).astype(np.int8)
y = quintrit.gemv(x, packed)                         # int32 array, exact for the full int8 range
assert np.array_equal(y, weights.astype(np.int64) @ x)
```

* `gemv(activations, packed, *, kernel=None, threads=None, dtype=None, scale=None)`
  * `activations`: 1-D int8 array of length `cols`, or 2-D `(batch, cols)`; the result has shape `(rows,)` or
    `(batch, rows)`.
  * `kernel`: `None` (the kernel the matrix was packed for), `"auto"`, `"quintrit"` or `"quintrit_512"`.
  * `threads`: worker threads (1 to 256), default is the number of logical CPUs, at most 256.
  * `dtype`: `"int32"` (default) or `"float32"`; `scale=...` multiplies the result and implies `float32`.
* The GIL is released while a kernel runs, so several Python threads can multiply concurrently.
* `pack` accepts arrays of any memory layout (C, Fortran, strided views). Inputs must be `int8`; other dtypes raise
  `DTypeError` instead of being converted. Values other than -1, 0, +1 raise `TritValueError`.
* Any number of rows and columns works. Rows are padded to the tile height internally (32, or 64 for the AVX-512
  kernel) and the padding never shows up in the result.
* `PackedMatrix` has `rows`, `cols`, `shape`, `kernel`, `nbytes`, `memory_bytes`, `bits_per_weight`, `unpack()`,
  `to_bytes()`, `PackedMatrix.from_bytes(data)`, equality and pickle support. `to_bytes()` stores a 24 byte header
  (`QPKM`, version, rows, cols) and the packed weights; it is a convenience format for the Python API and not a model file format.
* The kernels accumulate in float32, so integer results are exact while their magnitude stays below 2^24, which
  holds for `cols <= 131072` with full-range activations.
* Memory: the packed weights are kept once; the tiled layout a kernel reads is built on first use and cached per kernel
  (the layout of the kernel chosen at pack time is built immediately). `memory_bytes` reports the total.

```python
quintrit.available_kernels()   # ['quintrit'] or ['quintrit', 'quintrit_512']
quintrit.best_kernel()         # what "auto" resolves to
quintrit.cpu_features()        # {'avx2': True, 'avx512f': False, ...}
```

### Format

```python
with quintrit.open("ggml-model-i2_s.gguf") as gguf:
    for t in gguf.tensors():                       # TensorInfo: name, shape, dtype, is_ternary, nbytes, ...
        print(t.name, t.shape, t.dtype)
    trits = gguf.read_trits("blk.0.attn_q.weight")  # int8 array (rows, cols) with values -1/0/+1
    scale = gguf.tensor_scale("blk.0.attn_q.weight")
    packed = gguf.transcode("blk.0.attn_q.weight")  # PackedMatrix, ready for quintrit.gemv
    norm = quintrit.read_tensor(gguf, "blk.0.attn_norm.weight")   # float32 / float16 tensors too
    gguf.get_metadata("llama.context_length")
```

Supported ternary tensor types: `I2_S`, `TQ1_0`, `TQ2_0`. GGUF stores dimensions innermost first; `TensorInfo.shape`
is in numpy order (`(rows, cols)`), `TensorInfo.ggml_shape` is the stored order. The file is memory-mapped;
`close()` (or leaving the `with` block) unmaps it. GGUF is the only file format that is read.

### Runtime

```python
model = quintrit.Model.load("models/Falcon3-1B-Instruct-1.58bit", threads=8)   # file or directory
print(model.config["num_hidden_layers"], model.context_length, model.kernel)

prompt = "<|user|>\nWhat is ternary computing?\n<|assistant|>\n"
text = model.generate(prompt, max_tokens=64)             # greedy (temperature=0): deterministic

for piece in model.stream(prompt, max_tokens=64, temperature=0.8, top_p=0.9, seed=1):
    print(piece, end="", flush=True)

ids = model.tokenizer.encode(prompt)
logits = model.logits(ids)                               # float32 array (vocab_size,)
model.chat([{"role": "user", "content": "Hello"}], max_tokens=32)
```

* `generate` and `stream` take `max_tokens`, `temperature` (0 = greedy), `top_k`, `top_p`, `seed` and `stop` (a string
  or a list of strings; the text ends before the first match). Ctrl-C interrupts a running generation.
* A model has one context. Every `generate` / `stream` call starts from an empty context; starting a new one (or calling
  `reset()` / `logits()`) invalidates a stream that is still open, and its next step raises `StaleStreamError`.
* `TokenStream` exposes `text`, `token_ids`, `n_tokens`, `finish_reason` (`length`, `eos`, `stop`, `context`, `closed`)
  and `close()`. Text pieces never split a multi-byte character; a piece is held back until the character is complete.
* Loading a model is silent. `Model.load(path, verbose=True)` prints progress lines to the standard output of the
  process (they come from the Rust runtime and bypass Python's `sys.stdout`, so `contextlib.redirect_stdout` does not
  capture them).
* Use `kernel="quintrit"` or `kernel="quintrit_512"` in `Model.load` to force a kernel; all kernels give identical logits.

## Errors

All errors derive from `quintrit.QuintritError`; where a builtin is the natural parent they inherit from it too, so
`except ValueError` and `except TypeError` keep working.

| Class | Also a | Raised for |
|---|---|---|
| `ShapeError` | `ValueError` | wrong array rank, empty array, activation length mismatch |
| `DTypeError` | `TypeError` | wrong dtype or object type (for example a float array) |
| `InvalidArgumentError` | `ValueError` | thread count, sampling parameters, token ids |
| `TritValueError` | `ValueError` | trits outside {-1, 0, +1} |
| `KernelError`, `UnsupportedKernelError` | `ValueError` | unknown kernel name; kernel the CPU cannot run |
| `SerializationError` | `ValueError` | malformed `PackedMatrix.from_bytes` data |
| `FormatError`, `TensorNotFoundError` | `ValueError`, `KeyError` | bad or unsupported file content, unknown tensor |
| `QuintritIOError`, `QuintritFileNotFoundError` | `OSError`, `FileNotFoundError` | file access |
| `ModelLoadError`, `TokenizerError`, `GenerationError` | `RuntimeError` | runtime failures |
| `StaleStreamError`, `ContextOverflowError` | `GenerationError` | invalidated stream; prompt longer than the context |
| `InternalError` | `RuntimeError` | an unexpected failure inside the native code (a bug) |

Inputs are validated before any kernel runs, and a panic in the native code is caught at the boundary and raised as
`InternalError` instead of aborting the interpreter. The extension is therefore built with `panic = "unwind"`
(the SDK's own release profile uses `abort`).

## Supported platforms

| Platform | Kernels | Test status |
|---|---|---|
| Windows x86-64 | `quintrit` (AVX2), `quintrit_512` (AVX-512 BW + VBMI) | the test suite passes on an AVX2 host (the AVX-512 path needs an AVX-512 VBMI CPU) |
| Linux x86-64 | `quintrit` (AVX2), `quintrit_512` | same sources; run `pytest` on the target machine to confirm |
| Linux aarch64 | `quintrit` (NEON) | same sources; run `pytest` on the target machine to confirm |
| macOS arm64 / x86-64 | `quintrit` (NEON / AVX2) | same sources; run `pytest` on the target machine to confirm |

The CPU features are detected at run time: one wheel contains every kernel and picks the best one the machine can run.
On a CPU without AVX2 or NEON the `quintrit` kernel falls back to portable scalar code.

## Building wheels

```bash
pip install maturin
cd rust/crates/quintrit-py

# Linux, x86-64 or aarch64 (build on the matching machine or container; manylinux compliant)
maturin build --release --manylinux 2_28 --out dist
#   or with the official image:  docker run --rm -v "$PWD/../..":/io ghcr.io/pyo3/maturin build --release \
#       -m rust/crates/quintrit-py/Cargo.toml --out rust/crates/quintrit-py/dist

# Windows x86-64 (MSVC toolchain)
maturin build --release --out dist

# macOS: one wheel for both architectures
rustup target add aarch64-apple-darwin x86_64-apple-darwin
maturin build --release --target universal2-apple-darwin --out dist

pip install dist/quintrit-*.whl
```

The build needs the sibling crates (`quintrit-core`, `quintrit-format`, `quintrit-runtime`) next to this directory,
so build from a checkout of the SDK. The crate is a standalone Cargo package with its own `Cargo.lock`; it is not
part of the SDK's Cargo workspace, so `cargo test` of the SDK never needs Python.

## Tests

```bash
pytest                                        # default run: no model, no network
QUINTRIT_TEST_MODEL=/path/to/model.gguf pytest tests/test_model_real.py -s   # optional, real model
```

The default run checks the kernels against brute-force numpy ground truth for many shapes (including row and column
remainders and extreme activation values), memory layouts, error paths, GGUF parsing on synthetic files, and the whole
runtime (loading, tokenizing, generating, streaming, stop sequences) on a tiny random model that the tests write
themselves. The real-model test is skipped unless `QUINTRIT_TEST_MODEL` is set.

## Examples

* [`examples/01_pack_and_gemv.py`](examples/01_pack_and_gemv.py): pack, multiply, compare with numpy, serialize
* [`examples/02_inspect_gguf.py`](examples/02_inspect_gguf.py): list tensors of a GGUF file, transcode one
* [`examples/03_generate.py`](examples/03_generate.py): load a model, generate, stream

Standalone examples matching the C and C++ examples are also in [`examples/python/`](../../../examples/python/).

## Type information

The package ships `py.typed` and a stub file for the compiled module ([`python/quintrit/_native.pyi`](python/quintrit/_native.pyi)).

## License

Apache-2.0 (see [`LICENSE`](LICENSE))

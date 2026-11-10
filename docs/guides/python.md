# Using Quintrit from Python

The package `quintrit` (Python 3.9 and newer) wraps the kernels, the GGUF reader and
the runtime. One `abi3` wheel per platform works for every CPython from 3.9 on, and it contains all kernels: the CPU
features are detected at run time. The reference of the API is the
[`quintrit-py` README](../../rust/crates/quintrit-py/README.md).

## Install

```bash
pip install quintrit
```

The first release on PyPI is in preparation. Until then, build and install the package from
source using [maturin](https://www.maturin.rs) (Rust 1.98 or newer is required):

```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install maturin numpy pytest
cd rust/crates/quintrit-py
maturin develop --release
```

See the [`quintrit-py` README](../../rust/crates/quintrit-py/README.md) for building standalone distribution wheels and running tests.

## Kernel level

```python
import numpy as np
import quintrit

weights = np.random.default_rng(0).integers(-1, 2, size=(4096, 4096)).astype(np.int8)       # values -1, 0, +1
activations = np.random.default_rng(1).integers(-128, 128, size=4096).astype(np.int8)

packed = quintrit.pack(weights)                  # 1.60 bits per weight
print(packed)                                    # PackedMatrix(rows=4096, cols=4096, kernel="quintrit", ...)
result = quintrit.gemv(activations, packed)      # exact; same numbers as weights @ activations
assert np.array_equal(result, weights.astype(np.int64) @ activations.astype(np.int64))
```

- `quintrit.available_kernels()` lists the kernels this CPU can run, `quintrit.best_kernel()` the one `"auto"` picks.
- `gemv` accepts a batch of activation vectors (a 2-D array) and a `scale` to get `float32` results.
- The Python threads are free while a kernel runs, so several of them can multiply concurrently.
- The kernels accumulate in `float32`; integer results are exact while their magnitude stays below 2^24.

## Format level

```python
with quintrit.open("model.gguf") as gguf:                 # memory-mapped, validated; closed at the end of the block
    print(gguf.architecture, len(gguf), "tensors")
    ternary = [t.name for t in gguf.tensors() if t.is_ternary]
    trits = gguf.read_trits(ternary[0])                   # int8 array with values -1, 0, +1
    packed = gguf.transcode(ternary[0])                   # PackedMatrix, ready for quintrit.gemv
```

`GgufFile` lists the tensors and the metadata (`get_metadata`), reads ternary tensors (`I2_S`, `TQ1_0`, `TQ2_0`) and
converts them into packed matrices.

## Runtime level

```python
model = quintrit.Model.load("model.gguf")                # threads and kernel are chosen automatically
print(model.generate("<|user|>\nWhat is ternary computing?\n<|assistant|>\n", max_tokens=48))

for piece in model.stream("<|user|>\nHello\n<|assistant|>\n", max_tokens=48, temperature=0.8, top_k=40, seed=1):
    print(piece, end="", flush=True)
```

`model.tokenizer` encodes and decodes text. Loading prints nothing; pass `verbose=True` to `Model.load` to see the
progress. Errors are exceptions: `quintrit.QuintritError` and its subclasses (a malformed GGUF file, an unsupported
kernel, an invalid argument).

## Examples

- Standalone examples matching the C/C++ examples are in [`examples/python/`](../../examples/python/):
  [`gemv.py`](../../examples/python/gemv.py) (pack matrix and multiply with random activations) and
  [`generate.py`](../../examples/python/generate.py) (load model and stream text).
- An extended tutorial suite is in [`rust/crates/quintrit-py/examples/`](../../rust/crates/quintrit-py/examples/):
  `01_pack_and_gemv.py` (kernels, with a timing of every kernel), `02_inspect_gguf.py` (the format level: tensor
  inspection and transcoding) and `03_generate.py` (the runtime: generation, streaming, stop sequences).

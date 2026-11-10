# Python Examples

This directory contains standalone examples demonstrating Quintrit from Python.

## Prerequisites

Install the `quintrit` package and `numpy`:

```bash
# When released on PyPI:
pip install quintrit numpy

# Or build from source using maturin:
pip install maturin numpy
cd ../../rust/crates/quintrit-py
maturin develop --release
```

## Running the Examples

### 1. Matrix-Vector Multiplication (`gemv.py`)

Packs a ternary matrix `{-1, 0, +1}` using velurex encoding (5 trits per byte, 1.60 bits per weight), executes GEMV on `int8` activations, and checks the output against 64-bit integer matrix multiplication:

```bash
python gemv.py [rows] [cols]          # default: 512 2048
```

### 2. Text Generation (`generate.py`)

Loads a ternary GGUF model (`I2_S`, `TQ1_0`, or `TQ2_0`) and streams generated tokens to the terminal:

```bash
python generate.py path/to/model.gguf ["Your prompt here"]
```

## Extended Tutorials

For deeper walkthroughs covering GGUF format inspection and advanced generation controls, see the tutorial suite in [`../../rust/crates/quintrit-py/examples/`](../../rust/crates/quintrit-py/examples/):
- `01_pack_and_gemv.py`: benchmarks kernel performance across available instruction sets
- `02_inspect_gguf.py`: inspects tensor metadata and transcodes individual tensors
- `03_generate.py`: generation with streaming, sampling temperature, top-k, top-p, and stop sequences

"""Format level: inspect a GGUF file and turn one of its ternary tensors into a PackedMatrix.

    python examples/02_inspect_gguf.py path/to/ggml-model-i2_s.gguf [tensor-name]
"""

import sys

import numpy as np

import quintrit

if len(sys.argv) < 2:
    sys.exit(__doc__)

with quintrit.open(sys.argv[1]) as gguf:
    print(f"{gguf.path}: GGUF v{gguf.version}, architecture {gguf.architecture!r}, {len(gguf)} tensors")
    for key in ("general.name", f"{gguf.architecture}.block_count", f"{gguf.architecture}.context_length"):
        print(f"  {key} = {gguf.get_metadata(key)}")

    ternary = [t for t in gguf.tensors() if t.is_ternary]
    for info in gguf.tensors()[:12]:
        print(f"  {info.name:32s} {info.dtype:6s} shape={info.shape}")
    print(f"  ... {len(ternary)} ternary tensors in total")

    name = sys.argv[2] if len(sys.argv) > 2 else ternary[0].name
    trits = gguf.read_trits(name)  # int8, values -1/0/+1, shape (rows, cols)
    print(f"{name}: shape {trits.shape}, scale {gguf.tensor_scale(name)}, "
          f"zeros {np.mean(trits == 0):.1%}, +1 {np.mean(trits == 1):.1%}, -1 {np.mean(trits == -1):.1%}")

    packed = gguf.transcode(name)  # ready for quintrit.gemv
    rng = np.random.default_rng(0)
    x = rng.integers(-128, 128, size=packed.cols).astype(np.int8)
    y = quintrit.gemv(x, packed)
    assert np.array_equal(y, trits.astype(np.int64) @ x.astype(np.int64))
    print(f"gemv of the tensor with a random int8 vector OK: {y[:4]} ...")

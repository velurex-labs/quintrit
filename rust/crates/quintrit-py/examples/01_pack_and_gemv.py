"""Kernel level: pack a ternary matrix, multiply it with int8 activations, check against numpy.

    python examples/01_pack_and_gemv.py
"""

import time

import numpy as np

import quintrit

rng = np.random.default_rng(0)
rows, cols = 4096, 4096

# A ternary weight matrix (values -1, 0, +1) and an int8 activation vector.
weights = rng.integers(-1, 2, size=(rows, cols)).astype(np.int8)
activations = rng.integers(-128, 128, size=cols).astype(np.int8)

print("kernels on this CPU:", quintrit.available_kernels(), "-> auto =", quintrit.best_kernel())

packed = quintrit.pack(weights)  # kernel="auto"
print(packed)
print(f"{weights.nbytes / 2**20:.1f} MiB as int8  ->  {packed.nbytes / 2**20:.1f} MiB packed "
      f"({packed.bits_per_weight:.2f} bits per weight)")

result = quintrit.gemv(activations, packed)  # int32, exact
expected = weights.astype(np.int64) @ activations.astype(np.int64)
assert np.array_equal(result, expected)
print("gemv matches numpy:", result[:4], "...")

# Every kernel the CPU supports gives the same numbers.
for kernel in quintrit.available_kernels():
    start = time.perf_counter()
    for _ in range(20):
        out = quintrit.gemv(activations, packed, kernel=kernel)
    ms = (time.perf_counter() - start) / 20 * 1000
    assert np.array_equal(out, expected)
    print(f"  kernel {kernel:13s}: {ms:6.2f} ms per gemv")

# Several activation vectors at once, scaled to float32.
batch = rng.integers(-128, 128, size=(8, cols)).astype(np.int8)
scaled = quintrit.gemv(batch, packed, scale=0.02)
print("batched result:", scaled.shape, scaled.dtype)

# Serialize and restore.
restored = quintrit.PackedMatrix.from_bytes(packed.to_bytes())
assert restored == packed
assert np.array_equal(restored.unpack(), weights)
print("round trip through bytes OK")

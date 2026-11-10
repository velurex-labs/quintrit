"""Kernel level in Python: pack a ternary matrix, multiply it with int8 activations and check the result.

    python gemv.py [rows] [cols]          (defaults: 512 x 2048)
"""

import sys

import numpy as np

import quintrit


def main() -> int:
    rows = int(sys.argv[1]) if len(sys.argv) > 1 else 512
    cols = int(sys.argv[2]) if len(sys.argv) > 2 else 2048

    print(f"Quintrit {quintrit.__version__}")
    print(f"available kernels: {quintrit.available_kernels()} (auto: {quintrit.best_kernel()})")

    # Generate random ternary weights {-1, 0, +1} and int8 activations [-128, 127]
    rng = np.random.default_rng(1234)
    trits = rng.choice([-1, 0, 1], size=(rows, cols)).astype(np.int8)
    activations = rng.integers(-128, 128, size=cols).astype(np.int8)

    # Pack weights using velurex encoding (5 trits per byte)
    packed = quintrit.pack(trits)
    bits_per_weight = (packed.nbytes * 8.0) / (rows * cols)
    print(f"{rows * cols} trits -> {packed.nbytes} bytes ({bits_per_weight:.3f} bits per weight)")

    # Execute GEMV kernel
    out = quintrit.gemv(activations, packed)

    # Verify against 64-bit integer matrix multiplication
    expected = trits.astype(np.int64) @ activations.astype(np.int64)
    mismatches = int(np.sum(out != expected))

    if mismatches == 0:
        print("OK: matches the integer product (0 mismatching rows)")
        return 0
    else:
        print(f"MISMATCH: {mismatches} mismatching rows")
        return 1


if __name__ == "__main__":
    sys.exit(main())

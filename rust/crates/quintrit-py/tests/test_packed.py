"""PackedMatrix: serialization, pickling, equality, introspection."""

from __future__ import annotations

import copy
import pickle

import numpy as np
import pytest

import quintrit
from conftest import KERNELS, SHAPES, random_acts, random_trits, reference_gemv


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_bytes_roundtrip(rows, cols, rng):
    trits = random_trits(rng, rows, cols)
    packed = quintrit.pack(trits)
    blob = packed.to_bytes()
    assert isinstance(blob, bytes)
    assert blob[:4] == b"QPKM"
    assert len(blob) == 24 + packed.nbytes
    restored = quintrit.PackedMatrix.from_bytes(blob)
    assert restored.shape == (rows, cols)
    assert restored == packed
    np.testing.assert_array_equal(restored.unpack(), trits)
    acts = random_acts(rng, cols)
    np.testing.assert_array_equal(quintrit.gemv(acts, restored), reference_gemv(trits, acts))


def test_from_bytes_accepts_bytes_like_objects(rng):
    packed = quintrit.pack(random_trits(rng, 33, 170))
    blob = packed.to_bytes()
    for variant in (bytearray(blob), memoryview(blob), np.frombuffer(blob, dtype=np.uint8)):
        assert quintrit.PackedMatrix.from_bytes(variant) == packed


def test_from_bytes_kernel_argument(rng):
    packed = quintrit.pack(random_trits(rng, 64, 100))
    for kernel in KERNELS:
        assert quintrit.PackedMatrix.from_bytes(packed.to_bytes(), kernel=kernel).kernel == kernel


def test_pickle_and_copy(rng):
    trits = random_trits(rng, 70, 190)
    packed = quintrit.pack(trits)
    for restored in (pickle.loads(pickle.dumps(packed)), copy.copy(packed), copy.deepcopy(packed)):
        assert restored == packed
        np.testing.assert_array_equal(restored.unpack(), trits)


def test_equality(rng):
    trits = random_trits(rng, 32, 64)
    a, b = quintrit.pack(trits), quintrit.pack(trits.copy())
    assert a == b
    changed = trits.copy()
    changed[0, 0] = 1 if changed[0, 0] != 1 else -1
    assert a != quintrit.pack(changed)
    assert a != quintrit.pack(trits[:, :63])
    assert a != "something else"
    assert a != 5


def test_properties_and_repr(rng):
    packed = quintrit.pack(random_trits(rng, 64, 320), kernel="quintrit")
    assert (packed.rows, packed.cols, packed.shape) == (64, 320, (64, 320))
    assert packed.kernel == "quintrit"
    assert packed.nbytes == 64 * 320 // 5
    assert packed.memory_bytes >= packed.nbytes
    text = repr(packed)
    assert "PackedMatrix" in text and "rows=64" in text and "cols=320" in text and "quintrit" in text


def test_tile_cache_grows_on_demand(rng):
    if len(KERNELS) < 2:
        pytest.skip("needs both kernels")
    packed = quintrit.pack(random_trits(rng, 128, 256), kernel="quintrit")
    before = packed.memory_bytes
    quintrit.gemv(random_acts(rng, 256), packed, kernel="quintrit_512")
    assert packed.memory_bytes > before


def test_row_padding_does_not_leak_into_results(rng):
    """rows not a multiple of the tile height: only the real rows are returned."""
    for rows in (1, 5, 31, 33, 63, 65):
        trits = random_trits(rng, rows, 100)
        acts = random_acts(rng, 100)
        for kernel in KERNELS:
            out = quintrit.gemv(acts, quintrit.pack(trits, kernel=kernel))
            assert out.shape == (rows,)
            np.testing.assert_array_equal(out, reference_gemv(trits, acts))


def test_large_matrix_matches_bruteforce(rng):
    trits = random_trits(rng, 1024, 1500)
    acts = random_acts(rng, 1500)
    truth = reference_gemv(trits, acts)
    for kernel in KERNELS:
        np.testing.assert_array_equal(quintrit.gemv(acts, quintrit.pack(trits, kernel=kernel)), truth)

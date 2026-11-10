"""Error paths: every failure is a QuintritError (and the natural builtin exception), never a crash."""

from __future__ import annotations

import numpy as np
import pytest

import quintrit
from conftest import KERNELS, random_acts, random_trits


@pytest.fixture
def packed(rng):
    return quintrit.pack(random_trits(rng, 64, 200))


# ---- hierarchy -------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cls,builtin",
    [
        (quintrit.ShapeError, ValueError),
        (quintrit.DTypeError, TypeError),
        (quintrit.InvalidArgumentError, ValueError),
        (quintrit.TritValueError, ValueError),
        (quintrit.KernelError, ValueError),
        (quintrit.SerializationError, ValueError),
        (quintrit.FormatError, ValueError),
        (quintrit.TensorNotFoundError, KeyError),
        (quintrit.QuintritIOError, OSError),
        (quintrit.QuintritFileNotFoundError, FileNotFoundError),
        (quintrit.ModelLoadError, RuntimeError),
        (quintrit.InternalError, RuntimeError),
        (quintrit.ContextOverflowError, ValueError),
    ],
)
def test_exception_hierarchy(cls, builtin):
    assert issubclass(cls, quintrit.QuintritError)
    assert issubclass(cls, builtin)


def test_exception_subclass_relations():
    assert issubclass(quintrit.UnsupportedKernelError, quintrit.KernelError)
    assert issubclass(quintrit.StaleStreamError, quintrit.GenerationError)
    assert issubclass(quintrit.ContextOverflowError, quintrit.GenerationError)
    assert issubclass(quintrit.QuintritFileNotFoundError, quintrit.QuintritIOError)


# ---- pack ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.int32, np.float32, np.float64, np.bool_])
def test_pack_rejects_other_dtypes(dtype):
    with pytest.raises(quintrit.DTypeError, match="int8") as info:
        quintrit.pack(np.zeros((4, 4), dtype=dtype))
    assert isinstance(info.value, TypeError)


@pytest.mark.parametrize("bad", [[[0, 1], [1, 0]], "abc", 5, None, b"\x00\x01"])
def test_pack_rejects_non_arrays(bad):
    with pytest.raises(quintrit.DTypeError, match="numpy.ndarray"):
        quintrit.pack(bad)


@pytest.mark.parametrize("shape", [(5,), (2, 3, 4), ()])
def test_pack_rejects_wrong_dimensions(shape):
    with pytest.raises(quintrit.ShapeError, match="2-D"):
        quintrit.pack(np.zeros(shape, dtype=np.int8))


@pytest.mark.parametrize("shape", [(0, 5), (5, 0), (0, 0)])
def test_pack_rejects_empty_matrices(shape):
    with pytest.raises(quintrit.ShapeError, match="empty"):
        quintrit.pack(np.zeros(shape, dtype=np.int8))


@pytest.mark.parametrize("value", [2, -2, 127, -128])
def test_pack_rejects_non_trit_values(value):
    trits = np.zeros((8, 8), dtype=np.int8)
    trits[3, 5] = value
    with pytest.raises(quintrit.TritValueError, match="row 3, column 5"):
        quintrit.pack(trits)


def test_pack_rejects_unknown_kernel_and_bad_kernel_type():
    trits = np.zeros((4, 4), dtype=np.int8)
    with pytest.raises(quintrit.KernelError, match="unknown kernel"):
        quintrit.pack(trits, kernel="avx9000")
    with pytest.raises(TypeError):
        quintrit.pack(trits, kernel=5)
    with pytest.raises(TypeError):
        quintrit.pack(trits, "quintrit")  # kernel is keyword-only


@pytest.mark.skipif("quintrit_512" in KERNELS, reason="needs a CPU without AVX-512 VBMI")
def test_unsupported_kernel_is_reported():
    trits = np.zeros((64, 64), dtype=np.int8)
    with pytest.raises(quintrit.UnsupportedKernelError) as info:
        quintrit.pack(trits, kernel="quintrit_512")
    assert isinstance(info.value, quintrit.KernelError)
    packed = quintrit.pack(trits)
    with pytest.raises(quintrit.UnsupportedKernelError):
        quintrit.gemv(np.zeros(64, dtype=np.int8), packed, kernel="quintrit_512")


# ---- gemv ------------------------------------------------------------------------------------------------------------


@pytest.mark.parametrize("dtype", [np.uint8, np.int16, np.float32, np.bool_])
def test_gemv_rejects_other_activation_dtypes(packed, dtype):
    with pytest.raises(quintrit.DTypeError, match="int8"):
        quintrit.gemv(np.zeros(200, dtype=dtype), packed)


def test_gemv_rejects_non_arrays(packed):
    with pytest.raises(quintrit.DTypeError):
        quintrit.gemv([0] * 200, packed)
    with pytest.raises(quintrit.DTypeError):
        quintrit.gemv(None, packed)


def test_gemv_rejects_wrong_packed_argument(rng):
    acts = random_acts(rng, 10)
    with pytest.raises(TypeError):
        quintrit.gemv(acts, np.zeros((4, 4), dtype=np.int8))
    with pytest.raises(TypeError):
        quintrit.gemv(acts, None)


@pytest.mark.parametrize("length", [1, 199, 201, 400])
def test_gemv_rejects_length_mismatch(packed, length):
    with pytest.raises(quintrit.ShapeError, match="200 columns"):
        quintrit.gemv(np.zeros(length, dtype=np.int8), packed)
    with pytest.raises(quintrit.ShapeError):
        quintrit.gemv(np.zeros((3, length), dtype=np.int8), packed)


def test_gemv_rejects_bad_dimensions(packed):
    with pytest.raises(quintrit.ShapeError):
        quintrit.gemv(np.zeros((2, 2, 200), dtype=np.int8), packed)
    with pytest.raises(quintrit.ShapeError):
        quintrit.gemv(np.zeros((0, 200), dtype=np.int8), packed)
    with pytest.raises(quintrit.ShapeError):
        quintrit.gemv(np.zeros((), dtype=np.int8), packed)


@pytest.mark.parametrize("threads", [0, 257, 5000])
def test_gemv_rejects_bad_thread_counts(packed, threads):
    with pytest.raises(quintrit.InvalidArgumentError, match="threads"):
        quintrit.gemv(np.zeros(200, dtype=np.int8), packed, threads=threads)


def test_gemv_accepts_the_maximum_thread_count(packed):
    out = quintrit.gemv(np.zeros(200, dtype=np.int8), packed, threads=256)
    assert out.shape == (64,)


def test_default_threads_stay_within_the_limit():
    assert 1 <= quintrit.default_threads() <= 256


def test_gemv_column_limit_matches_the_c_api():
    # 131072 columns is the largest count with exact accumulators (the same limit as the C API).
    ok = quintrit.pack(np.zeros((1, 131072), dtype=np.int8))
    assert quintrit.gemv(np.full(131072, 127, dtype=np.int8), ok).tolist() == [0]
    too_wide = quintrit.pack(np.zeros((1, 131073), dtype=np.int8))
    with pytest.raises(quintrit.ShapeError, match="131072"):
        quintrit.gemv(np.zeros(131073, dtype=np.int8), too_wide)


def test_gemv_negative_threads_is_an_error(packed):
    with pytest.raises((OverflowError, quintrit.QuintritError)):
        quintrit.gemv(np.zeros(200, dtype=np.int8), packed, threads=-1)


def test_gemv_rejects_unknown_kernel(packed):
    with pytest.raises(quintrit.KernelError):
        quintrit.gemv(np.zeros(200, dtype=np.int8), packed, kernel="nope")


@pytest.mark.parametrize("dtype", ["int8", "float64", "bogus", np.float64, np.int16])
def test_gemv_rejects_unsupported_output_dtype(packed, dtype):
    with pytest.raises(quintrit.DTypeError):
        quintrit.gemv(np.zeros(200, dtype=np.int8), packed, dtype=dtype)


def test_gemv_scale_requires_float(packed):
    acts = np.zeros(200, dtype=np.int8)
    with pytest.raises(quintrit.DTypeError, match="float32"):
        quintrit.gemv(acts, packed, dtype="int32", scale=2.0)
    with pytest.raises(quintrit.InvalidArgumentError, match="finite"):
        quintrit.gemv(acts, packed, scale=float("nan"))
    with pytest.raises(quintrit.InvalidArgumentError, match="finite"):
        quintrit.gemv(acts, packed, scale=float("inf"))


# ---- serialization ---------------------------------------------------------------------------------------------------


def test_from_bytes_rejects_malformed_data(packed):
    good = packed.to_bytes()
    with pytest.raises(quintrit.SerializationError, match="too short"):
        quintrit.PackedMatrix.from_bytes(good[:10])
    with pytest.raises(quintrit.SerializationError, match="magic"):
        quintrit.PackedMatrix.from_bytes(b"XXXX" + good[4:])
    bad_version = good[:4] + (99).to_bytes(4, "little") + good[8:]
    with pytest.raises(quintrit.SerializationError, match="version"):
        quintrit.PackedMatrix.from_bytes(bad_version)
    with pytest.raises(quintrit.SerializationError, match="payload"):
        quintrit.PackedMatrix.from_bytes(good[:-1])
    with pytest.raises(quintrit.SerializationError, match="payload"):
        quintrit.PackedMatrix.from_bytes(good + b"\x00")
    zero_rows = good[:8] + (0).to_bytes(8, "little") + good[16:]
    with pytest.raises(quintrit.SerializationError, match="dimensions"):
        quintrit.PackedMatrix.from_bytes(zero_rows)
    huge = good[:8] + ((1 << 63) - 1).to_bytes(8, "little") + ((1 << 63) - 1).to_bytes(8, "little") + good[24:]
    with pytest.raises(quintrit.SerializationError):
        quintrit.PackedMatrix.from_bytes(huge)
    corrupted = bytearray(good)
    corrupted[40] = 251
    with pytest.raises(quintrit.SerializationError, match="251"):
        quintrit.PackedMatrix.from_bytes(bytes(corrupted))


def test_from_bytes_rejects_non_bytes():
    with pytest.raises(quintrit.DTypeError, match="bytes-like"):
        quintrit.PackedMatrix.from_bytes("text")
    with pytest.raises(quintrit.DTypeError, match="bytes-like"):
        quintrit.PackedMatrix.from_bytes(12)
    with pytest.raises(quintrit.DTypeError):
        quintrit.PackedMatrix.from_bytes(None)


def test_packed_matrix_cannot_be_constructed_directly():
    with pytest.raises(TypeError):
        quintrit.PackedMatrix()


def test_errors_never_leave_state_behind(packed, rng):
    """A failed call does not disturb later calls."""
    acts = random_acts(rng, 200)
    expected = quintrit.gemv(acts, packed)
    for _ in range(3):
        with pytest.raises(quintrit.QuintritError):
            quintrit.gemv(np.zeros(3, dtype=np.int8), packed)
    np.testing.assert_array_equal(quintrit.gemv(acts, packed), expected)

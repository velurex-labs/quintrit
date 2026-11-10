"""Kernel level: pack / unpack / gemv against brute-force ground truth (no model needed)."""

from __future__ import annotations

import threading
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import pytest

import quintrit
from conftest import KERNELS, SHAPES, random_acts, random_trits, reference_gemv


def test_version_and_kernel_listing():
    assert isinstance(quintrit.__version__, str) and quintrit.__version__.count(".") == 2
    kernels = quintrit.available_kernels()
    assert "quintrit" in kernels
    assert set(kernels) <= {"quintrit", "quintrit_512"}
    assert quintrit.best_kernel() in kernels
    features = quintrit.cpu_features()
    assert all(isinstance(k, str) and isinstance(v, bool) for k, v in features.items())
    # the AVX-512 kernel is listed exactly when the required features are present
    if "avx512vbmi" in features:
        expected = features["avx512f"] and features["avx512bw"] and features["avx512vbmi"]
        assert ("quintrit_512" in kernels) == expected
    else:
        assert "quintrit_512" not in kernels
    assert quintrit.default_threads() >= 1


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_pack_unpack_roundtrip(rows, cols, rng):
    trits = random_trits(rng, rows, cols)
    packed = quintrit.pack(trits)
    assert packed.shape == (rows, cols)
    out = quintrit.unpack(packed)
    assert out.dtype == np.int8 and out.shape == (rows, cols)
    np.testing.assert_array_equal(out, trits)
    np.testing.assert_array_equal(packed.unpack(), trits)


@pytest.mark.parametrize("rows,cols", SHAPES)
def test_gemv_matches_bruteforce(rows, cols, kernel, rng):
    trits = random_trits(rng, rows, cols)
    acts = random_acts(rng, cols)
    packed = quintrit.pack(trits, kernel=kernel)
    assert packed.kernel == kernel
    result = quintrit.gemv(acts, packed)
    assert result.dtype == np.int32 and result.shape == (rows,)
    np.testing.assert_array_equal(result, reference_gemv(trits, acts))


@pytest.mark.parametrize("rows,cols", [(33, 161), (64, 320), (130, 257)])
def test_every_kernel_works_on_every_packed_matrix(rows, cols, rng):
    """The layout of a kernel is built on demand, whichever kernel the matrix was packed for."""
    trits = random_trits(rng, rows, cols)
    acts = random_acts(rng, cols)
    truth = reference_gemv(trits, acts)
    for packed_for in KERNELS:
        packed = quintrit.pack(trits, kernel=packed_for)
        before = packed.memory_bytes
        for run_with in KERNELS:
            np.testing.assert_array_equal(quintrit.gemv(acts, packed, kernel=run_with), truth)
        if len(KERNELS) > 1:
            assert packed.memory_bytes > before


def test_extreme_activations_are_exact(kernel, rng):
    rows, cols = 96, 400
    trits = random_trits(rng, rows, cols)
    packed = quintrit.pack(trits, kernel=kernel)
    for fill in (-128, 127, 0, -1, 1):
        acts = np.full(cols, fill, dtype=np.int8)
        np.testing.assert_array_equal(quintrit.gemv(acts, packed), reference_gemv(trits, acts))
    alternating = np.where(np.arange(cols) % 2 == 0, -128, 127).astype(np.int8)
    np.testing.assert_array_equal(quintrit.gemv(alternating, packed), reference_gemv(trits, alternating))
    # all weights +1 / -1 against the extreme activations: the largest possible sums
    ones = np.ones((64, cols), dtype=np.int8)
    for sign in (1, -1):
        p = quintrit.pack(ones * sign, kernel=kernel)
        acts = np.full(cols, -128, dtype=np.int8)
        np.testing.assert_array_equal(quintrit.gemv(acts, p), reference_gemv(ones * sign, acts))


def test_all_zero_and_degenerate_inputs(kernel):
    packed = quintrit.pack(np.zeros((40, 200), dtype=np.int8), kernel=kernel)
    out = quintrit.gemv(np.full(200, 77, dtype=np.int8), packed)
    assert not out.any()
    one = quintrit.pack(np.array([[1]], dtype=np.int8), kernel=kernel)
    assert quintrit.gemv(np.array([-128], dtype=np.int8), one).tolist() == [-128]


@pytest.mark.parametrize("threads", [1, 2, 3, 8])
def test_threads_give_identical_results(threads, kernel, rng):
    rows, cols = 700, 330
    trits = random_trits(rng, rows, cols)
    acts = random_acts(rng, cols)
    packed = quintrit.pack(trits, kernel=kernel)
    np.testing.assert_array_equal(quintrit.gemv(acts, packed, threads=threads), reference_gemv(trits, acts))


def test_batch_of_activation_vectors(kernel, rng):
    rows, cols, batch = 70, 190, 5
    trits = random_trits(rng, rows, cols)
    acts = rng.integers(-128, 128, size=(batch, cols)).astype(np.int8)
    packed = quintrit.pack(trits, kernel=kernel)
    out = quintrit.gemv(acts, packed)
    assert out.shape == (batch, rows)
    np.testing.assert_array_equal(out, acts.astype(np.int64) @ trits.astype(np.int64).T)
    single = quintrit.gemv(acts[2], packed)
    np.testing.assert_array_equal(out[2], single)


def test_float_output_and_scale(kernel, rng):
    rows, cols = 50, 170
    trits = random_trits(rng, rows, cols)
    acts = random_acts(rng, cols)
    packed = quintrit.pack(trits, kernel=kernel)
    truth = reference_gemv(trits, acts)
    out = quintrit.gemv(acts, packed, dtype="float32")
    assert out.dtype == np.float32
    np.testing.assert_array_equal(out, truth.astype(np.float32))
    scaled = quintrit.gemv(acts, packed, scale=0.25)
    assert scaled.dtype == np.float32
    np.testing.assert_allclose(scaled, truth * 0.25, rtol=0, atol=0)
    # numpy dtypes and dtype objects are accepted as well
    assert quintrit.gemv(acts, packed, dtype=np.float32).dtype == np.float32
    assert quintrit.gemv(acts, packed, dtype=np.dtype("int32")).dtype == np.int32


def test_packed_method_equals_function(kernel, rng):
    trits = random_trits(rng, 64, 200)
    acts = random_acts(rng, 200)
    packed = quintrit.pack(trits, kernel=kernel)
    np.testing.assert_array_equal(packed.gemv(acts), quintrit.gemv(acts, packed))


def test_auto_kernel_selector(rng):
    trits = random_trits(rng, 64, 100)
    packed = quintrit.pack(trits)  # auto
    assert packed.kernel == quintrit.best_kernel()
    assert quintrit.pack(trits, kernel="auto").kernel == quintrit.best_kernel()
    assert quintrit.pack(trits, kernel="QUINTRIT").kernel == "quintrit"
    acts = random_acts(rng, 100)
    np.testing.assert_array_equal(quintrit.gemv(acts, packed, kernel="auto"), reference_gemv(trits, acts))


# ---------------------------------------------------------------------------------------------------------------------
# Memory layout and input hygiene
# ---------------------------------------------------------------------------------------------------------------------


def test_non_contiguous_inputs_are_accepted(kernel, rng):
    rows, cols = 66, 210
    trits = random_trits(rng, rows, cols)
    acts = random_acts(rng, cols)
    truth = reference_gemv(trits, acts)

    fortran = np.asfortranarray(trits)
    assert not fortran.flags.c_contiguous
    np.testing.assert_array_equal(quintrit.gemv(acts, quintrit.pack(fortran, kernel=kernel)), truth)

    wide = np.zeros((rows * 2, cols * 3), dtype=np.int8)
    wide[::2, ::3] = trits
    view = wide[::2, ::3]
    assert not view.flags.c_contiguous
    np.testing.assert_array_equal(quintrit.gemv(acts, quintrit.pack(view, kernel=kernel)), truth)

    reversed_trits = trits[::-1, ::-1]
    packed_rev = quintrit.pack(reversed_trits, kernel=kernel)
    np.testing.assert_array_equal(packed_rev.unpack(), reversed_trits)

    packed = quintrit.pack(trits, kernel=kernel)
    strided_acts = np.zeros(cols * 2, dtype=np.int8)
    strided_acts[::2] = acts
    np.testing.assert_array_equal(quintrit.gemv(strided_acts[::2], packed), truth)
    np.testing.assert_array_equal(quintrit.gemv(acts[::-1][::-1], packed), truth)


def test_read_only_and_subclass_inputs(kernel, rng):
    trits = random_trits(rng, 32, 64)
    trits.setflags(write=False)
    acts = random_acts(rng, 64)
    acts.setflags(write=False)
    out = quintrit.gemv(acts, quintrit.pack(trits, kernel=kernel))
    np.testing.assert_array_equal(out, reference_gemv(trits, acts))
    masked = np.ma.masked_array(random_acts(rng, 64))
    assert quintrit.gemv(masked, quintrit.pack(trits, kernel=kernel)).shape == (32,)


def test_inputs_are_not_modified(kernel, rng):
    trits = random_trits(rng, 40, 170)
    acts = random_acts(rng, 170)
    trits_copy, acts_copy = trits.copy(), acts.copy()
    packed = quintrit.pack(trits, kernel=kernel)
    quintrit.gemv(acts, packed)
    np.testing.assert_array_equal(trits, trits_copy)
    np.testing.assert_array_equal(acts, acts_copy)


def test_results_are_independent_writable_arrays(kernel, rng):
    packed = quintrit.pack(random_trits(rng, 32, 64), kernel=kernel)
    acts = random_acts(rng, 64)
    first = quintrit.gemv(acts, packed)
    second = quintrit.gemv(acts, packed)
    first[:] = 0
    assert second.any()
    assert first.flags.writeable and first.flags.c_contiguous


def test_memory_footprint_is_1_60_bpw(rng):
    # 160 columns = one block: exactly 5 trits per byte
    packed = quintrit.pack(random_trits(rng, 64, 320))
    assert packed.nbytes == 64 * 320 // 5
    assert packed.bits_per_weight == pytest.approx(1.6)
    ragged = quintrit.pack(random_trits(rng, 10, 161))
    assert ragged.nbytes == 10 * 2 * 32
    assert ragged.bits_per_weight > 1.6


# ---------------------------------------------------------------------------------------------------------------------
# Concurrency and the GIL
# ---------------------------------------------------------------------------------------------------------------------


def test_concurrent_gemv_from_python_threads(kernel, rng):
    trits = random_trits(rng, 256, 512)
    packed = quintrit.pack(trits, kernel=kernel)
    jobs = [random_acts(rng, 512) for _ in range(16)]
    with ThreadPoolExecutor(max_workers=4) as pool:
        results = list(pool.map(lambda a: quintrit.gemv(a, packed), jobs))
    for acts, out in zip(jobs, results):
        np.testing.assert_array_equal(out, reference_gemv(trits, acts))


def test_gil_is_released_during_gemv(rng):
    """A busy Python thread keeps running while a long gemv executes in another thread."""
    rows, cols = 3072, 3072
    packed = quintrit.pack(random_trits(rng, rows, cols), kernel="quintrit")
    acts = random_acts(rng, cols)
    quintrit.gemv(acts, packed, threads=1)  # build the layout and warm up

    ticks = 0
    stop = threading.Event()

    def ticker():
        nonlocal ticks
        while not stop.is_set():
            ticks += 1
            time.sleep(0.0005)

    def worker():
        deadline = time.perf_counter() + 0.5
        while time.perf_counter() < deadline:
            quintrit.gemv(acts, packed, threads=1)

    t = threading.Thread(target=ticker)
    t.start()
    try:
        worker()
    finally:
        stop.set()
        t.join()
    # with the GIL held for the whole 0.5 s the ticker would barely advance
    assert ticks > 20, f"ticker advanced only {ticks} times: the GIL seems to be held during gemv"

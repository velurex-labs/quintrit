"""Shared fixtures and ground-truth helpers for the quintrit test suite."""

from __future__ import annotations

import numpy as np
import pytest

import quintrit
from gguf_builder import write_tiny_model

#: Kernels the CPU running the tests can execute.
KERNELS = quintrit.available_kernels()

#: (rows, cols) shapes covering tile remainders (rows % 32, rows % 64) and the packed tail (cols % 160).
SHAPES = [
    (1, 1),
    (1, 5),
    (3, 7),
    (31, 33),
    (32, 160),
    (32, 161),
    (33, 159),
    (64, 320),
    (65, 31),
    (100, 480),
    (128, 128),
    (130, 257),
    (256, 640),
]


def random_trits(rng: np.random.Generator, rows: int, cols: int) -> np.ndarray:
    return rng.integers(-1, 2, size=(rows, cols)).astype(np.int8)


def random_acts(rng: np.random.Generator, n: int) -> np.ndarray:
    return rng.integers(-128, 128, size=n).astype(np.int8)


def reference_gemv(trits: np.ndarray, acts: np.ndarray) -> np.ndarray:
    """Brute-force ground truth in int64 arithmetic."""
    return trits.astype(np.int64) @ acts.astype(np.int64)


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(12345)


@pytest.fixture(params=KERNELS)
def kernel(request: pytest.FixtureRequest) -> str:
    return request.param


@pytest.fixture(scope="session")
def tiny_model_path(tmp_path_factory: pytest.TempPathFactory):
    return write_tiny_model(tmp_path_factory.mktemp("tiny") / "tiny-model.gguf")

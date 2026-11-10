"""Quintrit: ternary (1.60 bits per weight) matrix kernels, GGUF access and inference from Python.

Three levels are available:

* kernels: :func:`pack`, :func:`unpack`, :func:`gemv`, :class:`PackedMatrix` (numpy in, numpy out);
* format: :func:`open` / :class:`GgufFile` to inspect GGUF files and extract or transcode ternary tensors;
* runtime: :class:`Model` for tokenizing and generating text with a ternary GGUF language model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from ._errors import (
    ContextOverflowError,
    DTypeError,
    FormatError,
    GenerationError,
    InternalError,
    InvalidArgumentError,
    KernelError,
    ModelLoadError,
    QuintritError,
    QuintritFileNotFoundError,
    QuintritIOError,
    SerializationError,
    ShapeError,
    StaleStreamError,
    TensorNotFoundError,
    TokenizerError,
    TritValueError,
    UnsupportedKernelError,
)
from ._native import (
    GgufFile,
    Model,
    PackedMatrix,
    TensorInfo,
    Tokenizer,
    TokenStream,
    __version__,
    available_kernels,
    best_kernel,
    cpu_features,
    default_threads,
    gemv,
    open,
    pack,
    unpack,
)

__all__ = [
    "__version__",
    # kernel level
    "pack",
    "unpack",
    "gemv",
    "PackedMatrix",
    "available_kernels",
    "best_kernel",
    "cpu_features",
    "default_threads",
    # format level
    "open",
    "GgufFile",
    "TensorInfo",
    "read_tensor",
    # runtime level
    "Model",
    "Tokenizer",
    "TokenStream",
    # errors
    "QuintritError",
    "ShapeError",
    "DTypeError",
    "InvalidArgumentError",
    "TritValueError",
    "KernelError",
    "UnsupportedKernelError",
    "SerializationError",
    "FormatError",
    "TensorNotFoundError",
    "QuintritIOError",
    "QuintritFileNotFoundError",
    "ModelLoadError",
    "TokenizerError",
    "GenerationError",
    "StaleStreamError",
    "ContextOverflowError",
    "InternalError",
]

_FLOAT_DTYPES = {"F32": np.float32, "F16": np.float16}


def read_tensor(file: GgufFile, name: str) -> "np.ndarray[Any, Any]":
    """Reads a tensor of a GGUF file as a numpy array.

    Ternary tensors (``I2_S``, ``TQ1_0``, ``TQ2_0``) come back as ``int8`` arrays of trits with shape
    ``(rows, cols)``; ``F32`` and ``F16`` tensors as ``float32`` / ``float16`` arrays in numpy axis order.
    Other tensor types raise :class:`FormatError`.
    """
    info = file.tensor(name)
    if info.is_ternary:
        return file.read_trits(name)
    dtype = _FLOAT_DTYPES.get(info.dtype)
    if dtype is None:
        raise FormatError(f"tensor {name!r} has type {info.dtype}, which read_tensor cannot decode")
    raw = file.read_raw(name)
    return np.frombuffer(raw, dtype=dtype).reshape(info.shape).copy()

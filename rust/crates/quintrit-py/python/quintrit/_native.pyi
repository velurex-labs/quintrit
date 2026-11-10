"""Type stubs of the compiled module ``quintrit._native`` (re-exported by ``quintrit``)."""

import os
from typing import Any, Iterator, Literal, Optional, Sequence, Union

import numpy as np
from numpy.typing import NDArray

__version__: str

KernelName = Literal["auto", "quintrit", "quintrit_512"]
OutDType = Union[Literal["int32", "float32"], type, np.dtype[Any]]

# ---------------------------------------------------------------------------------------------------------------------
# Kernel level
# ---------------------------------------------------------------------------------------------------------------------

def available_kernels() -> list[str]:
    """Kernels this CPU can run: ``"quintrit"`` always, ``"quintrit_512"`` with AVX-512 BW + VBMI."""

def best_kernel() -> str:
    """The kernel ``"auto"`` resolves to."""

def cpu_features() -> dict[str, bool]:
    """Detected CPU features relevant for kernel selection."""

def default_threads() -> int:
    """Number of logical CPUs (at most 256)."""

class PackedMatrix:
    """A ternary matrix packed at 5 trits per byte (1.60 bits per weight)."""

    @property
    def rows(self) -> int: ...
    @property
    def cols(self) -> int: ...
    @property
    def shape(self) -> tuple[int, int]: ...
    @property
    def kernel(self) -> str:
        """Kernel the matrix was packed for (the default of :meth:`gemv`)."""
    @property
    def nbytes(self) -> int:
        """Size of the packed weights in bytes."""
    @property
    def memory_bytes(self) -> int:
        """Bytes held in memory, including the tiled layouts built so far."""
    @property
    def bits_per_weight(self) -> float: ...
    def unpack(self) -> NDArray[np.int8]:
        """Trits as an int8 array of shape ``(rows, cols)``."""
    def to_bytes(self) -> bytes:
        """Serialize: 24 byte header (magic ``QPKM``, version, rows, cols) followed by the packed weights."""
    @staticmethod
    def from_bytes(data: Union[bytes, bytearray, memoryview, NDArray[np.uint8]], *, kernel: Optional[KernelName] = None) -> "PackedMatrix": ...
    def gemv(
        self,
        activations: NDArray[np.int8],
        *,
        kernel: Optional[KernelName] = None,
        threads: Optional[int] = None,
        dtype: Optional[OutDType] = None,
        scale: Optional[float] = None,
    ) -> NDArray[Any]: ...
    def __eq__(self, other: object) -> bool: ...

def pack(trits: NDArray[np.int8], *, kernel: Optional[KernelName] = None) -> PackedMatrix:
    """Pack a 2-D int8 array with values -1, 0, +1."""

def unpack(packed: PackedMatrix) -> NDArray[np.int8]:
    """Inverse of :func:`pack`."""

def gemv(
    activations: NDArray[np.int8],
    packed: PackedMatrix,
    *,
    kernel: Optional[KernelName] = None,
    threads: Optional[int] = None,
    dtype: Optional[OutDType] = None,
    scale: Optional[float] = None,
) -> NDArray[Any]:
    """``W @ x`` for a packed ternary ``W`` and int8 activations (1-D, or 2-D ``(batch, cols)``).

    Exact integer results for the full int8 range; int32 output by default (float32 when ``scale`` is given).
    """

# ---------------------------------------------------------------------------------------------------------------------
# Format level
# ---------------------------------------------------------------------------------------------------------------------

class TensorInfo:
    @property
    def name(self) -> str: ...
    @property
    def shape(self) -> list[int]:
        """Shape in numpy order."""
    @property
    def ggml_shape(self) -> list[int]:
        """Shape as stored in the file (innermost dimension first)."""
    @property
    def dtype(self) -> str:
        """GGML type name, for example ``"I2_S"`` or ``"F16"``."""
    @property
    def dtype_id(self) -> int: ...
    @property
    def is_ternary(self) -> bool: ...
    @property
    def n_elements(self) -> int: ...
    @property
    def offset(self) -> int: ...
    @property
    def nbytes(self) -> Optional[int]: ...

class GgufFile:
    def __init__(self, path: Union[str, "os.PathLike[str]"]) -> None: ...
    @property
    def path(self) -> str: ...
    @property
    def version(self) -> int: ...
    @property
    def architecture(self) -> Optional[str]: ...
    @property
    def data_start(self) -> int: ...
    def __len__(self) -> int: ...
    def __contains__(self, name: str) -> bool: ...
    def tensors(self) -> list[TensorInfo]: ...
    def tensor(self, name: str) -> TensorInfo: ...
    def metadata_keys(self) -> list[str]: ...
    def get_metadata(self, key: str, default: Any = None) -> Any: ...
    def metadata(self) -> dict[str, Any]: ...
    def tensor_scale(self, name: str) -> Optional[float]: ...
    def read_raw(self, name: str) -> bytes: ...
    def read_trits(self, name: str) -> NDArray[np.int8]: ...
    def transcode(self, name: str, *, kernel: Optional[KernelName] = None) -> PackedMatrix: ...
    def close(self) -> None: ...
    def __enter__(self) -> "GgufFile": ...
    def __exit__(self, *args: Any) -> None: ...

def open(path: Union[str, "os.PathLike[str]"]) -> GgufFile:
    """Shorthand for ``GgufFile(path)``."""

# ---------------------------------------------------------------------------------------------------------------------
# Runtime level
# ---------------------------------------------------------------------------------------------------------------------

class Tokenizer:
    def encode(self, text: str, *, add_bos: bool = False) -> list[int]: ...
    def decode(self, ids: Sequence[int]) -> str: ...
    def decode_token(self, id: int) -> str: ...
    def decode_pieces(self, ids: Sequence[int]) -> list[str]:
        """The text pieces a token stream yields for these ids (one per id plus one for held-back text)."""
    def id_to_token(self, id: int) -> Optional[str]: ...
    @property
    def bos_token_id(self) -> Optional[int]: ...
    @property
    def eos_token_id(self) -> Optional[int]: ...
    @property
    def vocab_size(self) -> int: ...
    def __len__(self) -> int: ...

class TokenStream:
    """Iterator over the text pieces of one generation."""

    def __iter__(self) -> Iterator[str]: ...
    def __next__(self) -> str: ...
    @property
    def text(self) -> str:
        """Everything yielded so far."""
    @property
    def n_tokens(self) -> int: ...
    @property
    def token_ids(self) -> list[int]:
        """Ids of the tokens sampled so far (without the end-of-sequence token)."""
    @property
    def finish_reason(self) -> Optional[Literal["length", "eos", "stop", "context", "empty", "closed"]]: ...
    def close(self) -> None: ...
    def __enter__(self) -> "TokenStream": ...
    def __exit__(self, *args: Any) -> None: ...

class Model:
    @staticmethod
    def load(
        path: Union[str, "os.PathLike[str]"],
        *,
        threads: Optional[int] = None,
        kernel: Optional[KernelName] = None,
        verbose: bool = False,
    ) -> "Model": ...
    @property
    def path(self) -> str: ...
    @property
    def kernel(self) -> str: ...
    @property
    def threads(self) -> int: ...
    @threads.setter
    def threads(self, value: int) -> None: ...
    @property
    def config(self) -> dict[str, Any]: ...
    @property
    def architecture(self) -> str: ...
    @property
    def vocab_size(self) -> int: ...
    @property
    def context_length(self) -> int: ...
    @property
    def n_layers(self) -> int: ...
    @property
    def hidden_size(self) -> int: ...
    @property
    def tokenizer(self) -> Tokenizer: ...
    def reset(self) -> None: ...
    def generate(
        self,
        prompt: str,
        max_tokens: int = 128,
        *,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        seed: int = 42,
        stop: Union[str, Sequence[str], None] = None,
    ) -> str: ...
    def stream(
        self,
        prompt: str,
        max_tokens: int = 128,
        *,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        seed: int = 42,
        stop: Union[str, Sequence[str], None] = None,
    ) -> TokenStream: ...
    def format_chat(self, messages: Sequence[dict[str, str]]) -> str: ...
    def chat(
        self,
        messages: Sequence[dict[str, str]],
        max_tokens: int = 128,
        *,
        temperature: float = 0.0,
        top_k: int = 40,
        top_p: float = 0.9,
        seed: int = 42,
        stop: Union[str, Sequence[str], None] = None,
    ) -> str: ...
    def logits(self, prompt: Union[str, Sequence[int]]) -> NDArray[np.float32]: ...

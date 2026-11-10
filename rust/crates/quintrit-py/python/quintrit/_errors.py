"""Exception hierarchy of the ``quintrit`` package.

Every error raised by the native code is an instance of :class:`QuintritError`. Where a builtin exception is the
natural parent (``ValueError`` for bad values, ``TypeError`` for bad dtypes, ``FileNotFoundError`` ...), the classes
inherit from it too, so both ``except QuintritError`` and ``except ValueError`` work.

The classes are created here (not in Rust) because they have several bases; the native module imports this module and
instantiates them by name.
"""

from __future__ import annotations

__all__ = [
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


class QuintritError(Exception):
    """Base class of all errors raised by ``quintrit``."""


class ShapeError(QuintritError, ValueError):
    """An array or tensor has the wrong shape or number of dimensions."""


class DTypeError(QuintritError, TypeError):
    """An argument has the wrong type or numpy dtype (for example an array that is not ``int8``)."""


class InvalidArgumentError(QuintritError, ValueError):
    """A scalar argument is out of range (thread count, sampling parameter, ...)."""


class TritValueError(QuintritError, ValueError):
    """A trit array contains a value other than -1, 0 or +1."""


class KernelError(QuintritError, ValueError):
    """An unknown kernel name was requested."""


class UnsupportedKernelError(KernelError):
    """A known kernel was requested that this CPU cannot run."""


class SerializationError(QuintritError, ValueError):
    """Serialized ``PackedMatrix`` data is malformed."""


class FormatError(QuintritError, ValueError):
    """A file is not valid or uses an unsupported format or tensor type."""


class TensorNotFoundError(QuintritError, KeyError):
    """A tensor name is not present in the file."""

    def __str__(self) -> str:  # KeyError would show the repr of the message
        return Exception.__str__(self)


class QuintritIOError(QuintritError, OSError):
    """A file could not be read."""


class QuintritFileNotFoundError(QuintritIOError, FileNotFoundError):
    """A file or directory does not exist."""


class ModelLoadError(QuintritError, RuntimeError):
    """A model could not be loaded."""


class TokenizerError(QuintritError, RuntimeError):
    """The tokenizer failed."""


class GenerationError(QuintritError, RuntimeError):
    """Text generation failed."""


class StaleStreamError(GenerationError):
    """A token stream was invalidated because the model started another generation, or was reset."""


class ContextOverflowError(GenerationError, ValueError):
    """The prompt does not fit into the model's context window."""


class InternalError(QuintritError, RuntimeError):
    """An unexpected failure inside the native code (a bug); the message names the failing operation."""

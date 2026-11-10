"""Writers for small synthetic GGUF files, so that the tests need no model download.

``write_gguf`` writes a GGUF v3 container; ``pack_i2_s`` / ``pack_tq2_0`` produce the tensor payloads of the two
ternary formats the tests use; ``write_tiny_model`` builds a complete (random) language model with a byte-level BPE
tokenizer that the runtime can load and run.
"""

from __future__ import annotations

import struct
from pathlib import Path
from typing import Any, Sequence

import numpy as np

# GGML type ids (see quintrit_format::gguf)
F32, F16, TQ2_0, I2_S = 0, 1, 35, 36

ALIGNMENT = 32


def _string(s: str) -> bytes:
    raw = s.encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _value(v: Any) -> tuple[int, bytes]:
    if isinstance(v, bool):
        return 7, struct.pack("<B", int(v))
    if isinstance(v, int):
        return 4, struct.pack("<I", v)
    if isinstance(v, float):
        return 6, struct.pack("<f", v)
    if isinstance(v, str):
        return 8, _string(v)
    if isinstance(v, (list, tuple)):
        if not v:
            raise ValueError("empty arrays are not supported by the test writer")
        types = {_value(item)[0] for item in v}
        assert len(types) == 1, "array items must share one type"
        item_type = types.pop()
        body = b"".join(_value(item)[1] for item in v)
        return 9, struct.pack("<IQ", item_type, len(v)) + body
    raise TypeError(f"unsupported metadata value {v!r}")


def write_gguf(
    path: Path,
    metadata: dict[str, Any],
    tensors: Sequence[tuple[str, Sequence[int], int, bytes]],
) -> Path:
    """Writes a GGUF v3 file.

    ``tensors`` holds ``(name, ne, ggml_type, data)`` where ``ne`` is the GGUF dimension list (innermost first).
    """
    head = bytearray(b"GGUF")
    head += struct.pack("<IQQ", 3, len(tensors), len(metadata))
    for key, value in metadata.items():
        vtype, body = _value(value)
        head += _string(key) + struct.pack("<I", vtype) + body

    offsets = []
    cursor = 0
    for _, _, _, data in tensors:
        offsets.append(cursor)
        cursor += (len(data) + ALIGNMENT - 1) // ALIGNMENT * ALIGNMENT
    for (name, ne, ggml_type, _), offset in zip(tensors, offsets):
        head += _string(name) + struct.pack("<I", len(ne))
        head += b"".join(struct.pack("<Q", d) for d in ne)
        head += struct.pack("<IQ", ggml_type, offset)

    pad = (-len(head)) % ALIGNMENT
    blob = bytearray(head) + b"\x00" * pad
    for (_, _, _, data), _offset in zip(tensors, offsets):
        blob += data + b"\x00" * ((-len(data)) % ALIGNMENT)
    path.write_bytes(bytes(blob))
    return path


def pack_i2_s(trits: np.ndarray, scale: float) -> bytes:
    """``GGML_TYPE_I2_S`` payload of a ``(rows, cols)`` trit matrix (cols % 128 == 0) plus the trailing f32 scale."""
    rows, cols = trits.shape
    assert cols % 128 == 0
    codes = (trits.astype(np.uint8) + 1).reshape(rows, cols // 128, 4, 32)
    packed = (codes[:, :, 0] << 6) | (codes[:, :, 1] << 4) | (codes[:, :, 2] << 2) | codes[:, :, 3]
    return packed.astype(np.uint8).tobytes() + struct.pack("<f", scale)


def pack_tq2_0(trits: np.ndarray, scale: float) -> bytes:
    """``GGML_TYPE_TQ2_0`` payload (256 weights per 66 byte block, f16 scale in every block)."""
    flat = trits.reshape(-1)
    assert flat.size % 256 == 0
    codes = (flat.astype(np.uint8) + 1).reshape(-1, 2, 4, 32)  # block, half, plane l, byte m
    qs = np.zeros((codes.shape[0], 2, 32), dtype=np.uint8)
    for plane in range(4):
        qs |= codes[:, :, plane, :] << (2 * plane)
    d = np.float16(scale).tobytes()
    out = bytearray()
    for block in qs:
        out += block.tobytes() + d
    return bytes(out)


# ---------------------------------------------------------------------------------------------------------------------
# Tiny language model
# ---------------------------------------------------------------------------------------------------------------------

HIDDEN = 128
FFN = 256
HEADS = 4
KV_HEADS = 2
LAYERS = 2
CONTEXT = 64


def _bytes_to_unicode() -> dict[int, str]:
    """GPT-2 byte-to-character table used by byte-level BPE vocabularies."""
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(0xA1, 0xAD)) + list(range(0xAE, 0x100))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return {b: chr(c) for b, c in zip(bs, cs)}


def tiny_vocab() -> tuple[list[str], list[str], int, int]:
    """Byte-level vocabulary: 256 bytes, three merges and BOS / EOS markers. Returns (tokens, merges, bos, eos)."""
    table = _bytes_to_unicode()
    tokens = [table[b] for b in range(256)]
    space = table[ord(" ")]
    tokens += ["th", "the", space + "the"]
    merges = ["t h", "th e", f"{space} the"]
    bos = len(tokens)
    tokens.append("<s>")
    eos = len(tokens)
    tokens.append("</s>")
    return tokens, merges, bos, eos


def write_tiny_model(path: Path, seed: int = 7, overrides: dict[str, Any] | None = None) -> Path:
    """Writes a random two-layer llama-style model (hidden 128, 261 tokens) that loads with every kernel."""
    rng = np.random.default_rng(seed)
    tokens, merges, bos, eos = tiny_vocab()
    vocab = len(tokens)
    head_dim = HIDDEN // HEADS
    q_dim, kv_dim = HEADS * head_dim, KV_HEADS * head_dim

    meta: dict[str, Any] = {
        "general.architecture": "llama",
        "general.name": "tiny-test-model",
        "llama.vocab_size": vocab,
        "llama.embedding_length": HIDDEN,
        "llama.feed_forward_length": FFN,
        "llama.block_count": LAYERS,
        "llama.attention.head_count": HEADS,
        "llama.attention.head_count_kv": KV_HEADS,
        "llama.attention.layer_norm_rms_epsilon": 1e-5,
        "llama.rope.freq_base": 10000.0,
        "llama.context_length": CONTEXT,
        "tokenizer.ggml.model": "gpt2",
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.ggml.tokens": tokens,
        "tokenizer.ggml.merges": merges,
        "tokenizer.ggml.bos_token_id": bos,
        "tokenizer.ggml.eos_token_id": eos,
    }
    meta.update(overrides or {})

    def f32(shape: Sequence[int], value: float | None = None) -> bytes:
        arr = np.full(shape, value, np.float32) if value is not None else rng.normal(0, 0.5, shape).astype(np.float32)
        return arr.tobytes()

    def f16(shape: Sequence[int]) -> bytes:
        return rng.normal(0, 0.5, shape).astype(np.float16).tobytes()

    def ternary(rows: int, cols: int) -> bytes:
        trits = rng.integers(-1, 2, size=(rows, cols)).astype(np.int8)
        return pack_i2_s(trits, 0.05)

    tensors: list[tuple[str, Sequence[int], int, bytes]] = [
        ("token_embd.weight", [HIDDEN, vocab], F16, f16((vocab, HIDDEN))),
        ("output_norm.weight", [HIDDEN], F32, f32((HIDDEN,), 1.0)),
        ("output.weight", [HIDDEN, vocab], F16, f16((vocab, HIDDEN))),
    ]
    for layer in range(LAYERS):
        p = f"blk.{layer}."
        tensors += [
            (p + "attn_norm.weight", [HIDDEN], F32, f32((HIDDEN,), 1.0)),
            (p + "ffn_norm.weight", [HIDDEN], F32, f32((HIDDEN,), 1.0)),
            (p + "attn_q.weight", [HIDDEN, q_dim], I2_S, ternary(q_dim, HIDDEN)),
            (p + "attn_k.weight", [HIDDEN, kv_dim], I2_S, ternary(kv_dim, HIDDEN)),
            (p + "attn_v.weight", [HIDDEN, kv_dim], I2_S, ternary(kv_dim, HIDDEN)),
            (p + "attn_output.weight", [q_dim, HIDDEN], I2_S, ternary(HIDDEN, q_dim)),
            (p + "ffn_gate.weight", [HIDDEN, FFN], I2_S, ternary(FFN, HIDDEN)),
            (p + "ffn_up.weight", [HIDDEN, FFN], I2_S, ternary(FFN, HIDDEN)),
            (p + "ffn_down.weight", [FFN, HIDDEN], I2_S, ternary(HIDDEN, FFN)),
        ]
    return write_gguf(path, meta, tensors)

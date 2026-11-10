"""Format level: GGUF inspection, ternary tensor extraction and transcoding (synthetic files, no model needed)."""

from __future__ import annotations

import os
import struct

import numpy as np
import pytest

import quintrit
from conftest import KERNELS, random_acts, random_trits, reference_gemv
from gguf_builder import F16, F32, I2_S, TQ2_0, pack_i2_s, pack_tq2_0, write_gguf


@pytest.fixture
def sample(tmp_path, rng):
    """A small GGUF file with an I2_S, a TQ2_0, an F32 and an F16 tensor."""
    i2s = random_trits(rng, 96, 256)
    tq2 = random_trits(rng, 4, 256)
    norm = rng.normal(size=(24,)).astype(np.float32)
    half = rng.normal(size=(5, 8)).astype(np.float16)
    meta = {
        "general.architecture": "llama",
        "general.name": "sample",
        "llama.context_length": 2048,
        "llama.rope.freq_base": 10000.0,
        "tokenizer.ggml.add_bos_token": True,
        "tokenizer.ggml.tokens": ["a", "b", "c"],
        "sample.ints": [1, 2, 3],
    }
    tensors = [
        ("w.i2s", [256, 96], I2_S, pack_i2_s(i2s, 0.125)),
        ("w.tq2", [256, 4], TQ2_0, pack_tq2_0(tq2, 0.5)),
        ("norm", [24], F32, norm.tobytes()),
        ("half", [8, 5], F16, half.tobytes()),
    ]
    path = write_gguf(tmp_path / "sample.gguf", meta, tensors)
    return path, {"i2s": i2s, "tq2": tq2, "norm": norm, "half": half}


def test_open_and_basic_properties(sample):
    path, _ = sample
    with quintrit.open(path) as f:
        assert isinstance(f, quintrit.GgufFile)
        assert f.version == 3
        assert f.architecture == "llama"
        assert len(f) == 4
        assert "w.i2s" in f and "missing" not in f
        assert f.path == str(path)
        assert f.data_start % 32 == 0


def test_open_accepts_pathlike_and_str(sample):
    path, _ = sample
    for p in (path, str(path), os.fspath(path)):
        with quintrit.GgufFile(p) as f:
            assert len(f) == 4


def test_list_tensors(sample):
    path, _ = sample
    with quintrit.GgufFile(path) as f:
        infos = f.tensors()
        assert [t.name for t in infos] == ["w.i2s", "w.tq2", "norm", "half"]  # file order
        by_name = {t.name: t for t in infos}
        t = by_name["w.i2s"]
        assert t.shape == [96, 256] and t.ggml_shape == [256, 96]
        assert t.dtype == "I2_S" and t.dtype_id == 36 and t.is_ternary
        assert t.n_elements == 96 * 256
        assert t.nbytes == 96 * 256 // 4
        assert by_name["w.tq2"].dtype == "TQ2_0" and by_name["w.tq2"].nbytes == 4 * 256 // 256 * 66
        assert by_name["norm"].dtype == "F32" and not by_name["norm"].is_ternary and by_name["norm"].nbytes == 96
        assert by_name["half"].shape == [5, 8] and by_name["half"].dtype == "F16"
        assert "w.i2s" in repr(t)
        assert f.tensor("norm").name == "norm"


def test_metadata(sample):
    path, _ = sample
    with quintrit.GgufFile(path) as f:
        keys = f.metadata_keys()
        assert keys == sorted(keys) and "general.architecture" in keys
        assert f.get_metadata("llama.context_length") == 2048
        assert f.get_metadata("llama.rope.freq_base") == pytest.approx(10000.0)
        assert f.get_metadata("tokenizer.ggml.add_bos_token") is True
        assert f.get_metadata("tokenizer.ggml.tokens") == ["a", "b", "c"]
        assert f.get_metadata("sample.ints") == [1, 2, 3]
        assert f.get_metadata("nope") is None
        assert f.get_metadata("nope", "fallback") == "fallback"
        meta = f.metadata()
        assert meta["general.name"] == "sample" and set(meta) == set(keys)


def test_read_ternary_tensors(sample):
    path, truth = sample
    with quintrit.GgufFile(path) as f:
        i2s = f.read_trits("w.i2s")
        assert i2s.dtype == np.int8 and i2s.shape == (96, 256)
        np.testing.assert_array_equal(i2s, truth["i2s"])
        np.testing.assert_array_equal(f.read_trits("w.tq2"), truth["tq2"])
        np.testing.assert_array_equal(quintrit.read_tensor(f, "w.i2s"), truth["i2s"])


def test_tensor_scale(sample):
    path, _ = sample
    with quintrit.GgufFile(path) as f:
        assert f.tensor_scale("w.i2s") == pytest.approx(0.125)
        assert f.tensor_scale("w.tq2") == pytest.approx(0.5)
        assert f.tensor_scale("norm") is None
        with pytest.raises(quintrit.TensorNotFoundError):
            f.tensor_scale("missing")


def test_read_float_tensors(sample):
    path, truth = sample
    with quintrit.GgufFile(path) as f:
        norm = quintrit.read_tensor(f, "norm")
        assert norm.dtype == np.float32
        np.testing.assert_array_equal(norm, truth["norm"])
        half = quintrit.read_tensor(f, "half")
        assert half.dtype == np.float16 and half.shape == (5, 8)
        np.testing.assert_array_equal(half, truth["half"])
        assert f.read_raw("norm") == truth["norm"].tobytes()


@pytest.mark.parametrize("kernel", KERNELS)
def test_transcode_to_packed_matrix(sample, kernel, rng):
    path, truth = sample
    with quintrit.GgufFile(path) as f:
        packed = f.transcode("w.i2s", kernel=kernel)
    assert isinstance(packed, quintrit.PackedMatrix)
    assert packed.shape == (96, 256) and packed.kernel == kernel
    np.testing.assert_array_equal(packed.unpack(), truth["i2s"])
    acts = random_acts(rng, 256)
    np.testing.assert_array_equal(quintrit.gemv(acts, packed), reference_gemv(truth["i2s"], acts))
    # identical to packing the trits directly
    assert packed == quintrit.pack(truth["i2s"], kernel=kernel)


def test_transcode_tq2_with_row_remainder(sample, rng):
    path, truth = sample
    with quintrit.GgufFile(path) as f:
        packed = f.transcode("w.tq2")
    assert packed.shape == (4, 256)
    acts = random_acts(rng, 256)
    np.testing.assert_array_equal(quintrit.gemv(acts, packed), reference_gemv(truth["tq2"], acts))


def test_errors_for_missing_and_unsupported_tensors(sample):
    path, _ = sample
    with quintrit.GgufFile(path) as f:
        with pytest.raises(quintrit.TensorNotFoundError, match="missing"):
            f.tensor("missing")
        with pytest.raises(quintrit.TensorNotFoundError):
            f.read_trits("missing")
        with pytest.raises(KeyError):
            f.read_raw("missing")
        with pytest.raises(quintrit.FormatError, match="not a ternary type"):
            f.read_trits("norm")
        with pytest.raises(quintrit.FormatError):
            f.transcode("half")
        with pytest.raises(quintrit.KernelError):
            f.transcode("w.i2s", kernel="bogus")


def test_one_dimensional_ternary_tensor_is_rejected(tmp_path, rng):
    trits = random_trits(rng, 1, 128)
    path = write_gguf(tmp_path / "one_d.gguf", {"general.architecture": "llama"}, [("v", [128], I2_S, pack_i2_s(trits, 1.0))])
    with quintrit.GgufFile(path) as f:
        with pytest.raises(quintrit.ShapeError, match="2-D"):
            f.read_trits("v")


def test_i2_s_columns_must_be_multiple_of_128(tmp_path):
    data = bytes(100)  # claims 100 columns (100 bytes of trits), which the I2_S layout cannot hold
    path = write_gguf(tmp_path / "bad_cols.gguf", {"general.architecture": "llama"}, [("w", [100, 4], I2_S, data)])
    with quintrit.GgufFile(path) as f:
        with pytest.raises(quintrit.FormatError, match="multiple of 128"):
            f.read_trits("w")


def test_tensor_data_outside_the_file(tmp_path):
    # The reader checks every tensor against the file size when it opens the file.
    path = write_gguf(tmp_path / "short.gguf", {"general.architecture": "llama"}, [("w", [128, 64], I2_S, bytes(16))])
    with pytest.raises(quintrit.FormatError, match="beyond the end of the file"):
        quintrit.GgufFile(path)


def test_unreadable_tensor_type(tmp_path):
    path = write_gguf(tmp_path / "q8.gguf", {"general.architecture": "llama"}, [("w", [32, 2], 8, bytes(68))])
    with quintrit.GgufFile(path) as f:
        info = f.tensor("w")
        assert info.dtype == "Q8_0" and info.nbytes is None
        with pytest.raises(quintrit.FormatError):
            f.read_raw("w")
        with pytest.raises(quintrit.FormatError, match="cannot decode"):
            quintrit.read_tensor(f, "w")


def test_unknown_tensor_type_name(tmp_path):
    path = write_gguf(tmp_path / "odd.gguf", {"general.architecture": "llama"}, [("w", [32], 12, bytes(32))])
    with quintrit.GgufFile(path) as f:
        assert f.tensor("w").dtype == "UNKNOWN(12)"


# ---- file level errors ------------------------------------------------------------------------------------------------


def test_missing_file(tmp_path):
    with pytest.raises(quintrit.QuintritFileNotFoundError) as info:
        quintrit.open(tmp_path / "absent.gguf")
    assert isinstance(info.value, FileNotFoundError) and isinstance(info.value, quintrit.QuintritError)


def test_directory_is_not_a_gguf_file(tmp_path):
    with pytest.raises(quintrit.QuintritError):
        quintrit.open(tmp_path)


def test_not_a_gguf_file(tmp_path):
    bogus = tmp_path / "bogus.gguf"
    bogus.write_bytes(b"this is definitely not a model" * 4)
    with pytest.raises(quintrit.FormatError, match="not a valid GGUF"):
        quintrit.open(bogus)


def test_empty_and_truncated_files(tmp_path, sample):
    empty = tmp_path / "empty.gguf"
    empty.write_bytes(b"")
    with pytest.raises(quintrit.FormatError):
        quintrit.open(empty)
    path, _ = sample
    blob = path.read_bytes()
    truncated = tmp_path / "truncated.gguf"
    truncated.write_bytes(blob[:100])
    with pytest.raises(quintrit.FormatError):
        quintrit.open(truncated)
    only_magic = tmp_path / "magic.gguf"
    only_magic.write_bytes(b"GGUF" + struct.pack("<I", 3))
    with pytest.raises(quintrit.FormatError):
        quintrit.open(only_magic)


def test_quintrit_files_are_reported_as_unsupported(tmp_path):
    f = tmp_path / "model.quintrit"
    f.write_bytes(b"whatever")
    with pytest.raises(quintrit.FormatError, match=r"\.quintrit"):
        quintrit.open(f)


def test_wrong_argument_types():
    with pytest.raises(TypeError):
        quintrit.open(None)
    with pytest.raises(TypeError):
        quintrit.open(12)


def test_closed_file_raises(sample):
    path, _ = sample
    f = quintrit.GgufFile(path)
    f.close()
    f.close()  # idempotent
    with pytest.raises(quintrit.QuintritIOError, match="closed"):
        len(f)
    with pytest.raises(quintrit.QuintritIOError):
        f.tensors()
    with pytest.raises(quintrit.QuintritIOError):
        f.metadata()
    with pytest.raises(quintrit.QuintritIOError):
        f.read_trits("w.i2s")


def test_file_can_be_deleted_after_close(sample):
    """The memory map is released by close() (matters on Windows, which refuses to delete mapped files)."""
    path, _ = sample
    with quintrit.GgufFile(path) as f:
        assert len(f) == 4
    path.unlink()
    assert not path.exists()

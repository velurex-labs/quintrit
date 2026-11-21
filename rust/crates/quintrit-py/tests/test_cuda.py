"""GPU support on the tiny synthetic model. Without a build with the cargo feature ``cuda`` (or without a supported GPU) the GPU tests skip themselves;
the tests of the error behaviour always run."""

from __future__ import annotations

import numpy as np
import pytest

import quintrit

PROMPT = "the cat sat on the mat"


def supported_gpu() -> bool:
    return quintrit.cuda_built() and any(d["supported"] for d in quintrit.cuda_devices())


needs_gpu = pytest.mark.skipif(not supported_gpu(), reason="no build with GPU support or no supported NVIDIA GPU")


def test_cuda_queries_are_consistent():
    assert isinstance(quintrit.cuda_built(), bool)
    devices = quintrit.cuda_devices()
    assert isinstance(devices, list)
    if not quintrit.cuda_built():
        assert devices == []
    for d in devices:
        assert set(d) == {"index", "name", "compute_capability", "multiprocessors", "memory_bytes", "supported"}
        assert d["supported"] == (tuple(d["compute_capability"]) >= (8, 0))


def test_unknown_device_names_are_rejected(tiny_model_path):
    for bad in ("tpu", "cuda:x", "cuda0", "gpu"):
        with pytest.raises(quintrit.InvalidArgumentError):
            quintrit.Model.load(tiny_model_path, device=bad)
    with pytest.raises(quintrit.InvalidArgumentError):
        quintrit.Model.load(tiny_model_path, context=128)  # the context option belongs to the GPU


def test_cpu_model_reports_the_cpu(tiny_model_path):
    m = quintrit.Model.load(tiny_model_path, threads=1, device="cpu")
    assert m.device == "cpu"
    assert m.context_window == m.context_length


@pytest.mark.skipif(quintrit.cuda_built(), reason="checks the error of a build without GPU support")
def test_a_build_without_gpu_support_raises_device_error(tiny_model_path):
    with pytest.raises(quintrit.DeviceError):
        quintrit.Model.load(tiny_model_path, device="cuda")


@needs_gpu
def test_gpu_model_matches_the_cpu_model(tiny_model_path):
    cpu = quintrit.Model.load(tiny_model_path, threads=2)
    gpu = quintrit.Model.load(tiny_model_path, device="cuda")
    assert gpu.device == "cuda:0" and gpu.kernel == "cuda" and gpu.threads == 0
    assert "device=" in repr(gpu)
    assert gpu.architecture == cpu.architecture and gpu.vocab_size == cpu.vocab_size
    assert gpu.context_window <= gpu.context_length

    # Bit-identical logits for a text prompt and for token ids.
    a, b = cpu.logits(PROMPT), gpu.logits(PROMPT)
    assert a.dtype == b.dtype == np.float32
    assert np.array_equal(a.view(np.uint32), b.view(np.uint32))
    ids = cpu.tokenizer.encode(PROMPT)
    assert np.array_equal(cpu.logits(ids).view(np.uint32), gpu.logits(ids).view(np.uint32))

    # Greedy and sampled generation, streaming, a stop sequence and a chat prompt give the same text.
    assert gpu.generate(PROMPT, max_tokens=20) == cpu.generate(PROMPT, max_tokens=20)
    assert gpu.generate(PROMPT, max_tokens=20, temperature=0.8, seed=3) == cpu.generate(PROMPT, max_tokens=20, temperature=0.8, seed=3)
    assert "".join(gpu.stream(PROMPT, max_tokens=12)) == "".join(cpu.stream(PROMPT, max_tokens=12))
    msgs = [{"role": "user", "content": "hello"}]
    assert gpu.chat(msgs, max_tokens=10) == cpu.chat(msgs, max_tokens=10)

    # reset() starts a new conversation; the GPU path has no CPU threads to set.
    gpu.reset()
    assert gpu.generate(PROMPT, max_tokens=8) == cpu.generate(PROMPT, max_tokens=8)
    with pytest.raises(quintrit.InvalidArgumentError):
        gpu.threads = 2


@needs_gpu
def test_gpu_context_limits(tiny_model_path):
    gpu = quintrit.Model.load(tiny_model_path, device="cuda:0", context=32)
    # The synthetic model has a window of 64 positions; the GPU window is the smaller requested one.
    assert gpu.context_window <= 64
    with pytest.raises(quintrit.ContextOverflowError):
        gpu.generate("x" * 500, max_tokens=4)
    with pytest.raises(quintrit.DeviceError):
        quintrit.Model.load(tiny_model_path, device="cuda:99")

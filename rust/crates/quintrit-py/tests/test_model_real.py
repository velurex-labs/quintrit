"""Optional test against a real model; skipped unless ``QUINTRIT_TEST_MODEL`` points at a GGUF file (or directory).

    QUINTRIT_TEST_MODEL=/path/to/ggml-model-i2_s.gguf pytest tests/test_model_real.py -s

``QUINTRIT_TEST_TOKENS`` (default 16) sets the number of generated tokens. The test uses the evaluation prompt of the
SDK's parity checks.
"""

from __future__ import annotations

import os

import numpy as np
import pytest

import quintrit

MODEL = os.environ.get("QUINTRIT_TEST_MODEL")
pytestmark = pytest.mark.skipif(not MODEL, reason="set QUINTRIT_TEST_MODEL to run the real-model tests")

PROMPT = "<|user|>\nWhat is ternary computing?\n<|assistant|>\n"
TOKENS = int(os.environ.get("QUINTRIT_TEST_TOKENS", "16"))


@pytest.fixture(scope="module")
def model():
    return quintrit.Model.load(MODEL)


def test_real_model_metadata(model):
    assert model.vocab_size > 1000
    assert model.n_layers > 0
    assert model.tokenizer.vocab_size > 1000


def test_real_model_tokenizer_roundtrip(model):
    ids = model.tokenizer.encode(PROMPT)
    assert ids and model.tokenizer.decode(ids) == PROMPT


def test_real_model_greedy_generation(model):
    text = model.generate(PROMPT, TOKENS)
    assert text.strip(), "greedy generation produced no text"
    assert model.generate(PROMPT, TOKENS) == text
    assert "".join(model.stream(PROMPT, TOKENS)) == text
    print("\n[real model]", repr(text))


def test_real_model_logits_are_finite(model):
    logits = model.logits(PROMPT)
    assert logits.shape == (model.vocab_size,) and np.isfinite(logits).all()


@pytest.mark.parametrize("kernel", quintrit.available_kernels())
def test_real_model_kernels_agree(model, kernel):
    other = quintrit.Model.load(MODEL, kernel=kernel)
    assert other.generate(PROMPT, min(TOKENS, 8)) == model.generate(PROMPT, min(TOKENS, 8))

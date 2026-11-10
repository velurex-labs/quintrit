"""Runtime level on a tiny synthetic model (random weights, byte-level tokenizer): no model download needed."""

from __future__ import annotations

import numpy as np
import pytest

import quintrit
from conftest import KERNELS
from gguf_builder import CONTEXT, FFN, HEADS, HIDDEN, KV_HEADS, LAYERS, tiny_vocab

PROMPT = "the cat sat on the mat"


@pytest.fixture(scope="module")
def model(tiny_model_path):
    return quintrit.Model.load(tiny_model_path, threads=2)


def test_load_is_silent_by_default(tiny_model_path, capfd):
    quintrit.Model.load(tiny_model_path, threads=1)
    out = capfd.readouterr()
    assert out.out == ""
    assert out.err == ""


def test_load_verbose_prints_progress(tiny_model_path, capfd):
    quintrit.Model.load(tiny_model_path, threads=1, verbose=True)
    out = capfd.readouterr().out
    assert "Loading GGUF model" in out
    # The switch is restored: a following default load is silent again.
    quintrit.Model.load(tiny_model_path, threads=1)
    assert capfd.readouterr().out == ""


def test_load_and_properties(model, tiny_model_path):
    tokens, _, bos, eos = tiny_vocab()
    assert model.path == str(tiny_model_path)
    assert model.kernel == quintrit.best_kernel()
    assert model.threads == 2
    assert model.architecture == "llama"
    assert model.vocab_size == len(tokens)
    assert model.context_length == CONTEXT
    assert model.n_layers == LAYERS
    assert model.hidden_size == HIDDEN
    cfg = model.config
    assert cfg["num_attention_heads"] == HEADS and cfg["num_key_value_heads"] == KV_HEADS
    assert cfg["intermediate_size"] == FFN and cfg["head_dim"] == HIDDEN // HEADS
    assert cfg["bos_token_id"] is None or cfg["bos_token_id"] == bos  # BOS is only prepended for some tokenizers
    assert cfg["eos_token_id"] == eos
    assert cfg["hidden_act"] == "silu"
    assert "Model(" in repr(model)


def test_tokenizer_roundtrip(model):
    tok = model.tokenizer
    assert len(tok) == tok.vocab_size == model.vocab_size
    ids = tok.encode(PROMPT)
    assert ids and all(isinstance(i, int) for i in ids)
    assert tok.decode(ids) == PROMPT
    assert tok.decode([]) == ""
    # the registered merges are used: " the" is one token
    assert len(tok.encode(" the")) == 1
    assert len(tok.encode("the")) == 1
    assert tok.encode("") == []
    unicode_text = "naïve café ✓ \n\ttabs"
    assert tok.decode(tok.encode(unicode_text)) == unicode_text
    assert tok.decode_token(tok.encode("a")[0]) == "a"
    assert tok.id_to_token(10_000_000) is None


def test_decode_pieces_never_split_characters(model):
    tok = model.tokenizer
    text = "naïve café ✓ Příliš žluťoučký kůň 😀"
    ids = tok.encode(text)
    assert len(ids) > len(text)  # multi-byte characters take several byte-level tokens
    pieces = tok.decode_pieces(ids)
    assert len(pieces) == len(ids) + 1
    assert "".join(pieces) == text
    assert "�" not in "".join(pieces)
    assert "" in pieces  # tokens that end inside a character are held back
    assert tok.decode_pieces([]) == [""]


def test_tokenizer_bos_handling(model):
    tok = model.tokenizer
    if tok.bos_token_id is None:
        assert tok.encode("hi", add_bos=True) == tok.encode("hi")
    else:
        ids = tok.encode("hi", add_bos=True)
        assert ids[0] == tok.bos_token_id and ids[1:] == tok.encode("hi")
        assert tok.encode("hi", add_bos=True) == ids  # idempotent per call


def test_tokenizer_errors(model):
    tok = model.tokenizer
    with pytest.raises(TypeError):
        tok.encode(123)
    with pytest.raises(TypeError):
        tok.decode("abc")
    with pytest.raises(OverflowError):
        tok.decode([-1])


def test_greedy_generation_is_deterministic(model):
    a = model.generate(PROMPT, 12)
    b = model.generate(PROMPT, 12)
    assert a == b
    assert isinstance(a, str)
    assert model.generate(PROMPT, 12, temperature=0.0) == a


def test_stream_equals_generate(model):
    full = model.generate(PROMPT, 15)
    stream = model.stream(PROMPT, 15)
    pieces = list(stream)
    assert "".join(pieces) == full
    assert stream.text == full
    if stream.finish_reason != "stop":
        assert full == model.tokenizer.decode(stream.token_ids) or "�" in full
    assert len(stream.token_ids) == stream.n_tokens
    assert stream.finish_reason in ("length", "eos")
    assert stream.n_tokens <= 15
    # an exhausted stream stays exhausted
    assert list(stream) == []
    with pytest.raises(StopIteration):
        next(stream)


def test_max_tokens(model):
    assert model.generate(PROMPT, 0) == ""
    one = model.stream(PROMPT, 1)
    list(one)
    assert one.n_tokens <= 1
    short = model.generate(PROMPT, 3)
    longer = model.generate(PROMPT, 9)
    assert longer.startswith(short)
    stream = model.stream(PROMPT, 5)
    list(stream)
    assert stream.finish_reason in ("length", "eos")
    if stream.finish_reason == "length":
        assert stream.n_tokens == 5


def test_stream_context_manager_and_close(model):
    with model.stream(PROMPT, 20) as s:
        first = next(s)
        assert isinstance(first, str)
    assert s.finish_reason == "closed"
    assert list(s) == []


def test_sampling_is_seeded(model):
    kwargs = dict(temperature=1.0, top_k=20, top_p=0.95)
    a = model.generate(PROMPT, 12, seed=5, **kwargs)
    b = model.generate(PROMPT, 12, seed=5, **kwargs)
    assert a == b
    others = {model.generate(PROMPT, 12, seed=s, **kwargs) for s in range(6, 14)}
    assert len(others | {a}) > 1, "different seeds should give different samples"


def test_stop_sequences(model):
    free = model.generate(PROMPT, 20)
    if len(free) < 4:
        pytest.skip("the random model produced too little text")
    marker = free[2:4]
    cut = model.generate(PROMPT, 20, stop=marker)
    assert marker not in cut
    assert cut == free[: free.index(marker)]
    streamed = "".join(model.stream(PROMPT, 20, stop=[marker]))
    assert streamed == cut
    s = model.stream(PROMPT, 20, stop=marker)
    list(s)
    assert s.finish_reason == "stop"
    # a stop string that never occurs changes nothing
    assert model.generate(PROMPT, 20, stop="☃☃") == free
    assert model.generate(PROMPT, 20, stop=[""]) == free  # empty stop strings are ignored


def test_new_generation_invalidates_old_stream(model):
    old = model.stream(PROMPT, 10)
    next(old)
    new = model.stream(PROMPT, 10)
    with pytest.raises(quintrit.StaleStreamError):
        next(old)
    assert isinstance(next(new), str)
    assert old.finish_reason == "closed"
    # the new stream is unaffected; reset() invalidates it as well
    model.reset()
    with pytest.raises(quintrit.StaleStreamError):
        next(new)
    # the model is fully usable afterwards
    assert model.generate(PROMPT, 5) == model.generate(PROMPT, 5)


def test_generate_after_stream_matches(model):
    expected = model.generate(PROMPT, 8)
    s = model.stream("something else entirely", 4)
    next(s)
    assert model.generate(PROMPT, 8) == expected


def test_logits(model):
    logits = model.logits(PROMPT)
    assert logits.dtype == np.float32 and logits.shape == (model.vocab_size,)
    assert np.isfinite(logits).all()
    again = model.logits(PROMPT)
    np.testing.assert_array_equal(logits, again)
    ids = model.tokenizer.encode(PROMPT, add_bos=True)
    np.testing.assert_array_equal(model.logits(ids), logits)
    # greedy generation starts with the argmax of these logits
    first_id = int(np.argmax(logits))
    if first_id != model.config["eos_token_id"]:
        assert model.generate(PROMPT, 1) == model.tokenizer.decode_token(first_id)
    assert not np.array_equal(model.logits("a different prompt"), logits)


def test_logits_errors(model):
    with pytest.raises(quintrit.InvalidArgumentError, match="no tokens"):
        model.logits([])
    with pytest.raises(quintrit.InvalidArgumentError, match="vocabulary"):
        model.logits([model.vocab_size])
    with pytest.raises(quintrit.ContextOverflowError):
        model.logits([3] * (CONTEXT + 1))
    with pytest.raises(quintrit.DTypeError):
        model.logits(3.5)


def test_context_overflow_and_limit(model):
    with pytest.raises(quintrit.ContextOverflowError, match="context window"):
        model.generate("a" * (CONTEXT + 10), 5)
    # generation stops at the end of the context window instead of failing
    prompt = "a" * (CONTEXT - 6)
    s = model.stream(prompt, 100)
    text = "".join(s)
    assert s.finish_reason in ("context", "eos", "length")
    assert s.n_tokens <= 100
    assert isinstance(text, str)


def test_empty_prompt_is_handled(model):
    out = model.generate("", 4)
    assert isinstance(out, str)


@pytest.mark.parametrize(
    "kwargs,exc",
    [
        (dict(max_tokens=-1), quintrit.InvalidArgumentError),
        (dict(temperature=-0.5), quintrit.InvalidArgumentError),
        (dict(temperature=float("nan")), quintrit.InvalidArgumentError),
        (dict(top_p=0.0), quintrit.InvalidArgumentError),
        (dict(top_p=1.5), quintrit.InvalidArgumentError),
        (dict(top_k=-3), quintrit.InvalidArgumentError),
        (dict(stop=5), quintrit.DTypeError),
        (dict(stop=[1, 2]), quintrit.DTypeError),
    ],
)
def test_generation_argument_validation(model, kwargs, exc):
    with pytest.raises(exc):
        model.generate(PROMPT, **{"max_tokens": 3, **kwargs})
    with pytest.raises(exc):
        model.stream(PROMPT, **{"max_tokens": 3, **kwargs})


def test_generate_argument_types(model):
    with pytest.raises(TypeError):
        model.generate(123)
    with pytest.raises(TypeError):
        model.generate(PROMPT, "five")


def test_chat_format_and_generate(model):
    messages = [{"role": "system", "content": "be brief"}, {"role": "user", "content": "hi"}]
    prompt = model.format_chat(messages)
    assert "be brief" in prompt and "hi" in prompt and prompt.rstrip().endswith(":") or prompt.endswith("\n")
    assert model.chat(messages, 6) == model.generate(prompt, 6)
    with pytest.raises(quintrit.InvalidArgumentError, match="role"):
        model.format_chat([{"content": "x"}])
    with pytest.raises(quintrit.DTypeError):
        model.format_chat("hello")
    with pytest.raises(quintrit.DTypeError):
        model.format_chat([("user", "x")])
    with pytest.raises(quintrit.DTypeError):
        model.format_chat([{"role": "user", "content": 5}])


def test_thread_setter(model):
    expected = model.generate(PROMPT, 6)
    model.threads = 1
    assert model.threads == 1
    assert model.generate(PROMPT, 6) == expected
    model.threads = 3
    assert model.generate(PROMPT, 6) == expected
    with pytest.raises(quintrit.InvalidArgumentError):
        model.threads = 0
    model.threads = 2


@pytest.mark.parametrize("kernel", KERNELS)
def test_every_kernel_gives_identical_logits(kernel, tiny_model_path, model):
    other = quintrit.Model.load(tiny_model_path, threads=2, kernel=kernel)
    assert other.kernel == kernel
    np.testing.assert_array_equal(other.logits(PROMPT), model.logits(PROMPT))
    assert other.generate(PROMPT, 10) == model.generate(PROMPT, 10)


def test_thread_count_does_not_change_results(tiny_model_path, model):
    single = quintrit.Model.load(tiny_model_path, threads=1)
    np.testing.assert_array_equal(single.logits(PROMPT), model.logits(PROMPT))


def test_load_from_directory(tiny_model_path):
    m = quintrit.Model.load(tiny_model_path.parent, threads=1)
    assert m.path == str(tiny_model_path)


# ---- load errors -----------------------------------------------------------------------------------------------------


def test_load_missing_path(tmp_path):
    with pytest.raises(quintrit.QuintritFileNotFoundError):
        quintrit.Model.load(tmp_path / "nothing.gguf")


def test_load_empty_directory(tmp_path):
    with pytest.raises(quintrit.ModelLoadError):
        quintrit.Model.load(tmp_path)


def test_load_garbage_file(tmp_path):
    bogus = tmp_path / "garbage.gguf"
    bogus.write_bytes(b"not a gguf at all, just bytes" * 3)
    with pytest.raises(quintrit.ModelLoadError):
        quintrit.Model.load(bogus)


def test_load_gguf_without_model_tensors(tmp_path):
    from gguf_builder import F32, write_gguf

    path = write_gguf(
        tmp_path / "hollow.gguf",
        {"general.architecture": "llama", "tokenizer.ggml.tokens": ["a", "b"], "tokenizer.ggml.model": "gpt2"},
        [("something", [4], F32, bytes(16))],
    )
    with pytest.raises(quintrit.ModelLoadError):
        quintrit.Model.load(path)


def test_inconsistent_hyperparameters_are_a_load_error(tmp_path):
    """A head count of zero used to make the Rust runtime panic (division by zero in the config); the runtime now
    validates the hyper-parameters, so loading fails with a clear error and the interpreter is unaffected."""
    from gguf_builder import write_tiny_model

    path = write_tiny_model(tmp_path / "zero_heads.gguf", overrides={"llama.attention.head_count": 0})
    with pytest.raises(quintrit.ModelLoadError, match=r"attention\.head_count' = 0 is outside"):
        quintrit.Model.load(path)
    huge_context = write_tiny_model(tmp_path / "huge_ctx.gguf", overrides={"llama.context_length": 1 << 30})
    with pytest.raises(quintrit.ModelLoadError, match="context_length"):
        quintrit.Model.load(huge_context)
    # the interpreter and the library are still fine
    assert quintrit.gemv(
        np.ones(4, dtype=np.int8), quintrit.pack(np.ones((2, 4), dtype=np.int8))
    ).tolist() == [4, 4]


def test_load_argument_validation(tiny_model_path):
    with pytest.raises(quintrit.InvalidArgumentError):
        quintrit.Model.load(tiny_model_path, threads=0)
    with pytest.raises(quintrit.KernelError):
        quintrit.Model.load(tiny_model_path, kernel="nope")
    with pytest.raises(TypeError):
        quintrit.Model.load(None)
    with pytest.raises(TypeError):
        quintrit.Model()

//! Guards of the Rust engine (`quintrit-runtime`) against damaged models and over-long prompts, on the tiny synthetic
//! model of the C API tests. The Python package loads models through the same engine, so these are the checks it
//! relies on; before they existed, a damaged file or a prompt longer than the context window made the engine panic
//! (and abort the process with `panic = "abort"`).

mod common;

use common::synth::*;
use quintrit_runtime::{InferenceEngine, SamplingConfig};
use std::panic::{catch_unwind, AssertUnwindSafe};

fn load(path: &std::path::Path) -> Result<InferenceEngine, String> {
    catch_unwind(AssertUnwindSafe(|| InferenceEngine::load_from_gguf(path, 1, None)))
        .unwrap_or_else(|_| panic!("loading {} panicked instead of returning an error", path.display()))
}

#[test]
fn damaged_models_are_rejected_by_the_engine_with_an_error() {
    let dir = TempDir::new("engine-guards-load");
    for (name, flaws, needle) in [
        ("zero_heads", Flaws { zero_heads: true, ..Default::default() }, "attention.head_count"),
        ("no_tokens", Flaws { drop_tokens: true, ..Default::default() }, "tokenizer.ggml.tokens"),
        ("short_norm", Flaws { short_norm: true, ..Default::default() }, "attn_norm"),
        ("vocab_mismatch", Flaws { vocab_mismatch: true, ..Default::default() }, "vocab_size"),
        ("odd_cols", Flaws { odd_i2s_cols: true, ..Default::default() }, "I2_S"),
    ] {
        let path = dir.path(&format!("{name}.gguf"));
        write_model(&path, 5, &flaws);
        let err = load(&path).err().unwrap_or_else(|| panic!("{name}: the damaged model was accepted"));
        assert!(err.contains(needle), "{name}: error {err:?} does not mention {needle:?}");
    }
    let path = dir.path("ok.gguf");
    write_model(&path, 5, &Flaws::default());
    assert!(load(&path).is_ok());
}

#[test]
fn prompt_longer_than_the_context_is_an_error_and_generation_stops_at_the_context() {
    let dir = TempDir::new("engine-guards-ctx");
    let path = dir.path("model.gguf");
    write_model(&path, 9, &Flaws::default());
    let mut engine = load(&path).unwrap();
    let ctx = engine.kv_cache.max_seq_len();
    assert_eq!(ctx, CONTEXT);

    // The byte-level vocabulary has no merges: one token per byte, plus BOS.
    let too_long = "x".repeat(CONTEXT);
    let err = engine
        .generate_streaming(&too_long, 4, &SamplingConfig::greedy(), |_| {})
        .expect_err("a prompt of CONTEXT + 1 tokens must be rejected");
    assert!(err.contains("context window"), "{err}");
    let err = engine.benchmark(&too_long, 4).expect_err("the benchmark must reject it as well");
    assert!(err.contains("context window"), "{err}");

    // A prompt that fills the window exactly is accepted; generation then stops instead of running past the cache.
    let exact = "x".repeat(CONTEXT - 1);
    let mut pieces = 0;
    engine.generate_streaming(&exact, 100, &SamplingConfig::greedy(), |_| pieces += 1).unwrap();
    assert!(pieces <= 1, "at most the token sampled from the last position can be emitted, got {pieces}");

    let short = "xxxx";
    let report = engine.benchmark(short, 10_000).unwrap();
    assert_eq!(report.prompt_tokens, 5);
    assert!(report.generated_tokens <= CONTEXT - report.prompt_tokens, "{} tokens generated", report.generated_tokens);

    // The engine is still usable afterwards.
    engine.generate_streaming(short, 3, &SamplingConfig::greedy(), |_| {}).unwrap();
}

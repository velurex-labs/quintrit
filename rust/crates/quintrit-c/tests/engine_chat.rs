//! Multi-turn generation of the Rust engine (`quintrit-runtime`) on the tiny synthetic model of the C API tests: the
//! KV cache of the previous call is reused for the shared prefix of the next prompt, and that must not change the
//! result, and the streamed text must be exactly the returned text.

mod common;

use common::synth::*;
use quintrit_runtime::{InferenceEngine, SamplingConfig};

fn load(path: &std::path::Path) -> InferenceEngine {
    InferenceEngine::load_from_gguf(path, 1, None).unwrap()
}

fn sampled() -> SamplingConfig {
    SamplingConfig { temperature: 0.8, top_k: 40, top_p: 0.95, seed: 3 }
}

/// Generates with an emptied cache, the result every reuse must reproduce.
fn fresh(engine: &mut InferenceEngine, prompt: &str, max_tokens: usize, sampling: &SamplingConfig) -> String {
    engine.kv_cache.reset();
    engine.generate_streaming(prompt, max_tokens, sampling, |_| {}).unwrap()
}

#[test]
fn reusing_the_kv_cache_for_a_shared_prefix_gives_the_same_text() {
    let dir = TempDir::new("engine-chat-reuse");
    let path = dir.path("model.gguf");
    write_model(&path, 11, &Flaws::default());
    let mut engine = load(&path);
    let mut reference = load(&path);

    for sampling in [SamplingConfig::greedy(), sampled()] {
        engine.kv_cache.reset();
        let first = engine.generate_streaming("hi there", 6, &sampling, |_| {}).unwrap();
        assert_eq!(first, fresh(&mut reference, "hi there", 6, &sampling));

        // The second prompt extends the first one: only its new tokens may be computed.
        let resets = engine.kv_cache.reset_count();
        let second_prompt = "hi there, and more";
        let second = engine.generate_streaming(second_prompt, 6, &sampling, |_| {}).unwrap();
        assert_eq!(engine.kv_cache.reset_count(), resets, "the shared prefix must be reused, not recomputed");
        assert_eq!(second, fresh(&mut reference, second_prompt, 6, &sampling));

        // The same prompt again: everything but the last prompt token comes from the cache.
        let again = engine.generate_streaming(second_prompt, 6, &sampling, |_| {}).unwrap();
        assert_eq!(again, second);

        // A prompt that diverges early, and one that is shorter than the cached sequence.
        for prompt in ["ho", "hi"] {
            let text = engine.generate_streaming(prompt, 6, &sampling, |_| {}).unwrap();
            assert_eq!(text, fresh(&mut reference, prompt, 6, &sampling), "prompt {prompt:?}");
        }
    }
}

#[test]
fn the_cache_is_not_reused_after_someone_else_has_used_it() {
    let dir = TempDir::new("engine-chat-foreign");
    let path = dir.path("model.gguf");
    write_model(&path, 12, &Flaws::default());
    let mut engine = load(&path);
    let mut reference = load(&path);
    let greedy = SamplingConfig::greedy();
    let prompt = "abc def";
    let want = fresh(&mut reference, prompt, 8, &greedy);

    engine.generate_streaming(prompt, 8, &greedy, |_| {}).unwrap();
    // The benchmark refills the cache with another sequence.
    engine.benchmark("zzzzzz", 4).unwrap();
    assert_eq!(engine.generate_streaming(prompt, 8, &greedy, |_| {}).unwrap(), want);

    // An explicit reset (the /reset command of the chat) as well.
    engine.kv_cache.reset();
    assert_eq!(engine.generate_streaming(prompt, 8, &greedy, |_| {}).unwrap(), want);
}

#[test]
fn a_chat_with_a_reused_cache_matches_a_chat_that_recomputes_every_turn() {
    let dir = TempDir::new("engine-chat-turns");
    let path = dir.path("model.gguf");
    write_model(&path, 13, &Flaws::default());
    let mut engine = load(&path);
    let mut reference = load(&path);
    let sampling = sampled();
    let stops = ["U"];

    // The vocabulary has one token per byte and the context holds only CONTEXT tokens, so the turns use a minimal
    // template instead of `format_chat_prompt`; what matters is that each prompt re-sends the previous turns as text.
    let mut history = String::new();
    for user in ["x", "y", "z"] {
        let prompt = format!("{history}U{user}A");

        let mut streamed = String::new();
        let reply = engine
            .generate_streaming_with_stops(&prompt, 5, &sampling, &stops, |piece| streamed.push_str(piece))
            .unwrap();
        assert_eq!(streamed, reply, "the returned text must be exactly the streamed text");

        reference.kv_cache.reset();
        let want = reference.generate_streaming_with_stops(&prompt, 5, &sampling, &stops, |_| {}).unwrap();
        assert_eq!(reply, want, "turn {user:?}");

        history = format!("{prompt}{reply}E");
    }
}

#[test]
fn generation_stats_count_tokens_not_callback_pieces() {
    let dir = TempDir::new("engine-chat-stats");
    let path = dir.path("model.gguf");
    write_model(&path, 11, &Flaws::default());
    let mut engine = load(&path);
    let greedy = SamplingConfig::greedy();
    let prompt = "hi there";

    engine.kv_cache.reset();
    let mut pieces = 0usize;
    let text = engine.generate_streaming(prompt, 6, &greedy, |_| pieces += 1).unwrap();
    let stats = engine.last_generation_stats();
    assert_eq!(stats.prompt_tokens, prompt.len() + 1, "one token per byte plus BOS");
    assert_eq!(stats.reused_prompt_tokens, 0);
    assert!(stats.generated_tokens <= 6 && stats.generated_tokens >= pieces, "{stats:?}, {pieces} pieces");
    // The synthetic model emits arbitrary bytes, so every generated token must still be accounted for in the text.
    assert!(text.len() <= stats.generated_tokens * 3, "{stats:?}");

    // The same prompt again: all but the last prompt token come from the cache.
    engine.generate_streaming(prompt, 6, &greedy, |_| {}).unwrap();
    let again = engine.last_generation_stats();
    assert_eq!(again.reused_prompt_tokens, again.prompt_tokens - 1);
    assert_eq!(again.generated_tokens, stats.generated_tokens);

    // A failed call leaves no stale numbers behind.
    assert!(engine.generate_streaming(&"x".repeat(CONTEXT), 4, &greedy, |_| {}).is_err());
    assert_eq!(engine.last_generation_stats(), quintrit_runtime::GenerationStats::default());
}

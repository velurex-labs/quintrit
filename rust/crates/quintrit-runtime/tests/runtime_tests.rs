//! Integration tests for the Quintrit standalone transformer runtime.

use quintrit_format::decoders::i2_s::unpack_gguf_i2_s;
use quintrit_runtime::kv_cache::KVCache;
use quintrit_runtime::layer::{dispatch_weight_from_i2_s, TernaryKernelType};
use quintrit_runtime::ops::{f16_to_f32, f32_to_f16, relu2, rms_norm, silu, swiglu};
use quintrit_runtime::sampler::{Sampler, SamplingConfig};
use quintrit_runtime::tokenizer::{QuintritTokenizer, StreamDecoder, GGUF_TOKEN_TYPE_CONTROL};

#[test]
fn test_ops_rmsnorm_and_activations() {
    let x = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
    let weight = vec![1.0f32; 8];
    let mut out = vec![0.0f32; 8];
    rms_norm(&x, &weight, 1e-5, &mut out);

    // Sum of squares of normed output should equal length (8) since weight is 1.0
    let sum_sq: f32 = out.iter().map(|&v| v * v).sum();
    assert!((sum_sq - 8.0).abs() < 1e-3);

    // Squared ReLU
    let mut out_relu2 = vec![0.0f32; 8];
    relu2(&x, &mut out_relu2);
    assert_eq!(out_relu2[0], 1.0);
    assert_eq!(out_relu2[1], 0.0);
    assert_eq!(out_relu2[2], 9.0);
    assert_eq!(out_relu2[3], 0.0);

    // SiLU
    let mut out_silu = vec![0.0f32; 8];
    silu(&x, &mut out_silu);
    assert!(out_silu[0] > 0.0);
    assert!(out_silu[1] < 0.0);

    // SwiGLU
    let mut out_swiglu = vec![0.0f32; 8];
    let up = vec![2.0f32; 8];
    swiglu(&x, &up, &mut out_swiglu);
    for i in 0..8 {
        assert!((out_swiglu[i] - out_silu[i] * 2.0).abs() < 1e-4);
    }
}

#[test]
fn test_fp16_conversions() {
    let test_values = [-10.5f32, -1.0, -0.0, 0.0, 1.0, 2.5, 65504.0];
    for &val in &test_values {
        let f16_bits = f32_to_f16(val);
        let back = f16_to_f32(f16_bits);
        assert!((back - val).abs() < 1e-3 || (back - val).abs() / val.abs() < 1e-3);
    }
}

#[test]
fn test_kv_cache_append_and_retrieve() {
    let mut cache = KVCache::new(2, 16, 2, 64);
    assert!(cache.is_empty());

    let k0 = vec![1234u16; 128];
    let v0 = vec![5678u16; 128];
    cache.append(0, 0, &k0, &v0);
    assert_eq!(cache.len(), 1);

    let retrieved_k = cache.get_key(0, 0, 0);
    assert_eq!(retrieved_k.len(), 64);
    assert_eq!(retrieved_k[0], 1234);

    let retrieved_v = cache.get_val(0, 0, 1);
    assert_eq!(retrieved_v.len(), 64);
    assert_eq!(retrieved_v[0], 5678);

    cache.reset();
    assert!(cache.is_empty());
}

#[test]
fn test_sampler_greedy_and_stochastic() {
    let logits = vec![1.0f32, 5.0, 2.0, 0.5];
    let mut greedy_sampler = Sampler::new(SamplingConfig::greedy());
    let token = greedy_sampler.sample(&logits);
    assert_eq!(token, 1, "Greedy sampling must choose highest logit (index 1)");

    let mut temp_sampler = Sampler::new(SamplingConfig {
        temperature: 0.8,
        top_k: 4,
        top_p: 0.95,
        seed: 42,
    });
    // Token 1 carries most of the probability mass; with the same seed the draw is reproducible.
    let draws: Vec<u32> = (0..200).map(|_| temp_sampler.sample(&logits)).collect();
    assert!(draws.iter().all(|&t| t < 4));
    let ones = draws.iter().filter(|&&t| t == 1).count();
    assert!(ones > 150, "the most likely token was drawn only {ones} times out of 200");
    let mut again = Sampler::new(SamplingConfig { temperature: 0.8, top_k: 4, top_p: 0.95, seed: 42 });
    let replay: Vec<u32> = (0..200).map(|_| again.sample(&logits)).collect();
    assert_eq!(draws, replay, "the same seed must give the same tokens");
}

/// The nucleus filter must rank the candidates by probability also when top-k is disabled (0) or not truncating.
/// It used to accumulate the probabilities in token-id order, so a dominant last token was never drawn.
#[test]
fn test_sampler_top_p_ranks_by_probability_without_top_k() {
    let mut logits = vec![0.0f32; 10];
    logits[9] = 10.0; // about 99.9 % of the mass
    for top_k in [0usize, 10, 1000] {
        for seed in 1..=300u64 {
            let mut s = Sampler::new(SamplingConfig { temperature: 1.0, top_k, top_p: 0.5, seed });
            assert_eq!(s.sample(&logits), 9, "top_k={top_k}, seed={seed}");
        }
    }

    // Two dominant tokens at the end of the vocabulary: top-p 0.9 keeps exactly these two.
    let mut logits = vec![0.0f32; 64];
    logits[62] = 10.0;
    logits[63] = 10.0;
    let mut seen = [false; 64];
    let mut s = Sampler::new(SamplingConfig { temperature: 1.0, top_k: 0, top_p: 0.9, seed: 7 });
    for _ in 0..400 {
        seen[s.sample(&logits) as usize] = true;
    }
    assert!(seen[62] && seen[63], "both dominant tokens must be drawn");
    assert!(seen[..62].iter().all(|&s| !s), "tokens outside the nucleus must never be drawn");
}

#[test]
fn test_tokenizer_from_metadata() {
    let tokens = vec![
        "<|endoftext|>".to_string(),
        "Hello".to_string(),
        "world".to_string(),
        "!".to_string(),
    ];
    let tokenizer = QuintritTokenizer::from_gguf_metadata(&tokens, None, None, Some(0)).unwrap();
    assert_eq!(tokenizer.eos_token_id(), Some(0));
}

/// Byte-level vocabulary in which `ü` (UTF-8 `0xC3 0xBC`) is two tokens, as in the Falcon3 vocabulary: `Ã` and `¼` are
/// the byte-level characters of the bytes `0xC3` and `0xBC`.
fn byte_split_vocabulary() -> Vec<String> {
    ["<|endoftext|>", "T", "\u{C3}", "\u{BC}", "l", "u"].iter().map(|s| s.to_string()).collect()
}

#[test]
fn stream_decoder_keeps_multi_byte_characters_whole() {
    let tokenizer = QuintritTokenizer::from_gguf_metadata(&byte_split_vocabulary(), None, None, Some(0)).unwrap();
    let ids = [1u32, 2, 3, 4, 5]; // T 0xC3 0xBC l u

    // Decoding token by token splits the character: this is what produced "T\u{FFFD}\u{FFFD}lu" in the chat.
    let naive: String = ids.iter().map(|&id| tokenizer.decode_token(id)).collect();
    assert_eq!(naive, "T\u{FFFD}\u{FFFD}lu");

    let mut decoder = StreamDecoder::new();
    let pieces: Vec<String> = ids.iter().map(|&id| decoder.push(&tokenizer, id)).collect();
    assert_eq!(pieces, ["T", "", "\u{FC}", "l", "u"], "the first byte of \u{FC} must be held back");
    assert_eq!(decoder.finish(&tokenizer), "");
    assert_eq!(pieces.concat(), "T\u{FC}lu");
    assert_eq!(pieces.concat(), tokenizer.decode(&ids), "streamed text must equal decoding all ids at once");
    assert_eq!(decoder.ids(), ids);

    // A character still incomplete at the end of the sequence is flushed (lossily) by finish().
    let mut decoder = StreamDecoder::new();
    assert_eq!(decoder.push(&tokenizer, 1), "T");
    assert_eq!(decoder.push(&tokenizer, 2), "");
    assert_eq!(decoder.finish(&tokenizer), "\u{FFFD}");

    // Re-encoding the decoded text gives the original ids back, so a chat history built from it is not corrupted.
    assert_eq!(tokenizer.encode("T\u{FC}lu"), ids);
}

#[test]
fn control_tokens_from_gguf_types_are_encoded_as_single_ids() {
    let tokens = byte_split_vocabulary();
    let mut types = vec![1i32; tokens.len()];
    types[0] = GGUF_TOKEN_TYPE_CONTROL;
    let tokenizer = QuintritTokenizer::from_gguf_metadata_with_types(&tokens, None, Some(&types), None, Some(0)).unwrap();

    assert_eq!(tokenizer.encode("lu<|endoftext|>T"), [4, 5, 0, 1]);
    assert_eq!(tokenizer.decode(&[4, 5, 0, 1]), "luT", "decoding skips control tokens");
    assert_eq!(tokenizer.vocab_size(), tokens.len(), "a control token keeps its id from the vocabulary");

    // Without the token types the text is not recognised as the control token.
    let plain = QuintritTokenizer::from_gguf_metadata(&tokens, None, None, Some(0)).unwrap();
    assert!(!plain.encode("lu<|endoftext|>T").contains(&0));
}

#[test]
fn test_synthetic_transformer_layer_execution() {
    // 32 rows, 128 cols
    let rows = 32;
    let cols = 128;
    let planar_bytes = (rows * cols) / 4;
    // Fill with synthetic planar I2_S bytes (e.g. 0xAA)
    let synthetic_planar = vec![0xAAu8; planar_bytes];

    // Dispatch as 1.60 bpw Velurex tiled
    let layer_quintrit = dispatch_weight_from_i2_s(
        &synthetic_planar,
        rows,
        cols,
        1.0,
        TernaryKernelType::QuintritLut,
        1,
    );
    assert_eq!(layer_quintrit.kernel_type(), TernaryKernelType::QuintritLut);
    assert_eq!(layer_quintrit.shape(), (rows, cols));

    let activations = vec![1i8; cols];
    let mut out_quintrit = vec![0.0f32; rows];
    layer_quintrit.forward(&activations, 1.0, &mut out_quintrit);

    // Ground truth: exact integer dot product over the trits decoded from the I2_S bytes.
    let trits = unpack_gguf_i2_s(&synthetic_planar, rows, cols);
    for r in 0..rows {
        let expected: i64 = (0..cols).map(|c| trits[r * cols + c] as i64 * activations[c] as i64).sum();
        assert_eq!(
            out_quintrit[r], expected as f32,
            "Quintrit output must equal the exact integer dot product at row {}",
            r
        );
    }
}

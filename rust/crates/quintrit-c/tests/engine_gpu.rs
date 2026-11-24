//! The GPU decoder against the CPU decoder on the tiny synthetic model of the C API tests: every logit of every position and the greedy text must be
//! bit-identical. No model download is needed. The tests skip themselves (and print why) without a supported NVIDIA GPU or without GPU kernels.
#![cfg(feature = "cuda")]

mod common;

use common::synth::*;
use quintrit_runtime::{InferenceEngine, SamplingConfig};

/// A GPU engine for the model, or `None` (with a message) when this machine cannot run one.
fn gpu_engine(path: &std::path::Path) -> Option<InferenceEngine> {
    match InferenceEngine::load_from_gguf_on_gpu(path, 0, None) {
        Ok(e) => Some(e),
        Err(msg) if msg.contains("CUDA driver") || msg.contains("unsupported GPU") || msg.contains("no GPU kernels") => {
            eprintln!("skipped: {msg}");
            None
        }
        Err(msg) => panic!("loading on the GPU failed: {msg}"),
    }
}

#[test]
fn gpu_logits_equal_cpu_logits_position_by_position() {
    let dir = TempDir::new("engine-gpu");
    let path = dir.path("model.gguf");
    write_model(&path, 21, &Flaws::default());
    let Some(mut gpu) = gpu_engine(&path) else { return };
    let mut cpu = InferenceEngine::load_from_gguf(&path, 1, None).unwrap();
    assert!(gpu.gpu_device_name().is_some() && cpu.gpu_device_name().is_none());

    let (mut lc, mut lg) = (vec![0f32; VOCAB], vec![0f32; VOCAB]);
    // A sequence that fills the context window: 44 positions of a context of 48, tokens covering the byte vocabulary.
    let tokens: Vec<u32> = (0..44).map(|i| ((i * 37 + 11) % 256) as u32).collect();
    for (pos, &t) in tokens.iter().enumerate() {
        cpu.forward_token(t, pos, &mut lc).unwrap();
        gpu.forward_token(t, pos, &mut lg).unwrap();
        let (bc, bg): (Vec<u32>, Vec<u32>) = (lc.iter().map(|v| v.to_bits()).collect(), lg.iter().map(|v| v.to_bits()).collect());
        assert_eq!(bc, bg, "logits differ at position {pos}");
    }
    // Restarting from a shorter prefix overwrites the device cache positions and must again agree.
    for (pos, &t) in tokens.iter().enumerate().take(10) {
        cpu.forward_token(t ^ 1, pos, &mut lc).unwrap();
        gpu.forward_token(t ^ 1, pos, &mut lg).unwrap();
        assert_eq!(lc.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), lg.iter().map(|v| v.to_bits()).collect::<Vec<_>>(), "restart, position {pos}");
    }
    // A position beyond the window is an error, not a wrong result.
    assert!(gpu.forward_token(1, 48, &mut lg).is_err());
}

/// Prompt tokens other than the last one run without the final norm and the vocabulary projection (a second CUDA graph on the GPU): the cache they
/// leave must be the one of the full pass, so the logits of the last token and of the tokens that follow stay bit-identical.
#[test]
fn prefill_without_the_head_leaves_the_cache_of_the_full_pass() {
    let dir = TempDir::new("engine-gpu-prefill");
    let path = dir.path("model.gguf");
    write_model(&path, 21, &Flaws::default());
    let Some(mut gpu) = gpu_engine(&path) else { return };
    let mut cpu = InferenceEngine::load_from_gguf(&path, 1, None).unwrap();
    let tokens: Vec<u32> = (0..40).map(|i| ((i * 37 + 11) % 256) as u32).collect();
    let bits = |l: &[f32]| l.iter().map(|v| v.to_bits()).collect::<Vec<_>>();

    // Reference: the full pass on the CPU.
    let mut want = vec![0f32; VOCAB];
    for (pos, &t) in tokens.iter().enumerate() {
        cpu.forward_token(t, pos, &mut want).unwrap();
    }
    let mut next_want = vec![0f32; VOCAB];
    cpu.forward_token(7, tokens.len(), &mut next_want).unwrap();

    for (name, engine) in [("CPU", &mut cpu), ("GPU", &mut gpu)] {
        let mut got = vec![0f32; VOCAB];
        let (last, head) = tokens.split_last().unwrap();
        for (pos, &t) in head.iter().enumerate() {
            engine.prefill_token(t, pos).unwrap();
        }
        engine.forward_token(*last, head.len(), &mut got).unwrap();
        assert_eq!(bits(&got), bits(&want), "{name}: logits of the last prompt token");
        engine.forward_token(7, tokens.len(), &mut got).unwrap();
        assert_eq!(bits(&got), bits(&next_want), "{name}: logits of the token after the prompt");
    }
}

/// The whole decoder over a long context. The attended key-value axis is split into four chunks from 512 padded positions on (the first position that
/// attends more than 256 positions is position 256), which the context-48 model above never reaches. The model declares a context of 1280 positions,
/// a multiple of 256, so the CPU cache (the model's full length) and the GPU window give the same padded axis at every position, and 1100 positions
/// cross the chunk borders of the padded axes 512, 768, 1024 and 1280.
///
/// Limits: the synthetic model has a head dimension of 32 and two query heads per key-value head; the head dimensions 64, 128 and 256, other
/// group sizes and every chunk and tile border of the attention kernel are covered by `gpu_layers` in `quintrit-runtime`, not here.
#[test]
fn gpu_logits_equal_cpu_logits_over_a_long_context() {
    const CONTEXT_LONG: usize = 1280;
    const POSITIONS: usize = 1100;
    let dir = TempDir::new("engine-gpu-long");
    let path = dir.path("model.gguf");
    write_model_with_context(&path, 31, &Flaws::default(), CONTEXT_LONG);
    let Some(mut gpu) = gpu_engine(&path) else { return };
    let mut cpu = InferenceEngine::load_from_gguf(&path, 1, None).unwrap();
    assert!(gpu.gpu_device_name().is_some() && cpu.gpu_device_name().is_none());

    let (mut lc, mut lg) = (vec![0f32; VOCAB], vec![0f32; VOCAB]);
    for pos in 0..POSITIONS {
        // A deterministic pseudo-random sequence over the byte vocabulary.
        let t = ((pos * 7919 + 13) % 256) as u32;
        cpu.forward_token(t, pos, &mut lc).unwrap();
        gpu.forward_token(t, pos, &mut lg).unwrap();
        if let Some(i) = (0..VOCAB).find(|&i| lc[i].to_bits() != lg[i].to_bits()) {
            panic!("logits differ at position {pos}, first at vocabulary entry {i}: CPU {:#010x}, GPU {:#010x}", lc[i].to_bits(), lg[i].to_bits());
        }
    }
    // The last window position is accepted, the one beyond it is an error.
    gpu.forward_token(1, CONTEXT_LONG - 1, &mut lg).unwrap();
    assert!(gpu.forward_token(1, CONTEXT_LONG, &mut lg).is_err());
}

#[test]
fn gpu_generation_equals_cpu_generation() {
    let dir = TempDir::new("engine-gpu-gen");
    let path = dir.path("model.gguf");
    write_model(&path, 5, &Flaws::default());
    let Some(mut gpu) = gpu_engine(&path) else { return };
    let mut cpu = InferenceEngine::load_from_gguf(&path, 1, None).unwrap();
    for sampling in [SamplingConfig::greedy(), SamplingConfig { temperature: 0.8, top_k: 40, top_p: 0.95, seed: 3 }] {
        let a = cpu.generate_streaming("hello there", 20, &sampling, |_| {}).unwrap();
        let b = gpu.generate_streaming("hello there", 20, &sampling, |_| {}).unwrap();
        assert_eq!(a, b);
        // A second turn that extends the first one reuses the cached prefix on both.
        let a2 = cpu.generate_streaming("hello there, and more", 10, &sampling, |_| {}).unwrap();
        let b2 = gpu.generate_streaming("hello there, and more", 10, &sampling, |_| {}).unwrap();
        assert_eq!(a2, b2);
        assert_eq!(gpu.last_generation_stats().reused_prompt_tokens, cpu.last_generation_stats().reused_prompt_tokens);
    }
}

#[test]
fn unsupported_models_are_rejected_with_a_reason() {
    let dir = TempDir::new("engine-gpu-flaws");
    let path = dir.path("model.gguf");
    write_model(&path, 3, &Flaws { odd_i2s_cols: true, ..Flaws::default() });
    // The damaged file is rejected by the common validation before the GPU is involved.
    assert!(InferenceEngine::load_from_gguf_on_gpu(&path, 0, None).is_err());
    // A device that does not exist.
    let ok = dir.path("ok.gguf");
    write_model(&ok, 3, &Flaws::default());
    if gpu_engine(&ok).is_some() {
        let err = InferenceEngine::load_from_gguf_on_gpu(&ok, 99, None).err().expect("device 99 does not exist");
        assert!(err.contains("does not exist"), "{err}");
    }
}

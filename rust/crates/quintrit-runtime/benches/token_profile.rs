//! # Decode-step profile: where does the time of one token go?
//!
//! Replays the forward pass of one decode step phase by phase with timers, using only the public API of the
//! engine, and relates the time spent inside the ternary GEMVs to the bytes they stream. The sum of the phases is
//! compared with the engine's own `forward_token_with_scratchpad` so that the replay can be trusted.
//!
//! Phases per token: embedding, RMSNorm, attention (everything inside `AttentionBlock`), residual adds,
//! activation quantization (+ activation table for the velurex kernels), FFN gate+up GEMVs, SwiGLU, FFN down GEMV,
//! and the F16 vocabulary projection (`lm_head`). The four attention GEMVs are timed separately by a replay of just
//! those calls, and subtracted from the attention total.
//!
//! ```bash
//! cargo bench -p quintrit-runtime --bench token_profile -- --model "$MODEL" \
//!     --kernel quintrit --threads 4 --tokens 24 --suffix i7-7700
//! ```
//!
//! `$MODEL` is the path of the GGUF file. `--kernel` is `quintrit` (default), `quintrit_512` or, with the cargo feature
//! `bitnet-baseline`, `bitnet_mad` or `bitnet_mad_512` (BitNet's 4-trit multiply-add kernel).

mod common;
use common::run_header;
use quintrit_runtime::layer::{forward_linear_pair, forward_linear_triple, TernaryKernelType};
use quintrit_runtime::ops::{
    add_residual, elementwise_mul, f16_to_f32, ggml_vec_dot_f16, quantize_activations_int8_with_range, relu2, rms_norm,
    swiglu,
};
use quintrit_runtime::{ActivationType, InferenceEngine, Sampler, SamplingConfig};
use rayon::prelude::*;
use std::fmt::Write as _;
use std::time::Instant;

#[derive(Default, Clone)]
struct Phases {
    embed: f64,
    norm: f64,
    attn_total: f64,
    resid: f64,
    quant: f64,
    ffn_gate_up: f64,
    act: f64,
    ffn_down: f64,
    final_norm: f64,
    lm_head: f64,
}

impl Phases {
    fn total(&self) -> f64 {
        self.embed + self.norm + self.attn_total + self.resid + self.quant + self.ffn_gate_up + self.act + self.ffn_down + self.final_norm + self.lm_head
    }
}

/// Counts of activation values with magnitude above the exact range of a single int8 activation table.
#[derive(Default, Clone, Copy)]
struct ActStats {
    values: u64,
    above_42: u64,
    vectors: u64,
    vectors_above_5pct: u64,
    max_cols_above: u64,
}

impl ActStats {
    fn add(&mut self, v: &[i8]) {
        let n = v.iter().filter(|&&x| (x as i32).abs() > 42).count() as u64;
        self.values += v.len() as u64;
        self.above_42 += n;
        self.vectors += 1;
        if n * 20 > v.len() as u64 {
            self.vectors_above_5pct += 1;
        }
        self.max_cols_above = self.max_cols_above.max(n);
    }
}

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|p| a.get(p + 1).cloned())
}

fn main() {
    // The benchmarks keep the load-time progress lines; the library is silent by default.
    quintrit_runtime::set_verbose(true);
    let model_path = common::resolve_model(arg("--model"));
    let threads: usize = arg("--threads").and_then(|v| v.parse().ok()).unwrap_or(4);
    let tokens: usize = arg("--tokens").and_then(|v| v.parse().ok()).unwrap_or(24);
    let suffix = arg("--suffix").unwrap_or_else(|| "local".into());
    let out_dir = arg("--out").unwrap_or_else(|| "results".into());
    let kernel = match arg("--kernel").as_deref() {
        Some("quintrit_512") => TernaryKernelType::QuintritLutAVX512,
        #[cfg(feature = "bitnet-baseline")]
        Some("bitnet_mad") => TernaryKernelType::BitnetMad,
        #[cfg(feature = "bitnet-baseline")]
        Some("bitnet_mad_512") => TernaryKernelType::BitnetMad512,
        #[cfg(not(feature = "bitnet-baseline"))]
        Some("bitnet_mad") | Some("bitnet_mad_512") => {
            eprintln!("error: the BitNet kernels need the cargo feature `bitnet-baseline` (cargo bench --features bitnet-baseline)");
            std::process::exit(2);
        }
        _ => TernaryKernelType::QuintritLut,
    };
    #[cfg(target_arch = "x86_64")]
    if let Some(v) = arg("--avx512-variant").and_then(|v| v.parse::<u32>().ok()) {
        quintrit_core::simd::avx512::set_variant_override(v);
    }
    let mut engine = InferenceEngine::load_from_gguf(&model_path, threads, Some(kernel)).expect("load model");
    let prompt = "<|user|>\nWhat is ternary computing?\n<|assistant|>\n";
    let mut prompt_tokens = engine.tokenizer.encode(prompt);
    if let Some(bos) = engine.tokenizer.bos_token_id() {
        if prompt_tokens.first() != Some(&bos) {
            prompt_tokens.insert(0, bos);
        }
    }
    let vocab = engine.model.config.vocab_size;
    let hidden = engine.model.config.hidden_size;
    let inter = engine.model.config.intermediate_size;
    let n_layers = engine.model.blocks.len();
    let mut logits = vec![0.0f32; vocab];

    // Ternary weight bytes streamed per token, and the F16 vocabulary projection.
    let ternary_bytes: usize = engine.model.blocks.iter().map(|b| b.attn.memory_bytes() + b.gate_proj.memory_bytes() + b.up_proj.memory_bytes() + b.down_proj.memory_bytes()).sum();
    let lm_bytes = vocab * hidden * 2;

    // ---- Pass A: the engine's own forward -----------------------------------------------------------------
    let run_prefill = |engine: &mut InferenceEngine, logits: &mut Vec<f32>| {
        engine.kv_cache.reset();
        for (pos, &t) in prompt_tokens.iter().enumerate() {
            engine.model.forward_token_with_scratchpad(t, pos, &mut engine.kv_cache, &mut engine.scratchpad, logits);
        }
    };
    run_prefill(&mut engine, &mut logits);
    let mut sampler = Sampler::new(SamplingConfig::greedy());
    let mut engine_times = Vec::new();
    let mut pos = prompt_tokens.len();
    let mut ids = Vec::new();
    for _ in 0..tokens {
        let next = sampler.sample(&logits);
        ids.push(next);
        let t0 = Instant::now();
        engine.model.forward_token_with_scratchpad(next, pos, &mut engine.kv_cache, &mut engine.scratchpad, &mut logits);
        engine_times.push(t0.elapsed().as_secs_f64());
        pos += 1;
    }
    engine_times.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let engine_median = engine_times[engine_times.len() / 2];
    let engine_best = engine_times[0];

    // ---- Pass B: phase-by-phase replay of the same tokens -------------------------------------------------
    run_prefill(&mut engine, &mut logits);
    let mut per_token: Vec<Phases> = Vec::new();
    let mut stats = [ActStats::default(); 4]; // qkv input, o input, gate/up input, down input
    let mut pos = prompt_tokens.len();
    let num_threads = threads;
    for &next in &ids {
        let mut ph = Phases::default();
        let model = &engine.model;
        let sp = &mut engine.scratchpad;
        let kv = &mut engine.kv_cache;
        let t = Instant::now();
        let emb = &model.token_embeddings[(next as usize) * hidden..(next as usize + 1) * hidden];
        for i in 0..hidden {
            sp.x[i] = f16_to_f32(emb[i]);
        }
        ph.embed += t.elapsed().as_secs_f64();
        for (li, block) in model.blocks.iter().enumerate() {
            let t = Instant::now();
            rms_norm(&sp.x[..hidden], &block.attn_norm, block.rms_norm_eps, &mut sp.x_norm[..hidden]);
            ph.norm += t.elapsed().as_secs_f64();
            let t = Instant::now();
            block.attn.forward_with_scratchpad(li, pos, kv, sp);
            stats[0].add(&sp.x_int8[..hidden]);
            stats[1].add(&sp.o_int8[..model.config.num_attention_heads * model.config.head_dim]);
            ph.attn_total += t.elapsed().as_secs_f64();
            let t = Instant::now();
            add_residual(&mut sp.x[..hidden], &sp.attn_out[..hidden]);
            ph.resid += t.elapsed().as_secs_f64();
            let t = Instant::now();
            rms_norm(&sp.x[..hidden], &block.ffn_norm, block.rms_norm_eps, &mut sp.x_norm[..hidden]);
            ph.norm += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let scale = quantize_activations_int8_with_range(&sp.x_norm[..hidden], &mut sp.ffn_int8[..hidden], block.gate_proj.kernel_type().activation_quant_levels());
            let qlut_len = if matches!(block.gate_proj.kernel_type(), TernaryKernelType::QuintritLut) {
                quintrit_core::lut::build_activation_lut_wide_into(&sp.ffn_int8[..hidden], hidden, &mut sp.act_split, &mut sp.qlut_buf)
            } else {
                0
            };
            ph.quant += t.elapsed().as_secs_f64();
            stats[2].add(&sp.ffn_int8[..hidden]);
            let t = Instant::now();
            let qlut_ref = if qlut_len > 0 { Some(&sp.qlut_buf[..qlut_len]) } else { None };
            forward_linear_pair(&block.gate_proj, &mut sp.gate[..inter], &block.up_proj, &mut sp.up[..inter], &sp.ffn_int8[..hidden], scale, qlut_ref, num_threads);
            ph.ffn_gate_up += t.elapsed().as_secs_f64();
            let t = Instant::now();
            match block.hidden_act {
                ActivationType::Relu2 => {
                    relu2(&sp.gate[..inter], &mut sp.act_gate[..inter]);
                    elementwise_mul(&sp.act_gate[..inter], &sp.up[..inter], &mut sp.inter[..inter]);
                }
                ActivationType::Silu => swiglu(&sp.gate[..inter], &sp.up[..inter], &mut sp.inter[..inter]),
            }
            ph.act += t.elapsed().as_secs_f64();
            let t = Instant::now();
            let dscale = if let Some(sub) = &block.ffn_sub_norm {
                rms_norm(&sp.inter[..inter], sub, block.rms_norm_eps, &mut sp.ffn_sub_buf[..inter]);
                quantize_activations_int8_with_range(&sp.ffn_sub_buf[..inter], &mut sp.down_int8[..inter], block.down_proj.kernel_type().activation_quant_levels())
            } else {
                quantize_activations_int8_with_range(&sp.inter[..inter], &mut sp.down_int8[..inter], block.down_proj.kernel_type().activation_quant_levels())
            };
            ph.quant += t.elapsed().as_secs_f64();
            stats[3].add(&sp.down_int8[..inter]);
            let t = Instant::now();
            block.down_proj.forward(&sp.down_int8[..inter], dscale, &mut sp.ffn_out[..hidden]);
            ph.ffn_down += t.elapsed().as_secs_f64();
            let t = Instant::now();
            add_residual(&mut sp.x[..hidden], &sp.ffn_out[..hidden]);
            ph.resid += t.elapsed().as_secs_f64();
        }
        let t = Instant::now();
        rms_norm(&sp.x[..hidden], &model.output_norm, model.config.rms_norm_eps, &mut sp.x_final[..hidden]);
        ph.final_norm += t.elapsed().as_secs_f64();
        let t = Instant::now();
        for i in 0..hidden {
            sp.x_final_f16[i] = quintrit_runtime::ops::f32_to_f16(sp.x_final[i]);
        }
        let lm_weights = model.lm_head_f16.as_ref().unwrap_or(&model.token_embeddings);
        let x16 = &sp.x_final_f16[..hidden];
        quintrit_core::threading::get_thread_pool(num_threads).install(|| {
            logits.par_chunks_mut(512).enumerate().for_each(|(ci, chunk)| {
                for (i, l) in chunk.iter_mut().enumerate() {
                    let v = ci * 512 + i;
                    if v < vocab {
                        *l = ggml_vec_dot_f16(x16, &lm_weights[v * hidden..(v + 1) * hidden]);
                    }
                }
            });
        });
        ph.lm_head += t.elapsed().as_secs_f64();
        per_token.push(ph);
        pos += 1;
    }
    // Median of each phase over the tokens.
    let med = |f: fn(&Phases) -> f64| {
        let mut v: Vec<f64> = per_token.iter().map(f).collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        v[v.len() / 2]
    };
    let mut total_v: Vec<f64> = per_token.iter().map(|p| p.total()).collect();
    total_v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let replay_median = total_v[total_v.len() / 2];

    // ---- Pass C: attention GEMVs alone (q, k, v concurrently, then o), all layers -------------------------
    let q_dim = engine.model.config.num_attention_heads * engine.model.config.head_dim;
    let kv_dim = engine.model.config.num_key_value_heads * engine.model.config.head_dim;
    let acts_h: Vec<i8> = (0..hidden).map(|i| ((i * 7) % 127) as i8 - 63).collect();
    let acts_q: Vec<i8> = (0..q_dim).map(|i| ((i * 11) % 127) as i8 - 63).collect();
    let (mut oq, mut ok, mut ov, mut oo) = (vec![0f32; q_dim], vec![0f32; kv_dim], vec![0f32; kv_dim], vec![0f32; hidden]);
    let mut attn_gemv = Vec::new();
    for _rep in 0..5 {
        let t0 = Instant::now();
        for block in &engine.model.blocks {
            forward_linear_triple(block.attn.q_proj(), &mut oq, block.attn.k_proj(), &mut ok, block.attn.v_proj(), &mut ov, &acts_h, 1.0, None, num_threads);
            block.attn.o_proj().forward(&acts_q, 1.0, &mut oo);
        }
        attn_gemv.push(t0.elapsed().as_secs_f64());
    }
    attn_gemv.sort_by(|a, b| a.partial_cmp(b).unwrap());
    let attn_gemv_t = attn_gemv[0];

    // ---- Report ---------------------------------------------------------------------------------------------
    let ms = |s: f64| s * 1e3;
    let attn_rest = (med(|p| p.attn_total) - attn_gemv_t).max(0.0);
    let gemv_total = attn_gemv_t + med(|p| p.ffn_gate_up) + med(|p| p.ffn_down);
    let rows: Vec<(&str, f64)> = vec![
        ("ternary GEMVs: attention q,k,v,o", attn_gemv_t),
        ("ternary GEMVs: FFN gate+up", med(|p| p.ffn_gate_up)),
        ("ternary GEMVs: FFN down", med(|p| p.ffn_down)),
        ("F16 vocabulary projection (lm_head)", med(|p| p.lm_head)),
        ("attention without its GEMVs (RoPE, KV cache, scores, softmax, quantization, norms inside)", attn_rest),
        ("activation quantization (+ activation tables)", med(|p| p.quant)),
        ("RMSNorm", med(|p| p.norm) + med(|p| p.final_norm)),
        ("SwiGLU / activation function", med(|p| p.act)),
        ("residual adds + embedding lookup", med(|p| p.resid) + med(|p| p.embed)),
    ];
    let sum: f64 = rows.iter().map(|r| r.1).sum();
    let mut md = String::new();
    let _ = writeln!(md, "# Decode-step profile: {suffix}, kernel {:?}, {threads} threads\n", kernel);
    let _ = writeln!(md, "## Run

{}", run_header(&format!("{:?} (production path, no variant override)", kernel), threads, tokens));
    let _ = writeln!(md, "Falcon3-10B decode, {tokens} greedy tokens after an {}-token prompt; medians per token.\n", prompt_tokens.len());
    let _ = writeln!(md, "- Engine `forward_token_with_scratchpad`: median {:.1} ms/token ({:.2} tok/s), best {:.1} ms", ms(engine_median), 1.0 / engine_median, ms(engine_best));
    let _ = writeln!(md, "- Phase replay total: median {:.1} ms/token (replay/engine = {:.3})", ms(replay_median), replay_median / engine_median);
    let _ = writeln!(md, "- Ternary weight bytes per token: {:.0} MB; F16 `lm_head`: {:.0} MB; total {:.0} MB\n", ternary_bytes as f64 / 1e6, lm_bytes as f64 / 1e6, (ternary_bytes + lm_bytes) as f64 / 1e6);
    let _ = writeln!(md, "| Phase | ms/token | Share |\n|---|---:|---:|");
    for (n, t) in &rows {
        let _ = writeln!(md, "| {n} | {:.2} | {:.1}% |", ms(*t), t / sum * 100.0);
    }
    let _ = writeln!(md, "| **sum of phases** | **{:.2}** | |\n", ms(sum));
    let gemv_gbps = ternary_bytes as f64 / gemv_total / 1e9;
    let _ = writeln!(md, "- Effective rate inside the ternary GEMV regions only: **{:.2} GB/s** ({:.1} ms for {:.0} MB)", gemv_gbps, ms(gemv_total), ternary_bytes as f64 / 1e6);
    let lm = med(|p| p.lm_head);
    let _ = writeln!(md, "- Effective rate of the F16 `lm_head`: **{:.2} GB/s** ({:.1} ms for {:.0} MB)", lm_bytes as f64 / lm / 1e9, ms(lm), lm_bytes as f64 / 1e6);
    let _ = writeln!(md, "- Whole-token effective rate: **{:.2} GB/s** over {:.0} MB per token", (ternary_bytes + lm_bytes) as f64 / engine_median / 1e9, (ternary_bytes + lm_bytes) as f64 / 1e6);
    let _ = writeln!(md, "- F16 `lm_head` share of the phase sum: **{:.1} %**; ternary GEMVs: **{:.1} %**", lm / sum * 100.0, gemv_total / sum * 100.0);
    let _ = writeln!(md, "- GEMV regions (incl. lm_head share of memory-bound work) account for {:.0}% of the token; everything else is {:.1} ms", (gemv_total + lm) / sum * 100.0, ms(sum - gemv_total - lm));
    let _ = n_layers;
    {
        let names = ["qkv input", "o input", "gate/up input", "down input"];
        let _ = writeln!(md, "
### Activations above the single-table range (|x| > 42 after quantization)
");
        let _ = writeln!(md, "| Projection input | Share of values | Vectors with > 5 % | Max columns in one vector |
|---|---:|---:|---:|");
        for (n, s) in names.iter().zip(stats.iter()) {
            let _ = writeln!(md, "| {n} | {:.2} % | {} / {} | {} |", s.above_42 as f64 / s.values.max(1) as f64 * 100.0, s.vectors_above_5pct, s.vectors, s.max_cols_above);
        }
    }
    println!("{md}");
    let dir = std::path::Path::new(&out_dir);
    let _ = std::fs::create_dir_all(dir);
    let kname = format!("{:?}", kernel).to_lowercase();
    let path = dir.join(format!("token_profile_{kname}_{suffix}.md"));
    std::fs::write(&path, md).expect("write report");
    println!("report: {}", path.display());
}

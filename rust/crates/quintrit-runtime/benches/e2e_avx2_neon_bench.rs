//! # End-to-End Reference Model Benchmark (AVX2 & ARM NEON: 1.60 bpw Velurex)
//!
//! Evaluates end-to-end multi-token generation throughput (tokens/sec), per-token decoding latency,
//! Time to First Token (TTFT), DRAM memory streaming bandwidth (GB/s), GFLOPS, and 100% token-for-token
//! output parity across all 40 decoder layers of a reference model (**Falcon3-10B-Instruct-1.58bit**).
//! Serves as the primary end-to-end evaluation suite on AVX2 (x86_64) and ARM NEON / DotProd (AArch64) architectures.
//!
//! ## Zero-Warmup Cold DRAM Streaming Methodology
//!
//! In accordance with the project's zero-warmup policy for memory-bound LLM decoding:
//! - **Zero Artificial Warmup**: No artificial cache priming passes are performed. Artificial warmup is intentionally
//!   omitted because it distorts real-world autoregressive decoding where multi-gigabyte models exceed CPU caches.
//! - **Multi-Token Statistical Averaging**: Minimizes statistical jitter and timer variance by ingesting a standardized
//!   evaluation prompt followed by generating an answer sequence (default 64 tokens).
//! - **Deterministic Parity (Temp = 0.0)**: Evaluates under greedy argmax decoding (`temperature = 0.0`) and checks that
//!   the generated tokens are identical for every thread count.
//! - **Thread Scaling**: Sweeps powers of two up to the maximum available hardware threads (e.g. 1, 2, 4, 8...).
//!
//! ## Evaluated Architectures
//!
//! 1. **BitNet MAD (2.00 bpw)** (cargo feature `bitnet-baseline`): Microsoft BitNet's 4-trit multiply-add kernel; when it
//!    is built in, it is the baseline of the speed-up column and the reference of the parity check.
//! 2. **Quintrit Velurex (1.60 bpw)**: 32-row tiled Dual-Hash Activation-LUT kernel with velurex encoding.
//!
//! ## Execution Modes
//!
//! ```bash
//! # Default run (writes report to results/e2e_avx2_neon_bench.md)
//! cargo bench -p quintrit-runtime --bench e2e_avx2_neon_bench
//!
//! # Platform/device-tagged run via CLI flag (writes to results/e2e_avx2_neon_bench_<SUFFIX>.md)
//! cargo bench -p quintrit-runtime --bench e2e_avx2_neon_bench -- --suffix i7-7700 --tokens 64
//!
//! # Environment variable configuration
//! export QUINTRIT_BENCH_SUFFIX="i7-7700"   # or $env:QUINTRIT_BENCH_SUFFIX="i7-7700" on PowerShell
//! cargo bench -p quintrit-runtime --bench e2e_avx2_neon_bench
//! ```

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

use quintrit_format::GgufFile;
use quintrit_runtime::engine::InferenceEngine;
use quintrit_runtime::layer::TernaryKernelType;

/// Telemetry metrics for a single end-to-end model benchmark evaluation.
#[derive(Debug, Clone)]
pub struct E2eBenchmarkResult {
    pub algorithm_name: &'static str,
    pub paradigm: &'static str,
    pub bpw: f64,
    pub threads: usize,
    pub prompt_tokens: usize,
    pub gen_tokens: usize,
    pub ttft_ms: f64,
    pub gen_total_ms: f64,
    pub latency_ms: f64,
    pub tokens_per_sec: f64,
    pub weight_mb: f64,
    pub bandwidth_gbps: f64,
    pub gflops: f64,
}

/// Discovers hardware thread parallelism and computes scaling thread counts (powers of two up to max).
fn detect_thread_counts() -> (usize, Vec<usize>) {
    let max_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(1);

    let mut thread_counts = Vec::new();
    let mut t = 1;
    while t <= max_threads {
        thread_counts.push(t);
        t *= 2;
    }
    if thread_counts.last().copied() != Some(max_threads) && max_threads > 1 {
        thread_counts.push(max_threads);
    }
    (max_threads, thread_counts)
}

/// Detects active SIMD microarchitecture features.
fn detect_simd_target() -> String {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            "AVX2 (x86_64 / 256-bit SIMD)".to_string()
        } else {
            "Scalar Fallback (x86_64)".to_string()
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        "ARM NEON / ASIMD (AArch64)".to_string()
    }
    #[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
    {
        "Portable Scalar (Fallback)".to_string()
    }
}

/// Destination directory for benchmark reports: `results/` in the directory of this crate (ignored by git).
fn resolve_results_dir() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("results")
}

/// Extracts the optional output filename suffix from CLI arguments (`--suffix <VAL>`, `-s <VAL>`, `--suffix=<VAL>`)
/// or the environment variable `QUINTRIT_BENCH_SUFFIX`.
fn get_output_suffix() -> Option<String> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--suffix" | "-s" | "--output-suffix" => {
                if i + 1 < args.len() {
                    let val = args[i + 1].trim();
                    if !val.is_empty() {
                        return Some(val.to_string());
                    }
                    i += 1;
                }
            }
            arg if arg.starts_with("--suffix=") => {
                let val = arg.trim_start_matches("--suffix=").trim();
                if !val.is_empty() {
                    return Some(val.to_string());
                }
            }
            arg if arg.starts_with("-s=") => {
                let val = arg.trim_start_matches("-s=").trim();
                if !val.is_empty() {
                    return Some(val.to_string());
                }
            }
            arg if arg.starts_with("--output-suffix=") => {
                let val = arg.trim_start_matches("--output-suffix=").trim();
                if !val.is_empty() {
                    return Some(val.to_string());
                }
            }
            _ => {}
        }
        i += 1;
    }

    std::env::var("QUINTRIT_BENCH_SUFFIX")
        .ok()
        .map(|s| s.trim().to_string())
        .filter(|s| !s.is_empty())
}

/// Constructs the output markdown filename given an optional device/platform suffix.
fn resolve_output_filename(suffix: Option<&str>) -> String {
    match suffix {
        Some(s) if !s.trim().is_empty() => {
            let clean = s.trim();
            if clean.starts_with('_') || clean.starts_with('-') {
                format!("e2e_avx2_neon_bench{}.md", clean)
            } else {
                format!("e2e_avx2_neon_bench_{}.md", clean)
            }
        }
        _ => "e2e_avx2_neon_bench.md".to_string(),
    }
}

/// Ends the program with an error: the benchmark measures a real model and has no fallback without one.
fn exit_without_model(problem: &str) -> ! {
    eprintln!("error: {problem}");
    eprintln!();
    eprintln!("This benchmark needs a GGUF model and does not run without one: it measures real decoding, so a");
    eprintln!("report without a model would contain no measurement. Pass the model with --model <PATH>, or set");
    eprintln!("QUINTRIT_MODEL_PATH, or place it at models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf.");
    eprintln!("The download command is in docs/benchmarks.md (section \"Running the benchmarks\").");
    std::process::exit(2);
}

/// Discovers candidate path for the GGUF model file.
fn resolve_model_path() -> Option<PathBuf> {
    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" | "-m" => {
                if i + 1 < args.len() {
                    let p = PathBuf::from(&args[i + 1]);
                    if p.exists() {
                        return Some(p);
                    }
                }
            }
            arg if arg.starts_with("--model=") => {
                let p = PathBuf::from(arg.trim_start_matches("--model=").trim());
                if p.exists() {
                    return Some(p);
                }
            }
            _ => {}
        }
        i += 1;
    }

    if let Ok(env_path) = std::env::var("QUINTRIT_MODEL_PATH").or_else(|_| std::env::var("MODEL_PATH")) {
        let p = PathBuf::from(env_path.trim());
        if p.exists() {
            return Some(p);
        }
    }

    let mut cur = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
    while let Some(dir) = cur {
        let p = dir.join("models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf");
        if p.exists() {
            return Some(p);
        }
        cur = dir.parent();
    }

    let candidates = [
        "models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf",
        "../models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf",
        "../../models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf",
        "../../../models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf",
    ];

    for cand in &candidates {
        let p = PathBuf::from(cand);
        if p.exists() {
            return Some(p);
        }
    }

    None
}

/// Formats the current UTC timestamp into `YYYY-MM-DD HH:MM:SS UTC`.
fn format_current_timestamp() -> String {
    match std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH) {
        Ok(d) => {
            let total_secs = d.as_secs();
            let sec = total_secs % 60;
            let total_mins = total_secs / 60;
            let min = total_mins % 60;
            let total_hours = total_mins / 60;
            let hour = total_hours % 24;
            let total_days = total_hours / 24;

            let z = total_days as i64 + 719468;
            let era = (if z >= 0 { z } else { z - 146096 }) / 146097;
            let doe = (z - era * 146097) as u64;
            let yoe = (doe - doe / 1460 + doe / 36524 - doe / 146096) / 365;
            let y = yoe as i64 + era * 400;
            let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
            let mp = (5 * doy + 2) / 153;
            let d = doy - (153 * mp + 2) / 5 + 1;
            let m = if mp < 10 { mp + 3 } else { mp - 9 };
            let year = if m <= 2 { y + 1 } else { y };

            format!("{:04}-{:02}-{:02} {:02}:{:02}:{:02} UTC", year, m, d, hour, min, sec)
        }
        Err(_) => "Unknown".to_string(),
    }
}

/// Metadata description of a single ternary projection matrix in a transformer layer block.
#[allow(dead_code)]
#[derive(Clone)]
struct LayerProjectionMeta {
    name: &'static str,
    rows: usize,
    cols: usize,
}

/// Descriptor of a model architecture's projection dimensions per decoder layer block.
#[derive(Clone)]
struct ModelLayerPlan {
    name: String,
    num_layers: usize,
    hidden_size: usize,
    intermediate_size: usize,
    projections: Vec<LayerProjectionMeta>,
    total_mac_ops_per_token: f64,
}

impl ModelLayerPlan {
    /// Builds the execution plan for Falcon3-10B (3072 hidden, 23040 intermediate, 40 layers).
    fn falcon3_10b() -> Self {
        let hidden_size = 3072;
        let intermediate_size = 23040;
        let num_layers = 40;
        let q_dim = 12 * 256;  // 3072
        let kv_dim = 4 * 256;  // 1024

        let projections = vec![
            LayerProjectionMeta { name: "attn_q", rows: q_dim, cols: hidden_size },
            LayerProjectionMeta { name: "attn_k", rows: kv_dim, cols: hidden_size },
            LayerProjectionMeta { name: "attn_v", rows: kv_dim, cols: hidden_size },
            LayerProjectionMeta { name: "attn_output", rows: hidden_size, cols: q_dim },
            LayerProjectionMeta { name: "ffn_gate", rows: intermediate_size, cols: hidden_size },
            LayerProjectionMeta { name: "ffn_up", rows: intermediate_size, cols: hidden_size },
            LayerProjectionMeta { name: "ffn_down", rows: hidden_size, cols: intermediate_size },
        ];

        let ops_per_layer: f64 = projections.iter().map(|p| 2.0 * p.rows as f64 * p.cols as f64).sum();
        let total_mac_ops_per_token = ops_per_layer * num_layers as f64;

        Self {
            name: "Falcon3-10B-Instruct-1.58bit".to_string(),
            num_layers,
            hidden_size,
            intermediate_size,
            projections,
            total_mac_ops_per_token,
        }
    }
}

/// Evaluated algorithm specification.
struct AlgoSpec {
    kernel_type: TernaryKernelType,
    name: &'static str,
    paradigm: &'static str,
    bpw: f64,
}

fn main() {
    // The benchmarks keep the load-time progress lines; the library is silent by default.
    quintrit_runtime::set_verbose(true);
    println!("===============================================================================");
    println!("  Quintrit E2E Reference Model Benchmark (AVX2 & ARM NEON)");
    println!("===============================================================================");

    let suffix = get_output_suffix();
    let results_dir = resolve_results_dir();
    let output_filename = resolve_output_filename(suffix.as_deref());
    let output_path = results_dir.join(&output_filename);

    let (max_threads, thread_scaling) = detect_thread_counts();
    let simd_desc = detect_simd_target();

    // Default evaluation hyperparameters
    let mut gen_tokens = 64usize;
    let mut raw_prompt = "What is ternary computing?".to_string();

    let args: Vec<String> = std::env::args().collect();
    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--tokens" | "-n" => {
                if i + 1 < args.len() {
                    gen_tokens = args[i + 1].parse().unwrap_or(64);
                    i += 1;
                }
            }
            arg if arg.starts_with("--tokens=") => {
                gen_tokens = arg.trim_start_matches("--tokens=").parse().unwrap_or(64);
            }
            arg if arg.starts_with("-n=") => {
                gen_tokens = arg.trim_start_matches("-n=").parse().unwrap_or(64);
            }
            "--prompt" => {
                if i + 1 < args.len() {
                    raw_prompt = args[i + 1].clone();
                    i += 1;
                }
            }
            "-h" | "--help" => {
                println!("Usage: e2e_avx2_neon_bench [OPTIONS]");
                println!();
                println!("Options:");
                println!("  -n, --tokens <N>         Number of tokens to generate per forward evaluation (default: 64)");
                println!("      --prompt <TEXT>      Input prompt text (default: 'What is ternary computing?')");
                println!("  -s, --suffix <SUFFIX>    Suffix for output report filename (e.g. 'i7-7700', 'rpi5-cortex-a76')");
                println!("  -m, --model <PATH>       Path to the GGUF model file (*.gguf); required, the benchmark does not run without a model");
                println!("  -h, --help               Print this help message");
                println!();
                println!("Methodology: Cold DRAM streaming regime without artificial cache warmup.");
                std::process::exit(0);
            }
            _ => {}
        }
        i += 1;
    }

    // Standardized prompt formatting for Falcon3-10B-Instruct
    let chat_prompt = if raw_prompt.contains("<|assistant|>") {
        raw_prompt.clone()
    } else {
        format!("<|user|>\n{}\n<|assistant|>\n", raw_prompt.trim())
    };

    println!("  Results Output:      {}", output_path.display());
    println!("  Available Threads:   {}", max_threads);
    println!("  Thread Scaling:      {:?}", thread_scaling);
    println!("  SIMD Target:         {}", simd_desc);
    println!("  Warmup Policy:       Zero Warmup (Cold Streaming DRAM Regime)");
    println!("  Sampling Policy:     Deterministic Greedy Argmax (Temperature = 0.0)");
    println!("  Prompt:              {}", chat_prompt.escape_debug());
    println!("  Generation Target:   {} tokens (all 40 layers, 280 ternary GEMV passes/token)", gen_tokens);

    // 1. Locate reference model
    let model_path = resolve_model_path();
    let plan = ModelLayerPlan::falcon3_10b();

    println!("  Model Architecture:  {}", plan.name);
    println!("  Decoder Layers:      {}", plan.num_layers);
    println!("  Projections/Layer:   {}", plan.projections.len());
    println!("  Total Projections:   {}", plan.num_layers * plan.projections.len());
    println!("  Hidden Size:         {}", plan.hidden_size);
    println!("  Intermediate Size:   {}", plan.intermediate_size);
    println!("  Total MAC Ops/Token: {:.2} GFLOP", plan.total_mac_ops_per_token / 1e9);

    let model_path = match model_path {
        Some(p) => p,
        None => {
            let requested = std::env::args().skip(1).collect::<Vec<_>>();
            let given = requested.iter().enumerate().find_map(|(i, a)| {
                if a == "--model" || a == "-m" {
                    requested.get(i + 1).cloned()
                } else {
                    a.strip_prefix("--model=").map(str::to_string)
                }
            });
            match given {
                Some(p) => exit_without_model(&format!("the model file '{p}' does not exist")),
                None => exit_without_model("no GGUF model was found"),
            }
        }
    };
    println!("  Reference GGUF:      {}", model_path.display());
    let gguf = match GgufFile::open(&model_path) {
        Ok(g) => g,
        Err(e) => exit_without_model(&format!("the model file '{}' cannot be opened as GGUF ({e})", model_path.display())),
    };
    println!("  GGUF Ingest:         Memory-Mapped successfully ({:.2} GB file)", model_path.metadata().map(|m| m.len() as f64 / 1e9).unwrap_or(0.0));

    println!("===============================================================================");

    // 2. Define list of evaluated algorithms (AVX2 & ARM NEON regime)
    // The first entry is the baseline of the speed-up column and the reference of the parity check.
    let mut algo_specs: Vec<AlgoSpec> = Vec::new();
    #[cfg(feature = "bitnet-baseline")]
    algo_specs.push(AlgoSpec {
        kernel_type: TernaryKernelType::BitnetMad,
        name: "BitNet MAD (2.00 bpw)",
        paradigm: "MAD",
        bpw: 2.00,
    });
    algo_specs.push(AlgoSpec {
        kernel_type: TernaryKernelType::QuintritLut,
        name: "Quintrit Velurex (1.60 bpw)",
        paradigm: "LUT",
        bpw: 1.60,
    });

    let mut all_results: Vec<E2eBenchmarkResult> = Vec::new();
    let mut algo_generated_texts: HashMap<&'static str, String> = HashMap::new();
    // Generated text per thread count, in evaluation order (used for the determinism check).
    let mut thread_texts: Vec<(String, String)> = Vec::new();
    let mut prompt_token_count = 0usize;

    {
        let g = &gguf;
        // Execute End-to-End InferenceEngine across all evaluated algorithms
        for algo in &algo_specs {
            println!("\n==> Loading & Transcoding Reference Model for {}...", algo.name);
            let t_load = Instant::now();

            let mut engine = match InferenceEngine::load_from_parsed_gguf(
                g,
                max_threads,
                Some(algo.kernel_type),
                Some(&plan.name),
            ) {
                Ok(eng) => eng,
                Err(e) => {
                    eprintln!("  Failed to load engine for {}: {}", algo.name, e);
                    continue;
                }
            };
            println!("  Loaded & Transcoded into RAM in {:.2?}", t_load.elapsed());

            for &threads in &thread_scaling {
                engine.set_num_threads(threads);
                println!(
                    "  Evaluating {} ({} threads, {} tokens, Zero Warmup, temp = 0.0)...",
                    algo.name, threads, gen_tokens
                );

                // Zero-warmup benchmark pass under deterministic greedy decoding
                let rep = engine.benchmark(&chat_prompt, gen_tokens).expect("benchmark prompt must fit into the context");
                prompt_token_count = rep.prompt_tokens;

                let latency_ms = rep.generation_ms / gen_tokens as f64;
                let tokens_per_sec = rep.tokens_per_sec;

                let gflops = (plan.total_mac_ops_per_token * gen_tokens as f64 / (rep.generation_ms * 1e-3)) / 1e9;

                println!(
                    "  [{}] TTFT: {:6.1} ms | Gen: {:5.2} s | {:6.2} ms/tok | {:5.2} tok/s | {:6.2} GB/s | {:6.1} GFLOPS",
                    algo.name,
                    rep.ttft_ms,
                    rep.generation_ms / 1000.0,
                    latency_ms,
                    tokens_per_sec,
                    rep.memory_bandwidth_gb_s,
                    gflops
                );

                all_results.push(E2eBenchmarkResult {
                    algorithm_name: algo.name,
                    paradigm: algo.paradigm,
                    bpw: algo.bpw,
                    threads,
                    prompt_tokens: rep.prompt_tokens,
                    gen_tokens,
                    ttft_ms: rep.ttft_ms,
                    gen_total_ms: rep.generation_ms,
                    latency_ms,
                    tokens_per_sec,
                    weight_mb: rep.dram_traffic_per_token_mb,
                    bandwidth_gbps: rep.memory_bandwidth_gb_s,
                    gflops,
                });

                thread_texts.push((
                    format!("{} ({} thread{})", algo.name, threads, if threads == 1 { "" } else { "s" }),
                    rep.generated_text.clone(),
                ));
                algo_generated_texts.insert(algo.name, rep.generated_text.clone());
            }
        }
    }

    // 3. Output Token Parity Verification
    println!("\n===============================================================================");
    println!("  Output Token Parity Verification (Greedy Decoding, Temp = 0.0)");
    println!("===============================================================================");

    for algo in &algo_specs {
        let text = algo_generated_texts.get(algo.name).map(|s| s.as_str()).unwrap_or("N/A");
        println!("\n  --- [{}] Decoded Text ({} tokens) ---", algo.name, gen_tokens);
        for line in text.lines() {
            println!("  {}", line);
        }
    }
    println!("  -----------------------------------------------------------------------------");

    let (reference_label, reference_text) = thread_texts
        .first()
        .cloned()
        .unwrap_or(("N/A".to_string(), "N/A".to_string()));
    let mut parity_matrix: Vec<(String, String, bool)> = Vec::new();
    let mut all_matched = true;
    for (compared_label, text) in thread_texts.iter().skip(1) {
        let matched = *text == reference_text;
        if !matched {
            all_matched = false;
        }
        println!(
            "  [{}] {} vs {}: {}",
            if matched { "PASS" } else { "FAIL" },
            reference_label,
            compared_label,
            if matched { "100.0% Exact Parity (Tokens match bit-for-bit)" } else { "MISMATCH DETECTED" }
        );
        parity_matrix.push((reference_label.clone(), compared_label.clone(), matched));
    }
    println!("===============================================================================");
    if !all_matched {
        eprintln!("Warning: Output mismatch detected across kernels or thread counts!");
    }

    // 4. Generate Publication-Grade Markdown Telemetry Report
    println!("\n==> Generating Benchmark Report: {}", output_path.display());
    let mut md = String::new();

    let host_tag = suffix.as_deref().unwrap_or("generic");
    let timestamp = format_current_timestamp();

    // Title, introduction and architecture list of the report depend on whether the BitNet kernel was built in.
    let baseline_name = algo_specs[0].name;
    let (report_title, comparison_sentence, architectures_md) = if cfg!(feature = "bitnet-baseline") {
        (
            "End-to-End Reference Model Benchmark (AVX2 & ARM NEON): BitNet 4-trit kernel vs. Quintrit 5-trit kernel".to_string(),
            format!(
                "with Microsoft BitNet's 4-trit multiply-add kernel (2.00 bpw) and the Quintrit velurex encoding LUT kernel (1.60 bpw) \
                across power-of-two thread scaling tiers on AVX2 and ARM NEON architectures. The speed-up column is relative to **{}**.",
                baseline_name
            ),
            "- **BitNet MAD (2.00 bpw)**: the planar multiply-add kernel of Microsoft BitNet (4 trits per byte), the baseline of this report.\n\
             - **Quintrit Velurex (1.60 bpw, -20.0% memory)**: 32-row tiled Dual-Hash Activation-LUT kernel with velurex encoding (5 trits per byte).\n"
                .to_string(),
        )
    } else {
        (
            "End-to-End Reference Model Benchmark (AVX2 & ARM NEON): Quintrit (1.60 bpw)".to_string(),
            "with the Quintrit velurex encoding LUT kernel (1.60 bpw) across power-of-two thread scaling tiers on AVX2 and ARM NEON \
             architectures. To compare with the 4-trit BitNet kernel, build with the cargo feature `bitnet-baseline`."
                .to_string(),
            "- **Quintrit Velurex (1.60 bpw, -20.0% memory)**: 32-row tiled Dual-Hash Activation-LUT kernel with velurex encoding.\n".to_string(),
        )
    };

    md.push_str(&format!(
        "# {}\n\n\
        Comparative evaluation of end-to-end multi-token generation throughput (tokens/sec), per-token decoding latency, \
        Time to First Token (TTFT), and memory streaming bandwidth across all {} decoder layers ({} ternary projections) of **{}** \
        {}\n\n\
        ## Target Platform & Hardware Configuration\n\n\
        - **Host / Device Target**: `{}`\n\
        - **Execution Timestamp**: {}\n\
        - **SIMD Microarchitecture**: {}\n\
        - **Target OS / Architecture**: {} / {}\n\
        - **Available Hardware Threads**: {}\n\
        - **Reference Model**: {} ({} layers, {} hidden, {} intermediate)\n\
        - **Model Ingress Mode**: {}\n\
        - **Benchmark Sequence**: Prompt ({} tokens) -> Generation ({} tokens, total {} GEMV evaluations)\n\
        - **Cache / Warmup Policy**: **Zero Warmup Policy** (Cold DRAM streaming regime, zero artificial cache priming)\n\
        - **Sampling Policy**: Deterministic Greedy Argmax (`temperature = 0.0`, `top_k = 1`, `top_p = 1.0`)\n\
        - **Compiler Optimization**: Profile `bench` (`opt-level = 3`, LTO enabled, native SIMD)\n\n\
        ## Evaluated Architectures\n\n\
        {}\n",
        report_title,
        plan.num_layers,
        plan.num_layers * plan.projections.len(),
        plan.name,
        comparison_sentence,
        host_tag,
        timestamp,
        simd_desc,
        std::env::consts::OS,
        std::env::consts::ARCH,
        max_threads,
        plan.name,
        plan.num_layers,
        plan.hidden_size,
        plan.intermediate_size,
        "Standard GGUF Ingress (memory-mapped)",
        prompt_token_count,
        gen_tokens,
        gen_tokens * plan.num_layers * plan.projections.len(),
        architectures_md,
    ));

    // Per-thread count tables
    for &threads in &thread_scaling {
        let thread_results: Vec<&E2eBenchmarkResult> = all_results.iter().filter(|r| r.threads == threads).collect();
        if thread_results.is_empty() {
            continue;
        }

        let label = if threads == 1 { "1 Thread (Single-Threaded)".to_string() } else { format!("{} Threads", threads) };
        md.push_str(&format!("## Benchmark Telemetry Table — {}\n\n", label));
        let with_speedup = algo_specs.len() > 1;
        md.push_str(&format!(
            "| Implementation | Paradigm | BPW | Model Weights | TTFT | Gen ({} tok) | Latency / Token | Throughput | Memory Bandwidth | GFLOPS |{}\n",
            gen_tokens,
            if with_speedup { " vs. Baseline |" } else { "" }
        ));
        md.push_str(&format!("| :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: | :---: |{}\n", if with_speedup { " :---: |" } else { "" }));
        let baseline_latency = thread_results[0].latency_ms;

        for r in thread_results {
            let gen_sec = r.gen_total_ms / 1000.0;
            md.push_str(&format!(
                "| {} | {} | {:.2} | {:.2} MB | {:.1} ms | {:.2} s | {:.2} ms | **{:.2} tok/s** | {:.2} GB/s | {:.2} |{}\n",
                r.algorithm_name,
                r.paradigm,
                r.bpw,
                r.weight_mb,
                r.ttft_ms,
                gen_sec,
                r.latency_ms,
                r.tokens_per_sec,
                r.bandwidth_gbps,
                r.gflops,
                if !with_speedup {
                    String::new()
                } else if (r.latency_ms - baseline_latency).abs() < 1e-9 {
                    " 1.00x (Baseline) |".to_string()
                } else {
                    format!(" {:.2}x |", baseline_latency / r.latency_ms)
                },
            ));
        }
        md.push('\n');
    }

    // Thread scaling comparison table
    md.push_str("## Multi-Thread Scaling Summary (Tokens / Second)\n\n");
    let mut header = "| Implementation | BPW |".to_string();
    let mut separator = "| :--- | :---: |".to_string();
    for &threads in &thread_scaling {
        header.push_str(&format!(" {}T Latency | {}T Throughput |", threads, threads));
        separator.push_str(" :---: | :---: |");
    }
    md.push_str(&header);
    md.push('\n');
    md.push_str(&separator);
    md.push('\n');

    let unique_algorithms: Vec<&'static str> = {
        let mut algs = Vec::new();
        for r in &all_results {
            if !algs.contains(&r.algorithm_name) {
                algs.push(r.algorithm_name);
            }
        }
        algs
    };

    for alg in unique_algorithms {
        let alg_results: Vec<&E2eBenchmarkResult> = all_results.iter().filter(|r| r.algorithm_name == alg).collect();
        let bpw = alg_results.first().map(|r| r.bpw).unwrap_or(2.0);
        let mut row = format!("| {} | {:.2} |", alg, bpw);

        for &threads in &thread_scaling {
            if let Some(r) = alg_results.iter().find(|r| r.threads == threads) {
                row.push_str(&format!(" {:.1} ms | **{:.2} tok/s** |", r.latency_ms, r.tokens_per_sec));
            } else {
                row.push_str(" N/A | N/A |");
            }
        }
        md.push_str(&row);
        md.push('\n');
    }
    md.push('\n');

    // Output token parity verification section (100% token-for-token agreement)
    md.push_str("## Output Token Parity Verification (Greedy Decoding, Temp = 0.0)\n\n");
    md.push_str(&format!(
        "Under deterministic greedy argmax decoding (`temperature = 0.0`, `top_k = 1`, `top_p = 1.0`), \
        the generated text was compared across all kernels and thread counts for token-for-token output parity over \
        {} generated tokens.\n\n\
        ### Standardized Evaluation Prompt\n\n\
        ```text\n\
        {}\
        ```\n\n",
        gen_tokens,
        chat_prompt,
    ));

    md.push_str(&format!("### Decoded Output ({} Tokens)\n\n", gen_tokens));
    for algo in &algo_specs {
        let text = algo_generated_texts.get(algo.name).map(|s| s.as_str()).unwrap_or("N/A");
        md.push_str(&format!("#### {}\n\n```text\n{}\n```\n\n", algo.name, text.trim()));
    }

    md.push_str(
        "### Algorithmic Equivalence Matrix\n\n\
        | Reference Run | Compared Run | Tokens Compared | Token Parity | Status |\n\
        | :--- | :--- | :---: | :---: | :---: |\n",
    );

    for (base, comp, matched) in &parity_matrix {
        md.push_str(&format!(
            "| {} | {} | {} | {} | {} |\n",
            base,
            comp,
            gen_tokens,
            if *matched { "100.0%" } else { "Mismatch" },
            if *matched { "**PASSED**" } else { "**FAILED**" },
        ));
    }
    md.push('\n');

    if let Err(e) = std::fs::create_dir_all(&results_dir) {
        eprintln!("Error creating results directory: {}", e);
    }
    if let Err(e) = std::fs::write(&output_path, &md) {
        eprintln!("Error writing benchmark report to {}: {}", output_path.display(), e);
    } else {
        println!("==> Successfully wrote benchmark telemetry report to: {}", output_path.display());
    }

    println!("===============================================================================");
}

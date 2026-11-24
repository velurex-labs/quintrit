//! Standalone CLI local inference binary and benchmark harness (`quintrit`).
//!
//! Ultra-lean, high-performance local inference binary in pure Rust without
//! external C++ or GGML runtime dependencies. Ingests standard community GGUF models directly.

use std::env;
use std::path::PathBuf;
use std::process::exit;
use quintrit_runtime::engine::{format_chat_prompt, ChatMessage, InferenceEngine};
use quintrit_runtime::layer::TernaryKernelType;
use quintrit_runtime::sampler::SamplingConfig;

fn print_usage(program: &str) {
    println!(
        r#"
Quintrit Standalone Inference Engine (`quintrit`)
================================================
Ultra-lean, high-performance local inference binary in pure Rust without
external C++ or GGML runtime dependencies. Ingests standard community GGUF
models directly and dynamically transcodes weights into 1.60 bpw Velurex format in RAM.

USAGE:
    {program} [OPTIONS]

OPTIONS:
    --model <PATH>          Path to GGUF model file (*.gguf)
                            (default auto-detected from models/ directory)
    --prompt <TEXT>         Input prompt text for autoregressive generation
                            (default: "Once upon a time in a digital world,")
    --max-tokens <N>        Maximum number of tokens to generate (default: 32)
    --temp <FLOAT>          Sampling temperature (default: 0.7; 0.0 = greedy)
    --top-k <INT>           Top-K candidate filtering (default: 40)
    --top-p <FLOAT>         Nucleus Top-P cumulative threshold (default: 0.9)
    --threads <N>           Number of worker threads (default: CPU cores)
    --kernel <KERNEL>       Target ternary GEMV kernel implementation:
                            - quintrit:      Quintrit 1.60 bpw Velurex 32-row tiled activation LUT (default)
                            - quintrit_512: Quintrit 1.60 bpw 64-row tiled activation LUT on AVX-512
    --device <DEVICE>       Where the model runs (default: cpu):
                            - cpu:       the CPU kernels above
                            - cuda[:N]:  NVIDIA GPU N (default 0; compute capability 8.0 or newer; build with
                                         `--features cuda`). Results are bit-identical to the CPU path.
    --context <N>           Context window in tokens on the GPU (default 4096, at most 7936)
    --bench                 Run benchmark pass (greedy; measures TTFT and tokens/sec, on the CPU also the
                            weight traffic of dense models)
    --prompt-file <PATH>    Read the prompt from a text file instead of --prompt
    --chat-template         Wrap the prompt as one user message in the chat template of the model
    --warmup <N>            Benchmark runs that are executed and discarded first (default 0)
    --repeat <N>            Measured benchmark runs (default 1)
    --json                  After the benchmark print one line `QUINTRIT_BENCH_JSON {{...}}` with all runs and the
                            run conditions
    --chat                  Enter interactive multi-turn chat session
    --stop <TEXT>           Additional stop sequence to halt generation (can be repeated)
    -h, --help              Print this help information
"#,
        program = program
    );
}

/// Peak resident set size of this process in KiB (Linux only; `None` elsewhere).
fn peak_rss_kib() -> Option<u64> {
    let status = std::fs::read_to_string("/proc/self/status").ok()?;
    let line = status.lines().find(|l| l.starts_with("VmHWM:"))?;
    line.split_whitespace().nth(1)?.parse().ok()
}

fn main() {
    // The command-line tool reports loading progress; the library is silent by default.
    quintrit_runtime::set_verbose(true);
    let args: Vec<String> = env::args().collect();
    let program = &args[0];

    let default_candidates = [
        PathBuf::from("models/BitNet-b1.58-2B-4T.gguf"),
        PathBuf::from("models/bitnet-b1.58-2B-4T/ggml-model-i2_s.gguf"),
        PathBuf::from("models/ggml-model-i2_s.gguf"),
        PathBuf::from("ggml-model-i2_s.gguf"),
    ];
    let mut model_path = default_candidates
        .into_iter()
        .find(|p| p.exists())
        .unwrap_or_else(|| PathBuf::from("models/BitNet-b1.58-2B-4T.gguf"));

    let mut prompt = String::from("Once upon a time in a digital world,");
    let mut max_tokens: usize = 32;
    let mut temperature: f32 = 0.7;
    let mut top_k: usize = 40;
    let mut top_p: f32 = 0.9;
    let mut target_kernel = TernaryKernelType::QuintritLut;
    let mut gpu_device: Option<usize> = None;
    let mut gpu_context: Option<usize> = None;
    let mut num_threads = std::thread::available_parallelism()
        .map(|n| n.get())
        .unwrap_or(4);
    let mut run_bench = false;
    let mut bench_warmup: usize = 0;
    let mut bench_repeat: usize = 1;
    let mut bench_json = false;
    let mut chat_template = false;
    let mut prompt_file: Option<PathBuf> = None;
    let mut run_chat = false;
    let mut custom_stops: Vec<String> = Vec::new();

    let mut i = 1;
    while i < args.len() {
        match args[i].as_str() {
            "--model" => {
                i += 1;
                if i < args.len() {
                    model_path = PathBuf::from(&args[i]);
                }
            }
            "--prompt" => {
                i += 1;
                if i < args.len() {
                    prompt = args[i].clone();
                }
            }
            "--max-tokens" => {
                i += 1;
                if i < args.len() {
                    max_tokens = args[i].parse().unwrap_or(32);
                }
            }
            "--temp" => {
                i += 1;
                if i < args.len() {
                    temperature = args[i].parse().unwrap_or(0.7);
                }
            }
            "--top-k" => {
                i += 1;
                if i < args.len() {
                    top_k = args[i].parse().unwrap_or(40);
                }
            }
            "--top-p" => {
                i += 1;
                if i < args.len() {
                    top_p = args[i].parse().unwrap_or(0.9);
                }
            }
            "--kernel" => {
                i += 1;
                if i < args.len() {
                    match args[i].to_ascii_lowercase().as_str() {
                        "quintrit" | "velurex" => target_kernel = TernaryKernelType::QuintritLut,
                        "quintrit_512" | "velurex_512" | "quintrit_avx512" | "avx512" => target_kernel = TernaryKernelType::QuintritLutAVX512,
                        #[cfg(feature = "bitnet-baseline")]
                        "bitnet_mad" => target_kernel = TernaryKernelType::BitnetMad,
                        #[cfg(feature = "bitnet-baseline")]
                        "bitnet_mad_512" => target_kernel = TernaryKernelType::BitnetMad512,
                        other => {
                            eprintln!("Unknown kernel '{}'. Using default 'quintrit'.", other);
                        }
                    }
                }
            }
            "--threads" => {
                i += 1;
                if i < args.len() {
                    num_threads = args[i].parse().unwrap_or(num_threads);
                }
            }
            "--device" => {
                i += 1;
                if i < args.len() {
                    let d = args[i].to_ascii_lowercase();
                    if d == "cpu" {
                        gpu_device = None;
                    } else if let Some(rest) = d.strip_prefix("cuda") {
                        let ordinal = rest.strip_prefix(':').map(|n| n.parse::<usize>()).unwrap_or(Ok(0));
                        match ordinal {
                            Ok(n) => gpu_device = Some(n),
                            Err(_) => {
                                eprintln!("Invalid device '{}': use cpu, cuda or cuda:<N>.", d);
                                exit(1);
                            }
                        }
                    } else {
                        eprintln!("Unknown device '{}': use cpu, cuda or cuda:<N>.", d);
                        exit(1);
                    }
                }
            }
            "--context" => {
                i += 1;
                if i < args.len() {
                    gpu_context = args[i].parse().ok();
                }
            }
            "--bench" => {
                run_bench = true;
            }
            "--json" => {
                bench_json = true;
            }
            "--chat-template" => {
                chat_template = true;
            }
            "--prompt-file" => {
                i += 1;
                if i < args.len() {
                    prompt_file = Some(PathBuf::from(&args[i]));
                }
            }
            "--warmup" => {
                i += 1;
                if i < args.len() {
                    bench_warmup = args[i].parse().unwrap_or(0);
                }
            }
            "--repeat" => {
                i += 1;
                if i < args.len() {
                    bench_repeat = args[i].parse().unwrap_or(1).max(1);
                }
            }
            "--chat" => {
                run_chat = true;
            }
            "--stop" => {
                i += 1;
                if i < args.len() {
                    custom_stops.push(args[i].clone());
                }
            }
            "-h" | "--help" => {
                print_usage(program);
                exit(0);
            }
            arg => {
                eprintln!("Unknown option: {}", arg);
                print_usage(program);
                exit(1);
            }
        }
        i += 1;
    }

    println!("===============================================================================");
    println!("  Quintrit Standalone Local Inference Runtime (`quintrit`)");
    println!("===============================================================================");
    println!("  Model Path:      {}", model_path.display());
    println!("  Threads:         {}", num_threads);
    println!("  Target Kernel:   {:?} ({:.2} bpw)", target_kernel, target_kernel.bpw());
    println!("  Max Tokens:      {}", max_tokens);
    println!("  Temperature:     {:.2}", temperature);
    println!("  Top-P / Top-K:   {:.2} / {}", top_p, top_k);
    println!("===============================================================================");

    // Verify model file existence
    if !model_path.exists() {
        eprintln!("ERROR: Model file not found at: {}", model_path.display());
        eprintln!("Please specify a valid GGUF model path via: --model <PATH>");
        exit(1);
    }

    if model_path.is_dir() {
        let gguf_cand = model_path.join("ggml-model-i2_s.gguf");
        if gguf_cand.exists() {
            model_path = gguf_cand;
        }
    }

    // Load inference engine and transcode into RAM
    let loaded = match gpu_device {
        None => InferenceEngine::load_from_gguf(&model_path, num_threads, Some(target_kernel)),
        #[cfg(feature = "cuda")]
        Some(ordinal) => InferenceEngine::load_from_gguf_on_gpu(&model_path, ordinal, gpu_context),
        #[cfg(not(feature = "cuda"))]
        Some(_) => {
            let _ = gpu_context;
            Err("this build of `quintrit` has no GPU support; rebuild with `--features cuda`".to_string())
        }
    };
    let mut engine = match loaded {
        Ok(eng) => {
            if let Some(name) = eng.gpu_device_name() {
                println!("  Device:          {} (CUDA)", name);
            }
            eng
        }
        Err(err) => {
            eprintln!("Failed to load GGUF model: {}", err);
            exit(1);
        }
    };

    let sampling_cfg = SamplingConfig {
        temperature,
        top_k,
        top_p,
        seed: 42,
    };

    if let Some(path) = &prompt_file {
        prompt = match std::fs::read_to_string(path) {
            Ok(text) => text.trim_end_matches(['\r', '\n']).to_string(),
            Err(e) => {
                eprintln!("Cannot read the prompt file {}: {e}", path.display());
                exit(1);
            }
        };
    }
    if chat_template {
        let message = ChatMessage { role: "user".to_string(), content: prompt.clone(), name: None };
        prompt = format_chat_prompt(&[message], engine.model.config.architecture);
    }

    if run_bench {
        println!("\n==> Running Performance Benchmark Pass (Tokens: {}, warm-up runs: {}, measured runs: {})...", max_tokens, bench_warmup, bench_repeat);
        let mut reports = Vec::with_capacity(bench_repeat);
        for run in 0..bench_warmup + bench_repeat {
            let report = match engine.benchmark(&prompt, max_tokens) {
                Ok(report) => report,
                Err(e) => {
                    eprintln!("Error: {e}");
                    exit(1);
                }
            };
            let tag = if run < bench_warmup { "warm-up" } else { "measured" };
            println!(
                "  [{tag} {}] TTFT {:.2} ms, {} tokens in {:.2} ms, {:.2} tokens/sec",
                if run < bench_warmup { run + 1 } else { run + 1 - bench_warmup },
                report.ttft_ms,
                report.generated_tokens,
                report.generation_ms,
                report.tokens_per_sec
            );
            if run >= bench_warmup {
                reports.push(report);
            }
        }
        let report = reports.last().expect("at least one measured run");

        println!("\n===============================================================================");
        println!("  Inference Benchmark Results ({:?} Kernel)", report.kernel_type);
        println!("===============================================================================");
        println!("  Prompt Tokens Ingested:      {}", report.prompt_tokens);
        println!("  Tokens Generated:            {}", report.generated_tokens);
        println!("  Time to First Token (TTFT):  {:.2} ms", report.ttft_ms);
        println!("  Generation Latency:          {:.2} ms", report.generation_ms);
        println!("  Generation Throughput:       {:.2} tokens/sec", report.tokens_per_sec);
        match (report.dram_traffic_per_token_mb, report.memory_bandwidth_gb_s) {
            (Some(traffic), Some(bandwidth)) => {
                println!("  DRAM Weight Traffic/Token:   {:.2} MB", traffic);
                println!("  DRAM Memory Bandwidth:       {:.2} GB/s", bandwidth);
            }
            _ => println!("  Weight traffic:              not reported for this model or device"),
        }
        println!("===============================================================================");

        if bench_json {
            let mut rates: Vec<f64> = reports.iter().map(|r| r.tokens_per_sec).collect();
            rates.sort_by(|a, b| a.total_cmp(b));
            let median = if rates.len() % 2 == 1 {
                rates[rates.len() / 2]
            } else {
                (rates[rates.len() / 2 - 1] + rates[rates.len() / 2]) / 2.0
            };
            let runs: Vec<serde_json::Value> = reports
                .iter()
                .map(|r| {
                    serde_json::json!({
                        "ttft_ms": r.ttft_ms,
                        "generation_ms": r.generation_ms,
                        "generated_tokens": r.generated_tokens,
                        "tokens_per_sec": r.tokens_per_sec,
                    })
                })
                .collect();
            let gpu_name = engine.gpu_device_name();
            let json = serde_json::json!({
                "model_file": model_path.file_name().map(|n| n.to_string_lossy().to_string()),
                "model_bytes": std::fs::metadata(&model_path).map(|m| m.len()).ok(),
                "architecture": format!("{:?}", engine.model.config.architecture),
                "device": gpu_name.clone().unwrap_or_else(|| "cpu".to_string()),
                "kernel": if gpu_name.is_some() { "gpu".to_string() } else { format!("{:?}", report.kernel_type) },
                "threads": if gpu_name.is_some() { serde_json::Value::Null } else { serde_json::json!(num_threads) },
                "context": engine.kv_cache.max_seq_len(),
                "chat_template": chat_template,
                "prompt_tokens": report.prompt_tokens,
                "requested_tokens": max_tokens,
                "warmup_runs": bench_warmup,
                "runs": runs,
                "tokens_per_sec_median": median,
                "tokens_per_sec_min": rates.first(),
                "tokens_per_sec_max": rates.last(),
                "dram_traffic_per_token_mb": report.dram_traffic_per_token_mb,
                "memory_bandwidth_gb_s": report.memory_bandwidth_gb_s,
                "peak_rss_kib": peak_rss_kib(),
                "generated_text": report.generated_text,
            });
            println!("QUINTRIT_BENCH_JSON {json}");
        }
    } else if run_chat {
        println!("\n==> Entering Interactive Multi-Turn Chat Session");
        println!("    Model Architecture: {:?}", engine.model.config.architecture);
        println!("    Type your message and press Enter.");
        println!("    Commands: /reset (clears conversation history), /exit or /quit (exits)");
        println!("===============================================================================");

        let mut conversation: Vec<ChatMessage> = Vec::new();
        let arch = engine.model.config.architecture;

        let mut default_stops: Vec<String> = match arch {
            quintrit_runtime::config::ModelArchitecture::Qwen3
            | quintrit_runtime::config::ModelArchitecture::Qwen35
            | quintrit_runtime::config::ModelArchitecture::NemotronHMoe => vec![
                "<|im_end|>".to_string(),
                "<|im_start|>".to_string(),
                "<|endoftext|>".to_string(),
            ],
            quintrit_runtime::config::ModelArchitecture::Falcon3 => vec![
                "<|user|>".to_string(),
                "<|endoftext|>".to_string(),
                "<|im_end|>".to_string(),
                "<|assistant|>".to_string(),
                "\n<|user|>".to_string(),
                "\nUser:".to_string(),
                "User:".to_string(),
            ],
            quintrit_runtime::config::ModelArchitecture::BitNet | quintrit_runtime::config::ModelArchitecture::Llama => vec![
                "###".to_string(),
                "\n###".to_string(),
                "### User".to_string(),
                "### User:".to_string(),
                "### Assistant".to_string(),
                "### Response".to_string(),
                "\nUser:".to_string(),
                "User:".to_string(),
                "<|end_of_text|>".to_string(),
                "<|eot_id|>".to_string(),
                "<|im_end|>".to_string(),
            ],
        };
        default_stops.extend(custom_stops.clone());
        let stop_refs: Vec<&str> = default_stops.iter().map(|s| s.as_str()).collect();

        loop {
            print!("\nUser > ");
            let _ = std::io::Write::flush(&mut std::io::stdout());

            let mut input = String::new();
            if std::io::stdin().read_line(&mut input).is_err() {
                break;
            }

            let trimmed = input.trim();
            if trimmed.is_empty() {
                continue;
            }

            if trimmed == "/exit" || trimmed == "/quit" {
                println!("Exiting chat. Goodbye!");
                break;
            }

            if trimmed == "/reset" {
                conversation.clear();
                engine.kv_cache.reset();
                println!("==> Conversation history and KV cache reset.");
                continue;
            }

            conversation.push(ChatMessage {
                role: "user".to_string(),
                content: trimmed.to_string(),
                name: None,
            });

            let formatted_prompt = format_chat_prompt(&conversation, arch);
            print!("Assistant > ");
            let _ = std::io::Write::flush(&mut std::io::stdout());

            // The engine keeps the KV cache of the previous turn and computes only the tokens after the shared history.
            let result = engine.generate_streaming_with_stops(&formatted_prompt, max_tokens, &sampling_cfg, &stop_refs, |token_str| {
                print!("{}", token_str);
                let _ = std::io::Write::flush(&mut std::io::stdout());
            });
            println!();
            let assistant_response = match result {
                Ok(text) => text,
                Err(e) => {
                    // The conversation no longer fits into the context window: report it and drop the last user turn.
                    eprintln!("Error: {e}");
                    conversation.pop();
                    continue;
                }
            };

            conversation.push(ChatMessage {
                role: "assistant".to_string(),
                content: assistant_response.trim().to_string(),
                name: None,
            });
        }
    } else {
        println!("\n==> Prompt: \"{}\"", prompt);
        print!("==> Generation: ");
        std::io::Write::flush(&mut std::io::stdout()).unwrap();

        let stop_refs: Vec<&str> = custom_stops.iter().map(|s| s.as_str()).collect();
        let result = engine.generate_streaming_with_stops(&prompt, max_tokens, &sampling_cfg, &stop_refs, |token_str| {
            print!("{}", token_str);
            std::io::Write::flush(&mut std::io::stdout()).unwrap();
        });
        if let Err(e) = result {
            eprintln!("
Error: {e}");
            exit(1);
        }

        // The callback receives pieces of text, not tokens, so the counts come from the engine.
        let stats = engine.last_generation_stats();
        let per_sec = |n: usize, ms: f64| if ms > 0.0 { n as f64 * 1000.0 / ms } else { 0.0 };
        let computed_prompt = stats.prompt_tokens - stats.reused_prompt_tokens;
        println!("\n\n-------------------------------------------------------------------------------");
        println!(
            "Prompt:     {} tokens in {:.2} ms ({:.2} tokens/sec)",
            computed_prompt,
            stats.prompt_ms,
            per_sec(computed_prompt, stats.prompt_ms)
        );
        println!(
            "Generation: {} tokens in {:.2} ms ({:.2} tokens/sec)",
            stats.generated_tokens,
            stats.generation_ms,
            per_sec(stats.generated_tokens, stats.generation_ms)
        );
        println!("-------------------------------------------------------------------------------");
    }
}

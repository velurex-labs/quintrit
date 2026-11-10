//! # Energy per decoded token (Raspberry Pi 5, PMIC rail telemetry)
//!
//! Runs the real decode loop of the inference engine (production kernels, real model, greedy decoding) and integrates
//! the power read from the Raspberry Pi 5 power-management IC while it runs. Result: joules per generated token,
//! split into the CPU core rail, the DRAM rails and everything else, next to the same measurement of an idle
//! machine, so that the energy attributable to the decode work can be separated from the base load of the board.
//!
//! ## Sensor
//!
//! `vcgencmd pmic_read_adc` returns one current and one voltage per PMIC output rail (for example `VDD_CORE_A` and
//! `VDD_CORE_V`). The power of a rail is voltage times current; the sum over the rails is the power delivered by the
//! PMIC to the SoC, the DRAM and the 3.3 V / 1.8 V board rails. It does **not** include the conversion losses of the
//! PMIC and the 5 V input stage, USB peripherals powered from the board or the fan, so it is lower than the power
//! drawn at the wall. One reading takes about 22 ms; the sampler thread reads (about 6 Hz with the default pause of 150 ms) during the
//! whole run. The mean power of the samples inside a window multiplied by the window length is the energy of that
//! window. There is no fallback: when `vcgencmd pmic_read_adc` is unavailable the benchmark stops instead of
//! reporting a model estimate.
//!
//! ```bash
//! cargo bench -p quintrit-runtime --bench token_energy -- --model "$MODEL" \
//!     --kernel quintrit --threads 4 --tokens 48 --suffix rpi5-cortex-a76 --out results
//! # optional: --idle-secs 10  --sample-pause-ms 150
//! ```
//!
//! `$MODEL` is the path of the GGUF file. `--kernel` takes one or more kernels separated by commas: `quintrit`,
//! `quintrit_512` and, with the cargo feature `bitnet-baseline`, `bitnet_mad` and `bitnet_mad_512` (BitNet's 4-trit
//! multiply-add kernel). With several kernels the benchmark measures them one after the other, writes a report for each
//! and a comparison report (`token_energy_compare_t<threads>_<suffix>.md`) with the energy per token relative to the
//! first kernel and a check that all kernels generated the same tokens. Without `--kernel` it compares `bitnet_mad` with
//! `quintrit` when the feature is built in, and measures `quintrit` otherwise.

mod common;
use common::run_header;
use quintrit_runtime::layer::TernaryKernelType;
use quintrit_runtime::{InferenceEngine, Sampler, SamplingConfig};
use std::collections::BTreeMap;
use std::fmt::Write as _;
use std::process::Command;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// One reading of all rails: midpoint of the call and the power of every rail in watts.
type Sample = (Instant, BTreeMap<String, f64>);

/// Reads the PMIC once. Returns the power in watts per rail (rails with both a current and a voltage).
fn read_rails() -> Option<BTreeMap<String, f64>> {
    let out = Command::new("vcgencmd").arg("pmic_read_adc").output().ok()?;
    if !out.status.success() {
        return None;
    }
    let text = String::from_utf8_lossy(&out.stdout);
    let (mut amps, mut volts) = (BTreeMap::new(), BTreeMap::new());
    for line in text.lines() {
        let line = line.trim();
        let Some((lhs, rhs)) = line.split_once('=') else { continue };
        let Some(name) = lhs.split_whitespace().next() else { continue };
        let value: f64 = match rhs.trim().trim_end_matches(['A', 'V']).parse() {
            Ok(v) => v,
            Err(_) => continue,
        };
        if let Some(rail) = name.strip_suffix("_A") {
            amps.insert(rail.to_string(), value);
        } else if let Some(rail) = name.strip_suffix("_V") {
            volts.insert(rail.to_string(), value);
        }
    }
    let rails: BTreeMap<String, f64> = amps.iter().filter_map(|(r, a)| volts.get(r).map(|v| (r.clone(), a * v))).collect();
    if rails.is_empty() { None } else { Some(rails) }
}

fn vcgencmd_value(args: &[&str]) -> String {
    Command::new("vcgencmd").args(args).output().ok().map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string()).unwrap_or_default()
}

struct Sampler5 {
    stop: Arc<AtomicBool>,
    samples: Arc<Mutex<Vec<Sample>>>,
    worker: Option<std::thread::JoinHandle<()>>,
}

impl Sampler5 {
    fn start(pause_ms: u64) -> Self {
        let stop = Arc::new(AtomicBool::new(false));
        let samples: Arc<Mutex<Vec<Sample>>> = Arc::new(Mutex::new(Vec::new()));
        let (s2, v2) = (stop.clone(), samples.clone());
        let worker = std::thread::spawn(move || {
            while !s2.load(Ordering::Relaxed) {
                let t0 = Instant::now();
                if let Some(rails) = read_rails() {
                    let t1 = Instant::now();
                    v2.lock().unwrap().push((t0 + (t1 - t0) / 2, rails));
                }
                std::thread::sleep(Duration::from_millis(pause_ms));
            }
        });
        Self { stop, samples, worker: Some(worker) }
    }

    /// Mean power per rail and the number of samples with their midpoint inside `[from, to]`.
    fn window(&self, from: Instant, to: Instant) -> (BTreeMap<String, f64>, usize) {
        let guard = self.samples.lock().unwrap();
        let mut sum: BTreeMap<String, f64> = BTreeMap::new();
        let mut n = 0usize;
        for (t, rails) in guard.iter().filter(|(t, _)| *t >= from && *t <= to) {
            let _ = t;
            n += 1;
            for (r, p) in rails {
                *sum.entry(r.clone()).or_insert(0.0) += p;
            }
        }
        for v in sum.values_mut() {
            *v /= n.max(1) as f64;
        }
        (sum, n)
    }

    fn finish(mut self) {
        self.stop.store(true, Ordering::Relaxed);
        if let Some(w) = self.worker.take() {
            let _ = w.join();
        }
    }
}

fn group_of(rail: &str) -> &'static str {
    match rail {
        "VDD_CORE" => "core",
        "DDR_VDD2" | "DDR_VDDQ" => "dram",
        _ => "other",
    }
}

fn groups(rails: &BTreeMap<String, f64>) -> (f64, f64, f64) {
    let (mut c, mut d, mut o) = (0.0, 0.0, 0.0);
    for (r, p) in rails {
        match group_of(r) {
            "core" => c += p,
            "dram" => d += p,
            _ => o += p,
        }
    }
    (c, d, o)
}

fn arg(name: &str) -> Option<String> {
    let a: Vec<String> = std::env::args().collect();
    a.iter().position(|x| x == name).and_then(|p| a.get(p + 1).cloned())
}

/// Maps the name of a kernel on the command line to the runtime kernel.
fn parse_kernel(name: &str) -> Result<TernaryKernelType, String> {
    match name {
        "quintrit" => Ok(TernaryKernelType::QuintritLut),
        "quintrit_512" => Ok(TernaryKernelType::QuintritLutAVX512),
        #[cfg(feature = "bitnet-baseline")]
        "bitnet_mad" => Ok(TernaryKernelType::BitnetMad),
        #[cfg(feature = "bitnet-baseline")]
        "bitnet_mad_512" => Ok(TernaryKernelType::BitnetMad512),
        #[cfg(not(feature = "bitnet-baseline"))]
        "bitnet_mad" | "bitnet_mad_512" => {
            Err(format!("the kernel '{name}' needs the cargo feature `bitnet-baseline` (cargo bench --features bitnet-baseline)"))
        }
        other => Err(format!(
            "unknown kernel '{other}' (available: quintrit, quintrit_512, bitnet_mad, bitnet_mad_512; bitnet_* need --features bitnet-baseline)"
        )),
    }
}

/// Result of the energy measurement of one kernel.
struct Measurement {
    kernel_name: String,
    kernel: TernaryKernelType,
    md: String,
    ms_per_token: f64,
    tok_s: f64,
    p_active: f64,
    j_token: f64,
    j_net: f64,
    j_per_gb: f64,
    bytes_token: usize,
    generated: Vec<u32>,
}

/// Loads the model with one kernel and measures the energy of `tokens` greedy decode steps.
fn measure(
    kernel_name: &str,
    kernel: TernaryKernelType,
    model_path: &std::path::Path,
    threads: usize,
    tokens: usize,
    sample_pause_ms: u64,
    idle_secs: u64,
    suffix: &str,
) -> Measurement {
    let mut engine = InferenceEngine::load_from_gguf(model_path, threads, Some(kernel)).expect("load model");
    let prompt = "<|user|>\nWhat is ternary computing?\n<|assistant|>\n";
    let mut prompt_tokens = engine.tokenizer.encode(prompt);
    if let Some(bos) = engine.tokenizer.bos_token_id() {
        if prompt_tokens.first() != Some(&bos) {
            prompt_tokens.insert(0, bos);
        }
    }
    let vocab = engine.model.config.vocab_size;
    let hidden = engine.model.config.hidden_size;
    let ternary_bytes: usize = engine.model.blocks.iter().map(|b| b.attn.memory_bytes() + b.gate_proj.memory_bytes() + b.up_proj.memory_bytes() + b.down_proj.memory_bytes()).sum();
    let lm_bytes = vocab * hidden * 2;
    let mut logits = vec![0.0f32; vocab];

    let sampler5 = Sampler5::start(sample_pause_ms);

    // Prefill (not measured), then let the board settle after the model load.
    engine.kv_cache.reset();
    for (pos, &t) in prompt_tokens.iter().enumerate() {
        engine.model.forward_token_with_scratchpad(t, pos, &mut engine.kv_cache, &mut engine.scratchpad, &mut logits);
    }
    std::thread::sleep(Duration::from_secs(3));
    let temp_before = vcgencmd_value(&["measure_temp"]);

    // Idle window before the run (model resident, no work).
    let (i0, i1) = (Instant::now(), Instant::now() + Duration::from_secs(idle_secs));
    std::thread::sleep(i1 - i0);
    let (idle_before, idle_before_n) = sampler5.window(i0, i1);

    // Decode window: greedy decoding with the production engine path.
    let mut greedy = Sampler::new(SamplingConfig::greedy());
    let mut pos = prompt_tokens.len();
    let mut generated = Vec::with_capacity(tokens);
    let t_start = Instant::now();
    for _ in 0..tokens {
        let next = greedy.sample(&logits);
        generated.push(next);
        engine.model.forward_token_with_scratchpad(next, pos, &mut engine.kv_cache, &mut engine.scratchpad, &mut logits);
        pos += 1;
    }
    let t_end = Instant::now();
    let (active, active_n) = sampler5.window(t_start, t_end);
    let temp_after = vcgencmd_value(&["measure_temp"]);
    let throttled = vcgencmd_value(&["get_throttled"]);

    // Idle window after the run (drift check: temperature and leakage).
    std::thread::sleep(Duration::from_secs(2));
    let (a0, a1) = (Instant::now(), Instant::now() + Duration::from_secs(idle_secs));
    std::thread::sleep(a1 - a0);
    let (idle_after, idle_after_n) = sampler5.window(a0, a1);
    sampler5.finish();

    let secs = (t_end - t_start).as_secs_f64();
    let (ac, ad, ao) = groups(&active);
    let (ic, id, io) = groups(&idle_before);
    let (jc, jd, jo) = groups(&idle_after);
    let p_active = ac + ad + ao;
    let p_idle = ic + id + io;
    let p_idle_after = jc + jd + jo;
    let n = tokens as f64;
    let j_token = p_active * secs / n;
    let j_net = (p_active - p_idle) * secs / n;
    let (kernel_label, bytes_token) = (format!("{:?}", kernel), ternary_bytes + lm_bytes);
    let j_per_gb = j_net / (bytes_token as f64 / 1e9);

    let mut md = String::new();
    let _ = writeln!(md, "# Energy per token: {suffix}, kernel {kernel_label}, {threads} threads\n");
    let _ = writeln!(md, "## Run\n\n{}", run_header(&format!("{kernel_label} (production engine path)"), threads, tokens));
    let _ = writeln!(md, "Falcon3-10B decode, {tokens} greedy tokens after an {}-token prompt; PMIC rail power from `vcgencmd pmic_read_adc` (rails the PMIC delivers; without conversion losses, peripherals and fan).\n", prompt_tokens.len());
    let _ = writeln!(md, "| Quantity | Value |\n|---|---:|");
    let _ = writeln!(md, "| Decode window | {:.2} s ({:.1} ms/token, {:.2} tok/s) |", secs, secs / n * 1e3, n / secs);
    let _ = writeln!(md, "| PMIC samples (active / idle before / idle after) | {active_n} / {idle_before_n} / {idle_after_n} |");
    let _ = writeln!(md, "| Mean power, active | {p_active:.3} W (core {ac:.3}, DRAM {ad:.3}, other {ao:.3}) |");
    let _ = writeln!(md, "| Mean power, idle before the run (model resident) | {p_idle:.3} W (core {ic:.3}, DRAM {id:.3}, other {io:.3}) |");
    let _ = writeln!(md, "| Mean power, idle after the run | {p_idle_after:.3} W (core {jc:.3}, DRAM {jd:.3}, other {jo:.3}) |");
    let _ = writeln!(md, "| **Energy per token, total** | **{:.3} J** (core {:.3}, DRAM {:.4}, other {:.3}) |", j_token, ac * secs / n, ad * secs / n, ao * secs / n);
    let _ = writeln!(md, "| **Energy per token, above idle** | **{:.3} J** (core {:.3}, DRAM {:.4}, other {:.3}) |", j_net, (ac - ic) * secs / n, (ad - id) * secs / n, (ao - io) * secs / n);
    let _ = writeln!(md, "| Weight bytes read per token | {:.0} MB ({:.0} MB ternary + {:.0} MB F16 lm_head) |", bytes_token as f64 / 1e6, ternary_bytes as f64 / 1e6, lm_bytes as f64 / 1e6);
    let _ = writeln!(md, "| Energy per GB of weight bytes read (above idle) | {j_per_gb:.3} J/GB |");
    let _ = writeln!(md, "| SoC temperature before / after | {temp_before} / {temp_after} |");
    let _ = writeln!(md, "| `get_throttled` after the run | {throttled} |\n");
    let _ = writeln!(md, "### Mean power per rail (W)\n\n| Rail | Group | Active | Idle before | Idle after |\n|---|---|---:|---:|---:|");
    for (r, p) in &active {
        let _ = writeln!(md, "| {r} | {} | {p:.3} | {:.3} | {:.3} |", group_of(r), idle_before.get(r).copied().unwrap_or(0.0), idle_after.get(r).copied().unwrap_or(0.0));
    }
    println!(
        "ENERGY kernel={kernel_name} threads={threads} tokens={tokens} ms_per_token={:.2} p_active_w={p_active:.3} p_idle_w={p_idle:.3} p_idle_after_w={p_idle_after:.3} j_token={j_token:.4} j_token_net={j_net:.4} core_w={ac:.3} dram_w={ad:.4} other_w={ao:.3} idle_core_w={ic:.3} idle_dram_w={id:.4} idle_other_w={io:.3}",
        secs / n * 1e3
    );
    Measurement {
        kernel_name: kernel_name.to_string(),
        kernel,
        md,
        ms_per_token: secs / n * 1e3,
        tok_s: n / secs,
        p_active,
        j_token,
        j_net,
        j_per_gb,
        bytes_token,
        generated,
    }
}

fn main() {
    // The benchmarks keep the load-time progress lines; the library is silent by default.
    quintrit_runtime::set_verbose(true);
    let model_path = common::resolve_model(arg("--model"));
    let threads: usize = arg("--threads").and_then(|v| v.parse().ok()).unwrap_or(4);
    let tokens: usize = arg("--tokens").and_then(|v| v.parse().ok()).unwrap_or(48);
    // Pause between two PMIC readings (a reading itself takes about 22 ms); a longer pause disturbs the decode less.
    let sample_pause_ms: u64 = arg("--sample-pause-ms").and_then(|v| v.parse().ok()).unwrap_or(150);
    let idle_secs: u64 = arg("--idle-secs").and_then(|v| v.parse().ok()).unwrap_or(10);
    let suffix = arg("--suffix").unwrap_or_else(|| "local".into());
    let out_dir = arg("--out").unwrap_or_else(|| "results".into());
    // One or more kernels, separated by commas. The first one is the baseline of the comparison. With the cargo
    // feature `bitnet-baseline` the default compares BitNet's 4-trit kernel with the quintrit kernel.
    let default_kernels = if cfg!(feature = "bitnet-baseline") { "bitnet_mad,quintrit" } else { "quintrit" };
    let kernel_list = arg("--kernel").unwrap_or_else(|| default_kernels.to_string());
    let mut kernels: Vec<(String, TernaryKernelType)> = Vec::new();
    for name in kernel_list.split(',').map(str::trim).filter(|s| !s.is_empty()) {
        match parse_kernel(name) {
            Ok(k) => kernels.push((name.to_string(), k)),
            Err(e) => {
                eprintln!("error: {e}");
                std::process::exit(2);
            }
        }
    }
    if kernels.is_empty() {
        eprintln!("error: --kernel needs at least one kernel name");
        std::process::exit(2);
    }
    if read_rails().is_none() {
        eprintln!("`vcgencmd pmic_read_adc` is not available on this machine: no energy measurement is possible (there is no model fallback).");
        std::process::exit(2);
    }

    let dir = std::path::Path::new(&out_dir);
    let _ = std::fs::create_dir_all(dir);
    let mut results: Vec<Measurement> = Vec::new();
    for (name, kernel) in &kernels {
        println!("\n==> Measuring {name} ({kernel:?})");
        let m = measure(name, *kernel, &model_path, threads, tokens, sample_pause_ms, idle_secs, &suffix);
        println!("{}", m.md);
        let path = dir.join(format!("token_energy_{name}_t{threads}_{suffix}.md"));
        std::fs::write(&path, &m.md).expect("write report");
        println!("report: {}", path.display());
        results.push(m);
    }

    if results.len() > 1 {
        let base = &results[0];
        let mut md = String::new();
        let _ = writeln!(md, "# Energy per token, comparison: {suffix}, {threads} threads\n");
        let names: Vec<String> = results.iter().map(|m| format!("{:?}", m.kernel)).collect();
        let _ = writeln!(md, "## Run\n\n{}", run_header(&format!("{} (production engine path, one after the other)", names.join(", ")), threads, tokens));
        let _ = writeln!(md, "Falcon3-10B decode with each kernel, {tokens} greedy tokens after the same prompt; PMIC rail power from `vcgencmd pmic_read_adc` (rails the PMIC delivers; without conversion losses, peripherals and fan). The kernels run one after the other in this order; the reports of the single runs list the SoC temperature and the throttling state of each. The baseline is **{:?}**: the last column is the energy of the kernel above idle relative to the baseline (below 1 is less energy).\n", base.kernel);
        let _ = writeln!(md, "| Kernel | Encoding | Latency / token | Throughput | Mean power, active | Energy / token, total | Energy / token, above idle | Energy per GB of weights | Weight bytes / token | Tokens identical to baseline | Energy vs. baseline |");
        let _ = writeln!(md, "|---|---:|---:|---:|---:|---:|---:|---:|---:|:---:|---:|");
        for m in &results {
            let identical = if std::ptr::eq(m, base) { "(baseline)".to_string() } else if m.generated == base.generated { "**yes**".to_string() } else { "**NO**".to_string() };
            let ratio = if std::ptr::eq(m, base) { "1.00x".to_string() } else { format!("{:.2}x", m.j_net / base.j_net) };
            let _ = writeln!(
                md,
                "| `{}` ({:?}) | {:.2} bpw | {:.1} ms | {:.2} tok/s | {:.3} W | {:.3} J | **{:.3} J** | {:.3} J/GB | {:.0} MB | {} | {} |",
                m.kernel_name, m.kernel, m.kernel.bpw(), m.ms_per_token, m.tok_s, m.p_active, m.j_token, m.j_net, m.j_per_gb, m.bytes_token as f64 / 1e6, identical, ratio
            );
        }
        println!("{md}");
        let path = dir.join(format!("token_energy_compare_t{threads}_{suffix}.md"));
        std::fs::write(&path, md).expect("write comparison report");
        println!("comparison report: {}", path.display());
        if results.iter().any(|m| m.generated != base.generated) {
            eprintln!("Warning: the kernels generated different tokens; their energies are not comparable.");
        }
    }
}

//! Shared provenance-header helpers of the runtime benchmarks.

#![allow(dead_code)]

/// Description of the CPU features that select a kernel path.
pub fn detected_isa_features() -> String {
    let mut v: Vec<&str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        for (name, on) in [
            ("avx2", is_x86_feature_detected!("avx2")),
            ("avx512f", is_x86_feature_detected!("avx512f")),
            ("avx512bw", is_x86_feature_detected!("avx512bw")),
            ("avx512vbmi", is_x86_feature_detected!("avx512vbmi")),
            ("avx512vnni", is_x86_feature_detected!("avx512vnni")),
            ("f16c", is_x86_feature_detected!("f16c")),
        ] {
            if on {
                v.push(name);
            }
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        v.push("neon");
        for (name, on) in [
            ("dotprod", std::arch::is_aarch64_feature_detected!("dotprod")),
            ("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")),
            ("sve2", std::arch::is_aarch64_feature_detected!("sve2")),
        ] {
            if on {
                v.push(name);
            }
        }
    }
    v.join(", ")
}

pub fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program).args(args).output().ok()?;
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if out.status.success() && !s.is_empty() { Some(s) } else { None }
}

/// UTC date and time without external crates.
pub fn utc_timestamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    let z = days + 719_468;
    let era = z.div_euclid(146_097);
    let doe = z.rem_euclid(146_097);
    let yoe = (doe - doe / 1460 + doe / 36_524 - doe / 146_096) / 365;
    let doy = doe - (365 * yoe + yoe / 4 - yoe / 100);
    let mp = (5 * doy + 2) / 153;
    let d = doy - (153 * mp + 2) / 5 + 1;
    let m = if mp < 10 { mp + 3 } else { mp - 9 };
    let y = yoe + era * 400 + if m <= 2 { 1 } else { 0 };
    format!("{y:04}-{m:02}-{d:02} {:02}:{:02} UTC", rem / 3600, (rem % 3600) / 60)
}

/// Provenance table of the report; values come from `QUINTRIT_BENCH_*` environment variables when set.
pub fn run_header(kernel_path: &str, threads: usize, tokens: usize) -> String {
    let env = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
    let commit = env("QUINTRIT_BENCH_COMMIT").or_else(|| command_output("git", &["rev-parse", "--short", "HEAD"])).unwrap_or_else(|| "not recorded".into());
    let rustc = env("QUINTRIT_BENCH_RUSTC").or_else(|| command_output("rustc", &["--version"])).unwrap_or_else(|| "not recorded".into());
    let flags = env("QUINTRIT_BENCH_RUSTFLAGS").or_else(|| env("RUSTFLAGS")).unwrap_or_else(|| "(none)".into());
    let parity = env("QUINTRIT_BENCH_PARITY").unwrap_or_else(|| "not recorded".into());
    let device = env("QUINTRIT_BENCH_DEVICE").unwrap_or_else(|| std::env::consts::ARCH.to_string());
    let mut s = String::from("| Item | Value |
|---|---|
");
    s.push_str(&format!("| Commit | `{commit}` |
| Compiler | {rustc} |
| RUSTFLAGS | `{flags}` |
"));
    s.push_str(&format!("| Kernel / ISA path executed | {kernel_path} |
| CPU features detected | {} |
", detected_isa_features()));
    s.push_str(&format!("| Threads (rayon pool, unpinned) | {threads} |
| Repetitions | {tokens} greedy tokens, medians |
"));
    s.push_str(&format!("| Device | {device} |
| Date | {} |
", utc_timestamp()));
    s.push_str(&format!("| Parity with the reference runtime at this commit on this device | {parity} |\n"));
    s
}


/// Finds the GGUF model of a benchmark and ends the program with an error message when there is none.
///
/// Order: the path given with `--model` (`explicit`), then the environment variable `QUINTRIT_MODEL_PATH`, then
/// `models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf` in the crate directory or in any directory above it (the
/// recommended folder is `models/` at the root of the repository). A path that was given explicitly but does not
/// exist is an error; there is no silent fallback to another model.
pub fn resolve_model(explicit: Option<String>) -> std::path::PathBuf {
    use std::path::{Path, PathBuf};
    fn fail(problem: &str) -> ! {
        eprintln!("error: {problem}");
        eprintln!();
        eprintln!("This benchmark needs a GGUF model and does not run without one. Pass the model with --model <PATH>,");
        eprintln!("or set QUINTRIT_MODEL_PATH, or place it at models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf");
        eprintln!("in the repository root. The download command is in docs/benchmarks.md.");
        std::process::exit(2);
    }
    if let Some(p) = explicit {
        let p = PathBuf::from(p);
        return if p.exists() { p } else { fail(&format!("the model file '{}' does not exist", p.display())) };
    }
    if let Some(p) = std::env::var("QUINTRIT_MODEL_PATH").ok().filter(|v| !v.trim().is_empty()) {
        let p = PathBuf::from(p.trim());
        return if p.exists() { p } else { fail(&format!("QUINTRIT_MODEL_PATH points to '{}', which does not exist", p.display())) };
    }
    let mut dir = Some(Path::new(env!("CARGO_MANIFEST_DIR")));
    while let Some(d) = dir {
        let p = d.join("models/Falcon3-10B-Instruct-1.58bit/ggml-model-i2_s.gguf");
        if p.exists() {
            return p;
        }
        dir = d.parent();
    }
    fail("no GGUF model was found")
}

//! # Memory Bandwidth Ceiling Benchmark
//!
//! Measures how fast this machine can *read* memory, which is the ceiling a GEMV over weights streamed from DRAM can
//! reach. Every kernel benchmark in this repository is reported as a fraction of this ceiling, next to its absolute
//! speed, because a kernel that is limited by something other than memory bandwidth cannot gain from a smaller
//! weight encoding.
//!
//! ## What is measured
//!
//! - **Read sweep**: a read-only sum over a buffer of at least four times the last-level cache (default 1 GiB),
//!   partitioned into one contiguous chunk per thread, with eight independent vector accumulators per thread.
//!   Threads are pinned, one per physical core first and SMT siblings only after all physical cores are in use.
//!   The best, median and worst of the repetitions are reported.
//! - **Variants** at selected thread counts, interleaved round-robin so that drift affects all of them equally:
//!   unpinned threads, software prefetch, narrower loads, and a STREAM-style copy (read plus write; the write
//!   allocate traffic is not counted, so the copy rate is context only and not a GEMV ceiling).
//! - **Page size** (Linux only, `--pages`): the same read with transparent huge pages forbidden and requested.
//!
//! ## Usage
//!
//! ```bash
//! cargo bench -p quintrit-core --bench memory_bandwidth -- --suffix i7-7700 --label "Intel Core i7-7700"
//! # optional: --mib 2048  --reps 9  --max-threads 8  --pages  --peak-gbs 38.4  --dram "2 x DDR4-2400"  --out results
//! ```
//!
//! The report is written to `<out>/memory_bandwidth_<suffix>.md` (default `results/` in the crate directory).

mod common;

use common::*;
use std::fmt::Write as _;
use std::sync::Arc;
use std::sync::atomic::{AtomicU64, Ordering};

#[cfg(target_arch = "x86_64")]
use core::arch::x86_64::*;
#[cfg(target_arch = "aarch64")]
use core::arch::aarch64::*;

static SINK: AtomicU64 = AtomicU64::new(0);

type ReadFn = unsafe fn(*const u8, usize) -> u64;
type CopyFn = unsafe fn(*const u8, *mut u8, usize);

// ============================================================================
// Read kernels
// ============================================================================

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[allow(dead_code)]
enum Isa {
    Avx512,
    Avx2,
    Neon,
    Scalar,
}

impl Isa {
    fn name(self) -> &'static str {
        match self {
            Isa::Avx512 => "AVX-512 (64-byte loads)",
            Isa::Avx2 => "AVX2 (32-byte loads)",
            Isa::Neon => "NEON (16-byte loads)",
            Isa::Scalar => "scalar (8-byte loads)",
        }
    }

    fn available() -> Vec<Isa> {
        let mut v = Vec::new();
        #[cfg(target_arch = "x86_64")]
        {
            if is_x86_feature_detected!("avx512f") {
                v.push(Isa::Avx512);
            }
            if is_x86_feature_detected!("avx2") {
                v.push(Isa::Avx2);
            }
        }
        #[cfg(target_arch = "aarch64")]
        {
            v.push(Isa::Neon);
        }
        v.push(Isa::Scalar);
        v
    }
}

#[inline(always)]
unsafe fn tail_sum(p: *const u8, from: usize, len: usize) -> u64 {
    let mut s = 0u64;
    let mut i = from;
    while i + 8 <= len {
        s = s.wrapping_add(unsafe { (p.add(i) as *const u64).read_unaligned() });
        i += 8;
    }
    s
}

#[cfg(target_arch = "x86_64")]
#[inline(always)]
unsafe fn prefetch<const DIST: usize>(p: *const u8) {
    if DIST != 0 {
        unsafe { _mm_prefetch(p.add(DIST) as *const i8, _MM_HINT_T0) };
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx512f")]
unsafe fn read_avx512<const PF: usize>(p: *const u8, len: usize) -> u64 {
    unsafe {
        let mut a = [_mm512_setzero_si512(); 8];
        let mut i = 0usize;
        while i + 512 <= len {
            if PF != 0 {
                for k in 0..8 {
                    prefetch::<PF>(p.add(i + 64 * k));
                }
            }
            for k in 0..8 {
                a[k] = _mm512_add_epi64(a[k], _mm512_load_si512(p.add(i + 64 * k) as *const _));
            }
            i += 512;
        }
        let mut t = a[0];
        for k in 1..8 {
            t = _mm512_add_epi64(t, a[k]);
        }
        _mm512_reduce_add_epi64(t) as u64 ^ tail_sum(p, i, len)
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn read_avx2<const PF: usize>(p: *const u8, len: usize) -> u64 {
    unsafe {
        let mut a = [_mm256_setzero_si256(); 8];
        let mut i = 0usize;
        while i + 256 <= len {
            if PF != 0 {
                for k in 0..4 {
                    prefetch::<PF>(p.add(i + 64 * k));
                }
            }
            for k in 0..8 {
                a[k] = _mm256_add_epi64(a[k], _mm256_load_si256(p.add(i + 32 * k) as *const __m256i));
            }
            i += 256;
        }
        let mut t = a[0];
        for k in 1..8 {
            t = _mm256_add_epi64(t, a[k]);
        }
        let mut lanes = [0u64; 4];
        _mm256_storeu_si256(lanes.as_mut_ptr() as *mut __m256i, t);
        lanes.iter().fold(0u64, |s, v| s.wrapping_add(*v)) ^ tail_sum(p, i, len)
    }
}

#[cfg(target_arch = "aarch64")]
#[inline(always)]
unsafe fn prefetch_a64<const DIST: usize>(p: *const u8) {
    if DIST != 0 {
        unsafe {
            core::arch::asm!("prfm pldl1keep, [{0}]", in(reg) p.add(DIST), options(nostack, readonly, preserves_flags));
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn read_neon<const PF: usize>(p: *const u8, len: usize) -> u64 {
    unsafe {
        let mut a = [vdupq_n_u64(0); 8];
        let mut i = 0usize;
        while i + 128 <= len {
            if PF != 0 {
                for k in 0..2 {
                    prefetch_a64::<PF>(p.add(i + 64 * k));
                }
            }
            for k in 0..8 {
                a[k] = vaddq_u64(a[k], vld1q_u64(p.add(i + 16 * k) as *const u64));
            }
            i += 128;
        }
        let mut t = a[0];
        for k in 1..8 {
            t = vaddq_u64(t, a[k]);
        }
        (vgetq_lane_u64(t, 0).wrapping_add(vgetq_lane_u64(t, 1))) ^ tail_sum(p, i, len)
    }
}

unsafe fn read_scalar<const PF: usize>(p: *const u8, len: usize) -> u64 {
    let mut s = [0u64; 8];
    let mut i = 0usize;
    unsafe {
        while i + 64 <= len {
            for k in 0..8 {
                s[k] = s[k].wrapping_add((p.add(i + 8 * k) as *const u64).read_unaligned());
            }
            i += 64;
        }
        s.iter().fold(0u64, |a, v| a.wrapping_add(*v)) ^ tail_sum(p, i, len)
    }
}

fn read_kernel(isa: Isa, prefetch_dist: usize) -> Option<ReadFn> {
    match (isa, prefetch_dist) {
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx512, 0) => Some(read_avx512::<0>),
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx512, 1024) => Some(read_avx512::<1024>),
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx512, 4096) => Some(read_avx512::<4096>),
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx2, 0) => Some(read_avx2::<0>),
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx2, 1024) => Some(read_avx2::<1024>),
        #[cfg(target_arch = "x86_64")]
        (Isa::Avx2, 4096) => Some(read_avx2::<4096>),
        #[cfg(target_arch = "aarch64")]
        (Isa::Neon, 0) => Some(read_neon::<0>),
        #[cfg(target_arch = "aarch64")]
        (Isa::Neon, 1024) => Some(read_neon::<1024>),
        #[cfg(target_arch = "aarch64")]
        (Isa::Neon, 4096) => Some(read_neon::<4096>),
        (Isa::Scalar, 0) => Some(read_scalar::<0>),
        _ => None,
    }
}

#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn copy_avx2(src: *const u8, dst: *mut u8, len: usize) {
    unsafe {
        let mut i = 0usize;
        while i + 256 <= len {
            for k in 0..8 {
                let v = _mm256_load_si256(src.add(i + 32 * k) as *const __m256i);
                _mm256_store_si256(dst.add(i + 32 * k) as *mut __m256i, v);
            }
            i += 256;
        }
    }
}

#[cfg(target_arch = "aarch64")]
unsafe fn copy_neon(src: *const u8, dst: *mut u8, len: usize) {
    unsafe {
        let mut i = 0usize;
        while i + 128 <= len {
            for k in 0..8 {
                vst1q_u64(dst.add(i + 16 * k) as *mut u64, vld1q_u64(src.add(i + 16 * k) as *const u64));
            }
            i += 128;
        }
    }
}

unsafe fn copy_scalar(src: *const u8, dst: *mut u8, len: usize) {
    unsafe { core::ptr::copy_nonoverlapping(src, dst, len) }
}

fn copy_kernel() -> CopyFn {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            return copy_avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        return copy_neon;
    }
    #[allow(unreachable_code)]
    copy_scalar
}

// ============================================================================
// Measurement
// ============================================================================

fn gbps(bytes: usize, seconds: f64) -> f64 {
    bytes as f64 / seconds / 1e9
}

fn read_job(addr: usize, len: usize, threads: usize, kernel: ReadFn) -> Arc<dyn Fn(usize) + Send + Sync> {
    Arc::new(move |i| {
        let (s, e) = chunk_range(len, threads, i);
        let v = unsafe { kernel((addr + s) as *const u8, e - s) };
        SINK.fetch_xor(v, Ordering::Relaxed);
    })
}

fn copy_job(src: usize, dst: usize, len: usize, threads: usize, kernel: CopyFn) -> Arc<dyn Fn(usize) + Send + Sync> {
    Arc::new(move |i| {
        let (s, e) = chunk_range(len, threads, i);
        unsafe { kernel((src + s) as *const u8, (dst + s) as *mut u8, e - s) };
    })
}

struct SweepRow {
    threads: usize,
    cpus: String,
    stats: RateStats,
}

fn cpu_list(order: &[usize], threads: usize) -> String {
    order.iter().take(threads).map(|c| c.to_string()).collect::<Vec<_>>().join(",")
}

fn sweep(order: &[usize], addr: usize, len: usize, thread_list: &[usize], reps: usize, kernel: ReadFn) -> Vec<SweepRow> {
    let mut rows = Vec::new();
    for &t in thread_list {
        let pool = Pool::pinned(order, t);
        let job = read_job(addr, len, t, kernel);
        pool.run(job.clone()); // warm-up
        let samples: Vec<f64> = (0..reps).map(|_| gbps(len, pool.run(job.clone()).wall)).collect();
        rows.push(SweepRow { threads: t, cpus: cpu_list(order, t), stats: RateStats::from_samples(&samples) });
    }
    rows
}

fn write_sweep_table(out: &mut String, rows: &[SweepRow], cores: usize, peak: Option<f64>) {
    let _ = writeln!(out, "| Threads | Logical CPUs | Best GB/s | Median GB/s | Worst GB/s | Spread | Best per thread GB/s | Note |");
    let _ = writeln!(out, "|---:|---|---:|---:|---:|---:|---:|---|");
    for r in rows {
        let note = if r.threads > cores { "SMT siblings in use" } else { "" };
        let frac = peak.map(|p| format!(" ({:.0}% of theoretical)", r.stats.best / p * 100.0)).unwrap_or_default();
        let _ = writeln!(
            out,
            "| {} | {} | {:.2}{} | {:.2} | {:.2} | {:.1}% | {:.2} | {} |",
            r.threads, r.cpus, r.stats.best, frac, r.stats.median, r.stats.worst, r.stats.spread_pct(),
            r.stats.best / r.threads as f64, note
        );
    }
}

fn main() {
    let args = Args::from_env();
    let suffix = args.value("--suffix").or_else(|| std::env::var("QUINTRIT_BENCH_SUFFIX").ok()).unwrap_or_else(|| "local".into());
    let out_dir = args.value("--out").unwrap_or_else(|| "results".into());
    let reps = args.usize("--reps").unwrap_or(9).max(3);
    let peak = args.f64("--peak-gbs");
    let topo = Topology::detect();
    let order = topo.sweep_order();
    // `--threads 1,4` measures exactly those thread counts; without it every count up to `--max-threads` is swept.
    let explicit_threads: Option<Vec<usize>> = args.value("--threads").map(|s| {
        let mut v: Vec<usize> = s.split(',').filter_map(|t| t.trim().parse().ok()).filter(|&t| t >= 1 && t <= topo.logical).collect();
        v.sort();
        v.dedup();
        v
    });
    let max_threads = match &explicit_threads {
        Some(v) if !v.is_empty() => *v.last().unwrap(),
        _ => args.usize("--max-threads").unwrap_or(topo.logical).min(topo.logical).max(1),
    };

    let llc = topo.llc_bytes;
    let mut mib = args.usize("--mib").unwrap_or(1024);
    if let Some(l) = llc {
        mib = mib.max((4 * l + (1 << 20) - 1) >> 20);
    }
    if let Some(ram) = total_memory_bytes() {
        mib = mib.min(ram / (3 << 20)).max(256);
    }
    let len = mib << 20;

    let isas = Isa::available();
    let isa = isas[0];
    let read_main = read_kernel(isa, 0).expect("read kernel");

    println!("CPU: {}", args.value("--label").unwrap_or_else(cpu_name));
    println!("logical CPUs: {}, physical cores: {}, SMT ways: {}", topo.logical, topo.physical_cores(), topo.smt_ways());
    println!("last-level cache (sum over domains): {}", llc.map(|l| format!("{:.1} MiB", l as f64 / 1048576.0)).unwrap_or("unknown".into()));
    println!("buffer: {mib} MiB, read kernel: {}", isa.name());

    // Buffer, first touched by the pinned threads that will read their chunks later.
    let buf = AlignedBuf::new(len, PageAdvice::Default);
    let len = buf.len();
    {
        let pool = Pool::pinned(&order, max_threads);
        let addr = buf.addr();
        pool.run(Arc::new(move |i| {
            let (s, e) = chunk_range(len, max_threads, i);
            unsafe { fill_random(addr + s, e - s, 0x9E37_79B9_7F4A_7C15 ^ (i as u64 + 1)) };
        }));
    }

    // Core clock estimate (single thread, other cores idle).
    let ghz = {
        let pool = Pool::pinned(&order, 1);
        let cell = Arc::new(AtomicU64::new(0));
        let c2 = cell.clone();
        pool.run(Arc::new(move |_| c2.store(estimate_core_ghz().to_bits(), Ordering::Relaxed)));
        f64::from_bits(cell.load(Ordering::Relaxed))
    };
    println!("estimated single-core clock: {ghz:.2} GHz");

    // ---- Thread sweep ------------------------------------------------------
    let thread_list: Vec<usize> = match &explicit_threads {
        Some(v) if !v.is_empty() => v.clone(),
        _ => (1..=max_threads).collect(),
    };
    println!("\nread sweep ({} threads max) ...", max_threads);
    let rows = sweep(&order, buf.addr(), len, &thread_list, reps, read_main);
    for r in &rows {
        println!("  T={:<3} best {:7.2} GB/s  median {:7.2}  worst {:7.2}  ({})", r.threads, r.stats.best, r.stats.median, r.stats.worst, r.cpus);
    }
    let ceiling = rows.iter().map(|r| r.stats.best).fold(0.0, f64::max);
    let ceiling_t = rows.iter().find(|r| r.stats.best == ceiling).map(|r| r.threads).unwrap_or(1);

    // ---- Variants at fixed thread counts, interleaved ----------------------
    let mut variant_thread_counts = match &explicit_threads {
        Some(v) if !v.is_empty() => v.clone(),
        _ => vec![1usize, 2, topo.physical_cores(), max_threads],
    };
    variant_thread_counts.retain(|&t| t >= 1 && t <= max_threads);
    variant_thread_counts.sort();
    variant_thread_counts.dedup();

    let mut variant_report = String::new();
    let half = len / 2;
    let (src_addr, dst_addr) = (buf.addr(), buf.addr() + half);
    for &t in &variant_thread_counts {
        let pinned = Pool::pinned(&order, t);
        let unpinned = Pool::unpinned(t);
        struct Variant {
            name: String,
            pinned: bool,
            bytes: usize,
            job: Arc<dyn Fn(usize) + Send + Sync>,
            samples: Vec<f64>,
        }
        let mut variants: Vec<Variant> = Vec::new();
        let mut add = |name: &str, pinned: bool, bytes: usize, job: Arc<dyn Fn(usize) + Send + Sync>| {
            variants.push(Variant { name: name.to_string(), pinned, bytes, job, samples: Vec::new() });
        };
        add(&format!("read, {}, pinned", isa.name()), true, len, read_job(buf.addr(), len, t, read_main));
        add("read, same kernel, unpinned threads", false, len, read_job(buf.addr(), len, t, read_main));
        if let Some(k) = read_kernel(isa, 1024) {
            add("read, pinned, software prefetch 1 KiB ahead", true, len, read_job(buf.addr(), len, t, k));
        }
        if let Some(k) = read_kernel(isa, 4096) {
            add("read, pinned, software prefetch 4 KiB ahead", true, len, read_job(buf.addr(), len, t, k));
        }
        for &other in isas.iter().skip(1) {
            if other == Isa::Scalar && isas.len() > 2 {
                continue;
            }
            if let Some(k) = read_kernel(other, 0) {
                add(&format!("read, pinned, {}", other.name()), true, len, read_job(buf.addr(), len, t, k));
            }
        }
        add("copy, pinned (read + write bytes counted)", true, 2 * (half & !4095), copy_job(src_addr, dst_addr, half & !4095, t, copy_kernel()));

        for v in &variants {
            let pool = if v.pinned { &pinned } else { &unpinned };
            pool.run(v.job.clone());
        }
        for round in 0..reps {
            for step in 0..variants.len() {
                let idx = (step + round) % variants.len();
                let v = &mut variants[idx];
                let pool = if v.pinned { &pinned } else { &unpinned };
                let wall = pool.run(v.job.clone()).wall;
                v.samples.push(gbps(v.bytes, wall));
            }
        }
        println!("\nvariants at {t} thread(s):");
        let _ = writeln!(variant_report, "\n#### {t} thread(s)\n");
        let _ = writeln!(variant_report, "| Variant | Best GB/s | Median GB/s | Worst GB/s | Spread |");
        let _ = writeln!(variant_report, "|---|---:|---:|---:|---:|");
        for v in &variants {
            let s = RateStats::from_samples(&v.samples);
            println!("  {:<58} best {:7.2}  median {:7.2}  worst {:7.2}", v.name, s.best, s.median, s.worst);
            let _ = writeln!(variant_report, "| {} | {:.2} | {:.2} | {:.2} | {:.1}% |", v.name, s.best, s.median, s.worst, s.spread_pct());
        }
    }

    // ---- Page size (Linux transparent huge pages) --------------------------
    let mut pages_report = String::new();
    if args.flag("--pages") {
        if cfg!(target_os = "linux") {
            let mut counts = vec![1usize, topo.physical_cores(), max_threads];
            counts.sort();
            counts.dedup();
            let _ = writeln!(pages_report, "| Page advice | Threads | Best GB/s | Median GB/s | Worst GB/s |");
            let _ = writeln!(pages_report, "|---|---:|---:|---:|---:|");
            for advice in [PageAdvice::NoHuge, PageAdvice::Huge] {
                let b = AlignedBuf::new(len, advice);
                let a = b.addr();
                let bl = b.len();
                let pool = Pool::pinned(&order, max_threads);
                pool.run(Arc::new(move |i| {
                    let (s, e) = chunk_range(bl, max_threads, i);
                    unsafe { fill_random(a + s, e - s, 7 + i as u64) };
                }));
                drop(pool);
                for r in sweep(&order, a, bl, &counts, reps, read_main) {
                    println!("pages {:?} T={} best {:.2} GB/s", advice, r.threads, r.stats.best);
                    let _ = writeln!(pages_report, "| {:?} | {} | {:.2} | {:.2} | {:.2} |", advice, r.threads, r.stats.best, r.stats.median, r.stats.worst);
                }
            }
        } else {
            pages_report.push_str("Page-size variants are only implemented for Linux transparent huge pages.\n");
        }
    }

    // ---- Report -------------------------------------------------------------
    let mut md = String::new();
    let label = args.value("--label").unwrap_or_else(cpu_name);
    let _ = writeln!(md, "# Memory Bandwidth Ceiling: {suffix}\n");
    let _ = writeln!(md, "Read-only streaming bandwidth of the machine: the ceiling a GEMV over DRAM-resident weights can reach.");
    let _ = writeln!(md, "Generated by `cargo bench -p quintrit-core --bench memory_bandwidth`. Rates are decimal GB/s (10^9 bytes/s).\n");
    let _ = writeln!(md, "## Run\n");
    let thread_text = thread_list.iter().map(|t| t.to_string()).collect::<Vec<_>>().join(", ");
    md.push_str(&run_header(&format!("read-only stream, {}", isa.name()), &thread_text, &reps.to_string()));
    let _ = writeln!(md, "\n## Machine\n");
    let _ = writeln!(md, "| Item | Value |\n|---|---|");
    let _ = writeln!(md, "| CPU | {label} |");
    let _ = writeln!(md, "| Architecture / OS | {} / {} |", std::env::consts::ARCH, std::env::consts::OS);
    let _ = writeln!(md, "| Logical CPUs | {} |", topo.logical);
    let _ = writeln!(md, "| Physical cores | {} ({} thread(s) per core) |", topo.physical_cores(), topo.smt_ways());
    let _ = writeln!(md, "| Last-level cache (sum over sharing domains) | {} |", llc.map(|l| format!("{:.1} MiB", l as f64 / 1048576.0)).unwrap_or("unknown".into()));
    let _ = writeln!(md, "| Total memory | {} |", total_memory_bytes().map(|m| format!("{:.1} GiB", m as f64 / 1073741824.0)).unwrap_or("not read".into()));
    let _ = writeln!(md, "| DRAM (as given by the operator) | {} |", args.value("--dram").unwrap_or_else(|| "not given".into()));
    let _ = writeln!(md, "| Theoretical DRAM peak | {} |", peak.map(|p| format!("{p:.1} GB/s")).unwrap_or_else(|| "not given".into()));
    let _ = writeln!(md, "| Estimated single-core clock | {ghz:.2} GHz |");
    let _ = writeln!(md, "| Buffer | {mib} MiB ({} x last-level cache) |", llc.map(|l| format!("{:.1}", len as f64 / l as f64)).unwrap_or("?".into()));
    let _ = writeln!(md, "| Read kernel | {} |", isa.name());
    let _ = writeln!(md, "| Repetitions per point | {reps} (one warm-up run discarded) |\n");
    let _ = writeln!(md, "## Result\n");
    let _ = writeln!(md, "**Ceiling: {ceiling:.2} GB/s** (best of {reps}, {ceiling_t} thread(s)).{}\n", peak.map(|p| format!(" That is {:.0}% of the theoretical peak of {p:.1} GB/s.", ceiling / p * 100.0)).unwrap_or_default());
    let _ = writeln!(md, "### Thread sweep (pinned, one thread per physical core first)\n");
    write_sweep_table(&mut md, &rows, topo.physical_cores(), peak);
    let _ = writeln!(md, "\n### Variants (interleaved round-robin)\n{variant_report}");
    if !pages_report.is_empty() {
        let _ = writeln!(md, "\n### Page size\n\n{pages_report}");
    }
    let _ = writeln!(md, "\n## Method\n");
    let _ = writeln!(md, "- Each thread sums its own contiguous chunk of the buffer with eight independent accumulators; the buffer is filled with pseudo-random data by the threads that later read it (first touch).");
    let _ = writeln!(md, "- All threads start on a barrier; the rate is total bytes divided by the wall time until the last thread finishes.");
    let _ = writeln!(md, "- Threads are pinned to one logical CPU each, physical cores first. Unpinned runs show the effect of OS scheduling.");
    let _ = writeln!(md, "- The copy rate counts one read and one write per byte; the write-allocate read of the destination is not counted, so a copy is not a GEMV ceiling.");
    let _ = writeln!(md, "- Best-of-N is the ceiling estimate; the spread shows how noisy the machine was during the run.");

    let dir = std::path::Path::new(&out_dir);
    let _ = std::fs::create_dir_all(dir);
    let path = dir.join(format!("memory_bandwidth_{suffix}.md"));
    std::fs::write(&path, &md).expect("write report");
    println!("\nceiling: {ceiling:.2} GB/s at {ceiling_t} thread(s); report written to {}", path.display());
    std::hint::black_box(SINK.load(Ordering::Relaxed));
}

//! Shared helpers of the memory-bandwidth and GEMV-ladder benchmarks.
//!
//! Provides CPU topology discovery (physical cores versus SMT siblings, last-level cache size), thread pinning
//! without external crates, a persistent pool of pinned worker threads that start and stop on barriers, 2 MiB
//! aligned buffers with an optional page-size advice, a core-clock estimate and small statistics helpers.
//!
//! Everything here is benchmark infrastructure: it is deliberately independent of the kernels under test.

#![allow(dead_code)]

use std::alloc::{Layout, alloc, dealloc};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, Barrier, Mutex};
use std::thread::JoinHandle;
use std::time::Instant;

// ============================================================================
// Operating-system interface (pinning, topology, page advice)
// ============================================================================

/// Page-size advice given to the operating system for a freshly allocated buffer.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PageAdvice {
    /// Leave the system default.
    Default,
    /// Ask for transparent huge pages (Linux `MADV_HUGEPAGE`).
    Huge,
    /// Forbid transparent huge pages (Linux `MADV_NOHUGEPAGE`).
    NoHuge,
}

#[cfg(windows)]
mod os {
    use super::PageAdvice;

    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
        fn GetLogicalProcessorInformation(buffer: *mut u8, length: *mut u32) -> i32;
    }

    /// Pins the calling thread to one logical CPU (CPUs 0..63 of processor group 0 only).
    pub fn pin_current_thread(cpu: usize) -> bool {
        if cpu >= 64 {
            return false;
        }
        unsafe { SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu) != 0 }
    }

    pub fn advise(_ptr: *mut u8, _len: usize, _advice: PageAdvice) {}

    /// Returns `(cores, last_level_cache_bytes)`; each core lists the logical CPUs that share it.
    pub fn topology() -> (Vec<Vec<usize>>, Option<usize>) {
        let mut len: u32 = 0;
        unsafe {
            GetLogicalProcessorInformation(core::ptr::null_mut(), &mut len);
        }
        if len == 0 {
            return (Vec::new(), None);
        }
        let mut buf = vec![0u8; len as usize];
        let ok = unsafe { GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut len) };
        if ok == 0 {
            return (Vec::new(), None);
        }
        // SYSTEM_LOGICAL_PROCESSOR_INFORMATION on 64-bit Windows: mask (8), relationship (4 + 4 padding), union (16).
        const ENTRY: usize = 32;
        let mut cores = Vec::new();
        let mut l3 = 0usize;
        let mut best_level = 0u8;
        let mut best_total = 0usize;
        let mut off = 0usize;
        while off + ENTRY <= len as usize {
            let mask = usize::from_ne_bytes(buf[off..off + 8].try_into().unwrap());
            let rel = u32::from_ne_bytes(buf[off + 8..off + 12].try_into().unwrap());
            match rel {
                0 => {
                    let cpus: Vec<usize> = (0..64).filter(|b| mask >> b & 1 == 1).collect();
                    if !cpus.is_empty() {
                        cores.push(cpus);
                    }
                }
                2 => {
                    let level = buf[off + 16];
                    let size = u32::from_ne_bytes(buf[off + 20..off + 24].try_into().unwrap()) as usize;
                    let kind = u32::from_ne_bytes(buf[off + 24..off + 28].try_into().unwrap());
                    if kind != 1 {
                        // Not an instruction-only cache.
                        if level > best_level {
                            best_level = level;
                            best_total = 0;
                        }
                        if level == best_level {
                            best_total += size;
                        }
                        if level == 3 {
                            l3 += size;
                        }
                    }
                }
                _ => {}
            }
            off += ENTRY;
        }
        let _ = l3;
        (cores, if best_total > 0 { Some(best_total) } else { None })
    }
}

#[cfg(target_os = "linux")]
mod os {
    use super::PageAdvice;
    use std::collections::BTreeSet;

    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
        fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
        fn madvise(addr: *mut u8, len: usize, advice: i32) -> i32;
    }

    pub fn pin_current_thread(cpu: usize) -> bool {
        let mut mask = [0u64; 16]; // 1024 CPUs
        if cpu >= 1024 {
            return false;
        }
        mask[cpu / 64] = 1u64 << (cpu % 64);
        unsafe { sched_setaffinity(0, core::mem::size_of_val(&mask), mask.as_ptr()) == 0 }
    }

    pub fn advise(ptr: *mut u8, len: usize, advice: PageAdvice) {
        let code = match advice {
            PageAdvice::Default => return,
            PageAdvice::Huge => 14,   // MADV_HUGEPAGE
            PageAdvice::NoHuge => 15, // MADV_NOHUGEPAGE
        };
        unsafe {
            madvise(ptr, len, code);
        }
    }

    fn allowed_cpus() -> Vec<usize> {
        let mut mask = [0u64; 16];
        let rc = unsafe { sched_getaffinity(0, core::mem::size_of_val(&mask), mask.as_mut_ptr()) };
        if rc != 0 {
            return Vec::new();
        }
        (0..1024).filter(|c| mask[c / 64] >> (c % 64) & 1 == 1).collect()
    }

    fn parse_cpu_list(s: &str) -> Vec<usize> {
        let mut out = Vec::new();
        for part in s.trim().split(',') {
            if let Some((a, b)) = part.split_once('-') {
                if let (Ok(a), Ok(b)) = (a.parse::<usize>(), b.parse::<usize>()) {
                    out.extend(a..=b);
                }
            } else if let Ok(v) = part.parse::<usize>() {
                out.push(v);
            }
        }
        out
    }

    fn parse_size(s: &str) -> usize {
        let s = s.trim();
        let (num, mul) = if let Some(n) = s.strip_suffix('K') {
            (n, 1024)
        } else if let Some(n) = s.strip_suffix('M') {
            (n, 1024 * 1024)
        } else {
            (s, 1)
        };
        num.parse::<usize>().unwrap_or(0) * mul
    }

    pub fn topology() -> (Vec<Vec<usize>>, Option<usize>) {
        let allowed = allowed_cpus();
        let mut seen: BTreeSet<Vec<usize>> = BTreeSet::new();
        for &cpu in &allowed {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
            let mut sib = std::fs::read_to_string(&path)
                .map(|s| parse_cpu_list(&s))
                .unwrap_or_else(|_| vec![cpu]);
            sib.retain(|c| allowed.contains(c));
            if sib.is_empty() {
                sib = vec![cpu];
            }
            seen.insert(sib);
        }
        let mut cores: Vec<Vec<usize>> = seen.into_iter().collect();
        cores.sort_by_key(|c| c[0]);

        // Last-level cache: highest cache index, summed over distinct sharing domains.
        let mut llc = 0usize;
        let mut domains: BTreeSet<(usize, Vec<usize>)> = BTreeSet::new();
        let mut best_level = 0usize;
        let mut entries: Vec<(usize, Vec<usize>, usize)> = Vec::new();
        for &cpu in &allowed {
            for idx in 0..8 {
                let base = format!("/sys/devices/system/cpu/cpu{cpu}/cache/index{idx}");
                let Ok(level) = std::fs::read_to_string(format!("{base}/level")) else { break };
                let level: usize = level.trim().parse().unwrap_or(0);
                let kind = std::fs::read_to_string(format!("{base}/type")).unwrap_or_default();
                if kind.trim() == "Instruction" {
                    continue;
                }
                let size = parse_size(&std::fs::read_to_string(format!("{base}/size")).unwrap_or_default());
                let shared = std::fs::read_to_string(format!("{base}/shared_cpu_list"))
                    .map(|s| parse_cpu_list(&s))
                    .unwrap_or_default();
                entries.push((level, shared, size));
                best_level = best_level.max(level);
            }
        }
        for (level, shared, size) in entries {
            if level == best_level && domains.insert((level, shared)) {
                llc += size;
            }
        }
        (cores, if llc > 0 { Some(llc) } else { None })
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod os {
    use super::PageAdvice;
    pub fn pin_current_thread(_cpu: usize) -> bool {
        false
    }
    pub fn advise(_ptr: *mut u8, _len: usize, _advice: PageAdvice) {}
    pub fn topology() -> (Vec<Vec<usize>>, Option<usize>) {
        (Vec::new(), None)
    }
}

/// Pins the calling thread to one logical CPU. Returns `false` when pinning is unsupported or failed.
pub fn pin_current_thread(cpu: usize) -> bool {
    os::pin_current_thread(cpu)
}

/// CPU topology as seen by the process.
#[derive(Clone, Debug)]
pub struct Topology {
    /// Number of logical CPUs available to the process.
    pub logical: usize,
    /// Logical CPUs grouped by physical core (SMT siblings share a group).
    pub cores: Vec<Vec<usize>>,
    /// Total size of the last cache level across all sharing domains, if known.
    pub llc_bytes: Option<usize>,
}

impl Topology {
    pub fn detect() -> Self {
        let (mut cores, llc_bytes) = os::topology();
        let logical = std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1);
        if cores.is_empty() {
            cores = (0..logical).map(|c| vec![c]).collect();
        }
        let logical = cores.iter().map(|c| c.len()).sum::<usize>().max(1);
        Self { logical, cores, llc_bytes }
    }

    pub fn physical_cores(&self) -> usize {
        self.cores.len()
    }

    pub fn smt_ways(&self) -> usize {
        self.cores.iter().map(|c| c.len()).max().unwrap_or(1)
    }

    /// Logical CPUs in sweep order: the first thread of every physical core, then the SMT siblings.
    pub fn sweep_order(&self) -> Vec<usize> {
        let mut order = Vec::new();
        for way in 0..self.smt_ways() {
            for core in &self.cores {
                if let Some(&cpu) = core.get(way) {
                    order.push(cpu);
                }
            }
        }
        order
    }
}

// ============================================================================
// Buffers
// ============================================================================

const BUF_ALIGN: usize = 2 << 20;

/// Heap buffer aligned to 2 MiB (so that transparent huge pages can back it).
pub struct AlignedBuf {
    ptr: *mut u8,
    len: usize,
    layout: Layout,
}

unsafe impl Send for AlignedBuf {}
unsafe impl Sync for AlignedBuf {}

impl AlignedBuf {
    pub fn new(len: usize, advice: PageAdvice) -> Self {
        let size = (len.max(1) + BUF_ALIGN - 1) & !(BUF_ALIGN - 1);
        let layout = Layout::from_size_align(size, BUF_ALIGN).expect("layout");
        let ptr = unsafe { alloc(layout) };
        assert!(!ptr.is_null(), "allocation of {size} bytes failed");
        os::advise(ptr, size, advice);
        Self { ptr, len: size, layout }
    }

    pub fn addr(&self) -> usize {
        self.ptr as usize
    }

    pub fn len(&self) -> usize {
        self.len
    }
}

impl Drop for AlignedBuf {
    fn drop(&mut self) {
        unsafe { dealloc(self.ptr, self.layout) }
    }
}

/// Fills `[addr, addr + len)` with xorshift noise (also faults the pages in).
///
/// # Safety
/// The range must be valid, writable memory that no other thread accesses.
pub unsafe fn fill_random(addr: usize, len: usize, seed: u64) {
    let mut s = seed | 1;
    let words = len / 8;
    let p = addr as *mut u64;
    for i in 0..words {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        unsafe { p.add(i).write(s) };
    }
}

/// Splits `len` bytes into `parts` contiguous chunks (multiples of 4 KiB except the last one).
pub fn chunk_range(len: usize, parts: usize, index: usize) -> (usize, usize) {
    let per = (len / parts) & !4095;
    let start = per * index;
    let end = if index + 1 == parts { len } else { start + per };
    (start, end)
}

// ============================================================================
// Pinned worker pool
// ============================================================================

type Job = Arc<dyn Fn(usize) + Send + Sync + 'static>;

/// Persistent worker threads, each optionally pinned to one logical CPU, released together by a barrier.
pub struct Pool {
    n: usize,
    job: Arc<Mutex<Option<Job>>>,
    start: Arc<Barrier>,
    end: Arc<Barrier>,
    stop: Arc<std::sync::atomic::AtomicBool>,
    times_ns: Arc<Vec<AtomicU64>>,
    handles: Vec<JoinHandle<()>>,
    pub pinned: bool,
}

/// Timing of one `Pool::run`.
pub struct RunTiming {
    /// Wall-clock seconds between the common start and the last worker finishing.
    pub wall: f64,
    /// Seconds each worker spent inside the job.
    pub per_thread: Vec<f64>,
}

impl Pool {
    /// Creates one worker per entry; `Some(cpu)` pins the worker to that logical CPU.
    pub fn new(cpus: &[Option<usize>]) -> Self {
        let n = cpus.len();
        let job: Arc<Mutex<Option<Job>>> = Arc::new(Mutex::new(None));
        let start = Arc::new(Barrier::new(n + 1));
        let end = Arc::new(Barrier::new(n + 1));
        let stop = Arc::new(std::sync::atomic::AtomicBool::new(false));
        let times_ns: Arc<Vec<AtomicU64>> = Arc::new((0..n).map(|_| AtomicU64::new(0)).collect());
        let mut handles = Vec::new();
        let mut pinned = n > 0;
        for (idx, cpu) in cpus.iter().enumerate() {
            if cpu.is_none() {
                pinned = false;
            }
            let (job, start, end, stop, times) =
                (job.clone(), start.clone(), end.clone(), stop.clone(), times_ns.clone());
            let cpu = *cpu;
            handles.push(std::thread::spawn(move || {
                if let Some(c) = cpu {
                    if !pin_current_thread(c) {
                        eprintln!("warning: could not pin worker {idx} to CPU {c}");
                    }
                }
                loop {
                    start.wait();
                    if stop.load(Ordering::Acquire) {
                        break;
                    }
                    let f = job.lock().unwrap().clone().expect("job");
                    let t0 = Instant::now();
                    f(idx);
                    times[idx].store(t0.elapsed().as_nanos() as u64, Ordering::Release);
                    end.wait();
                }
            }));
        }
        Self { n, job, start, end, stop, times_ns, handles, pinned }
    }

    /// Pool of `threads` workers pinned to the first `threads` entries of `order`.
    pub fn pinned(order: &[usize], threads: usize) -> Self {
        let cpus: Vec<Option<usize>> = order.iter().take(threads).map(|&c| Some(c)).collect();
        Self::new(&cpus)
    }

    /// Pool of `threads` unpinned workers.
    pub fn unpinned(threads: usize) -> Self {
        Self::new(&vec![None; threads])
    }

    pub fn threads(&self) -> usize {
        self.n
    }

    /// Runs `f(worker_index)` on every worker simultaneously and waits for all of them.
    pub fn run(&self, f: Job) -> RunTiming {
        *self.job.lock().unwrap() = Some(f);
        // The timestamp is taken before the workers are released. When every logical CPU runs a worker, this thread
        // gets a CPU only after some workers have started; a later timestamp would shorten the measured time and
        // inflate the rate (the rates of runs with as many threads as logical CPUs could exceed the DRAM peak).
        let t0 = Instant::now();
        self.start.wait();
        self.end.wait();
        let wall = t0.elapsed().as_secs_f64();
        let per_thread = self.times_ns.iter().map(|t| t.load(Ordering::Acquire) as f64 * 1e-9).collect();
        RunTiming { wall, per_thread }
    }
}

impl Drop for Pool {
    fn drop(&mut self) {
        self.stop.store(true, Ordering::Release);
        self.start.wait();
        for h in self.handles.drain(..) {
            let _ = h.join();
        }
    }
}

// ============================================================================
// Statistics and clock
// ============================================================================

/// Summary of repeated measurements of a rate (larger is better).
#[derive(Clone, Copy, Debug)]
pub struct RateStats {
    pub best: f64,
    pub median: f64,
    pub worst: f64,
}

impl RateStats {
    pub fn from_samples(samples: &[f64]) -> Self {
        let mut v = samples.to_vec();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap());
        Self {
            best: *v.last().unwrap_or(&0.0),
            median: v[v.len() / 2],
            worst: *v.first().unwrap_or(&0.0),
        }
    }

    /// `(best - worst) / median` in percent.
    pub fn spread_pct(&self) -> f64 {
        if self.median > 0.0 { (self.best - self.worst) / self.median * 100.0 } else { 0.0 }
    }
}

/// Runs a chain of 8 dependent single-cycle additions per iteration and returns the core clock in GHz
/// as seen by the calling thread. Uses the best of a few trials; the chain is light, so on parts that lower
/// the clock under wide-vector load the value is an upper bound for the clock during the kernels.
pub fn estimate_core_ghz() -> f64 {
    let iters: u64 = 2_000_000;
    let mut best = 0.0f64;
    for _ in 0..7 {
        let t0 = Instant::now();
        let sink = dependent_add_chain(iters);
        let dt = t0.elapsed().as_secs_f64();
        std::hint::black_box(sink);
        best = best.max(iters as f64 * 8.0 / dt / 1e9);
    }
    best
}

#[cfg(target_arch = "x86_64")]
fn dependent_add_chain(iters: u64) -> u64 {
    let mut x = 0u64;
    unsafe {
        core::arch::asm!(
            "2:",
            "add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1",
            "add {x}, 1", "add {x}, 1", "add {x}, 1", "add {x}, 1",
            "dec {n}",
            "jnz 2b",
            x = inout(reg) x,
            n = inout(reg) iters => _,
            options(nostack),
        );
    }
    x
}

#[cfg(target_arch = "aarch64")]
fn dependent_add_chain(iters: u64) -> u64 {
    let mut x = 0u64;
    unsafe {
        core::arch::asm!(
            "2:",
            "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1",
            "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1", "add {x}, {x}, #1",
            "subs {n}, {n}, #1",
            "b.ne 2b",
            x = inout(reg) x,
            n = inout(reg) iters => _,
            options(nostack),
        );
    }
    x
}

#[cfg(not(any(target_arch = "x86_64", target_arch = "aarch64")))]
fn dependent_add_chain(iters: u64) -> u64 {
    let mut x = 0u64;
    for _ in 0..iters * 8 {
        x = std::hint::black_box(x + 1);
    }
    x
}

/// Name of the host CPU as far as it can be read without external tools.
pub fn cpu_name() -> String {
    #[cfg(target_os = "linux")]
    {
        if let Ok(text) = std::fs::read_to_string("/proc/cpuinfo") {
            for key in ["model name", "Model", "Hardware"] {
                if let Some(line) = text.lines().find(|l| l.starts_with(key)) {
                    if let Some((_, v)) = line.split_once(':') {
                        return v.trim().to_string();
                    }
                }
            }
        }
    }
    std::env::var("PROCESSOR_IDENTIFIER").unwrap_or_else(|_| std::env::consts::ARCH.to_string())
}

/// Total physical memory in bytes, if it can be read cheaply.
pub fn total_memory_bytes() -> Option<usize> {
    #[cfg(target_os = "linux")]
    {
        let text = std::fs::read_to_string("/proc/meminfo").ok()?;
        let line = text.lines().find(|l| l.starts_with("MemTotal"))?;
        let kb: usize = line.split_whitespace().nth(1)?.parse().ok()?;
        return Some(kb * 1024);
    }
    #[allow(unreachable_code)]
    None
}

/// Parses `--name value` style options; unknown arguments (such as cargo's `--bench`) are ignored.
pub struct Args {
    args: Vec<String>,
}

impl Args {
    pub fn from_env() -> Self {
        Self { args: std::env::args().skip(1).collect() }
    }

    pub fn value(&self, name: &str) -> Option<String> {
        let pos = self.args.iter().position(|a| a == name)?;
        self.args.get(pos + 1).cloned()
    }

    pub fn flag(&self, name: &str) -> bool {
        self.args.iter().any(|a| a == name)
    }

    pub fn usize(&self, name: &str) -> Option<usize> {
        self.value(name).and_then(|v| v.parse().ok())
    }

    pub fn f64(&self, name: &str) -> Option<f64> {
        self.value(name).and_then(|v| v.parse().ok())
    }
}

// ============================================================================
// Report header (provenance of a measurement)
// ============================================================================

/// UTC date and time (`YYYY-MM-DD HH:MM`) from the system clock, without external crates.
pub fn utc_timestamp() -> String {
    let secs = std::time::SystemTime::now().duration_since(std::time::UNIX_EPOCH).map(|d| d.as_secs()).unwrap_or(0) as i64;
    let (days, rem) = (secs.div_euclid(86_400), secs.rem_euclid(86_400));
    // Civil-from-days (Howard Hinnant's algorithm).
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

/// Instruction-set features of the running CPU that select a kernel path, as a comma-separated list.
pub fn detected_isa_features() -> String {
    let mut v: Vec<&str> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx2") {
            v.push("avx2");
        }
        if is_x86_feature_detected!("avx512f") {
            v.push("avx512f");
        }
        if is_x86_feature_detected!("avx512bw") {
            v.push("avx512bw");
        }
        if is_x86_feature_detected!("avx512vbmi") {
            v.push("avx512vbmi");
        }
        if is_x86_feature_detected!("avx512vnni") {
            v.push("avx512vnni");
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        v.push("neon");
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            v.push("dotprod");
        }
        if std::arch::is_aarch64_feature_detected!("i8mm") {
            v.push("i8mm");
        }
        if std::arch::is_aarch64_feature_detected!("sve2") {
            v.push("sve2");
        }
    }
    v.join(", ")
}

fn command_output(program: &str, args: &[&str]) -> Option<String> {
    let out = std::process::Command::new(program).args(args).output().ok()?;
    if !out.status.success() {
        return None;
    }
    let s = String::from_utf8_lossy(&out.stdout).trim().to_string();
    if s.is_empty() { None } else { Some(s) }
}

/// Markdown table with the provenance fields every report carries: commit, toolchain, flags, executed kernel
/// and ISA path, threads, repetitions, device, date and parity status.
///
/// The commit, compiler version, flags, parity status and device description are taken from the environment
/// variables `QUINTRIT_BENCH_COMMIT`, `QUINTRIT_BENCH_RUSTC`, `QUINTRIT_BENCH_RUSTFLAGS`, `QUINTRIT_BENCH_PARITY`
/// and `QUINTRIT_BENCH_DEVICE`; when a variable is unset the value is probed (`git`, `rustc`, `RUSTFLAGS`, CPU
/// name) or reported as not recorded.
pub fn run_header(kernel_path: &str, threads: &str, repetitions: &str) -> String {
    let env = |name: &str| std::env::var(name).ok().filter(|v| !v.trim().is_empty());
    let commit = env("QUINTRIT_BENCH_COMMIT").or_else(|| command_output("git", &["rev-parse", "--short", "HEAD"])).unwrap_or_else(|| "not recorded".into());
    let rustc = env("QUINTRIT_BENCH_RUSTC").or_else(|| command_output("rustc", &["--version"])).unwrap_or_else(|| "not recorded".into());
    let flags = env("QUINTRIT_BENCH_RUSTFLAGS").or_else(|| env("RUSTFLAGS")).unwrap_or_else(|| "(none)".into());
    let parity = env("QUINTRIT_BENCH_PARITY").unwrap_or_else(|| "not recorded".into());
    let device = env("QUINTRIT_BENCH_DEVICE").unwrap_or_else(cpu_name);
    let mut s = String::new();
    s.push_str("| Item | Value |\n|---|---|\n");
    s.push_str(&format!("| Commit | `{commit}` |\n"));
    s.push_str(&format!("| Compiler | {rustc} |\n"));
    s.push_str(&format!("| RUSTFLAGS | `{flags}` |\n"));
    s.push_str(&format!("| Kernel / ISA path executed | {kernel_path} |\n"));
    s.push_str(&format!("| CPU features detected | {} |\n", detected_isa_features()));
    s.push_str(&format!("| Threads | {threads} |\n"));
    s.push_str(&format!("| Repetitions | {repetitions} |\n"));
    s.push_str(&format!("| Device | {device} |\n"));
    s.push_str(&format!("| Date | {} |\n", utc_timestamp()));
    s.push_str(&format!("| Parity with the reference runtime at this commit on this device | {parity} |\n"));
    s
}

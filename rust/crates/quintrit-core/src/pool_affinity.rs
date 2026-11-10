//! Optional CPU pinning of the workers of the shared thread pool.
//!
//! Pins pool workers to distinct logical CPUs, physical cores first (SMT siblings only after every core is used).
//! Enabled by the environment variable `QUINTRIT_POOL_PIN=1` when a pool is created; without it the operating system
//! places the workers.

use std::sync::OnceLock;

/// `true` when pinning was requested through `QUINTRIT_POOL_PIN=1`.
pub fn requested() -> bool {
    static FLAG: OnceLock<bool> = OnceLock::new();
    *FLAG.get_or_init(|| std::env::var("QUINTRIT_POOL_PIN").map(|v| v.trim() == "1").unwrap_or(false))
}

/// Logical CPUs in pinning order: one CPU per physical core first, then the SMT siblings.
pub fn order() -> &'static [usize] {
    static ORDER: OnceLock<Vec<usize>> = OnceLock::new();
    ORDER.get_or_init(|| {
        let cores = os::cores();
        let mut order = Vec::new();
        let ways = cores.iter().map(|c| c.len()).max().unwrap_or(0);
        for rank in 0..ways {
            for core in &cores {
                if let Some(&cpu) = core.get(rank) {
                    order.push(cpu);
                }
            }
        }
        order
    })
}

/// Pins worker number `index` of a pool; does nothing when the CPU order is unknown.
pub fn pin_worker(index: usize) {
    let order = order();
    if !order.is_empty() {
        os::pin_current(order[index % order.len()]);
    }
}

#[cfg(target_os = "linux")]
mod os {
    use std::collections::BTreeSet;

    unsafe extern "C" {
        fn sched_setaffinity(pid: i32, size: usize, mask: *const u64) -> i32;
        fn sched_getaffinity(pid: i32, size: usize, mask: *mut u64) -> i32;
    }

    pub fn pin_current(cpu: usize) {
        if cpu >= 1024 {
            return;
        }
        let mut mask = [0u64; 16];
        mask[cpu / 64] = 1u64 << (cpu % 64);
        unsafe {
            sched_setaffinity(0, core::mem::size_of_val(&mask), mask.as_ptr());
        }
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

    /// Logical CPUs of every physical core among the CPUs this process may use.
    pub fn cores() -> Vec<Vec<usize>> {
        let mut mask = [0u64; 16];
        if unsafe { sched_getaffinity(0, core::mem::size_of_val(&mask), mask.as_mut_ptr()) } != 0 {
            return Vec::new();
        }
        let allowed: Vec<usize> = (0..1024).filter(|c| mask[c / 64] >> (c % 64) & 1 == 1).collect();
        let mut seen: BTreeSet<Vec<usize>> = BTreeSet::new();
        for &cpu in &allowed {
            let path = format!("/sys/devices/system/cpu/cpu{cpu}/topology/thread_siblings_list");
            let mut sib = std::fs::read_to_string(&path).map(|s| parse_cpu_list(&s)).unwrap_or_else(|_| vec![cpu]);
            sib.retain(|c| allowed.contains(c));
            if sib.is_empty() {
                sib = vec![cpu];
            }
            seen.insert(sib);
        }
        let mut cores: Vec<Vec<usize>> = seen.into_iter().collect();
        cores.sort_by_key(|c| c[0]);
        cores
    }
}

#[cfg(windows)]
mod os {
    #[link(name = "kernel32")]
    unsafe extern "system" {
        fn GetCurrentThread() -> isize;
        fn SetThreadAffinityMask(thread: isize, mask: usize) -> usize;
        fn GetLogicalProcessorInformation(buffer: *mut u8, length: *mut u32) -> i32;
    }

    pub fn pin_current(cpu: usize) {
        if cpu < 64 {
            unsafe {
                SetThreadAffinityMask(GetCurrentThread(), 1usize << cpu);
            }
        }
    }

    /// Logical CPUs of every physical core (processor group 0 only).
    pub fn cores() -> Vec<Vec<usize>> {
        let mut len: u32 = 0;
        unsafe {
            GetLogicalProcessorInformation(core::ptr::null_mut(), &mut len);
        }
        if len == 0 {
            return Vec::new();
        }
        let mut buf = vec![0u8; len as usize];
        if unsafe { GetLogicalProcessorInformation(buf.as_mut_ptr(), &mut len) } == 0 {
            return Vec::new();
        }
        // SYSTEM_LOGICAL_PROCESSOR_INFORMATION on 64-bit Windows: mask (8), relationship (4 + 4 padding), union (16).
        const ENTRY: usize = 32;
        let mut cores = Vec::new();
        let mut off = 0usize;
        while off + ENTRY <= len as usize {
            let mask = usize::from_ne_bytes(buf[off..off + 8].try_into().unwrap());
            let rel = u32::from_ne_bytes(buf[off + 8..off + 12].try_into().unwrap());
            if rel == 0 {
                let cpus: Vec<usize> = (0..64).filter(|b| mask >> b & 1 == 1).collect();
                if !cpus.is_empty() {
                    cores.push(cpus);
                }
            }
            off += ENTRY;
        }
        cores
    }
}

#[cfg(not(any(windows, target_os = "linux")))]
mod os {
    pub fn pin_current(_cpu: usize) {}
    pub fn cores() -> Vec<Vec<usize>> {
        Vec::new()
    }
}

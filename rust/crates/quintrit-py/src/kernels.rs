//! Kernel selection and runtime CPU feature detection.

use crate::error::{Kind, QError, QResult};
use quintrit_core::MAX_THREADS;
use quintrit_runtime::TernaryKernelType;

/// The two Quintrit kernels (both use 5 trits per byte, 1.60 bpw).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// 32-row tiles; AVX2 on x86-64, NEON on AArch64, portable scalar code elsewhere.
    Quintrit,
    /// 64-row tiles; AVX-512 BW + VBMI (x86-64 only).
    QuintritAvx512,
}

impl Kernel {
    pub const ALL: [Kernel; 2] = [Kernel::Quintrit, Kernel::QuintritAvx512];

    pub fn name(self) -> &'static str {
        match self {
            Kernel::Quintrit => "quintrit",
            Kernel::QuintritAvx512 => "quintrit_512",
        }
    }

    /// Rows per weight tile of the kernel's layout.
    pub fn tile_rows(self) -> usize {
        match self {
            Kernel::Quintrit => 32,
            Kernel::QuintritAvx512 => 64,
        }
    }

    /// Index into the per-matrix tile cache.
    pub fn slot(self) -> usize {
        match self {
            Kernel::Quintrit => 0,
            Kernel::QuintritAvx512 => 1,
        }
    }

    pub fn is_supported(self) -> bool {
        match self {
            Kernel::Quintrit => true,
            Kernel::QuintritAvx512 => avx512_supported(),
        }
    }

    pub fn runtime_kind(self) -> TernaryKernelType {
        match self {
            Kernel::Quintrit => TernaryKernelType::QuintritLut,
            Kernel::QuintritAvx512 => TernaryKernelType::QuintritLutAVX512,
        }
    }

    /// The fastest kernel this CPU can run.
    pub fn best() -> Kernel {
        if Kernel::QuintritAvx512.is_supported() { Kernel::QuintritAvx512 } else { Kernel::Quintrit }
    }

    /// Parses a user supplied kernel name (`None` and `"auto"` select [`Kernel::best`]) and checks that the CPU
    /// can run it.
    pub fn parse(name: Option<&str>) -> QResult<Kernel> {
        let kernel = match name.map(|s| s.trim().to_ascii_lowercase()).as_deref() {
            None | Some("auto") => return Ok(Kernel::best()),
            Some("quintrit") => Kernel::Quintrit,
            Some("quintrit_512") => Kernel::QuintritAvx512,
            Some(other) => {
                return Err(QError::new(
                    Kind::Kernel,
                    format!("unknown kernel {other:?}; expected 'auto', 'quintrit' or 'quintrit_512'"),
                ));
            }
        };
        if !kernel.is_supported() {
            return Err(QError::new(
                Kind::UnsupportedKernel,
                format!(
                    "kernel {:?} is not supported by this CPU (available: {:?})",
                    kernel.name(),
                    available_names()
                ),
            ));
        }
        Ok(kernel)
    }
}

/// True when the AVX-512 kernels can run: AVX-512 F, BW and VBMI (x86-64 only).
pub fn avx512_supported() -> bool {
    #[cfg(target_arch = "x86_64")]
    {
        is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vbmi")
    }
    #[cfg(not(target_arch = "x86_64"))]
    {
        false
    }
}

/// Names of the kernels this CPU can run.
pub fn available_names() -> Vec<&'static str> {
    Kernel::ALL.iter().filter(|k| k.is_supported()).map(|k| k.name()).collect()
}

/// Detected CPU features relevant for the kernels, as `(name, present)` pairs.
pub fn cpu_features() -> Vec<(&'static str, bool)> {
    let mut v: Vec<(&'static str, bool)> = Vec::new();
    #[cfg(target_arch = "x86_64")]
    {
        v.push(("avx2", is_x86_feature_detected!("avx2")));
        v.push(("avx512f", is_x86_feature_detected!("avx512f")));
        v.push(("avx512bw", is_x86_feature_detected!("avx512bw")));
        v.push(("avx512vbmi", is_x86_feature_detected!("avx512vbmi")));
        v.push(("avx512vnni", is_x86_feature_detected!("avx512vnni")));
    }
    #[cfg(target_arch = "aarch64")]
    {
        v.push(("neon", std::arch::is_aarch64_feature_detected!("neon")));
        v.push(("dotprod", std::arch::is_aarch64_feature_detected!("dotprod")));
        v.push(("i8mm", std::arch::is_aarch64_feature_detected!("i8mm")));
    }
    v
}

/// Default worker thread count: the number of logical CPUs, at most `MAX_THREADS` (the same rule as the C API).
pub fn default_threads() -> usize {
    std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1).clamp(1, MAX_THREADS)
}

/// Validates an explicit thread count.
pub fn resolve_threads(threads: Option<usize>) -> QResult<usize> {
    match threads {
        None => Ok(default_threads()),
        Some(0) => Err(QError::new(Kind::InvalidArgument, "threads must be at least 1")),
        Some(n) if n > MAX_THREADS => {
            Err(QError::new(Kind::InvalidArgument, format!("threads must be at most {MAX_THREADS}, got {n}")))
        }
        Some(n) => Ok(n),
    }
}

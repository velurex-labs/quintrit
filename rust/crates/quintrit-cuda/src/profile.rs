//! Kernel configurations chosen by device resources.
//!
//! The best launch configuration of the projection kernel (rows per warp, warps per block, how the first weight word and
//! the activations are loaded) depends on what the device offers, not on its name: a GPU with few streaming
//! multiprocessors and a large L2 cache is memory bound and prefers large blocks, one with many multiprocessors and a
//! small L2 cache is limited by shared-memory traffic and prefers small ones. The table below holds one profile per
//! device that was measured. A device that was never measured gets the profile whose reference resources (multiprocessor
//! count and L2 size) are nearest, with the distance `|ln(sms / ref_sms)| + |ln(l2 / ref_l2)|`.
//!
//! Every configuration produces the same exact integer results; only the speed differs. A shape the chosen
//! configuration cannot tile (the row count is not a multiple of its rows per block) falls back to a smaller
//! configuration, see [`DeviceProfile::config`].

use crate::device::DeviceInfo;
use crate::error::{Error, Result};

/// Launch configuration of the projection kernel.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct GemvConfig {
    /// Matrix rows handled by one warp.
    pub rows_per_warp: u32,
    /// Warps per block.
    pub warps: u32,
    /// Load the first weight word of every row before the prologue (hides DRAM latency of small shapes).
    pub first_word_prefetch: bool,
    /// Read activations through the L1 / L2 caches instead of staging them in shared memory.
    pub act_from_global: bool,
    /// Four-byte digit table with the fifth digit computed arithmetically (halves the shared-memory wavefronts).
    pub d4: bool,
}

impl GemvConfig {
    /// Rows of the matrix covered by one block.
    pub const fn rows_per_block(&self) -> u32 {
        self.rows_per_warp * self.warps
    }

    /// Threads per block.
    pub const fn threads(&self) -> u32 {
        32 * self.warps
    }

    /// Name fragment of the kernel instantiation, for example `r4_w8_pf` or `r2_w4_g_d4`.
    pub fn tag(&self) -> String {
        let mode = match (self.first_word_prefetch, self.act_from_global) {
            (true, false) => "_pf",
            (false, true) => "_g",
            (false, false) => "",
            (true, true) => "_pfg",
        };
        format!("r{}_w{}{}{}", self.rows_per_warp, self.warps, mode, if self.d4 { "_d4" } else { "" })
    }
}

const fn cfg(rows_per_warp: u32, warps: u32, pf: bool, g: bool, d4: bool) -> GemvConfig {
    GemvConfig { rows_per_warp, warps, first_word_prefetch: pf, act_from_global: g, d4 }
}

/// 4 rows per warp, 8 warps, first-word prefetch.
pub const R4_W8_PF: GemvConfig = cfg(4, 8, true, false, false);
/// 4 rows per warp, 8 warps, activations from global memory.
pub const R4_W8_G: GemvConfig = cfg(4, 8, false, true, false);
/// 2 rows per warp, 4 warps, activations from global memory, four-byte digit table.
pub const R2_W4_G_D4: GemvConfig = cfg(2, 4, false, true, true);
/// 4 rows per warp, 4 warps, activations from global memory, four-byte digit table.
pub const R4_W4_G_D4: GemvConfig = cfg(4, 4, false, true, true);
/// 2 rows per warp, 2 warps (fallback for row counts that are a multiple of 4).
pub const R2_W2_G_D4: GemvConfig = cfg(2, 2, false, true, true);
/// 2 rows per warp, 1 warp (fallback for any even row count).
pub const R2_W1_G_D4: GemvConfig = cfg(2, 1, false, true, true);

/// Every configuration the kernels are instantiated for. Profiles may only use these.
pub const ALL_CONFIGS: [GemvConfig; 6] = [R4_W8_PF, R4_W8_G, R2_W4_G_D4, R4_W4_G_D4, R2_W2_G_D4, R2_W1_G_D4];

/// Fallbacks tried in this order when a configuration cannot tile the row count.
const FALLBACKS: [GemvConfig; 3] = [R2_W4_G_D4, R2_W2_G_D4, R2_W1_G_D4];

/// What a projection of a transformer layer does with its result.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Role {
    /// Attention output projection (plain result).
    Output,
    /// Feed-forward down projection (plain result).
    Down,
    /// Stacked query, key and value projection with the rotary embedding and the KV-cache append in the epilogue.
    QkvRope,
    /// Stacked, row-interleaved gate and up projection with the gated activation and its block maximum in the epilogue.
    GateUp,
}

/// The configurations measured on one reference device.
#[derive(Clone, Copy, Debug)]
pub struct DeviceProfile {
    /// Name of the profile (reference multiprocessor count and L2 size).
    pub name: &'static str,
    /// Multiprocessor count of the reference device.
    pub ref_sms: u32,
    /// L2 cache of the reference device in MiB.
    pub ref_l2_mib: u32,
    /// Configuration of [`Role::Output`].
    pub output: GemvConfig,
    /// Configuration of [`Role::Down`].
    pub down: GemvConfig,
    /// Configuration of [`Role::QkvRope`].
    pub qkv_rope: GemvConfig,
    /// Configuration of [`Role::GateUp`].
    pub gate_up: GemvConfig,
    /// Rows of the next weight matrix whose first cache lines the norm kernels prefetch into L2 (0 switches the prefetch
    /// off). It pays off only when the L2 cache can hold the prefetched lines.
    pub prefetch_rows: u32,
}

/// All profiles. Index 0 is the default for devices without a nearer match.
pub const PROFILES: [DeviceProfile; 2] = [
    // Memory bound, 24 multiprocessors, 24 MiB L2 (RTX 4060 class).
    DeviceProfile {
        name: "24sm_24mib",
        ref_sms: 24,
        ref_l2_mib: 24,
        output: R4_W8_PF,
        down: R4_W8_G,
        qkv_rope: R2_W4_G_D4,
        gate_up: R4_W8_PF,
        prefetch_rows: 46080,
    },
    // Limited by shared-memory traffic, 46 multiprocessors, 4 MiB L2 (RTX 3070 class).
    DeviceProfile {
        name: "46sm_4mib",
        ref_sms: 46,
        ref_l2_mib: 4,
        output: R2_W4_G_D4,
        down: R2_W4_G_D4,
        qkv_rope: R4_W4_G_D4,
        gate_up: R4_W4_G_D4,
        prefetch_rows: 0,
    },
];

impl DeviceProfile {
    /// Picks the profile for a device by its resources. The environment variable `QUINTRIT_CUDA_PROFILE=<index>` selects
    /// one explicitly (for experiments).
    pub fn select(info: &DeviceInfo) -> &'static DeviceProfile {
        if let Some(i) = std::env::var("QUINTRIT_CUDA_PROFILE").ok().and_then(|v| v.trim().parse::<usize>().ok()) {
            if let Some(p) = PROFILES.get(i) {
                return p;
            }
        }
        Self::nearest(info.sm_count, info.l2_cache_bytes)
    }

    /// The profile nearest to a device with `sms` multiprocessors and `l2_bytes` of L2 cache.
    pub fn nearest(sms: u32, l2_bytes: u64) -> &'static DeviceProfile {
        if sms == 0 || l2_bytes == 0 {
            return &PROFILES[0];
        }
        let mut best = (&PROFILES[0], f64::MAX);
        for p in &PROFILES {
            let d = (sms as f64 / p.ref_sms as f64).ln().abs() + (l2_bytes as f64 / (p.ref_l2_mib as f64 * 1_048_576.0)).ln().abs();
            if d < best.1 {
                best = (p, d);
            }
        }
        best.0
    }

    /// The configuration for a projection with `rows` output rows. When the profile's configuration cannot tile `rows`
    /// (not a multiple of its rows per block) the first fallback that can is used: 8, 4 or 2 rows per block. The
    /// fallbacks are correct on every device but not tuned. `rows` must be even.
    pub fn config(&self, role: Role, rows: u32) -> Result<GemvConfig> {
        let wanted = match role {
            Role::Output => self.output,
            Role::Down => self.down,
            Role::QkvRope => self.qkv_rope,
            Role::GateUp => self.gate_up,
        };
        if rows > 0 && rows % wanted.rows_per_block() == 0 {
            return Ok(wanted);
        }
        FALLBACKS
            .iter()
            .copied()
            .find(|c| rows > 0 && rows % c.rows_per_block() == 0)
            .ok_or_else(|| Error::InvalidArgument(format!("a projection needs an even, non-zero number of rows, got {rows}")))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn nearest_profile_follows_resources() {
        // RTX 4060 (24 SMs, 24 MiB), RTX 3070 (46 SMs, 4 MiB), A100 (108 SMs, 40 MiB): the values of the measured devices.
        assert_eq!(DeviceProfile::nearest(24, 24 << 20).name, "24sm_24mib");
        assert_eq!(DeviceProfile::nearest(46, 4 << 20).name, "46sm_4mib");
        assert_eq!(DeviceProfile::nearest(108, 40 << 20).name, "24sm_24mib");
        // Unknown resources fall back to the first profile.
        assert_eq!(DeviceProfile::nearest(0, 0).name, PROFILES[0].name);
    }

    #[test]
    fn profiles_use_only_instantiated_configs() {
        for p in &PROFILES {
            for c in [p.output, p.down, p.qkv_rope, p.gate_up] {
                assert!(ALL_CONFIGS.contains(&c), "{} uses {} which is not instantiated", p.name, c.tag());
            }
        }
    }

    #[test]
    fn epilogue_configs_have_even_rows_per_warp() {
        for p in &PROFILES {
            assert_eq!(p.qkv_rope.rows_per_warp % 2, 0);
            assert_eq!(p.gate_up.rows_per_warp % 2, 0);
        }
        for c in FALLBACKS {
            assert_eq!(c.rows_per_warp % 2, 0);
        }
    }

    #[test]
    fn config_falls_back_to_a_tiling_one() {
        let p = &PROFILES[0];
        // 3072 rows divide by 32 (R4_W8): the profile's own choice.
        assert_eq!(p.config(Role::Output, 3072).unwrap(), R4_W8_PF);
        // 2560 = 80 x 32 as well.
        assert_eq!(p.config(Role::Output, 2560).unwrap(), R4_W8_PF);
        // 4 * 8 = 32 does not divide 40: falls back to 8 rows per block.
        assert_eq!(p.config(Role::Output, 40).unwrap(), R2_W4_G_D4);
        // 12 is a multiple of 4 only.
        assert_eq!(p.config(Role::Output, 12).unwrap(), R2_W2_G_D4);
        // 6 is even only.
        assert_eq!(p.config(Role::Output, 6).unwrap(), R2_W1_G_D4);
        assert!(p.config(Role::Output, 7).is_err());
        assert!(p.config(Role::Output, 0).is_err());
    }

    #[test]
    fn tags_are_distinct() {
        let mut tags: Vec<String> = ALL_CONFIGS.iter().map(|c| c.tag()).collect();
        tags.sort();
        tags.dedup();
        assert_eq!(tags.len(), ALL_CONFIGS.len());
        assert_eq!(R4_W8_PF.tag(), "r4_w8_pf");
        assert_eq!(R2_W4_G_D4.tag(), "r2_w4_g_d4");
    }
}

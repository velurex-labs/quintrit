//! Block-scaled ternary GEMV: ternary weights with one scale per 64 weights of a row, multiplied with activations
//! quantized in blocks of 32 (one int8 code per value and one scale per block).
//!
//! This is the arithmetic of llama.cpp's `Q2_0` weights against `Q8_0` activations. The weights are stored in the
//! velurex encoding (5 trits per byte, the 5-planar layout of [`crate::pack`]), so plane `p` of a 160-trit block is
//! exactly the 32-value activation block `5 * block + p`, and the scales are kept as `f32`.
//!
//! The result reproduces the reference CPU kernel (`ggml_vec_dot_q2_0_q8_0`, AVX2 code path) bit for bit, on every
//! instruction set:
//!
//! - every 32-value activation block `s` gives eight integer lane sums, lane `j` covering the values `4j..4j+4`;
//! - each lane sum is converted to `f32` and accumulated with one fused multiply-add,
//!   `acc[j] = fma(w_scale[s / 2] * a_scale[s], lane[j], acc[j])`, into one of two accumulators: the 64-weight blocks
//!   with an even index go to the first, the odd ones to the second;
//! - the two accumulators are added lane by lane and the eight lanes are reduced as `((l0+l4) + (l2+l6)) +
//!   ((l1+l5) + (l3+l7))`.
//!
//! The integer lane sums are exact, so the decoding of the weights (table, arithmetic or SIMD) does not matter; only
//! the floating-point order above does, and it is the same in the scalar and the SIMD implementations.

use crate::pack::{BLOCK_BYTES_5, BYTE_TO_TRITS, QK_5};
use rayon::prelude::*;

/// Values per activation block (llama.cpp `QK8_0`).
pub const ACT_BLOCK: usize = 32;
/// Weights per scale block (llama.cpp `QK2_0`).
pub const SCALE_BLOCK: usize = 64;

/// Bytes of one packed row of `cols` weights (whole 160-trit blocks).
#[inline]
pub fn row_bytes(cols: usize) -> usize {
    cols.div_ceil(QK_5) * BLOCK_BYTES_5
}

/// Number of `f32` lane sums [`activation_lane_sums`] produces for `cols` activations.
#[inline]
pub fn lane_sums_len(cols: usize) -> usize {
    cols / 4
}

/// Sums of every four consecutive activation codes (`out[i] = a[4i] + ... + a[4i+3]`), which the SIMD kernels use to
/// turn the weight codes `{0, 1, 2}` back into trits. Computed once per activation vector.
pub fn activation_lane_sums(acts: &[i8], out: &mut [i32]) {
    for (o, c) in out.iter_mut().zip(acts.chunks_exact(4)) {
        *o = c.iter().map(|&v| v as i32).sum();
    }
}

/// The implementation the entry points run on this CPU.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Path {
    /// AVX-512 F/BW/VL/VBMI/VNNI (decode with `vpermb`, products with `vpdpbusd`).
    #[cfg(target_arch = "x86_64")]
    Avx512,
    /// AVX2 + FMA.
    #[cfg(target_arch = "x86_64")]
    Avx2,
    /// NEON with the dot-product extension.
    #[cfg(target_arch = "aarch64")]
    Neon,
    /// The definition.
    Portable,
}

fn path() -> Path {
    #[cfg(target_arch = "x86_64")]
    {
        if is_x86_feature_detected!("avx512f")
            && is_x86_feature_detected!("avx512bw")
            && is_x86_feature_detected!("avx512vl")
            && is_x86_feature_detected!("avx512vbmi")
            && is_x86_feature_detected!("avx512vnni")
            && is_x86_feature_detected!("avx2")
            && is_x86_feature_detected!("fma")
        {
            return Path::Avx512;
        }
        if is_x86_feature_detected!("avx2") && is_x86_feature_detected!("fma") {
            return Path::Avx2;
        }
    }
    #[cfg(target_arch = "aarch64")]
    {
        if std::arch::is_aarch64_feature_detected!("dotprod") {
            return Path::Neon;
        }
    }
    Path::Portable
}

/// Per-call data of the SIMD kernels, computed once per activation vector.
struct Lanes {
    /// One value per four activations: the activation lane sums (AVX2), minus them (AVX-512), the start value of the
    /// dot products (NEON); empty for the portable code.
    ints: Vec<i32>,
    /// AVX-512: every activation scale repeated eight times; empty otherwise.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    scales: Vec<f32>,
    /// AVX-512: the activations, `ints` and `scales` in the pair order of the 512-bit loop; empty otherwise.
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pair_acts: Vec<i8>,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pair_ints: Vec<i32>,
    #[cfg_attr(not(target_arch = "x86_64"), allow(dead_code))]
    pair_scales: Vec<f32>,
}

fn lane_data(acts: &[i8], a_scales: &[f32], path: Path) -> Lanes {
    #[cfg(not(target_arch = "x86_64"))]
    let _ = a_scales;
    let n = lane_sums_len(acts.len());
    let mut lanes = Lanes {
        ints: Vec::new(),
        scales: Vec::new(),
        pair_acts: Vec::new(),
        pair_ints: Vec::new(),
        pair_scales: Vec::new(),
    };
    match path {
        #[cfg(target_arch = "x86_64")]
        Path::Avx512 => {
            lanes.ints = vec![0; n];
            activation_lane_sums(acts, &mut lanes.ints);
            lanes.ints.iter_mut().for_each(|v| *v = -*v);
            lanes.scales = a_scales.iter().flat_map(|&x| [x; 8]).collect();
            avx512::pair_layout(acts, &mut lanes);
        }
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => {
            lanes.ints = vec![0; n];
            activation_lane_sums(acts, &mut lanes.ints);
        }
        #[cfg(target_arch = "aarch64")]
        Path::Neon => lanes.ints = vec![neon::MAGIC; n],
        Path::Portable => {}
    }
    lanes
}

/// `((l0+l4) + (l2+l6)) + ((l1+l5) + (l3+l7))`: the horizontal sum of ggml's `hsum_float_8`.
#[inline(always)]
fn hsum8(l: &[f32; 8]) -> f32 {
    let r = [l[4] + l[0], l[5] + l[1], l[6] + l[2], l[7] + l[3]];
    (r[0] + r[2]) + (r[1] + r[3])
}

/// Checks the shapes shared by all entry points.
fn check(
    packed: &[u8],
    w_scales: &[f32],
    acts: &[i8],
    a_scales: &[f32],
    rows: usize,
    cols: usize,
    out: &[f32],
) {
    assert!(
        cols.is_multiple_of(SCALE_BLOCK),
        "the column count must be a multiple of {SCALE_BLOCK}"
    );
    assert!(
        packed.len() >= rows * row_bytes(cols),
        "packed weights too short"
    );
    assert!(
        w_scales.len() >= rows * (cols / SCALE_BLOCK),
        "weight scales too short"
    );
    assert_eq!(acts.len(), cols, "activation length");
    assert_eq!(a_scales.len(), cols / ACT_BLOCK, "activation scale count");
    assert_eq!(out.len(), rows, "output length");
}

/// Dot product of one packed row with the activations, portable implementation (the definition).
pub fn dot_row_scalar(
    row: &[u8],
    w_scales: &[f32],
    acts: &[i8],
    a_scales: &[f32],
    cols: usize,
) -> f32 {
    let mut acc = [[0.0f32; 8]; 2];
    for s in 0..cols / ACT_BLOCK {
        let block = &row[(s / 5) * BLOCK_BYTES_5..(s / 5 + 1) * BLOCK_BYTES_5];
        let plane = s % 5;
        let a = &acts[s * ACT_BLOCK..(s + 1) * ACT_BLOCK];
        let d = w_scales[s / 2] * a_scales[s];
        let chain = &mut acc[(s / 2) & 1];
        for (j, lane) in chain.iter_mut().enumerate() {
            let mut p = 0i32;
            for k in 4 * j..4 * j + 4 {
                p += BYTE_TO_TRITS[block[k] as usize][plane] as i32 * a[k] as i32;
            }
            *lane = d.mul_add(p as f32, *lane);
        }
    }
    let mut l = [0.0f32; 8];
    for j in 0..8 {
        l[j] = acc[0][j] + acc[1][j];
    }
    hsum8(&l)
}

/// Row-by-row GEMV `out[r] = dot(row r, acts)` on one thread. `w_scales` holds `cols / 64` scales per row,
/// `a_scales` one scale per 32 activations.
pub fn gemv_block_scaled_1xn(
    packed: &[u8],
    w_scales: &[f32],
    acts: &[i8],
    a_scales: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
) {
    check(packed, w_scales, acts, a_scales, rows, cols, out);
    let p = path();
    let lanes = lane_data(acts, a_scales, p);
    gemv_rows(p, packed, w_scales, acts, &lanes, a_scales, 0, cols, out);
}

/// Multi-threaded [`gemv_block_scaled_1xn`]: the rows are split between `threads` workers of the persistent pool.
/// Every row is computed by one worker, so the result does not depend on the thread count.
#[allow(clippy::too_many_arguments)]
pub fn gemv_block_scaled_parallel(
    packed: &[u8],
    w_scales: &[f32],
    acts: &[i8],
    a_scales: &[f32],
    rows: usize,
    cols: usize,
    out: &mut [f32],
    threads: usize,
) {
    check(packed, w_scales, acts, a_scales, rows, cols, out);
    let p = path();
    let lanes = lane_data(acts, a_scales, p);
    if threads <= 1 || rows < 64 {
        gemv_rows(p, packed, w_scales, acts, &lanes, a_scales, 0, cols, out);
        return;
    }
    let chunk = rows.div_ceil(threads * 4).max(16);
    crate::threading::get_thread_pool(threads).install(|| {
        out.par_chunks_mut(chunk).enumerate().for_each(|(i, o)| {
            gemv_rows(p, packed, w_scales, acts, &lanes, a_scales, i * chunk, cols, o);
        });
    });
}

/// Computes `out.len()` rows starting at row `first` on `path`. `lanes` is [`lane_data`] of `acts` for `path`.
#[allow(clippy::too_many_arguments)]
fn gemv_rows(
    path: Path,
    packed: &[u8],
    w_scales: &[f32],
    acts: &[i8],
    lanes: &Lanes,
    a_scales: &[f32],
    first: usize,
    cols: usize,
    out: &mut [f32],
) {
    let rb = row_bytes(cols);
    let sb = cols / SCALE_BLOCK;
    let row = |r: usize| &packed[r * rb..(r + 1) * rb];
    let scales = |r: usize| &w_scales[r * sb..(r + 1) * sb];
    match path {
        #[cfg(target_arch = "x86_64")]
        Path::Avx512 => {
            const R: usize = avx512::ROWS;
            let mut i = 0;
            while i + R <= out.len() {
                let r = first + i;
                // SAFETY: the AVX-512 extensions are available (`path`); the slices have the lengths checked by
                // `check`, and `lanes` was built for this path.
                unsafe {
                    avx512::dot_rows::<R>(
                        core::array::from_fn(|k| row(r + k)),
                        core::array::from_fn(|k| scales(r + k)),
                        acts,
                        lanes,
                        cols,
                        &mut out[i..i + R],
                    )
                };
                i += R;
            }
            while i < out.len() {
                let r = first + i;
                // SAFETY: as above.
                unsafe {
                    avx512::dot_rows::<1>([row(r)], [scales(r)], acts, lanes, cols, &mut out[i..i + 1])
                };
                i += 1;
            }
        }
        #[cfg(target_arch = "x86_64")]
        Path::Avx2 => {
            for (i, o) in out.iter_mut().enumerate() {
                let r = first + i;
                // SAFETY: AVX2 and FMA are available (`path`); the slices have the lengths checked by `check`.
                *o = unsafe { avx2::dot_row(row(r), scales(r), acts, &lanes.ints, a_scales, cols) };
            }
        }
        #[cfg(target_arch = "aarch64")]
        Path::Neon => {
            for (i, o) in out.iter_mut().enumerate() {
                let r = first + i;
                // SAFETY: the dot-product extension is available (`path`); the slices have the lengths checked by
                // `check`.
                *o = unsafe { neon::dot_row(row(r), scales(r), acts, &lanes.ints, a_scales, cols) };
            }
        }
        Path::Portable => {
            for (i, o) in out.iter_mut().enumerate() {
                let r = first + i;
                *o = dot_row_scalar(row(r), scales(r), acts, a_scales, cols);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx512 {
    use super::{ACT_BLOCK, BLOCK_BYTES_5, Lanes, hsum8};
    use std::arch::x86_64::*;

    /// Rows per pass: several weight streams keep the memory system busier at the same compute cost per row.
    pub(super) const ROWS: usize = 2;
    /// Software prefetch distance in bytes ahead of each weight stream.
    const PREFETCH: usize = 4096;

    /// `floor(w / 28) = floor((w >> 2) / 7)` because 28 is a multiple of 4, so everything that depends on the high
    /// part `U = d3 + 3 d4` is a 64-entry table indexed by `w >> 2` (the lookups ignore the two foreign bits of the
    /// 16-bit shift): `what` 0 gives `28 U`, 1 the code `U % 3`, 2 the code `U / 3`.
    const fn high_table(what: u32) -> [u8; 64] {
        let mut t = [0u8; 64];
        let mut q = 0;
        while q < 64 {
            let u = q / 7;
            t[q as usize] = match what {
                0 => 28 * u,
                1 => u % 3,
                _ => u / 3,
            } as u8;
            q += 1;
        }
        t
    }
    /// The code of the digit `(L / div) % 3` of the low part `L = w - 28 U` (`0..=26`), 32 entries.
    const fn low_table(div: u32) -> [u8; 32] {
        let mut t = [0u8; 32];
        let mut l = 0;
        while l < 32 {
            t[l as usize] = ((l / div) % 3) as u8;
            l += 1;
        }
        t
    }
    static T28: [u8; 64] = high_table(0);
    static TD3: [u8; 64] = high_table(1);
    static TD4: [u8; 64] = high_table(2);
    static TD0: [u8; 32] = low_table(1);
    static TD1: [u8; 32] = low_table(3);
    static TD2: [u8; 32] = low_table(9);

    /// Lookup tables of the 256-bit decode (`vpermt2b` for the 64-entry tables).
    struct Tables {
        t28: [__m256i; 2],
        d3: [__m256i; 2],
        d4: [__m256i; 2],
        d0: __m256i,
        d1: __m256i,
        d2: __m256i,
    }

    /// Lookup tables of the 512-bit decode: `vpermb` on a 512-bit register uses the low six bits of the index, so
    /// each 64-entry table fits one register and the 32-entry tables are repeated.
    struct Tables512 {
        t28: __m512i,
        d3: __m512i,
        d4: __m512i,
        d0: __m512i,
        d1: __m512i,
        d2: __m512i,
    }

    #[inline(always)]
    unsafe fn load2(t: &[u8; 64]) -> [__m256i; 2] {
        unsafe {
            [
                _mm256_loadu_si256(t.as_ptr() as *const __m256i),
                _mm256_loadu_si256(t.as_ptr().add(32) as *const __m256i),
            ]
        }
    }

    #[inline(always)]
    unsafe fn load32(t: &[u8; 32]) -> __m256i {
        unsafe { _mm256_loadu_si256(t.as_ptr() as *const __m256i) }
    }

    #[inline(always)]
    unsafe fn tables() -> Tables {
        unsafe {
            Tables {
                t28: load2(&T28),
                d3: load2(&TD3),
                d4: load2(&TD4),
                d0: load32(&TD0),
                d1: load32(&TD1),
                d2: load32(&TD2),
            }
        }
    }

    #[inline(always)]
    unsafe fn tables512() -> Tables512 {
        unsafe {
            Tables512 {
                t28: _mm512_loadu_si512(T28.as_ptr() as *const _),
                d3: _mm512_loadu_si512(TD3.as_ptr() as *const _),
                d4: _mm512_loadu_si512(TD4.as_ptr() as *const _),
                d0: _mm512_broadcast_i64x4(load32(&TD0)),
                d1: _mm512_broadcast_i64x4(load32(&TD1)),
                d2: _mm512_broadcast_i64x4(load32(&TD2)),
            }
        }
    }

    /// The five planes of weight codes `{0, 1, 2}` (trit + 1) of one 160-trit block: eight instructions.
    #[inline(always)]
    unsafe fn decode(w: __m256i, t: &Tables) -> [__m256i; 5] {
        unsafe {
            let q = _mm256_srli_epi16::<2>(w);
            let l = _mm256_sub_epi8(w, _mm256_permutex2var_epi8(t.t28[0], q, t.t28[1]));
            [
                _mm256_permutexvar_epi8(l, t.d0),
                _mm256_permutexvar_epi8(l, t.d1),
                _mm256_permutexvar_epi8(l, t.d2),
                _mm256_permutex2var_epi8(t.d3[0], q, t.d3[1]),
                _mm256_permutex2var_epi8(t.d4[0], q, t.d4[1]),
            ]
        }
    }

    /// The same for two consecutive 160-trit blocks (64 bytes): block `b` in the lower, `b + 1` in the upper half
    /// of every plane.
    #[inline(always)]
    unsafe fn decode512(w: __m512i, t: &Tables512) -> [__m512i; 5] {
        unsafe {
            let q = _mm512_srli_epi16::<2>(w);
            let l = _mm512_sub_epi8(w, _mm512_permutexvar_epi8(q, t.t28));
            [
                _mm512_permutexvar_epi8(l, t.d0),
                _mm512_permutexvar_epi8(l, t.d1),
                _mm512_permutexvar_epi8(l, t.d2),
                _mm512_permutexvar_epi8(q, t.d3),
                _mm512_permutexvar_epi8(q, t.d4),
            ]
        }
    }

    /// The activation data of the 512-bit loop in pair order: pair `k` holds the activation blocks
    /// `s = 4 (k / 2) + k % 2` (lower half) and `s + 2` (upper half), for every complete group of four blocks.
    pub(super) fn pair_layout(acts: &[i8], lanes: &mut Lanes) {
        let pairs = 2 * (acts.len() / (4 * ACT_BLOCK));
        lanes.pair_acts.reserve(pairs * 2 * ACT_BLOCK);
        lanes.pair_ints.reserve(pairs * 16);
        lanes.pair_scales.reserve(pairs * 16);
        for k in 0..pairs {
            let lo = 4 * (k / 2) + k % 2;
            for s in [lo, lo + 2] {
                lanes.pair_acts.extend_from_slice(&acts[s * ACT_BLOCK..(s + 1) * ACT_BLOCK]);
                lanes.pair_ints.extend_from_slice(&lanes.ints[s * 8..s * 8 + 8]);
                lanes.pair_scales.extend_from_slice(&lanes.scales[s * 8..s * 8 + 8]);
            }
        }
    }

    /// `R` rows of [`super::dot_row_scalar`] with the same floating-point order. `vpdpbusd` (VNNI) forms the eight
    /// lane sums of four products of the codes with the activations, starting from minus the activation lane sums
    /// (`lanes.ints`), which turns the codes back into trits; each lane then takes one fused multiply-add with
    /// `w_scale * a_scale` (a broadcast `w_scale` times the precomputed broadcast `a_scale`).
    ///
    /// The main loop takes 20 activation blocks (four 160-trit blocks) per step on 512-bit registers. The
    /// accumulator register of a row holds both chains of the definition, `[acc0 | acc1]`: block `s` belongs to
    /// chain `(s / 2) & 1`, so the step pairs the blocks `s` (lower half, chain 0) and `s + 2` (upper half, chain 1)
    /// for `s % 4 in {0, 1}` and every chain takes its blocks in the definition's order. Two 160-trit blocks are
    /// decoded per 512-bit register and `vshufi64x2` gathers the two planes of a pair; the activations are stored
    /// in pair order once per vector ([`pair_layout`]). The weight-scale vector of a pair is built from two
    /// broadcasts: a single load of the ten scales of a step makes every pair wait for it, which costs a quarter of
    /// the rate when the scales come from memory. The remaining (at most 19) blocks run on 256-bit registers.
    #[target_feature(enable = "avx2,fma,avx512f,avx512bw,avx512vl,avx512vbmi,avx512vnni")]
    #[allow(clippy::needless_range_loop)]
    pub(super) unsafe fn dot_rows<const R: usize>(
        rows: [&[u8]; R],
        w_scales: [&[f32]; R],
        acts: &[i8],
        lanes: &Lanes,
        cols: usize,
        out: &mut [f32],
    ) {
        unsafe {
            let t = tables();
            let tz = tables512();
            let n_sub = cols / ACT_BLOCK;
            let rp: [*const u8; R] = core::array::from_fn(|r| rows[r].as_ptr());
            let wp: [*const f32; R] = core::array::from_fn(|r| w_scales[r].as_ptr());
            let ap = acts.as_ptr();
            let ip = lanes.ints.as_ptr();
            let bp = lanes.scales.as_ptr();
            let (zap, zip, zbp) = (lanes.pair_acts.as_ptr(), lanes.pair_ints.as_ptr(), lanes.pair_scales.as_ptr());
            let mut accz = [_mm512_setzero_ps(); R];
            let mut s = 0usize;
            while s + 20 <= n_sub {
                let block = s / 5;
                for r in 0..R {
                    let p = rp[r].wrapping_add(block * BLOCK_BYTES_5 + PREFETCH);
                    _mm_prefetch::<_MM_HINT_T0>(p as *const i8);
                    _mm_prefetch::<_MM_HINT_T0>(p.wrapping_add(64) as *const i8);
                }
                let wsp: [*const f32; R] = core::array::from_fn(|r| wp[r].add(s / 2));
                let src: [*const u8; R] = core::array::from_fn(|r| rp[r].add(block * BLOCK_BYTES_5));
                // Pair `$k` of this step from the decoded planes `$dl` (lower block) and `$dh` (upper block).
                macro_rules! pair {
                    ($k:literal, $dl:ident, $dh:ident) => {{
                        const LO: usize = 4 * ($k / 2) + $k % 2;
                        const HI: usize = LO + 2;
                        const LP: usize = LO % 5;
                        const HP: usize = HI % 5;
                        // Lower 256 bits from the half of `$dl` that holds block `LO / 5`, upper from `$dh`.
                        const IMM: i32 =
                            (if (LO / 5) % 2 == 1 { 0x0e } else { 0x04 }) | (if (HI / 5) % 2 == 1 { 0xe0 } else { 0x40 });
                        let kp = s / 2 + $k;
                        let av = _mm512_loadu_si512(zap.add(kp * 2 * ACT_BLOCK) as *const _);
                        let start = _mm512_loadu_si512(zip.add(kp * 16) as *const _);
                        let ab = _mm512_loadu_ps(zbp.add(kp * 16));
                        for r in 0..R {
                            let w = _mm512_shuffle_i64x2::<IMM>($dl[r][LP], $dh[r][HP]);
                            let p = _mm512_cvtepi32_ps(_mm512_dpbusd_epi32(start, w, av));
                            let wsv = _mm512_castpd_ps(_mm512_insertf64x4::<1>(
                                _mm512_castps_pd(_mm512_set1_ps(*wsp[r].add(2 * ($k / 2)))),
                                _mm256_castps_pd(_mm256_set1_ps(*wsp[r].add(2 * ($k / 2) + 1))),
                            ));
                            accz[r] = _mm512_fmadd_ps(_mm512_mul_ps(wsv, ab), p, accz[r]);
                        }
                    }};
                }
                let dx: [[__m512i; 5]; R] =
                    core::array::from_fn(|r| decode512(_mm512_loadu_si512(src[r] as *const _), &tz));
                pair!(0, dx, dx);
                pair!(1, dx, dx);
                pair!(2, dx, dx);
                pair!(3, dx, dx);
                let dy: [[__m512i; 5]; R] =
                    core::array::from_fn(|r| decode512(_mm512_loadu_si512(src[r].add(64) as *const _), &tz));
                pair!(4, dx, dy);
                pair!(5, dx, dy);
                pair!(6, dy, dy);
                pair!(7, dy, dy);
                pair!(8, dy, dy);
                pair!(9, dy, dy);
                s += 20;
            }
            // The remaining blocks on the two 256-bit chains, with the chain chosen at run time.
            let mut acc: [[__m256; 2]; R] = core::array::from_fn(|r| {
                [
                    _mm512_castps512_ps256(accz[r]),
                    _mm256_castpd_ps(_mm512_extractf64x4_pd::<1>(_mm512_castps_pd(accz[r]))),
                ]
            });
            let mut block = s / 5;
            while s < n_sub {
                let dec: [[__m256i; 5]; R] = core::array::from_fn(|r| {
                    decode(_mm256_loadu_si256(rp[r].add(block * BLOCK_BYTES_5) as *const __m256i), &t)
                });
                for p in 0..(n_sub - s).min(5) {
                    let av = _mm256_loadu_si256(ap.add(s * ACT_BLOCK) as *const __m256i);
                    let start = _mm256_loadu_si256(ip.add(s * 8) as *const __m256i);
                    let ab = _mm256_loadu_ps(bp.add(s * 8));
                    for r in 0..R {
                        let pr = _mm256_cvtepi32_ps(_mm256_dpbusd_epi32(start, dec[r][p], av));
                        let d = _mm256_mul_ps(_mm256_set1_ps(*wp[r].add(s / 2)), ab);
                        let c = (s / 2) & 1;
                        acc[r][c] = _mm256_fmadd_ps(d, pr, acc[r][c]);
                    }
                    s += 1;
                }
                block += 1;
            }
            for r in 0..R {
                let mut l = [0.0f32; 8];
                _mm256_storeu_ps(l.as_mut_ptr(), _mm256_add_ps(acc[r][0], acc[r][1]));
                out[r] = hsum8(&l);
            }
        }
    }

    #[cfg(test)]
    pub(super) unsafe fn decode_for_test(bytes: &[u8; 64]) -> [[[u8; 32]; 5]; 2] {
        unsafe {
            let mut out = [[[0u8; 32]; 5]; 2];
            let narrow = decode(_mm256_loadu_si256(bytes.as_ptr() as *const __m256i), &tables());
            for (o, p) in out[0].iter_mut().zip(narrow) {
                _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, p);
            }
            // The wide decode of the 64 bytes; its lower half must equal the narrow decode of the first 32.
            let wide = decode512(_mm512_loadu_si512(bytes.as_ptr() as *const _), &tables512());
            for (p, w) in wide.into_iter().enumerate() {
                let mut b = [0u8; 64];
                _mm512_storeu_si512(b.as_mut_ptr() as *mut _, w);
                assert_eq!(b[..32], out[0][p], "plane {p}: 512-bit and 256-bit decode differ");
                out[1][p].copy_from_slice(&b[32..]);
            }
            out
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod avx2 {
    use super::{ACT_BLOCK, BLOCK_BYTES_5, hsum8};
    use std::arch::x86_64::*;

    /// Decodes the 32 bytes of a 160-trit block into its five planes of weight codes `{0, 1, 2}` (trit + 1).
    ///
    /// `byte = L + 28 U` with `L = d0 + 3 d1 + 9 d2` and `U = d3 + 3 d4`; the divisions by 28, 3 and 9 are exact
    /// multiply-high operations for the byte range `0..=250` (checked exhaustively by the tests).
    #[inline]
    #[target_feature(enable = "avx2")]
    unsafe fn decode_planes(bytes: __m256i) -> [__m256i; 5] {
        let digits = |x: __m256i| -> [__m256i; 5] {
            let u = _mm256_mulhi_epu16(x, _mm256_set1_epi16(2341));
            let l = _mm256_sub_epi16(x, _mm256_mullo_epi16(u, _mm256_set1_epi16(28)));
            let l3 = _mm256_mulhi_epu16(l, _mm256_set1_epi16(21846));
            let d0 = _mm256_sub_epi16(l, _mm256_mullo_epi16(l3, _mm256_set1_epi16(3)));
            let d2 = _mm256_mulhi_epu16(l, _mm256_set1_epi16(7282));
            let d1 = _mm256_sub_epi16(l3, _mm256_mullo_epi16(d2, _mm256_set1_epi16(3)));
            let d4 = _mm256_mulhi_epu16(u, _mm256_set1_epi16(21846));
            let d3 = _mm256_sub_epi16(u, _mm256_mullo_epi16(d4, _mm256_set1_epi16(3)));
            [d0, d1, d2, d3, d4]
        };
        let lo = digits(_mm256_cvtepu8_epi16(_mm256_castsi256_si128(bytes)));
        let hi = digits(_mm256_cvtepu8_epi16(_mm256_extracti128_si256::<1>(bytes)));
        let pack = |a: __m256i, b: __m256i| {
            _mm256_permute4x64_epi64::<0xD8>(_mm256_packus_epi16(a, b))
        };
        [
            pack(lo[0], hi[0]),
            pack(lo[1], hi[1]),
            pack(lo[2], hi[2]),
            pack(lo[3], hi[3]),
            pack(lo[4], hi[4]),
        ]
    }

    /// AVX2 + FMA implementation of [`super::dot_row_scalar`] (same floating-point order).
    #[target_feature(enable = "avx2,fma")]
    pub(super) unsafe fn dot_row(
        row: &[u8],
        w_scales: &[f32],
        acts: &[i8],
        sums: &[i32],
        a_scales: &[f32],
        cols: usize,
    ) -> f32 {
        unsafe {
            let n_sub = cols / ACT_BLOCK;
            let ones = _mm256_set1_epi16(1);
            let mut acc = [_mm256_setzero_ps(); 2];
            let mut s = 0usize;
            let mut block = 0usize;
            while s < n_sub {
                let bytes =
                    _mm256_loadu_si256(row.as_ptr().add(block * BLOCK_BYTES_5) as *const __m256i);
                let planes = decode_planes(bytes);
                for plane in planes.iter().take((n_sub - s).min(5)) {
                    let a = _mm256_loadu_si256(acts.as_ptr().add(s * ACT_BLOCK) as *const __m256i);
                    let prod = _mm256_madd_epi16(_mm256_maddubs_epi16(*plane, a), ones);
                    let asum = _mm256_loadu_si256(sums.as_ptr().add(s * 8) as *const __m256i);
                    let p = _mm256_cvtepi32_ps(_mm256_sub_epi32(prod, asum));
                    let d = _mm256_set1_ps(w_scales[s / 2] * a_scales[s]);
                    let chain = (s / 2) & 1;
                    acc[chain] = _mm256_fmadd_ps(d, p, acc[chain]);
                    s += 1;
                }
                block += 1;
            }
            let mut l = [0.0f32; 8];
            _mm256_storeu_ps(l.as_mut_ptr(), _mm256_add_ps(acc[0], acc[1]));
            hsum8(&l)
        }
    }

    #[cfg(test)]
    pub(super) unsafe fn decode_for_test(bytes: &[u8; 32]) -> [[u8; 32]; 5] {
        unsafe {
            let planes = decode_planes(_mm256_loadu_si256(bytes.as_ptr() as *const __m256i));
            let mut out = [[0u8; 32]; 5];
            for (o, p) in out.iter_mut().zip(planes) {
                _mm256_storeu_si256(o.as_mut_ptr() as *mut __m256i, p);
            }
            out
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod neon {
    use super::{ACT_BLOCK, BLOCK_BYTES_5, hsum8};
    use core::arch::aarch64::*;

    /// Start value of every dot product: the bits of `1.5 * 2^23`. A lane sum `p` (at most `4 * 128` in magnitude)
    /// added to it in the integer domain gives the bits of the `f32` value `1.5 * 2^23 + p`, so subtracting
    /// `1.5 * 2^23` yields `p` exactly. This replaces the integer-to-float conversion, which issues at half the rate
    /// of an addition, and lets the start value come from memory instead of a vector register.
    pub(super) const MAGIC: i32 = 0x4B40_0000;
    const MAGIC_F: f32 = 12_582_912.0;

    /// Trit (`digit - 1`) of the digits `L % 3`, `(L / 3) % 3` and `L / 9` of the low part `L = d0 + 3 d1 + 9 d2`
    /// (`0..=26`, padded to 32 entries for `tbl` with two registers).
    const fn low_table(digit: u32) -> [i8; 32] {
        let mut t = [0i8; 32];
        let mut l = 0;
        while l < 27 {
            t[l as usize] = ((l / [1, 3, 9][digit as usize]) % 3) as i8 - 1;
            l += 1;
        }
        t
    }
    /// Trit of the digits `U % 3` and `U / 3` of the high part `U = d3 + 3 d4` (`0..=8`).
    const fn high_table(digit: u32) -> [i8; 16] {
        let mut t = [0i8; 16];
        let mut u = 0;
        while u < 9 {
            t[u as usize] = ((u / [1, 3][digit as usize]) % 3) as i8 - 1;
            u += 1;
        }
        t
    }
    static T0: [i8; 32] = low_table(0);
    static T1: [i8; 32] = low_table(1);
    static T2: [i8; 32] = low_table(2);
    static T3: [i8; 16] = high_table(0);
    static T4: [i8; 16] = high_table(1);
    /// `28 U`, so that `L = byte - 28 U` needs no multiply.
    static T28: [u8; 16] = [0, 28, 56, 84, 112, 140, 168, 196, 224, 0, 0, 0, 0, 0, 0, 0];

    /// Lookup tables of the decode, loaded once per row.
    struct Tables {
        t0: int8x16x2_t,
        t1: int8x16x2_t,
        t2: int8x16x2_t,
        t3: int8x16_t,
        t4: int8x16_t,
        t28: uint8x16_t,
    }

    #[inline(always)]
    unsafe fn tables() -> Tables {
        unsafe {
            Tables {
                t0: vld1q_s8_x2(T0.as_ptr()),
                t1: vld1q_s8_x2(T1.as_ptr()),
                t2: vld1q_s8_x2(T2.as_ptr()),
                t3: vld1q_s8(T3.as_ptr()),
                t4: vld1q_s8(T4.as_ptr()),
                t28: vld1q_u8(T28.as_ptr()),
            }
        }
    }

    /// The parts `(L, U)` of 16 bytes, `byte = L + 28 U`. `U = (byte * 147) >> 12` is exact for every valid byte
    /// (low part at most 26, checked exhaustively by the tests): the high bytes of the 16-bit products are taken
    /// with `uzp2`, and `28 U` comes from a table.
    #[inline(always)]
    unsafe fn parts(b: uint8x16_t, t: &Tables) -> [uint8x16_t; 2] {
        unsafe {
            let lo = vmull_u8(vget_low_u8(b), vdup_n_u8(147));
            let hi = vmull_high_u8(b, vdupq_n_u8(147));
            let u = vshrq_n_u8::<4>(vuzp2q_u8(vreinterpretq_u8_u16(lo), vreinterpretq_u8_u16(hi)));
            [vsubq_u8(b, vqtbl1q_u8(t.t28, u)), u]
        }
    }

    /// The trits `{-1, 0, 1}` of plane `P`.
    #[inline(always)]
    unsafe fn plane<const P: usize>(x: &[uint8x16_t; 2], t: &Tables) -> int8x16_t {
        unsafe {
            match P {
                0 => vqtbl2q_s8(t.t0, x[0]),
                1 => vqtbl2q_s8(t.t1, x[0]),
                2 => vqtbl2q_s8(t.t2, x[0]),
                3 => vqtbl1q_s8(t.t3, x[1]),
                _ => vqtbl1q_s8(t.t4, x[1]),
            }
        }
    }

    /// Decodes 16 bytes into the trits of the five planes.
    #[inline(always)]
    unsafe fn decode16(b: uint8x16_t, t: &Tables) -> [int8x16_t; 5] {
        unsafe {
            let x = parts(b, t);
            [
                plane::<0>(&x, t),
                plane::<1>(&x, t),
                plane::<2>(&x, t),
                plane::<3>(&x, t),
                plane::<4>(&x, t),
            ]
        }
    }

    /// NEON implementation of [`super::dot_row_scalar`] with the same floating-point order: `vdotq_s32` gives the
    /// eight lane sums of four products directly (lanes 0..3 from the first 16 values of a block, 4..7 from the
    /// second), and each lane takes one fused multiply-add into its chain. `starts` holds [`MAGIC`] once per four
    /// activations.
    ///
    /// The main loop takes 20 activation blocks (four 160-trit blocks) per step, so that the chain of every block
    /// and the lane of its scale product are known at compile time, computes the 20 products
    /// `w_scale[s / 2] * a_scale[s]` four per vector multiply, and decodes the next half block while the current
    /// one is multiplied (the kernel is limited by the latency of the decode, not by memory).
    #[target_feature(enable = "neon,dotprod")]
    pub(super) unsafe fn dot_row(
        row: &[u8],
        w_scales: &[f32],
        acts: &[i8],
        starts: &[i32],
        a_scales: &[f32],
        cols: usize,
    ) -> f32 {
        unsafe {
            let t = tables();
            let n_sub = cols / ACT_BLOCK;
            let ap = acts.as_ptr();
            let ip = starts.as_ptr();
            let asc = a_scales.as_ptr();
            let wsp = w_scales.as_ptr();
            let magic = vdupq_n_f32(MAGIC_F);
            // acc[chain][half]: lanes 0..3 and 4..7 of each of the two chains.
            let mut acc = [[vdupq_n_f32(0.0); 2]; 2];
            let mut s = 0usize;
            while s + 20 <= n_sub {
                let base = row.as_ptr().add(s / 5 * BLOCK_BYTES_5);
                let w0 = vld1q_f32(wsp.add(s / 2));
                let w1 = vld1q_f32(wsp.add(s / 2 + 4));
                let w2 = vld1_f32(wsp.add(s / 2 + 8));
                let w2 = vcombine_f32(w2, w2);
                // d[k] lane j = w_scale[(s + 4k + j) / 2] * a_scale[s + 4k + j].
                let d = [
                    vmulq_f32(vzip1q_f32(w0, w0), vld1q_f32(asc.add(s))),
                    vmulq_f32(vzip2q_f32(w0, w0), vld1q_f32(asc.add(s + 4))),
                    vmulq_f32(vzip1q_f32(w1, w1), vld1q_f32(asc.add(s + 8))),
                    vmulq_f32(vzip2q_f32(w1, w1), vld1q_f32(asc.add(s + 12))),
                    vmulq_f32(vzip1q_f32(w2, w2), vld1q_f32(asc.add(s + 16))),
                ];
                let x = |b: usize, h: usize| parts(vld1q_u8(base.add(b * BLOCK_BYTES_5 + 16 * h)), &t);
                // Activation block `s + $i` (plane `$p` of its 160-trit block), half `$h`.
                macro_rules! sub {
                    ($i:expr, $p:literal, $h:literal, $x:ident) => {{
                        const I: usize = $i;
                        let raw = vdotq_s32(
                            vld1q_s32(ip.add((s + I) * 8 + 4 * $h)),
                            plane::<$p>(&$x, &t),
                            vld1q_s8(ap.add((s + I) * ACT_BLOCK + 16 * $h)),
                        );
                        let prod = vsubq_f32(vreinterpretq_f32_s32(raw), magic);
                        let c = &mut acc[(I / 2) & 1][$h];
                        *c = vfmaq_laneq_f32::<{ (I % 4) as i32 }>(*c, prod, d[I / 4]);
                    }};
                }
                macro_rules! half {
                    ($b:literal, $h:literal, $x:ident) => {{
                        sub!(5 * $b, 0, $h, $x);
                        sub!(5 * $b + 1, 1, $h, $x);
                        sub!(5 * $b + 2, 2, $h, $x);
                        sub!(5 * $b + 3, 3, $h, $x);
                        sub!(5 * $b + 4, 4, $h, $x);
                    }};
                }
                let x0 = x(0, 0);
                let x1 = x(0, 1);
                half!(0, 0, x0);
                let x2 = x(1, 0);
                half!(0, 1, x1);
                let x3 = x(1, 1);
                half!(1, 0, x2);
                let x4 = x(2, 0);
                half!(1, 1, x3);
                let x5 = x(2, 1);
                half!(2, 0, x4);
                let x6 = x(3, 0);
                half!(2, 1, x5);
                let x7 = x(3, 1);
                half!(3, 0, x6);
                half!(3, 1, x7);
                s += 20;
            }
            // The remaining (at most 19) activation blocks, with the chain chosen at run time.
            let zero = vdupq_n_s32(0);
            let mut block = s / 5;
            while s < n_sub {
                let base = row.as_ptr().add(block * BLOCK_BYTES_5);
                let lo = decode16(vld1q_u8(base), &t);
                let hi = decode16(vld1q_u8(base.add(16)), &t);
                for p in 0..(n_sub - s).min(5) {
                    let a = ap.add(s * ACT_BLOCK);
                    let p_lo = vcvtq_f32_s32(vdotq_s32(zero, lo[p], vld1q_s8(a)));
                    let p_hi = vcvtq_f32_s32(vdotq_s32(zero, hi[p], vld1q_s8(a.add(16))));
                    let d = vdupq_n_f32(*w_scales.get_unchecked(s / 2) * *a_scales.get_unchecked(s));
                    let c = &mut acc[(s / 2) & 1];
                    c[0] = vfmaq_f32(c[0], p_lo, d);
                    c[1] = vfmaq_f32(c[1], p_hi, d);
                    s += 1;
                }
                block += 1;
            }
            let mut l = [0.0f32; 8];
            vst1q_f32(l.as_mut_ptr(), vaddq_f32(acc[0][0], acc[1][0]));
            vst1q_f32(l.as_mut_ptr().add(4), vaddq_f32(acc[0][1], acc[1][1]));
            hsum8(&l)
        }
    }

    #[cfg(test)]
    pub(super) unsafe fn decode_for_test(bytes: &[u8; 32]) -> [[i8; 32]; 5] {
        unsafe {
            let t = tables();
            let lo = decode16(vld1q_u8(bytes.as_ptr()), &t);
            let hi = decode16(vld1q_u8(bytes.as_ptr().add(16)), &t);
            let mut out = [[0i8; 32]; 5];
            for p in 0..5 {
                vst1q_s8(out[p].as_mut_ptr(), lo[p]);
                vst1q_s8(out[p].as_mut_ptr().add(16), hi[p]);
            }
            out
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pack::pack_matrix;

    #[cfg(target_arch = "aarch64")]
    #[test]
    fn neon_decode_is_exact_for_every_byte() {
        let valid: Vec<u8> = (0..243u32)
            .map(|s| {
                crate::pack::pack_5trits([0, 1, 2, 3, 4].map(|k| ((s / 3u32.pow(k)) % 3) as i8 - 1))
            })
            .collect();
        for start in (0..valid.len()).step_by(32) {
            let mut bytes = [0u8; 32];
            for (j, b) in bytes.iter_mut().enumerate() {
                *b = valid[(start + j) % valid.len()];
            }
            let planes = unsafe { neon::decode_for_test(&bytes) };
            for (j, &b) in bytes.iter().enumerate() {
                for (p, plane) in planes.iter().enumerate() {
                    assert_eq!(plane[j], BYTE_TO_TRITS[b as usize][p], "byte {b} plane {p}");
                }
            }
        }
    }

    fn lcg(seed: &mut u64) -> u64 {
        *seed = seed
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *seed >> 33
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn simd_decode_is_exact_for_every_byte() {
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        // All 243 valid bytes (the radix-28 gap leaves 27, 55, ... unused).
        let valid: Vec<u8> = (0..243u32)
            .map(|s| {
                crate::pack::pack_5trits([0, 1, 2, 3, 4].map(|k| ((s / 3u32.pow(k)) % 3) as i8 - 1))
            })
            .collect();
        for start in (0..valid.len()).step_by(32) {
            let mut bytes = [0u8; 32];
            for (j, b) in bytes.iter_mut().enumerate() {
                *b = valid[(start + j) % valid.len()];
            }
            let planes = unsafe { avx2::decode_for_test(&bytes) };
            for (j, &b) in bytes.iter().enumerate() {
                for (p, plane) in planes.iter().enumerate() {
                    assert_eq!(
                        plane[j] as i8 - 1,
                        BYTE_TO_TRITS[b as usize][p],
                        "byte {b} plane {p}"
                    );
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn avx512_decode_is_exact_for_every_byte() {
        if path() != Path::Avx512 {
            return;
        }
        let valid: Vec<u8> = (0..243u32)
            .map(|s| {
                crate::pack::pack_5trits([0, 1, 2, 3, 4].map(|k| ((s / 3u32.pow(k)) % 3) as i8 - 1))
            })
            .collect();
        for start in (0..valid.len()).step_by(32) {
            let mut bytes = [0u8; 64];
            for (j, b) in bytes.iter_mut().enumerate() {
                *b = valid[(start + j) % valid.len()];
            }
            // The 256-bit decode of the first 32 bytes and the 512-bit decode of all 64 (the helper also checks
            // that both decodes agree on the first 32).
            let planes = unsafe { avx512::decode_for_test(&bytes) };
            for (j, &b) in bytes.iter().enumerate() {
                for p in 0..5 {
                    let code = planes[j / 32][p][j % 32];
                    assert_eq!(code as i8 - 1, BYTE_TO_TRITS[b as usize][p], "byte {b} plane {p}");
                }
            }
        }
    }

    /// The definition, written out with f64-free plain loops over dequantized values, for the integer part, and the
    /// documented accumulation order for the floating-point part.
    fn reference(trits: &[i8], ws: &[f32], acts: &[i8], as_: &[f32], cols: usize) -> f32 {
        let mut acc = [[0.0f32; 8]; 2];
        for s in 0..cols / 32 {
            let d = ws[s / 2] * as_[s];
            for j in 0..8 {
                let p: i32 = (0..4)
                    .map(|k| trits[s * 32 + 4 * j + k] as i32 * acts[s * 32 + 4 * j + k] as i32)
                    .sum();
                acc[(s / 2) & 1][j] = d.mul_add(p as f32, acc[(s / 2) & 1][j]);
            }
        }
        let l: Vec<f32> = (0..8).map(|j| acc[0][j] + acc[1][j]).collect();
        ((l[4] + l[0]) + (l[6] + l[2])) + ((l[5] + l[1]) + (l[7] + l[3]))
    }

    #[test]
    fn gemv_matches_definition_for_all_paths() {
        let mut seed = 7u64;
        for &(rows, cols) in &[
            (1usize, 64usize),
            (3, 128),
            (70, 192),
            (65, 320),
            (33, 2048),
            (130, 2560),
            (17, 640),
            (9, 704),
            (66, 1280),
            (5, 9728),
        ] {
            let trits: Vec<i8> = (0..rows * cols)
                .map(|_| (lcg(&mut seed) % 3) as i8 - 1)
                .collect();
            let packed = pack_matrix(&trits, rows, cols);
            let ws: Vec<f32> = (0..rows * cols / 64)
                .map(|_| (lcg(&mut seed) % 1000) as f32 * 1e-4 + 1e-3)
                .collect();
            let acts: Vec<i8> = (0..cols).map(|_| (lcg(&mut seed) % 255) as i8).collect();
            let as_: Vec<f32> = (0..cols / 32)
                .map(|_| (lcg(&mut seed) % 977) as f32 * 3e-3 + 1e-2)
                .collect();
            let want: Vec<f32> = (0..rows)
                .map(|r| {
                    reference(
                        &trits[r * cols..(r + 1) * cols],
                        &ws[r * cols / 64..(r + 1) * cols / 64],
                        &acts,
                        &as_,
                        cols,
                    )
                })
                .collect();
            let rb = row_bytes(cols);
            for r in 0..rows {
                let got = dot_row_scalar(
                    &packed[r * rb..(r + 1) * rb],
                    &ws[r * cols / 64..],
                    &acts,
                    &as_,
                    cols,
                );
                assert_eq!(
                    got.to_bits(),
                    want[r].to_bits(),
                    "scalar row {r} of {rows}x{cols}"
                );
            }
            let mut out = vec![0.0f32; rows];
            gemv_block_scaled_1xn(&packed, &ws, &acts, &as_, rows, cols, &mut out);
            for r in 0..rows {
                assert_eq!(
                    out[r].to_bits(),
                    want[r].to_bits(),
                    "1xn row {r} of {rows}x{cols}"
                );
            }
            for threads in [2, 4] {
                let mut out = vec![0.0f32; rows];
                gemv_block_scaled_parallel(
                    &packed, &ws, &acts, &as_, rows, cols, &mut out, threads,
                );
                for r in 0..rows {
                    assert_eq!(
                        out[r].to_bits(),
                        want[r].to_bits(),
                        "parallel({threads}) row {r} of {rows}x{cols}"
                    );
                }
            }
        }
    }
}

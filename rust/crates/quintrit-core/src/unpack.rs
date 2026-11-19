//! Load-time conversion of 2-bit weight codes into trits.
//!
//! Several GGUF formats (`I2_S`, `TQ2_0`) store a group of 32 bytes as four bit planes: bits `s..s + 2` of byte `j`
//! are the code of weight `32 * plane + j`. The codes 0, 1, 2 mean `-1, 0, +1`; the unused code 3 decodes to 0. This
//! module turns such groups into one byte per trit. It runs once per tensor while a model is loaded, so it is a plain
//! conversion kernel: AVX2 on x86-64 (chosen at run time), NEON on AArch64, and a portable fallback. All three give
//! identical bytes.

/// Which end of a byte holds the first plane of a group.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PlaneOrder {
    /// Plane 0 is in bits 6..7, plane 3 in bits 0..1 (`I2_S`).
    HighFirst,
    /// Plane 0 is in bits 0..1, plane 3 in bits 6..7 (`TQ2_0`).
    LowFirst,
}

/// Bytes of one group of bit planes.
pub const PLANE_GROUP_BYTES: usize = 32;
/// Trits of one group of bit planes.
pub const PLANE_GROUP_TRITS: usize = 4 * PLANE_GROUP_BYTES;

/// Decodes groups of 32 bytes (`src.len()` must be a multiple of 32) into `4 * src.len()` trits in `out`.
///
/// # Panics
/// If the lengths do not fit.
pub fn decode_2bit_planes(src: &[u8], out: &mut [i8], order: PlaneOrder) {
    assert_eq!(src.len() % PLANE_GROUP_BYTES, 0, "source must be a whole number of 32-byte groups");
    assert_eq!(out.len(), src.len() * 4, "output must hold four trits per source byte");
    match order {
        PlaneOrder::HighFirst => dispatch::<true>(src, out),
        PlaneOrder::LowFirst => dispatch::<false>(src, out),
    }
}

fn dispatch<const HIGH_FIRST: bool>(src: &[u8], out: &mut [i8]) {
    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") {
        // SAFETY: the CPU supports AVX2 and the lengths were checked by the caller.
        unsafe { x86::decode::<HIGH_FIRST>(src, out) };
        return;
    }
    #[cfg(target_arch = "aarch64")]
    {
        // SAFETY: NEON is part of the AArch64 baseline and the lengths were checked by the caller.
        unsafe { arm::decode::<HIGH_FIRST>(src, out) };
        return;
    }
    #[allow(unreachable_code)]
    decode_portable::<HIGH_FIRST>(src, out);
}

/// Shift of plane `p` and the trit of a code.
#[inline(always)]
fn plane_shift<const HIGH_FIRST: bool>(p: usize) -> usize {
    if HIGH_FIRST { 6 - 2 * p } else { 2 * p }
}

#[inline(always)]
fn trit_of(code: u8) -> i8 {
    if code == 3 { 0 } else { code as i8 - 1 }
}

/// Portable implementation (the definition the SIMD versions must reproduce).
pub fn decode_portable<const HIGH_FIRST: bool>(src: &[u8], out: &mut [i8]) {
    for (group, trits) in src.chunks_exact(PLANE_GROUP_BYTES).zip(out.chunks_exact_mut(PLANE_GROUP_TRITS)) {
        for (p, plane) in trits.chunks_exact_mut(PLANE_GROUP_BYTES).enumerate() {
            let shift = plane_shift::<HIGH_FIRST>(p);
            for (w, &byte) in plane.iter_mut().zip(group) {
                *w = trit_of((byte >> shift) & 3);
            }
        }
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    use super::{plane_shift, PLANE_GROUP_BYTES, PLANE_GROUP_TRITS};
    use core::arch::x86_64::*;

    /// # Safety
    /// The CPU must support AVX2; `src.len()` is a multiple of 32 and `out.len() == 4 * src.len()`.
    #[target_feature(enable = "avx2")]
    pub unsafe fn decode<const HIGH_FIRST: bool>(src: &[u8], out: &mut [i8]) {
        // Code -> trit table for `vpshufb` (per 128-bit lane): 0 -> -1, 1 -> 0, 2 -> +1, 3 -> 0.
        let lut = _mm256_setr_epi8(
            -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, -1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
        );
        let mask = _mm256_set1_epi8(3);
        for (group, trits) in src.chunks_exact(PLANE_GROUP_BYTES).zip(out.chunks_exact_mut(PLANE_GROUP_TRITS)) {
            // SAFETY: a group is 32 bytes and four planes of 32 trits follow each other in `trits`.
            unsafe {
                let v = _mm256_loadu_si256(group.as_ptr() as *const __m256i);
                let dst = trits.as_mut_ptr() as *mut __m256i;
                macro_rules! plane {
                    ($p:literal) => {{
                        let shifted = match plane_shift::<HIGH_FIRST>($p) {
                            0 => v,
                            2 => _mm256_srli_epi16::<2>(v),
                            4 => _mm256_srli_epi16::<4>(v),
                            _ => _mm256_srli_epi16::<6>(v),
                        };
                        let code = _mm256_and_si256(shifted, mask);
                        _mm256_storeu_si256(dst.add($p), _mm256_shuffle_epi8(lut, code));
                    }};
                }
                plane!(0);
                plane!(1);
                plane!(2);
                plane!(3);
            }
        }
    }
}

#[cfg(target_arch = "aarch64")]
mod arm {
    use super::{plane_shift, PLANE_GROUP_BYTES, PLANE_GROUP_TRITS};
    use core::arch::aarch64::*;

    /// # Safety
    /// `src.len()` is a multiple of 32 and `out.len() == 4 * src.len()`. NEON is part of the AArch64 baseline.
    #[target_feature(enable = "neon")]
    pub unsafe fn decode<const HIGH_FIRST: bool>(src: &[u8], out: &mut [i8]) {
        // Code -> trit table for `vqtbl1q`: 0 -> -1, 1 -> 0, 2 -> +1, 3 -> 0.
        let table: [i8; 16] = [-1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0];
        let lut = unsafe { vld1q_s8(table.as_ptr()) };
        let mask = vdupq_n_u8(3);
        for (group, trits) in src.chunks_exact(PLANE_GROUP_BYTES).zip(out.chunks_exact_mut(PLANE_GROUP_TRITS)) {
            for half in 0..2 {
                // SAFETY: 16 source bytes at `16 * half` and 16 output bytes per plane at `32 * p + 16 * half`.
                unsafe {
                    let v = vld1q_u8(group.as_ptr().add(16 * half));
                    let dst = trits.as_mut_ptr().add(16 * half);
                    macro_rules! plane {
                        ($p:literal) => {{
                            let shifted = match plane_shift::<HIGH_FIRST>($p) {
                                0 => v,
                                2 => vshrq_n_u8::<2>(v),
                                4 => vshrq_n_u8::<4>(v),
                                _ => vshrq_n_u8::<6>(v),
                            };
                            let code = vandq_u8(shifted, mask);
                            vst1q_s8(dst.add(32 * $p), vqtbl1q_s8(lut, code));
                        }};
                    }
                    plane!(0);
                    plane!(1);
                    plane!(2);
                    plane!(3);
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Every byte value in every position, so that all 256 byte values and all 4 codes of each plane are covered.
    fn all_bytes() -> Vec<u8> {
        (0..256 * 32).map(|i| ((i / 32 + i % 32 * 7) % 256) as u8).collect()
    }

    #[test]
    fn dispatched_decoder_matches_the_portable_one_and_the_definition() {
        let src = all_bytes();
        for order in [PlaneOrder::HighFirst, PlaneOrder::LowFirst] {
            let mut got = vec![0i8; src.len() * 4];
            decode_2bit_planes(&src, &mut got, order);
            let mut portable = vec![0i8; src.len() * 4];
            match order {
                PlaneOrder::HighFirst => decode_portable::<true>(&src, &mut portable),
                PlaneOrder::LowFirst => decode_portable::<false>(&src, &mut portable),
            }
            assert_eq!(got, portable, "{order:?}");
            // Independent check of the definition for every weight.
            for (g, group) in src.chunks_exact(32).enumerate() {
                for w in 0..128 {
                    let (plane, j) = (w / 32, w % 32);
                    let shift = if order == PlaneOrder::HighFirst { 6 - 2 * plane } else { 2 * plane };
                    let want = match (group[j] >> shift) & 3 {
                        0 => -1,
                        2 => 1,
                        _ => 0,
                    };
                    assert_eq!(got[g * 128 + w], want, "{order:?} group {g} weight {w}");
                }
            }
        }
    }

    #[test]
    #[should_panic]
    fn rejects_partial_groups() {
        decode_2bit_planes(&[0u8; 31], &mut [0i8; 124], PlaneOrder::HighFirst);
    }
}

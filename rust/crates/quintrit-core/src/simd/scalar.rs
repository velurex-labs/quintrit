//! Portable scalar reference kernel for GEMV.
//!
//! Evaluates a 32-row block of weights without architecture-specific SIMD intrinsics.

/// Evaluates a 32-row block of weights against the precomputed Dual-Hash QLUT in pure scalar Rust.
///
/// # Safety
/// Pointers `block_weights`, `qlut`, and `out_32` must be valid, properly aligned,
/// and have sufficient memory allocated for `32 * bytes_per_row`, `bytes_per_row * 65`,
/// and `32 * sizeof(f32)` respectively.
pub unsafe fn gemv_lut_block32_scalar(
    block_weights: *const u8,
    qlut: *const i8,
    bytes_per_row: usize,
    out_32: *mut f32,
) {
    let a2_ptr = unsafe { qlut.add(bytes_per_row * 64) };

    for r in 0..32 {
        let mut row_acc: i32 = 0;

        for c in 0..bytes_per_row {
            let x = unsafe { *block_weights.add(c * 32 + r) };

            // Dual-Hash and ALU extraction exactly matching hardware SIMD:
            let v11 = (x >> 2) & 0x3F;
            let v15 = (v11 >> 3) & 0x1F;
            let v17 = ((v11 as u16 + v15 as u16 + 1) >> 1) as u8;
            let h34 = ((v17 >> 2) & 0x0F) as usize;
            let p34 = unsafe { *qlut.add(c * 64 + h34) };

            let w2 = x.wrapping_add(x);
            let diff1 = w2.wrapping_sub(v11);
            let diff2 = diff1.wrapping_sub(h34 as u8);
            let h01 = (diff2 & 0x0F) as usize;
            let p01 = unsafe { *qlut.add(c * 64 + 32 + h01) };

            let xm1 = x.wrapping_sub(1);
            let u = (((xm1 as u16 + v17 as u16 + 1) >> 1) as u8) & 0x0F;
            let t2 = if u < 5 {
                -1i8
            } else if u > 9 {
                1i8
            } else {
                0i8
            };

            let a2 = unsafe { *a2_ptr.add(c) };
            let p2 = if t2 < 0 {
                -a2
            } else if t2 > 0 {
                a2
            } else {
                0
            };

            row_acc += p34 as i32 + p01 as i32 + p2 as i32;
        }

        unsafe {
            *out_32.add(r) = row_acc as f32;
        }
    }
}

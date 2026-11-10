//! Limits shared by the language bindings (C ABI, Python) so that they cannot drift apart.

/// Largest column count for which the GEMV kernels return exact integer sums.
///
/// The kernels accumulate in `i32` and convert the sums to `f32`, which is exact while the magnitude stays below
/// `2^24`. With full-range int8 activations the largest magnitude is `127 * cols`, and `127 * 131072 < 2^24`.
pub const MAX_GEMV_COLS: usize = 131_072;

/// Largest worker thread count a binding accepts. Every distinct thread count creates a persistent pool that lives
/// until the process ends, so the bindings cap the value.
pub const MAX_THREADS: usize = 256;

/// Returns `true` when a GEMV over `cols` columns is guaranteed to be exact.
#[inline]
pub const fn gemv_cols_supported(cols: usize) -> bool {
    cols <= MAX_GEMV_COLS
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn max_gemv_cols_keeps_the_accumulator_exact() {
        assert!(127 * MAX_GEMV_COLS < 1 << 24);
        assert!(128 * MAX_GEMV_COLS <= 1 << 24);
        assert!(gemv_cols_supported(MAX_GEMV_COLS));
        assert!(!gemv_cols_supported(MAX_GEMV_COLS + 1));
    }
}

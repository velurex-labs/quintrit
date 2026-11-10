//! Tests of the kernel-level C API against brute-force ground truth, including every error path.
//!
//! The ground truth is plain scalar code on standard CPU operations: a simple loop over the unpacked trits and
//! activations with `i64` accumulation, without SIMD and without the library's packing, tiling or lookup tables.

mod common;

use common::*;
use quintrit_c::*;
use std::ptr::{null, null_mut};

const ALL_KERNELS: [i32; 3] = [QUINTRIT_KERNEL_SCALAR, QUINTRIT_KERNEL_SIMD32, QUINTRIT_KERNEL_AVX512];

fn supported(kernel: i32) -> bool {
    let mut s = -1;
    expect_status(unsafe { quintrit_kernel_supported(kernel, &mut s) }, QUINTRIT_OK);
    s == 1
}

fn tiled_size(kernel: i32, rows: usize, cols: usize) -> usize {
    let mut n = 0usize;
    expect_status(unsafe { quintrit_tiled_size(kernel, rows, cols, &mut n) }, QUINTRIT_OK);
    n
}

fn pack(kernel: i32, trits: &[i8], rows: usize, cols: usize) -> Vec<u8> {
    let mut out = vec![0u8; tiled_size(kernel, rows, cols)];
    expect_status(
        unsafe { quintrit_pack_matrix(kernel, trits.as_ptr(), trits.len(), rows, cols, out.as_mut_ptr(), out.len()) },
        QUINTRIT_OK,
    );
    out
}

/// Shapes with full tiles, remainder rows, remainder columns, partial 160-trit blocks and degenerate sizes.
const SHAPES: &[(usize, usize)] = &[
    (1, 1),
    (3, 5),
    (5, 7),
    (32, 160),
    (33, 161),
    (31, 159),
    (64, 320),
    (65, 250),
    (96, 3072),
    (100, 1000),
    (130, 640),
    (192, 33),
];

#[test]
fn version_status_and_error_slot() {
    assert_eq!(quintrit_abi_version(), QUINTRIT_ABI_VERSION);
    assert_eq!(from_c(quintrit_version()), env!("CARGO_PKG_VERSION"));
    for code in 0..=QUINTRIT_ERR_INVALID_HANDLE {
        assert!(!from_c(quintrit_status_string(code)).is_empty());
    }
    assert_eq!(from_c(quintrit_status_string(12345)), "unknown status code");

    // A failing call sets the slot, the next successful call clears it.
    let mut n = 0usize;
    expect_status(unsafe { quintrit_tiled_size(QUINTRIT_KERNEL_SIMD32, 0, 5, &mut n) }, QUINTRIT_ERR_INVALID_SHAPE);
    expect_status(unsafe { quintrit_tiled_size(QUINTRIT_KERNEL_SIMD32, 5, 5, &mut n) }, QUINTRIT_OK);

    // The slot is per thread.
    expect_status(unsafe { quintrit_tiled_size(QUINTRIT_KERNEL_SIMD32, 0, 5, &mut n) }, QUINTRIT_ERR_INVALID_SHAPE);
    std::thread::spawn(|| assert_eq!(quintrit_last_error_code(), 0)).join().unwrap();
    assert_eq!(quintrit_last_error_code(), QUINTRIT_ERR_INVALID_SHAPE);
}

#[test]
fn cpu_features_and_kernel_queries_are_consistent() {
    let mut mask = 0u64;
    expect_status(unsafe { quintrit_cpu_features(&mut mask) }, QUINTRIT_OK);
    #[cfg(target_arch = "x86_64")]
    {
        assert_eq!(mask & QUINTRIT_CPU_AVX2 != 0, is_x86_feature_detected!("avx2"));
        assert_eq!(mask & QUINTRIT_CPU_AVX512VBMI != 0, is_x86_feature_detected!("avx512vbmi"));
        assert_eq!(mask & QUINTRIT_CPU_NEON, 0);
    }
    #[cfg(target_arch = "aarch64")]
    assert_ne!(mask & QUINTRIT_CPU_NEON, 0);

    // AUTO and SCALAR are always supported.
    assert!(supported(QUINTRIT_KERNEL_AUTO));
    assert!(supported(QUINTRIT_KERNEL_SCALAR));
    assert_eq!(supported(QUINTRIT_KERNEL_SIMD32), mask & (QUINTRIT_CPU_AVX2 | QUINTRIT_CPU_NEON) != 0);
    assert_eq!(
        supported(QUINTRIT_KERNEL_AVX512),
        mask & (QUINTRIT_CPU_AVX512F | QUINTRIT_CPU_AVX512BW | QUINTRIT_CPU_AVX512VBMI)
            == (QUINTRIT_CPU_AVX512F | QUINTRIT_CPU_AVX512BW | QUINTRIT_CPU_AVX512VBMI)
    );

    // The supported list is best first, ends with SCALAR and agrees with the single queries.
    let mut count = 0usize;
    assert_eq!(unsafe { quintrit_supported_kernels(null_mut(), 0, &mut count) }, QUINTRIT_ERR_BUFFER_TOO_SMALL);
    let mut list = vec![0i32; count];
    expect_status(unsafe { quintrit_supported_kernels(list.as_mut_ptr(), list.len(), &mut count) }, QUINTRIT_OK);
    eprintln!("usable kernels on this CPU: {:?}", list.iter().map(|&k| from_c(quintrit_kernel_name(k))).collect::<Vec<_>>());
    assert_eq!(list.last(), Some(&QUINTRIT_KERNEL_SCALAR));
    for k in ALL_KERNELS {
        assert_eq!(list.contains(&k), supported(k), "kernel {k}");
    }
    let mut resolved = 0i32;
    expect_status(unsafe { quintrit_kernel_resolve(QUINTRIT_KERNEL_AUTO, &mut resolved) }, QUINTRIT_OK);
    assert_eq!(resolved, list[0]);
    expect_status(unsafe { quintrit_kernel_resolve(QUINTRIT_KERNEL_AVX512, &mut resolved) }, QUINTRIT_OK);
    assert_eq!(resolved, QUINTRIT_KERNEL_AVX512);

    let mut tile = 0usize;
    for (k, expected) in [(QUINTRIT_KERNEL_SCALAR, 32), (QUINTRIT_KERNEL_SIMD32, 32), (QUINTRIT_KERNEL_AVX512, 64)] {
        expect_status(unsafe { quintrit_kernel_tile_rows(k, &mut tile) }, QUINTRIT_OK);
        assert_eq!(tile, expected);
    }
    assert_eq!(from_c(quintrit_kernel_name(QUINTRIT_KERNEL_AVX512)), "avx512");
    assert!(quintrit_kernel_name(99).is_null());

    let mut threads = 0u32;
    expect_status(unsafe { quintrit_default_threads(&mut threads) }, QUINTRIT_OK);
    assert!((1..=QUINTRIT_MAX_THREADS).contains(&threads));

    // Errors.
    let mut s = 0;
    expect_status(unsafe { quintrit_kernel_supported(77, &mut s) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    expect_status(unsafe { quintrit_kernel_supported(1, null_mut()) }, QUINTRIT_ERR_NULL_POINTER);
    expect_status(unsafe { quintrit_cpu_features(null_mut()) }, QUINTRIT_ERR_NULL_POINTER);
}

#[test]
fn layout_matches_core_for_every_kernel() {
    for k in ALL_KERNELS {
        for &(rows, cols) in SHAPES {
            let mut l = quintrit_c::TiledLayout { struct_size: size_of::<quintrit_c::TiledLayout>() as u32, ..Default::default() };
            expect_status(unsafe { quintrit_tiled_layout(k, rows, cols, &mut l) }, QUINTRIT_OK);
            let tile = if k == QUINTRIT_KERNEL_AVX512 { 64 } else { 32 };
            assert_eq!(l.tile_rows, tile);
            assert_eq!(l.padded_rows as usize, rows.next_multiple_of(tile as usize));
            assert_eq!(l.bytes_per_row as usize, quintrit_core::pack::tile_row_bytes(cols));
            assert_eq!(l.total_bytes, l.padded_rows * l.bytes_per_row);
            assert_eq!(l.kernel, k);
            assert_eq!(tiled_size(k, rows, cols) as u64, l.total_bytes);
        }
    }
    // AUTO resolves inside the layout.
    let mut l = quintrit_c::TiledLayout { struct_size: size_of::<quintrit_c::TiledLayout>() as u32, ..Default::default() };
    expect_status(unsafe { quintrit_tiled_layout(QUINTRIT_KERNEL_AUTO, 40, 40, &mut l) }, QUINTRIT_OK);
    assert_ne!(l.kernel, QUINTRIT_KERNEL_AUTO);
}

#[test]
fn pack_unpack_round_trip_and_agree_with_core_layout() {
    let mut rng = Rng::new(1);
    for k in ALL_KERNELS {
        for &(rows, cols) in SHAPES {
            let trits = rng.trits(rows * cols);
            let tiled = pack(k, &trits, rows, cols);

            // Reference: pad the rows with zero weights, pack with core, tile with core.
            let tile = if k == QUINTRIT_KERNEL_AVX512 { 64 } else { 32 };
            let padded_rows = rows.next_multiple_of(tile);
            let mut padded = trits.clone();
            padded.resize(padded_rows * cols, 0);
            let planar = quintrit_core::pack::pack_matrix(&padded, padded_rows, cols);
            let expected = if tile == 32 {
                quintrit_core::lut::transform_weights_lut(&planar, padded_rows, cols)
            } else {
                #[cfg(target_arch = "x86_64")]
                {
                    quintrit_core::lut::transform_weights_lut_64(&planar, padded_rows, cols)
                }
                #[cfg(not(target_arch = "x86_64"))]
                {
                    Vec::new()
                }
            };
            if !expected.is_empty() {
                assert_eq!(tiled, expected, "layout of kernel {k} for {rows}x{cols}");
            }
            // Padding rows hold zero weights, so the byte of an all-zero group (125) fills them.
            assert!(tiled.iter().all(|&b| b <= 250));

            let mut back = vec![9i8; rows * cols];
            expect_status(
                unsafe { quintrit_unpack_matrix(k, tiled.as_ptr(), tiled.len(), rows, cols, back.as_mut_ptr(), back.len()) },
                QUINTRIT_OK,
            );
            assert_eq!(back, trits, "round trip of kernel {k} for {rows}x{cols}");
        }
    }
    // The 32-row layouts of SCALAR and SIMD32 are identical.
    let trits = rng.trits(40 * 300);
    assert_eq!(pack(QUINTRIT_KERNEL_SCALAR, &trits, 40, 300), pack(QUINTRIT_KERNEL_SIMD32, &trits, 40, 300));
}

fn check_gemv(kernel: i32, rows: usize, cols: usize, threads: u32, rng: &mut Rng) {
    let trits = rng.trits(rows * cols);
    let acts = rng.acts(cols);
    let tiled = pack(kernel, &trits, rows, cols);
    let expected = truth(&trits, &acts, rows, cols);

    let mut out = vec![i32::MIN; rows];
    expect_status(
        unsafe {
            quintrit_gemv_i32(kernel, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), threads)
        },
        QUINTRIT_OK,
    );
    let got: Vec<i64> = out.iter().map(|&v| v as i64).collect();
    assert_eq!(got, expected, "gemv_i32 kernel {kernel} {rows}x{cols} threads {threads}");

    let mut outf = vec![f32::NAN; rows];
    expect_status(
        unsafe {
            quintrit_gemv_f32(kernel, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), 0.25, outf.as_mut_ptr(), outf.len(), threads)
        },
        QUINTRIT_OK,
    );
    let want: Vec<f32> = expected.iter().map(|&v| v as f32 * 0.25).collect();
    assert_eq!(outf, want, "gemv_f32 kernel {kernel} {rows}x{cols} threads {threads}");
}

#[test]
fn gemv_matches_brute_force_for_every_supported_kernel() {
    let mut rng = Rng::new(2);
    for k in ALL_KERNELS {
        if !supported(k) {
            continue;
        }
        for &(rows, cols) in SHAPES {
            for threads in [1u32, 0, 3] {
                check_gemv(k, rows, cols, threads, &mut rng);
            }
        }
        // A realistic projection shape: many tiles, a partial last block, more threads than tiles on a small matrix.
        check_gemv(k, 512, 3072, 4, &mut rng);
        check_gemv(k, 96, 6000, 16, &mut rng);
    }
    // AUTO runs the best kernel and must agree.
    check_gemv(QUINTRIT_KERNEL_AUTO, 70, 777, 2, &mut rng);
}

#[test]
fn gemv_is_exact_for_extreme_activations() {
    for k in ALL_KERNELS {
        if !supported(k) {
            continue;
        }
        let (rows, cols) = (64usize, 1000usize);
        for fill in [i8::MIN, i8::MAX, 0, -1, 1] {
            // All weights +1 and -1 against a constant activation: the largest possible sums.
            for w in [1i8, -1i8] {
                let trits = vec![w; rows * cols];
                let acts = vec![fill; cols];
                let tiled = pack(k, &trits, rows, cols);
                let mut out = vec![0i32; rows];
                expect_status(
                    unsafe { quintrit_gemv_i32(k, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 1) },
                    QUINTRIT_OK,
                );
                assert!(out.iter().all(|&v| v == w as i32 * fill as i32 * cols as i32), "kernel {k} fill {fill} w {w}");
            }
        }
    }
}

#[test]
fn pack_and_unpack_error_paths() {
    let k = QUINTRIT_KERNEL_SIMD32;
    let trits = vec![0i8; 40 * 40];
    let mut out = vec![0u8; tiled_size(k, 40, 40)];
    let pack_call = |kernel: i32, t: *const i8, tl: usize, rows: usize, cols: usize, o: *mut u8, ol: usize| unsafe {
        quintrit_pack_matrix(kernel, t, tl, rows, cols, o, ol)
    };

    // Null pointers.
    expect_status(pack_call(k, null(), trits.len(), 40, 40, out.as_mut_ptr(), out.len()), QUINTRIT_ERR_NULL_POINTER);
    expect_status(pack_call(k, trits.as_ptr(), trits.len(), 40, 40, null_mut(), out.len()), QUINTRIT_ERR_NULL_POINTER);
    // Shapes.
    for (rows, cols) in [(0usize, 40usize), (40, 0), (usize::MAX, 40), (40, usize::MAX), (1 << 32, 1 << 32), ((1 << 31) + 1, 1)] {
        expect_status(pack_call(k, trits.as_ptr(), trits.len(), rows, cols, out.as_mut_ptr(), out.len()), QUINTRIT_ERR_INVALID_SHAPE);
    }
    // Buffers that are too small, never read or written beyond their length.
    expect_status(pack_call(k, trits.as_ptr(), trits.len() - 1, 40, 40, out.as_mut_ptr(), out.len()), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    expect_status(pack_call(k, trits.as_ptr(), trits.len(), 40, 40, out.as_mut_ptr(), out.len() - 1), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    expect_status(pack_call(k, trits.as_ptr(), 0, 40, 40, out.as_mut_ptr(), out.len()), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    // Kernel ids.
    expect_status(pack_call(42, trits.as_ptr(), trits.len(), 40, 40, out.as_mut_ptr(), out.len()), QUINTRIT_ERR_INVALID_ARGUMENT);
    expect_status(pack_call(-1, trits.as_ptr(), trits.len(), 40, 40, out.as_mut_ptr(), out.len()), QUINTRIT_ERR_INVALID_ARGUMENT);
    // Values outside {-1, 0, 1}.
    for bad in [2i8, -2, 127, -128] {
        let mut t = trits.clone();
        t[7 * 40 + 3] = bad;
        expect_status(pack_call(k, t.as_ptr(), t.len(), 40, 40, out.as_mut_ptr(), out.len()), QUINTRIT_ERR_INVALID_VALUE);
        assert!(last_error().1.contains("row 7, column 3"), "{:?}", last_error());
    }
    // Overlap between input and output.
    let mut shared = vec![0u8; 4000];
    let p = shared.as_mut_ptr();
    expect_status(pack_call(k, p as *const i8, 1600, 40, 40, p, 4000), QUINTRIT_ERR_INVALID_ARGUMENT);
    // One row pads to a whole tile: the AVX-512 layout is twice the 32-row layout.
    assert!(tiled_size(QUINTRIT_KERNEL_AVX512, 1, 40) == 2 * tiled_size(QUINTRIT_KERNEL_SIMD32, 1, 40));

    // Unpack.
    let tiled = pack(k, &trits, 40, 40);
    let mut back = vec![0i8; 40 * 40];
    let unpack_call = |t: *const u8, tl: usize, rows: usize, cols: usize, o: *mut i8, ol: usize| unsafe {
        quintrit_unpack_matrix(k, t, tl, rows, cols, o, ol)
    };
    expect_status(unpack_call(tiled.as_ptr(), tiled.len(), 40, 40, back.as_mut_ptr(), back.len()), QUINTRIT_OK);
    expect_status(unpack_call(null(), tiled.len(), 40, 40, back.as_mut_ptr(), back.len()), QUINTRIT_ERR_NULL_POINTER);
    expect_status(unpack_call(tiled.as_ptr(), tiled.len(), 40, 40, null_mut(), back.len()), QUINTRIT_ERR_NULL_POINTER);
    expect_status(unpack_call(tiled.as_ptr(), tiled.len() - 1, 40, 40, back.as_mut_ptr(), back.len()), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    expect_status(unpack_call(tiled.as_ptr(), tiled.len(), 40, 40, back.as_mut_ptr(), back.len() - 1), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    expect_status(unpack_call(tiled.as_ptr(), tiled.len(), 0, 40, back.as_mut_ptr(), back.len()), QUINTRIT_ERR_INVALID_SHAPE);
    for bad in [251u8, 255] {
        let mut t = tiled.clone();
        t[17] = bad;
        expect_status(unpack_call(t.as_ptr(), t.len(), 40, 40, back.as_mut_ptr(), back.len()), QUINTRIT_ERR_INVALID_VALUE);
    }
}

#[test]
fn gemv_error_paths() {
    let (rows, cols) = (64usize, 320usize);
    let mut rng = Rng::new(3);
    let trits = rng.trits(rows * cols);
    let acts = rng.acts(cols);
    let k = QUINTRIT_KERNEL_SCALAR;
    let tiled = pack(k, &trits, rows, cols);
    let mut out = vec![0i32; rows];
    let call = |kernel: i32, w: *const u8, wl: usize, r: usize, c: usize, a: *const i8, al: usize, o: *mut i32, ol: usize, th: u32| unsafe {
        quintrit_gemv_i32(kernel, w, wl, r, c, a, al, o, ol, th)
    };
    let ok = |o: &mut Vec<i32>| call(k, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), o.as_mut_ptr(), o.len(), 1);
    expect_status(ok(&mut out), QUINTRIT_OK);

    expect_status(call(k, null(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 1), QUINTRIT_ERR_NULL_POINTER);
    expect_status(call(k, tiled.as_ptr(), tiled.len(), rows, cols, null(), acts.len(), out.as_mut_ptr(), out.len(), 1), QUINTRIT_ERR_NULL_POINTER);
    expect_status(call(k, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), null_mut(), out.len(), 1), QUINTRIT_ERR_NULL_POINTER);
    expect_status(call(k, tiled.as_ptr(), tiled.len() - 1, rows, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 1), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    expect_status(call(k, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len() - 1, out.as_mut_ptr(), out.len(), 1), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    expect_status(call(k, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), rows - 1, 1), QUINTRIT_ERR_BUFFER_TOO_SMALL);
    expect_status(call(k, tiled.as_ptr(), tiled.len(), 0, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 1), QUINTRIT_ERR_INVALID_SHAPE);
    expect_status(call(k, tiled.as_ptr(), tiled.len(), rows, 0, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 1), QUINTRIT_ERR_INVALID_SHAPE);
    // More columns than the exactness limit.
    expect_status(call(k, tiled.as_ptr(), tiled.len(), rows, 131_073, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 1), QUINTRIT_ERR_INVALID_SHAPE);
    expect_status(call(k, tiled.as_ptr(), tiled.len(), usize::MAX, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 1), QUINTRIT_ERR_INVALID_SHAPE);
    expect_status(call(99, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 1), QUINTRIT_ERR_INVALID_ARGUMENT);
    // Thread counts.
    expect_status(call(k, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), QUINTRIT_MAX_THREADS + 1), QUINTRIT_ERR_INVALID_ARGUMENT);
    expect_status(call(k, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), u32::MAX), QUINTRIT_ERR_INVALID_ARGUMENT);

    // Misaligned output (int32 needs 4-byte alignment).
    let mut raw = vec![0u8; rows * 4 + 8];
    let misaligned = unsafe { raw.as_mut_ptr().add(1 + (raw.as_ptr() as usize).wrapping_neg() % 4) } as *mut i32;
    assert_ne!(misaligned as usize % 4, 0);
    expect_status(call(k, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), misaligned, rows, 1), QUINTRIT_ERR_INVALID_ARGUMENT);

    // An unsupported kernel is reported, not run.
    for kernel in [QUINTRIT_KERNEL_SIMD32, QUINTRIT_KERNEL_AVX512] {
        if !supported(kernel) {
            let t = pack(kernel, &trits, rows, cols);
            expect_status(call(kernel, t.as_ptr(), t.len(), rows, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 1), QUINTRIT_ERR_UNSUPPORTED);
        }
    }
    // Weights packed for the 64-row layout are large enough for a 32-row run only if the size check allows it;
    // a buffer shorter than the layout never passes.
    let short = [125u8; 10];
    expect_status(call(k, short.as_ptr(), short.len(), rows, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 1), QUINTRIT_ERR_BUFFER_TOO_SMALL);
}

#[test]
fn layout_struct_size_is_checked() {
    let mut l = quintrit_c::TiledLayout::default();
    expect_status(unsafe { quintrit_tiled_layout(QUINTRIT_KERNEL_SIMD32, 4, 4, &mut l) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    l.struct_size = 4;
    expect_status(unsafe { quintrit_tiled_layout(QUINTRIT_KERNEL_SIMD32, 4, 4, &mut l) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    l.struct_size = size_of::<quintrit_c::TiledLayout>() as u32;
    expect_status(unsafe { quintrit_tiled_layout(QUINTRIT_KERNEL_SIMD32, 4, 4, &mut l) }, QUINTRIT_OK);
    expect_status(unsafe { quintrit_tiled_layout(QUINTRIT_KERNEL_SIMD32, 4, 4, null_mut()) }, QUINTRIT_ERR_NULL_POINTER);
}

#[cfg(panic = "unwind")]
#[test]
fn a_panic_inside_an_entry_point_becomes_an_error_code() {
    let status = quintrit_c::error::guard(|| {
        let v: Vec<i32> = Vec::new();
        let i = std::hint::black_box(3usize);
        let _ = v[i];
        Ok(())
    });
    assert_eq!(status, QUINTRIT_ERR_INTERNAL);
    let (code, msg) = last_error();
    assert_eq!(code, QUINTRIT_ERR_INTERNAL);
    assert!(msg.contains("panic"), "{msg}");
}

#[test]
fn concurrent_gemv_calls_from_several_threads() {
    let (rows, cols) = (256usize, 1000usize);
    let mut rng = Rng::new(9);
    let trits = rng.trits(rows * cols);
    let k = QUINTRIT_KERNEL_AUTO;
    let tiled = pack(k, &trits, rows, cols);
    std::thread::scope(|s| {
        for t in 0..4u64 {
            let (trits, tiled) = (&trits, &tiled);
            s.spawn(move || {
                let mut rng = Rng::new(100 + t);
                for _ in 0..5 {
                    let acts = rng.acts(cols);
                    let mut out = vec![0i32; rows];
                    expect_status(
                        unsafe { quintrit_gemv_i32(k, tiled.as_ptr(), tiled.len(), rows, cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 2) },
                        QUINTRIT_OK,
                    );
                    let want = truth(trits, &acts, rows, cols);
                    assert_eq!(out.iter().map(|&v| v as i64).collect::<Vec<_>>(), want);
                }
            });
        }
    });
}

/// The C limits are the ones defined in `quintrit-core`, which the Python binding uses as well.
#[test]
fn limits_come_from_the_core_crate() {
    assert_eq!(QUINTRIT_MAX_THREADS as usize, quintrit_core::MAX_THREADS);
    assert_eq!(quintrit_c::tiles::MAX_GEMV_COLS, quintrit_core::MAX_GEMV_COLS);
}

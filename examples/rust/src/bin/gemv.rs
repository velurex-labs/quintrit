//! Kernel level in Rust: pack a ternary matrix, multiply it with int8 activations and check the result against a
//! plain integer product.
//!
//! ```text
//! cargo run --release --bin gemv [rows] [cols]      (defaults: 512 x 2048; rows must be a multiple of 32)
//! ```

use quintrit_core::lut::{build_activation_lut_wide, gemv_lut_wide_1xn, transform_weights_lut};
use quintrit_core::pack::pack_matrix;

/// A small deterministic generator keeps the example free of dependencies.
struct XorShift(u64);

impl XorShift {
    fn next(&mut self) -> u32 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        (self.0.wrapping_mul(0x2545_F491_4F6C_DD1D) >> 32) as u32
    }
}

fn main() {
    let mut args = std::env::args().skip(1);
    let rows: usize = args.next().map_or(512, |a| a.parse().expect("rows"));
    let cols: usize = args.next().map_or(2048, |a| a.parse().expect("cols"));
    assert_eq!(rows % 32, 0, "rows must be a multiple of 32");

    // Ternary weights in {-1, 0, +1} (row-major) and int8 activations over the whole range.
    let mut rng = XorShift(0x243F_6A88_85A3_08D3);
    let trits: Vec<i8> = (0..rows * cols).map(|_| (rng.next() % 3) as i8 - 1).collect();
    let activations: Vec<i8> = (0..cols).map(|_| rng.next() as u8 as i8).collect();

    // 1. Pack five trits per byte (1.60 bits per weight), then arrange the bytes in 32-row tiles.
    let packed = pack_matrix(&trits, rows, cols);
    let tiled = transform_weights_lut(&packed, rows, cols);
    println!(
        "{} x {} trits: {} bytes as int8, {} bytes tiled ({:.2} bits per weight)",
        rows,
        cols,
        rows * cols,
        tiled.len(),
        8.0 * tiled.len() as f64 / (rows * cols) as f64
    );

    // 2. Tabulate the activations once; the "wide" table makes the product exact for every int8 value.
    let table = build_activation_lut_wide(&activations, cols);

    // 3. The matrix-vector product: table lookups instead of multiplications.
    let mut out = vec![0.0f32; rows];
    gemv_lut_wide_1xn(&tiled, &table, rows, cols, &mut out);

    // The result equals the integer product (an f32 holds integers up to 2^24 exactly).
    let mismatches = (0..rows)
        .filter(|&r| {
            let sum: i64 = (0..cols).map(|c| i64::from(trits[r * cols + c]) * i64::from(activations[c])).sum();
            sum as f32 != out[r]
        })
        .count();
    println!("first rows: {:?}", &out[..4]);
    println!(
        "{} ({} mismatching rows)",
        if mismatches == 0 { "OK: matches the integer product" } else { "MISMATCH" },
        mismatches
    );
    std::process::exit(i32::from(mismatches != 0));
}

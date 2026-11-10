//! Helpers shared by the integration tests of the C API.
#![allow(dead_code)]

pub mod synth;

use std::ffi::{c_char, CStr, CString};

/// Small deterministic generator (xorshift64*), so the tests need no external crates.
pub struct Rng(u64);

impl Rng {
    pub fn new(seed: u64) -> Self {
        Self(seed.wrapping_mul(0x9E37_79B9_7F4A_7C15) | 1)
    }

    pub fn next(&mut self) -> u64 {
        self.0 ^= self.0 >> 12;
        self.0 ^= self.0 << 25;
        self.0 ^= self.0 >> 27;
        self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
    }

    pub fn trits(&mut self, n: usize) -> Vec<i8> {
        (0..n).map(|_| (self.next() % 3) as i8 - 1).collect()
    }

    /// Activations over the full int8 range, with both extremes present.
    pub fn acts(&mut self, n: usize) -> Vec<i8> {
        (0..n)
            .map(|i| match i % 7 {
                0 => i8::MIN,
                1 => i8::MAX,
                _ => (self.next() & 0xFF) as u8 as i8,
            })
            .collect()
    }
}

/// Tells the user, once per test binary, what the ground truth is. It writes to the raw standard error stream because
/// the test harness swallows `println!` and `eprintln!` output of passing tests.
fn announce_ground_truth() {
    static ONCE: std::sync::Once = std::sync::Once::new();
    ONCE.call_once(|| {
        use std::io::Write;
        let _ = std::io::stderr().write_all(
            b"\nGround truth: brute-force product in plain scalar CPU code (standard integer operations, no SIMD).\n\
              Every SIMD kernel is compared with it and must return exactly the same integers.\n\n",
        );
    });
}

/// Brute-force `W x` in 64-bit integers.
pub fn truth(trits: &[i8], acts: &[i8], rows: usize, cols: usize) -> Vec<i64> {
    announce_ground_truth();
    (0..rows)
        .map(|r| (0..cols).map(|c| trits[r * cols + c] as i64 * acts[c] as i64).sum())
        .collect()
}

pub fn last_error() -> (i32, String) {
    let code = quintrit_c::quintrit_last_error_code();
    let msg = unsafe { CStr::from_ptr(quintrit_c::quintrit_last_error_message()) }.to_string_lossy().into_owned();
    (code, msg)
}

pub fn cstr(s: &str) -> CString {
    CString::new(s).unwrap()
}

pub fn from_c(p: *const c_char) -> String {
    unsafe { CStr::from_ptr(p) }.to_string_lossy().into_owned()
}

/// Asserts a status code and checks that the last-error slot agrees with it.
#[track_caller]
pub fn expect_status(status: i32, expected: i32) {
    assert_eq!(status, expected, "status {} ({}), last error: {:?}", status, from_c(quintrit_c::quintrit_status_string(status)), last_error());
    if expected == 0 {
        assert_eq!(quintrit_c::quintrit_last_error_code(), 0);
        assert_eq!(from_c(quintrit_c::quintrit_last_error_message()), "");
    } else {
        let (code, msg) = last_error();
        assert_eq!(code, expected);
        assert!(!msg.is_empty(), "an error must carry a message");
    }
}

//! Serial against parallel attention over the query heads of one layer, per context length: the evidence for `PARALLEL_ATTENTION_MIN_POSITIONS`.
//!
//! For several head layouts (query heads, key/value heads, head dimension) and both value numerics it times `attend_heads_with` with `parallel`
//! false and true at growing numbers of cached positions, and prints the time per call and the ratio. The first position from which the parallel
//! variant is faster, and stays faster, is the threshold for that layout and thread count.
//!
//! `QUINTRIT_THREADS` sets the pool size (default: the available parallelism). Run it in a release build:
//! `cargo bench -p quintrit-runtime --bench attention_threshold`.

use std::time::Instant;

use quintrit_runtime::attention::{attend_heads_with, HeadsLayout};
use quintrit_runtime::kv_cache::KVCache;
use quintrit_runtime::ops::f32_to_f16;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    }
}

fn median(mut v: Vec<f64>) -> f64 {
    v.sort_by(|a, b| a.partial_cmp(b).unwrap());
    v[v.len() / 2]
}

fn main() {
    let threads = std::env::var("QUINTRIT_THREADS").ok().and_then(|v| v.parse().ok()).unwrap_or_else(|| std::thread::available_parallelism().map(|n| n.get()).unwrap_or(1));
    // (name, query heads, key/value heads, head dimension)
    let layouts = [("20/5x128 (BitNet 2B)", 20, 5, 128), ("32/8x128 (Bonsai 4B)", 32, 8, 128), ("16/2x256 (hybrid)", 16, 2, 256), ("8/2x128 (small)", 8, 2, 128)];
    let positions = [8usize, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 512];
    let max_len = 1024;
    println!("threads: {threads}");
    for f32_values in [false, true] {
        println!("\nvalue numerics: {}", if f32_values { "llama.cpp (F32 accumulator)" } else { "bitnet.cpp (F16 accumulator)" });
        for (name, nq, nkv, hd) in layouts {
            let mut rng = Lcg(7);
            let mut cache = KVCache::new(1, max_len, nkv, hd);
            let row = |rng: &mut Lcg, n: usize| (0..n).map(|_| f32_to_f16(rng.next())).collect::<Vec<u16>>();
            for pos in 0..max_len {
                let (k, v) = (row(&mut rng, nkv * hd), row(&mut rng, nkv * hd));
                cache.append(0, pos, &k, &v);
            }
            let q: Vec<f32> = (0..nq * hd).map(|_| rng.next()).collect();
            let (mut q16, mut vkq, mut acc, mut out) = (vec![0u16; nq * hd], vec![0u16; nq * hd], vec![0f32; nq * hd], vec![0f32; nq * hd]);
            println!("{name}: positions -> serial us, parallel us, parallel/serial");
            for &seq_len in &positions {
                let layout = HeadsLayout { head_dim: hd, gqa_ratio: nq / nkv, f32_values, layer: 0, seq_len, scale: 1.0 / (hd as f32).sqrt() };
                let mut time = |parallel: bool| -> f64 {
                    let reps = (40_000 / seq_len).clamp(50, 2_000);
                    let mut run = || {
                        for _ in 0..reps {
                            attend_heads_with(&layout, &q, &mut q16, &mut vkq, &mut acc, &cache, &mut out, threads, parallel);
                        }
                    };
                    run();
                    median((0..5).map(|_| {
                        let t = Instant::now();
                        run();
                        t.elapsed().as_secs_f64() * 1e6 / reps as f64
                    }).collect())
                };
                // Interleave the two variants so that drift hits both.
                let (s1, p1) = (time(false), time(true));
                let (s2, p2) = (time(false), time(true));
                let (s, p) = (s1.min(s2), p1.min(p2));
                println!("  {seq_len:4}: {s:9.2} {p:9.2}  {:5.2}", p / s);
            }
        }
    }
}

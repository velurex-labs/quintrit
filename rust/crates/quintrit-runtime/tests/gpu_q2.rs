//! The GPU kernels of block-scaled (`Q2_0`) models against the CPU runtime: every result must be bit-identical. No model is needed; the inputs are random. The
//! tests skip themselves (and print why) when the machine has no supported NVIDIA GPU or the kernels were not built.
#![cfg(feature = "cuda")]

use quintrit_core::block_scaled::gemv_block_scaled_1xn;
use quintrit_core::pack::pack_matrix;
use quintrit_cuda::{
    kernels, AttentionLayout, AttentionWorkspace, BlockScaledKernels, BlockScaledMatrix, Device, Error, LayerKernels, RopeKvLayout, RopeMode as GpuRopeMode, Stream,
};
use quintrit_runtime::attention::attend_head;
use quintrit_runtime::config::{RopeConfig, RopeMode};
use quintrit_runtime::ops::{apply_rope_cached, f16_to_f32, f32_to_f16, quantize_q8_0_scalar, rms_norm, rope_cache, swiglu};
use quintrit_runtime::KVCache;

struct Gpu {
    device: Device,
    stream: Stream,
    q2: BlockScaledKernels,
    layer: LayerKernels,
}

fn gpu() -> Option<Gpu> {
    let device = match Device::open(0) {
        Ok(d) => d,
        Err(e @ (Error::NoDriver(_) | Error::Unsupported(_))) => {
            eprintln!("skipped: {e}");
            return None;
        }
        Err(e) => panic!("unexpected error opening device 0: {e}"),
    };
    if !kernels::available() {
        eprintln!("skipped: built without CUDA kernels");
        return None;
    }
    let stream = device.create_stream().unwrap();
    let q2 = BlockScaledKernels::new(&device).unwrap();
    let layer = LayerKernels::new(&device).unwrap();
    Some(Gpu { device, stream, q2, layer })
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    /// Uniform in [-1, 1).
    fn unit(&mut self) -> f32 {
        (self.next() % 2_000_000) as f32 / 1_000_000.0 - 1.0
    }
    fn trit(&mut self) -> i8 {
        (self.next() % 3) as i8 - 1
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A vector with a few large outliers, as activations of a transformer have.
fn activations_f32(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|i| rng.unit() * if i % 97 == 3 { 40.0 } else { 2.0 }).collect()
}

fn run_quantize(g: &Gpu, x: &[f32]) -> (Vec<i8>, Vec<f32>) {
    let n = x.len();
    let xb = g.device.upload(x).unwrap();
    let qs = g.device.alloc_zeroed::<i8>(n).unwrap();
    let d = g.device.alloc_zeroed::<f32>(n / 32).unwrap();
    g.q2.quantize(&g.stream, &xb, &qs, &d, n).unwrap();
    g.stream.synchronize().unwrap();
    (qs.to_vec().unwrap(), d.to_vec().unwrap())
}

#[test]
fn q8_0_quantization_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(3);
    for &n in &[32usize, 64, 2048, 3072, 4096, 9728] {
        for _ in 0..3 {
            let x = activations_f32(n, &mut rng);
            let (qs, d) = run_quantize(&g, &x);
            let (mut wq, mut wd) = (vec![0i8; n], vec![0f32; n / 32]);
            quantize_q8_0_scalar(&x, &mut wq, &mut wd);
            assert_eq!(qs, wq, "codes, n = {n}");
            assert_eq!(bits(&d), bits(&wd), "scales, n = {n}");
        }
    }
    // Edge cases in one vector of 32 blocks: ties at .5 after scaling, zeros, negative zero, subnormals, tiny and huge blocks, a NaN, an infinity.
    let mut x = vec![0f32; 32 * 32];
    for (b, block) in x.chunks_mut(32).enumerate() {
        for (i, v) in block.iter_mut().enumerate() {
            *v = match b {
                0 => 0.0,
                1 => -0.0,
                2 => f32::from_bits(1 + i as u32),           // subnormals
                3 => (i as f32 - 15.5) * 1e-30,               // tiny
                4 => (i as f32 - 15.5) * 1e30,                // huge
                5 => if i == 7 { f32::NAN } else { i as f32 * 0.25 },
                6 => if i == 7 { f32::INFINITY } else { 1.0 },
                7 => if i == 3 { f32::NEG_INFINITY } else { -2.0 },
                8 => 127.0 * (i as f32 - 15.5) / 15.5,        // exact multiples of the scale
                9 => (i as f32) * 0.5 + 0.25,                 // halves
                10 => 63.5 + (i % 2) as f32,                  // ties (x.5) against amax 64.5
                _ => (rng.unit() * 50.0).powi(3),
            };
        }
    }
    let (qs, d) = run_quantize(&g, &x);
    let (mut wq, mut wd) = (vec![0i8; x.len()], vec![0f32; 32]);
    quantize_q8_0_scalar(&x, &mut wq, &mut wd);
    assert_eq!(qs, wq, "edge-case codes");
    assert_eq!(bits(&d), bits(&wd), "edge-case scales");
}

/// A scale that an F16 file can hold: random magnitude, sometimes zero, subnormal or large.
fn weight_scale(rng: &mut Rng, i: usize) -> f32 {
    let v = match i % 17 {
        0 => 0.0,
        1 => f32::from_bits(0x3380_0000), // 2^-24, the smallest F16 subnormal
        2 => 60000.0,
        _ => rng.unit() * 0.1,
    };
    f16_to_f32(f32_to_f16(v))
}

#[test]
fn block_scaled_projection_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(11);
    // (rows, cols): a column count that is not a multiple of 160 (the partial last block), one that is, odd row counts, and shapes of real models (Bonsai 1.7B and 8B).
    let shapes = [(5usize, 64usize), (33, 192), (64, 320), (37, 2048), (96, 3072), (130, 5120), (2048, 2048), (4096, 6144), (1000, 12288), (151936 / 8, 2048)];
    for &(rows, cols) in &shapes {
        let trits: Vec<i8> = (0..rows * cols).map(|_| rng.trit()).collect();
        let packed = pack_matrix(&trits, rows, cols);
        let sb = cols / 64;
        let w_scales: Vec<f32> = (0..rows * sb).map(|i| weight_scale(&mut rng, i)).collect();
        let w_f16: Vec<u16> = w_scales.iter().map(|&s| f32_to_f16(s)).collect();
        let x = activations_f32(cols, &mut rng);
        let (mut acts, mut a_scales) = (vec![0i8; cols], vec![0f32; cols / 32]);
        quantize_q8_0_scalar(&x, &mut acts, &mut a_scales);
        let mut want = vec![0f32; rows];
        gemv_block_scaled_1xn(&packed, &w_scales, &acts, &a_scales, rows, cols, &mut want);

        let m = BlockScaledMatrix::new(&g.device, &packed, &w_f16, rows, cols).unwrap();
        let (xb, qs, d) = (g.device.upload(&x).unwrap(), g.device.alloc_zeroed::<i8>(cols).unwrap(), g.device.alloc_zeroed::<f32>(cols / 32).unwrap());
        let out = g.device.alloc_zeroed::<f32>(rows).unwrap();
        g.q2.quantize(&g.stream, &xb, &qs, &d, cols).unwrap();
        g.q2.gemv(&g.stream, &m, &qs, &d, &out).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(qs.to_vec().unwrap(), acts, "codes, {rows}x{cols}");
        assert_eq!(bits(&out.to_vec().unwrap()), bits(&want), "projection, {rows}x{cols}");
    }
}

#[test]
fn block_scaled_projection_with_extreme_activations_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(29);
    let (rows, cols) = (64usize, 640usize);
    let trits: Vec<i8> = (0..rows * cols).map(|_| rng.trit()).collect();
    let packed = pack_matrix(&trits, rows, cols);
    let w_scales: Vec<f32> = (0..rows * (cols / 64)).map(|i| weight_scale(&mut rng, i)).collect();
    let w_f16: Vec<u16> = w_scales.iter().map(|&s| f32_to_f16(s)).collect();
    // Every code at its extreme, and scales that make the lane sums and products overflow the F16 range of the weights' scales times 127 * 4.
    for pattern in 0..4 {
        let acts: Vec<i8> = (0..cols)
            .map(|i| match pattern {
                0 => -128,
                1 => 127,
                2 => if i % 2 == 0 { -128 } else { 127 },
                _ => (rng.next() % 256) as u8 as i8,
            })
            .collect();
        let a_scales: Vec<f32> = (0..cols / 32).map(|i| f16_to_f32(f32_to_f16(if i % 3 == 0 { 1e-3 } else { 40.0 * rng.unit().abs() }))).collect();
        let mut want = vec![0f32; rows];
        gemv_block_scaled_1xn(&packed, &w_scales, &acts, &a_scales, rows, cols, &mut want);
        let m = BlockScaledMatrix::new(&g.device, &packed, &w_f16, rows, cols).unwrap();
        let (qs, d) = (g.device.upload(&acts).unwrap(), g.device.upload(&a_scales).unwrap());
        let out = g.device.alloc_zeroed::<f32>(rows).unwrap();
        g.q2.gemv(&g.stream, &m, &qs, &d, &out).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&out.to_vec().unwrap()), bits(&want), "pattern {pattern}");
    }
}

#[test]
fn rms_norm_rows_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(31);
    // (rows, cols, offset): the whole-vector norm of the hidden state, and the per-head norms of the query and the key inside the stacked projection result.
    for &(rows, cols, off) in &[(1usize, 2048usize, 0usize), (1, 4096, 0), (1, 5120, 0), (16, 128, 0), (8, 128, 16 * 128), (40, 128, 0), (3, 256, 40)] {
        let total = off + rows * cols + 64;
        let x = activations_f32(total, &mut rng);
        let w: Vec<f32> = (0..cols).map(|_| 0.5 + rng.unit().abs()).collect();
        let eps = 1e-6f32;
        let mut want = x.clone();
        for r in 0..rows {
            let span = off + r * cols..off + (r + 1) * cols;
            let mut y = vec![0f32; cols];
            rms_norm(&x[span.clone()], &w, eps, &mut y);
            want[span].copy_from_slice(&y);
        }
        let (xb, wb) = (g.device.upload(&x).unwrap(), g.device.upload(&w).unwrap());
        // Separate output buffer.
        let out = g.device.upload(&x).unwrap();
        g.q2.rms_norm_rows(&g.stream, &xb, off, &wb, &out, off, rows, cols, eps).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&out.to_vec().unwrap()), bits(&want), "norm {rows} x {cols} at {off}");
        // In place.
        g.q2.rms_norm_rows(&g.stream, &xb, off, &wb, &xb, off, rows, cols, eps).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&xb.to_vec().unwrap()), bits(&want), "in-place norm {rows} x {cols} at {off}");
    }
}

#[test]
fn swiglu_and_residual_add_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(37);
    for &n in &[8usize, 2048, 6144, 9728, 12288] {
        let gate_up: Vec<f32> = (0..2 * n).map(|i| rng.unit() * if i % 31 == 0 { 30.0 } else { 4.0 }).collect();
        let mut want = vec![0f32; n];
        swiglu(&gate_up[..n], &gate_up[n..], &mut want);
        let (gb, out) = (g.device.upload(&gate_up).unwrap(), g.device.alloc_zeroed::<f32>(n).unwrap());
        g.q2.swiglu(&g.stream, &gb, &out, n).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&out.to_vec().unwrap()), bits(&want), "swiglu, n = {n}");

        let h: Vec<f32> = (0..n).map(|_| rng.unit() * 8.0).collect();
        let r: Vec<f32> = (0..n).map(|_| rng.unit() * 3.0).collect();
        let want: Vec<f32> = h.iter().zip(&r).map(|(a, b)| a + b).collect();
        let (hb, rb) = (g.device.upload(&h).unwrap(), g.device.upload(&r).unwrap());
        g.q2.add(&g.stream, &hb, &rb, n).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&hb.to_vec().unwrap()), bits(&want), "add, n = {n}");
    }
}

struct RopeCase {
    q_heads: usize,
    kv_heads: usize,
    hd: usize,
    rope_dims: usize,
    mode: RopeMode,
}

#[test]
fn rope_and_cache_append_of_f32_input_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(41);
    let cache_len = 64usize;
    let cases = [
        RopeCase { q_heads: 16, kv_heads: 8, hd: 128, rope_dims: 128, mode: RopeMode::Neox },
        RopeCase { q_heads: 32, kv_heads: 8, hd: 128, rope_dims: 128, mode: RopeMode::Neox },
        RopeCase { q_heads: 12, kv_heads: 4, hd: 256, rope_dims: 64, mode: RopeMode::Neox },
        RopeCase { q_heads: 8, kv_heads: 2, hd: 128, rope_dims: 128, mode: RopeMode::Normal },
    ];
    for case in &cases {
        let RopeCase { q_heads, kv_heads, hd, rope_dims, mode } = *case;
        let rope = RopeConfig::plain(mode, rope_dims, 1_000_000.0, 4096);
        let (q_rows, kv_rows) = (q_heads * hd, kv_heads * hd);
        let qkv: Vec<f32> = (0..q_rows + 2 * kv_rows).map(|_| rng.unit() * 5.0).collect();
        let pos = 37usize;
        let (mut q, mut k) = (qkv[..q_rows].to_vec(), qkv[q_rows..q_rows + kv_rows].to_vec());
        let v = &qkv[q_rows + kv_rows..];
        let mut cache = Vec::new();
        rope_cache(pos, &rope, &mut cache);
        apply_rope_cached(&mut q, q_heads, hd, mode, &cache);
        apply_rope_cached(&mut k, kv_heads, hd, mode, &cache);
        let mut table = Vec::new();
        let mut c = Vec::new();
        for p in 0..cache_len {
            rope_cache(p, &rope, &mut c);
            table.extend_from_slice(&c);
        }
        let (qb, tb) = (g.device.upload(&qkv).unwrap(), g.device.upload(&table).unwrap());
        let pos_dev = g.device.upload(&[pos as i32]).unwrap();
        let q_out = g.device.alloc_zeroed::<f32>(q_rows).unwrap();
        let k_cache = g.device.alloc_zeroed::<u16>(kv_heads * cache_len * hd).unwrap();
        let v_cache = g.device.alloc_zeroed::<u16>(kv_heads * cache_len * hd).unwrap();
        let layout = RopeKvLayout {
            q_heads,
            kv_heads,
            head_dim: hd,
            rope_dims,
            mode: if mode == RopeMode::Neox { GpuRopeMode::HalfSplit } else { GpuRopeMode::Interleaved },
            cache_len,
        };
        g.q2.rope_kv(&g.stream, layout, &qb, &tb, &pos_dev, &q_out, &k_cache, &v_cache).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&q_out.to_vec().unwrap()), bits(&q), "q, {q_heads}x{hd} {mode:?} dims {rope_dims}");
        let (gk, gv) = (k_cache.to_vec().unwrap(), v_cache.to_vec().unwrap());
        for h in 0..kv_heads {
            let at = (h * cache_len + pos) * hd;
            let wk: Vec<u16> = k[h * hd..(h + 1) * hd].iter().map(|&x| f32_to_f16(x)).collect();
            let wv: Vec<u16> = v[h * hd..(h + 1) * hd].iter().map(|&x| f32_to_f16(x)).collect();
            assert_eq!(&gk[at..at + hd], &wk[..], "k head {h}, {mode:?}");
            assert_eq!(&gv[at..at + hd], &wv[..], "v head {h}, {mode:?}");
        }
    }
}

struct AttnCase {
    q_heads: usize,
    kv_heads: usize,
    hd: usize,
    cache_len: usize,
}

#[test]
fn attention_with_f32_accumulator_matches_the_cpu_including_the_split_kv_axis() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(23);
    let cases = [
        AttnCase { q_heads: 16, kv_heads: 8, hd: 128, cache_len: 4096 },
        AttnCase { q_heads: 12, kv_heads: 4, hd: 256, cache_len: 7936 },
        AttnCase { q_heads: 8, kv_heads: 2, hd: 64, cache_len: 1000 },
        AttnCase { q_heads: 8, kv_heads: 4, hd: 32, cache_len: 1000 },
        AttnCase { q_heads: 10, kv_heads: 2, hd: 128, cache_len: 800 },
        AttnCase { q_heads: 4, kv_heads: 1, hd: 256, cache_len: 511 },
    ];
    let contexts = [
        1usize, 2, 15, 16, 17, 31, 32, 33, 127, 128, 129, 255, 256, 257, 300, 384, 511, 512, 513, 576, 700, 768, 769, 1000, 1023, 1024, 1025, 1537, 2047, 4000, 4097, 7000,
        7936,
    ];
    for case in &cases {
        let AttnCase { q_heads, kv_heads, hd, cache_len } = *case;
        let fill = cache_len;
        let mut cpu = KVCache::new(1, cache_len, kv_heads, hd);
        let mut kg = vec![0u16; kv_heads * cache_len * hd];
        let mut vg = vec![0u16; kv_heads * cache_len * hd];
        for p in 0..fill {
            let kscale = if p % 61 == 7 { 3.0 } else { 0.6 };
            let k: Vec<u16> = (0..kv_heads * hd).map(|_| f32_to_f16(rng.unit() * kscale)).collect();
            let v: Vec<u16> = (0..kv_heads * hd).map(|_| f32_to_f16(rng.unit() * 2.0)).collect();
            cpu.append(0, p, &k, &v);
            for h in 0..kv_heads {
                let at = (h * cache_len + p) * hd;
                kg[at..at + hd].copy_from_slice(&k[h * hd..(h + 1) * hd]);
                vg[at..at + hd].copy_from_slice(&v[h * hd..(h + 1) * hd]);
            }
        }
        let (kb, vb) = (g.device.upload(&kg).unwrap(), g.device.upload(&vg).unwrap());
        let q: Vec<f32> = (0..q_heads * hd).map(|_| rng.unit() * 3.0).collect();
        let qb = g.device.upload(&q).unwrap();
        let out = g.device.alloc_zeroed::<f32>(q_heads * hd).unwrap();
        let pos_dev = g.device.alloc_zeroed::<i32>(1).unwrap();
        let scale = 1.0 / (hd as f32).sqrt();
        let layout = AttentionLayout { q_heads, kv_heads, head_dim: hd, cache_len, max_len: cache_len };
        let ws = AttentionWorkspace::new(&g.device, q_heads, hd).unwrap();
        for &len in contexts.iter().filter(|&&l| l <= fill) {
            pos_dev.copy_from_host(&[(len - 1) as i32]).unwrap();
            g.layer.attention_f32(&g.stream, layout, scale, &kb, &vb, &qb, &out, &pos_dev, &ws).unwrap();
            g.stream.synchronize().unwrap();
            let got = out.to_vec().unwrap();
            let mut want = vec![0f32; q_heads * hd];
            let (mut vkq, mut acc) = (vec![0u16; hd], vec![0f32; hd]);
            for h in 0..q_heads {
                let qs = &q[h * hd..(h + 1) * hd];
                let q16: Vec<u16> = qs.iter().map(|&v| f32_to_f16(v)).collect();
                attend_head(&q16, qs, &cpu, 0, h / (q_heads / kv_heads), len, scale, &mut vkq, &mut acc, &mut want[h * hd..(h + 1) * hd], true);
            }
            assert_eq!(bits(&got), bits(&want), "F32 attention, {q_heads}/{kv_heads} heads of {hd}, cache {cache_len}, context {len}");
        }
    }
}

//! The GPU kernels of a decoder layer against the CPU runtime: every result must be bit-identical. No model is needed; the inputs are random. The tests
//! skip themselves (and print why) when the machine has no supported NVIDIA GPU or the kernels were not built.
#![cfg(feature = "cuda")]

use quintrit_cuda::pack::{act_entries, activation_sum, interleave_rows, pack_planar, permute_activations, reference_gemv, row_bytes, stack_rows};
use quintrit_cuda::{kernels, Activation, AttentionLayout, AttentionWorkspace, Device, Error, Gemv, LayerKernels, RopeKvLayout, RopeLayout, RopeMode as GpuRopeMode, Stream};
use quintrit_runtime::attention::attend_head;
use quintrit_runtime::config::{RopeConfig, RopeMode};
use quintrit_runtime::ops::{
    apply_rope_cached, elementwise_mul, f16_to_f32, f32_to_f16, ggml_vec_dot_f16, quantize_activations_int8_with_range, relu2, rms_norm, rope_cache, swiglu,
};
use quintrit_runtime::KVCache;

struct Gpu {
    device: Device,
    stream: Stream,
    gemv: Gemv,
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
    let gemv = Gemv::new(&device).unwrap();
    let layer = LayerKernels::new(&device).unwrap();
    Some(Gpu { device, stream, gemv, layer })
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
    fn act(&mut self) -> i8 {
        (self.next() % 256) as u8 as i8
    }
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

/// A vector with a few large outliers, as activations of a transformer have.
fn activations_f32(n: usize, rng: &mut Rng) -> Vec<f32> {
    (0..n).map(|i| rng.unit() * if i % 97 == 3 { 40.0 } else { 2.0 }).collect()
}

fn cpu_quantize(y: &[f32]) -> (Vec<i8>, f32) {
    let mut q = vec![0i8; y.len()];
    let s = quantize_activations_int8_with_range(y, &mut q, 127.0);
    (q, s)
}

#[test]
fn rms_quant_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(5);
    // 2048 (Falcon-E 1B), 2560 (BitNet 2B), 3072 (Falcon3), 4096 (Llama3 8B), 6912 (BitNet feed-forward sub-norm).
    for &n in &[2048usize, 2560, 3072, 4096, 6912] {
        for round in 0..3 {
            let x = activations_f32(n, &mut rng);
            let gamma: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit().abs()).collect();
            let eps = if round == 0 { 1e-5 } else { 1e-6 };
            // Normalized.
            let mut y = vec![0f32; n];
            rms_norm(&x, &gamma, eps, &mut y);
            let (q, s) = cpu_quantize(&y);
            let (xb, gb) = (g.device.upload(&x).unwrap(), g.device.upload(&gamma).unwrap());
            let act = g.device.alloc_zeroed::<i8>(act_entries(n) * 8).unwrap();
            let (sum, scale) = (g.device.alloc_zeroed::<i32>(1).unwrap(), g.device.alloc_zeroed::<f32>(1).unwrap());
            g.layer.rms_quant(&g.stream, &xb, Some(&gb), &act, &sum, &scale, n, eps, None, None).unwrap();
            g.stream.synchronize().unwrap();
            assert_eq!(act.to_vec().unwrap(), permute_activations(&q), "codes, n = {n}");
            assert_eq!(sum.to_vec().unwrap()[0], activation_sum(&q), "sum, n = {n}");
            assert_eq!(scale.to_vec().unwrap()[0].to_bits(), s.to_bits(), "scale, n = {n}");
            // The L2 prefetch of a weight matrix runs in extra blocks and must not change anything.
            let weights = g.device.alloc_zeroed::<u8>(64 * 640).unwrap();
            let pf = quintrit_cuda::layer::Prefetch { base: weights.ptr(), row_bytes: 640, rows: 64 };
            let act_pf = g.device.alloc_zeroed::<i8>(act_entries(n) * 8).unwrap();
            g.layer.rms_quant(&g.stream, &xb, Some(&gb), &act_pf, &sum, &scale, n, eps, None, Some(pf)).unwrap();
            g.stream.synchronize().unwrap();
            assert_eq!(act_pf.to_vec().unwrap(), permute_activations(&q), "codes with prefetch, n = {n}");
            // Not normalized (quantization of the attention output).
            let (q0, s0) = cpu_quantize(&x);
            g.layer.rms_quant(&g.stream, &xb, None, &act, &sum, &scale, n, eps, None, None).unwrap();
            g.stream.synchronize().unwrap();
            assert_eq!(act.to_vec().unwrap(), permute_activations(&q0), "plain codes, n = {n}");
            assert_eq!(scale.to_vec().unwrap()[0].to_bits(), s0.to_bits(), "plain scale, n = {n}");
        }
    }
    // An all-zero vector gives scale 0 and zero codes.
    let zeros = g.device.alloc_zeroed::<f32>(3072).unwrap();
    let act = g.device.upload(&vec![5i8; act_entries(3072) * 8]).unwrap();
    let (sum, scale) = (g.device.alloc_zeroed::<i32>(1).unwrap(), g.device.upload(&[1.0f32]).unwrap());
    g.layer.rms_quant(&g.stream, &zeros, None, &act, &sum, &scale, 3072, 1e-5, None, None).unwrap();
    g.stream.synchronize().unwrap();
    assert_eq!(scale.to_vec().unwrap()[0], 0.0);
    assert!(act.to_vec().unwrap()[..160 * 8].iter().all(|&c| c == 0));
}

#[test]
fn residual_fused_into_the_norm_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(9);
    for &n in &[2048usize, 3072, 4096] {
        let h: Vec<f32> = activations_f32(n, &mut rng);
        let delta: Vec<i32> = (0..n).map(|_| (rng.next() % 40001) as i32 - 20000).collect();
        let gamma: Vec<f32> = (0..n).map(|_| 0.5 + rng.unit().abs()).collect();
        let (ws, prev_scale, eps) = (0.0371f32, 11.37f32, 1e-5f32);
        let ps = ws / prev_scale;
        let h2: Vec<f32> = h.iter().zip(&delta).map(|(&a, &d)| a + d as f32 * ps).collect();
        let mut y = vec![0f32; n];
        rms_norm(&h2, &gamma, eps, &mut y);
        let (q, s) = cpu_quantize(&y);
        let (hb, db, gb) = (g.device.upload(&h).unwrap(), g.device.upload(&delta).unwrap(), g.device.upload(&gamma).unwrap());
        let act = g.device.alloc_zeroed::<i8>(act_entries(n) * 8).unwrap();
        let (sum, scale) = (g.device.alloc_zeroed::<i32>(1).unwrap(), g.device.upload(&[prev_scale]).unwrap());
        g.layer.rms_quant(&g.stream, &hb, Some(&gb), &act, &sum, &scale, n, eps, Some((&db, ws)), None).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&hb.to_vec().unwrap()), bits(&h2), "residual stream, n = {n}");
        assert_eq!(act.to_vec().unwrap(), permute_activations(&q), "codes, n = {n}");
        assert_eq!(scale.to_vec().unwrap()[0].to_bits(), s.to_bits());
        // The standalone residual add.
        let hb2 = g.device.upload(&h).unwrap();
        let scale2 = g.device.upload(&[prev_scale]).unwrap();
        g.layer.residual_add(&g.stream, &hb2, &db, &scale2, ws, n).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&hb2.to_vec().unwrap()), bits(&h2));
    }
}

#[test]
fn final_norm_and_head_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(13);
    for &(k, vocab) in &[(2048usize, 1000usize), (2560, 1003), (3072, 4096), (4096, 777)] {
        let h = activations_f32(k, &mut rng);
        let w: Vec<f32> = (0..k).map(|_| 0.5 + rng.unit().abs()).collect();
        let eps = 1e-5f32;
        let mut y = vec![0f32; k];
        rms_norm(&h, &w, eps, &mut y);
        let y16: Vec<u16> = y.iter().map(|&v| f32_to_f16(v)).collect();
        let yf: Vec<f32> = y16.iter().map(|&v| f16_to_f32(v)).collect();
        let (hb, wb, xb) = (g.device.upload(&h).unwrap(), g.device.upload(&w).unwrap(), g.device.alloc_zeroed::<f32>(k).unwrap());
        g.layer.final_norm(&g.stream, &hb, &wb, &xb, k, eps).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&xb.to_vec().unwrap()), bits(&yf), "final norm, k = {k}");
        // Output head over random F16 weights.
        let head: Vec<u16> = (0..vocab * k).map(|_| f32_to_f16(rng.unit() * 0.1)).collect();
        let (hw, logits) = (g.device.upload(&head).unwrap(), g.device.alloc_zeroed::<f32>(vocab).unwrap());
        g.layer.lm_head(&g.stream, &hw, &xb, &logits, vocab, k).unwrap();
        g.stream.synchronize().unwrap();
        let want: Vec<f32> = (0..vocab).map(|v| ggml_vec_dot_f16(&y16, &head[v * k..(v + 1) * k])).collect();
        assert_eq!(bits(&logits.to_vec().unwrap()), bits(&want), "logits, k = {k}");
    }
}

fn cos_sin_table(rope: &RopeConfig, positions: usize) -> Vec<f32> {
    let mut all = Vec::new();
    let mut c = Vec::new();
    for p in 0..positions {
        rope_cache(p, rope, &mut c);
        all.extend_from_slice(&c);
    }
    all
}

/// CPU result of the rotary embedding and the cache append for one token: (rotated q, K cache entry, V cache entry as F16).
#[allow(clippy::too_many_arguments)]
fn cpu_rope(raw: &[i64], q_heads: usize, kv_heads: usize, hd: usize, ws: (f32, f32, f32), s: f32, pos: usize, rope: &RopeConfig) -> (Vec<f32>, Vec<u16>, Vec<u16>) {
    let (q_rows, kv_rows) = (q_heads * hd, kv_heads * hd);
    let scaled = |slice: &[i64], w: f32| -> Vec<f32> {
        let ps = w / s;
        slice.iter().map(|&r| r as f32 * ps).collect()
    };
    let mut q = scaled(&raw[..q_rows], ws.0);
    let mut k = scaled(&raw[q_rows..q_rows + kv_rows], ws.1);
    let v = scaled(&raw[q_rows + kv_rows..], ws.2);
    let mut cache = Vec::new();
    rope_cache(pos, rope, &mut cache);
    apply_rope_cached(&mut q, q_heads, hd, rope.mode, &cache);
    apply_rope_cached(&mut k, kv_heads, hd, rope.mode, &cache);
    (q, k.iter().map(|&x| f32_to_f16(x)).collect(), v.iter().map(|&x| f32_to_f16(x)).collect())
}

struct RopeCase {
    q_heads: usize,
    kv_heads: usize,
    hd: usize,
    rope_dims: usize,
    mode: RopeMode,
}

const ROPE_CASES: [RopeCase; 4] = [
    RopeCase { q_heads: 12, kv_heads: 4, hd: 256, rope_dims: 256, mode: RopeMode::Normal },
    RopeCase { q_heads: 16, kv_heads: 2, hd: 128, rope_dims: 128, mode: RopeMode::Normal },
    RopeCase { q_heads: 20, kv_heads: 5, hd: 128, rope_dims: 128, mode: RopeMode::Neox },
    RopeCase { q_heads: 4, kv_heads: 2, hd: 64, rope_dims: 32, mode: RopeMode::Neox },
];

#[test]
fn rope_and_cache_append_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(17);
    let cache_len = 64usize;
    for case in &ROPE_CASES {
        let RopeCase { q_heads, kv_heads, hd, rope_dims, mode } = *case;
        let rope = RopeConfig::plain(mode, rope_dims, 10000.0, 4096);
        let rows = (q_heads + 2 * kv_heads) * hd;
        let raw: Vec<i64> = (0..rows).map(|_| (rng.next() % 20001) as i64 - 10000).collect();
        let qkv: Vec<i32> = raw.iter().map(|&r| r as i32).collect();
        let (s, ws) = (7.31f32, (0.021f32, 0.017f32, 0.013f32));
        let pos = 37usize;
        let (want_q, want_k, want_v) = cpu_rope(&raw, q_heads, kv_heads, hd, ws, s, pos, &rope);
        let (qb, sb) = (g.device.upload(&qkv).unwrap(), g.device.upload(&[s]).unwrap());
        let table = g.device.upload(&cos_sin_table(&rope, cache_len)).unwrap();
        let pos_dev = g.device.upload(&[pos as i32]).unwrap();
        let q_out = g.device.alloc_zeroed::<f32>(q_heads * hd).unwrap();
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
        g.layer.rope_kv(&g.stream, layout, &qb, &sb, ws, &table, &pos_dev, &q_out, &k_cache, &v_cache).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&q_out.to_vec().unwrap()), bits(&want_q), "q, {q_heads}x{hd} {mode:?} dims {rope_dims}");
        let (gk, gv) = (k_cache.to_vec().unwrap(), v_cache.to_vec().unwrap());
        for h in 0..kv_heads {
            let at = (h * cache_len + pos) * hd;
            assert_eq!(&gk[at..at + hd], &want_k[h * hd..(h + 1) * hd], "k head {h}, {mode:?}");
            assert_eq!(&gv[at..at + hd], &want_v[h * hd..(h + 1) * hd], "v head {h}, {mode:?}");
        }
    }
}

#[test]
fn projection_epilogues_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(19);
    // The fused rotary epilogue (interleaved pairing only).
    for case in ROPE_CASES.iter().filter(|c| c.mode == RopeMode::Normal) {
        let RopeCase { q_heads, kv_heads, hd, rope_dims, mode } = *case;
        let cols = 2048;
        let rope = RopeConfig::plain(mode, rope_dims, 500000.0, 4096);
        let (q_rows, kv_rows) = (q_heads * hd, kv_heads * hd);
        let rows = q_rows + 2 * kv_rows;
        let trits: Vec<i8> = (0..rows * cols).map(|_| rng.trit()).collect();
        let act: Vec<i8> = (0..cols).map(|_| rng.act()).collect();
        let raw = reference_gemv(&trits, rows, cols, &act);
        let (s, ws) = (9.5f32, (0.021f32, 0.0017f32, 0.0013f32));
        let pos = 11usize;
        let (want_q, want_k, want_v) = cpu_rope(&raw, q_heads, kv_heads, hd, ws, s, pos, &rope);
        let cache_len = 16;
        let packed = g.device.upload(&pack_planar(&trits, rows, cols)).unwrap();
        let (ab, sumb) = (g.device.upload(&permute_activations(&act)).unwrap(), g.device.upload(&[activation_sum(&act)]).unwrap());
        let (sb, pos_dev, table) = (g.device.upload(&[s]).unwrap(), g.device.upload(&[pos as i32]).unwrap(), g.device.upload(&cos_sin_table(&rope, cache_len)).unwrap());
        let q_out = g.device.alloc_zeroed::<f32>(q_rows).unwrap();
        let k_cache = g.device.alloc_zeroed::<u16>(kv_heads * cache_len * hd).unwrap();
        let v_cache = g.device.alloc_zeroed::<u16>(kv_heads * cache_len * hd).unwrap();
        let layout = RopeLayout { q_rows, kv_rows, head_dim: hd, rope_dims, cache_len };
        g.gemv.qkv_rope(&g.stream, layout, cols, &packed, &ab, &sumb, &sb, ws, &table, &pos_dev, &q_out, &k_cache, &v_cache).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&q_out.to_vec().unwrap()), bits(&want_q), "fused q, {q_heads}x{hd}");
        let (gk, gv) = (k_cache.to_vec().unwrap(), v_cache.to_vec().unwrap());
        for h in 0..kv_heads {
            let at = (h * cache_len + pos) * hd;
            assert_eq!(&gk[at..at + hd], &want_k[h * hd..(h + 1) * hd], "fused k head {h}");
            assert_eq!(&gv[at..at + hd], &want_v[h * hd..(h + 1) * hd], "fused v head {h}");
        }
    }
    // The gated epilogue: SwiGLU and squared ReLU. The intermediate size is a multiple of 8 (the CPU's vector path).
    for &(inter, activation) in &[(1536usize, Activation::Silu), (1728, Activation::Relu2), (2304, Activation::Silu)] {
        let cols = 2560;
        let gate: Vec<i8> = (0..inter * cols).map(|_| rng.trit()).collect();
        let up: Vec<i8> = (0..inter * cols).map(|_| rng.trit()).collect();
        let act: Vec<i8> = (0..cols).map(|_| rng.act()).collect();
        let (s, ws_g, ws_u) = (12.25f32, 0.0043f32, 0.0051f32);
        let (rg, ru) = (reference_gemv(&gate, inter, cols, &act), reference_gemv(&up, inter, cols, &act));
        let (psg, psu) = (ws_g / s, ws_u / s);
        let gv: Vec<f32> = rg.iter().map(|&r| r as f32 * psg).collect();
        let uv: Vec<f32> = ru.iter().map(|&r| r as f32 * psu).collect();
        let mut want = vec![0f32; inter];
        match activation {
            Activation::Silu => swiglu(&gv, &uv, &mut want),
            Activation::Relu2 => {
                let mut a = vec![0f32; inter];
                relu2(&gv, &mut a);
                elementwise_mul(&a, &uv, &mut want);
            }
        }
        let rb = row_bytes(cols);
        let packed = g.device.upload(&interleave_rows(&pack_planar(&gate, inter, cols), &pack_planar(&up, inter, cols), rb)).unwrap();
        let (ab, sumb) = (g.device.upload(&permute_activations(&act)).unwrap(), g.device.upload(&[activation_sum(&act)]).unwrap());
        let sb = g.device.upload(&[s]).unwrap();
        let v = g.device.alloc_zeroed::<f32>(inter).unwrap();
        let cells = g.device.alloc_zeroed::<i32>(2).unwrap();
        let sum_out = g.device.upload(&[1i32]).unwrap();
        g.gemv.gated(&g.stream, 2 * inter, cols, &packed, &ab, &sumb, &sb, (ws_g, ws_u), activation, &v, &cells, 0, &sum_out).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(bits(&v.to_vec().unwrap()), bits(&want), "{activation:?}, inter = {inter}");
        // The quantization that follows: the maximum collected by the epilogue gives the scale of the CPU's quantizer.
        let (q, s2) = cpu_quantize(&want);
        let act2 = g.device.alloc_zeroed::<i8>(act_entries(inter) * 8).unwrap();
        let scale2 = g.device.alloc_zeroed::<f32>(1).unwrap();
        g.layer.quant_cells(&g.stream, &v, &act2, &sum_out, &scale2, &cells, 0, inter, None).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(act2.to_vec().unwrap(), permute_activations(&q), "codes after {activation:?}");
        assert_eq!(scale2.to_vec().unwrap()[0].to_bits(), s2.to_bits());
        assert_eq!(sum_out.to_vec().unwrap()[0], activation_sum(&q));
    }
    let _ = stack_rows;
}

struct AttnCase {
    q_heads: usize,
    kv_heads: usize,
    hd: usize,
    cache_len: usize,
}

#[test]
fn attention_matches_the_cpu_including_the_split_kv_axis() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(23);
    let cases = [
        AttnCase { q_heads: 12, kv_heads: 4, hd: 256, cache_len: 7936 },
        AttnCase { q_heads: 20, kv_heads: 5, hd: 128, cache_len: 4096 },
        AttnCase { q_heads: 8, kv_heads: 2, hd: 64, cache_len: 1000 },
        AttnCase { q_heads: 12, kv_heads: 4, hd: 256, cache_len: 300 },
        AttnCase { q_heads: 8, kv_heads: 4, hd: 32, cache_len: 1000 },
        // Cache lengths that are not multiples of 256 (the padded axis is cut by the cache), five query heads per key-value head, a cache just below
        // the split threshold.
        AttnCase { q_heads: 10, kv_heads: 2, hd: 128, cache_len: 800 },
        AttnCase { q_heads: 6, kv_heads: 2, hd: 64, cache_len: 513 },
        AttnCase { q_heads: 4, kv_heads: 1, hd: 256, cache_len: 511 },
    ];
    // Contexts around the padding (256), the split threshold (512), the chunk borders of the padded lengths 512 (chunks of 128), 768 (192), 1024 (256) and
    // the cache lengths above, the borders of the tiles of 32 positions, and long ones.
    let contexts = [
        1usize, 2, 15, 16, 17, 31, 32, 33, 127, 128, 129, 191, 192, 193, 255, 256, 257, 300, 383, 384, 385, 511, 512, 513, 575, 576, 577, 700, 767, 768, 769, 800, 1000,
        1023, 1024, 1025, 1536, 1537, 2047, 4000, 4097, 7000, 7936,
    ];
    for case in &cases {
        let AttnCase { q_heads, kv_heads, hd, cache_len } = *case;
        let fill = cache_len;
        let mut cpu = KVCache::new(1, cache_len, kv_heads, hd);
        let mut kg = vec![0u16; kv_heads * cache_len * hd];
        let mut vg = vec![0u16; kv_heads * cache_len * hd];
        for p in 0..fill {
            // Scores of different sizes, with the occasional larger key so that the running maximum grows at various positions.
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
            g.layer.attention(&g.stream, layout, scale, &kb, &vb, &qb, &out, &pos_dev, &ws).unwrap();
            g.stream.synchronize().unwrap();
            let got = out.to_vec().unwrap();
            let mut want = vec![0f32; q_heads * hd];
            let (mut vkq, mut acc) = (vec![0u16; hd], vec![0f32; hd]);
            for h in 0..q_heads {
                let qs = &q[h * hd..(h + 1) * hd];
                let q16: Vec<u16> = qs.iter().map(|&v| f32_to_f16(v)).collect();
                attend_head(&q16, qs, &cpu, 0, h / (q_heads / kv_heads), len, scale, &mut vkq, &mut acc, &mut want[h * hd..(h + 1) * hd], false);
            }
            assert_eq!(bits(&got), bits(&want), "attention, {q_heads}/{kv_heads} heads of {hd}, cache {cache_len}, context {len}");
        }
    }
}

#[test]
fn invalid_layer_launches_are_rejected() {
    let Some(g) = gpu() else { return };
    let x = g.device.alloc_zeroed::<f32>(100).unwrap();
    let act = g.device.alloc_zeroed::<i8>(act_entries(100) * 8).unwrap();
    let (sum, scale) = (g.device.alloc_zeroed::<i32>(1).unwrap(), g.device.alloc_zeroed::<f32>(1).unwrap());
    // Buffer too small, a norm over too many columns, a residual without a norm.
    assert!(g.layer.rms_quant(&g.stream, &x, None, &act, &sum, &scale, 4000, 1e-5, None, None).is_err());
    let big = g.device.alloc_zeroed::<f32>(8000).unwrap();
    let big_act = g.device.alloc_zeroed::<i8>(act_entries(8000) * 8).unwrap();
    let gamma = g.device.alloc_zeroed::<f32>(8000).unwrap();
    assert!(g.layer.rms_quant(&g.stream, &big, Some(&gamma), &big_act, &sum, &scale, 8000, 1e-5, None, None).is_err());
    let delta = g.device.alloc_zeroed::<i32>(100).unwrap();
    assert!(g.layer.rms_quant(&g.stream, &x, None, &act, &sum, &scale, 100, 1e-5, Some((&delta, 1.0)), None).is_err());
    let kv = g.device.alloc_zeroed::<u16>(10).unwrap();
    let q = g.device.alloc_zeroed::<f32>(10).unwrap();
    let pos = g.device.alloc_zeroed::<i32>(1).unwrap();
    let bad_head = AttentionLayout { q_heads: 2, kv_heads: 1, head_dim: 96, cache_len: 4, max_len: 4 };
    let ws = AttentionWorkspace::new(&g.device, 2, 96).unwrap();
    assert!(g.layer.attention(&g.stream, bad_head, 0.1, &kv, &kv, &q, &q, &pos, &ws).is_err());
}

//! Ground-truth tests of the projection kernels against brute-force integer and `f64` arithmetic. They skip themselves (and print why)
//! when the machine has no supported GPU or the crate was built without kernels.

use quintrit_cuda::pack::{act_entries, activation_sum, interleave_rows, pack_planar, permute_activations, reference_gemv, row_bytes, stack_rows};
use quintrit_cuda::profile::{Role, ALL_CONFIGS};
use quintrit_cuda::{kernels, Activation, Device, DeviceBuffer, Error, Gemv, RopeLayout, Stream};

struct Gpu {
    device: Device,
    stream: Stream,
    gemv: Gemv,
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
    Some(Gpu { device, stream, gemv })
}

struct Rng(u64);
impl Rng {
    fn next(&mut self) -> u64 {
        self.0 = self.0.wrapping_mul(6364136223846793005).wrapping_add(1442695040888963407);
        self.0 >> 33
    }
    fn trit(&mut self) -> i8 {
        (self.next() % 3) as i8 - 1
    }
    fn act(&mut self) -> i8 {
        (self.next() % 256) as i64 as u8 as i8
    }
}

fn weights(kind: &str, n: usize, rng: &mut Rng) -> Vec<i8> {
    match kind {
        "random" => (0..n).map(|_| rng.trit()).collect(),
        "plus" => vec![1; n],
        "minus" => vec![-1; n],
        "zero" => vec![0; n],
        _ => unreachable!(),
    }
}

fn activations(kind: &str, n: usize, rng: &mut Rng) -> Vec<i8> {
    match kind {
        "random" => (0..n).map(|_| rng.act()).collect(),
        "max" => vec![127; n],
        "min" => vec![-128; n],
        "alternating" => (0..n).map(|i| if i % 2 == 0 { 127 } else { -128 }).collect(),
        _ => unreachable!(),
    }
}

struct Uploaded {
    act: DeviceBuffer<i8>,
    sum: DeviceBuffer<i32>,
}

fn upload_activations(g: &Gpu, act: &[i8]) -> Uploaded {
    let perm = permute_activations(act);
    assert_eq!(perm.len(), act_entries(act.len()) * 8);
    Uploaded { act: g.device.upload(&perm).unwrap(), sum: g.device.upload(&[activation_sum(act)]).unwrap() }
}

#[test]
fn projection_is_exact_for_every_configuration_shape_and_pattern() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(7);
    let rows = 64;
    // Columns: a single trit, exactly one block, one trit more, two blocks, a non-multiple of 160 (K = 3072 of Falcon3 and 2560 of BitNet),
    // a power of two, a size with a partial last block (13312 = 83.2 blocks) and the widest shape of the 10B model (23040).
    for &cols in &[1usize, 160, 161, 320, 2560, 3072, 4096, 13312, 23040] {
        for wk in ["random", "plus", "minus", "zero"] {
            let trits = weights(wk, rows * cols, &mut rng);
            let packed = g.device.upload(&pack_planar(&trits, rows, cols)).unwrap();
            for ak in ["random", "max", "min", "alternating"] {
                let act = activations(ak, cols, &mut rng);
                let want = reference_gemv(&trits, rows, cols, &act);
                let up = upload_activations(&g, &act);
                for cfg in ALL_CONFIGS {
                    let out = g.device.alloc_zeroed::<i32>(rows).unwrap();
                    g.gemv.plain_with(&g.stream, &cfg, rows, cols, &packed, &up.act, &up.sum, &out).unwrap();
                    g.stream.synchronize().unwrap();
                    let got = out.to_vec().unwrap();
                    for r in 0..rows {
                        assert_eq!(got[r] as i64, want[r], "{} cols={cols} weights={wk} acts={ak} row {r}", cfg.tag());
                    }
                }
            }
        }
    }
}

#[test]
fn profile_configuration_tiles_awkward_row_counts() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(11);
    // 6 and 12 rows need the smallest fallbacks, 40 the 8-row one, 2560 and 3456 the profile's own configuration.
    for &(rows, cols) in &[(2usize, 160usize), (6, 3072), (12, 2560), (40, 3072), (2560, 2560), (3456, 2560)] {
        let trits = weights("random", rows * cols, &mut rng);
        let act = activations("random", cols, &mut rng);
        let want = reference_gemv(&trits, rows, cols, &act);
        let packed = g.device.upload(&pack_planar(&trits, rows, cols)).unwrap();
        let up = upload_activations(&g, &act);
        for role in [Role::Output, Role::Down] {
            let out = g.device.alloc_zeroed::<i32>(rows).unwrap();
            g.gemv.plain(&g.stream, role, rows, cols, &packed, &up.act, &up.sum, &out).unwrap();
            g.stream.synchronize().unwrap();
            let got: Vec<i64> = out.to_vec().unwrap().iter().map(|&v| v as i64).collect();
            assert_eq!(got, want, "{rows}x{cols} {role:?}");
        }
    }
}

#[test]
fn invalid_launches_are_rejected() {
    let Some(g) = gpu() else { return };
    let trits = vec![1i8; 8 * 160];
    let packed = g.device.upload(&pack_planar(&trits, 8, 160)).unwrap();
    let up = upload_activations(&g, &vec![1i8; 160]);
    let out = g.device.alloc_zeroed::<i32>(8).unwrap();
    // A row count that the configuration cannot tile, a matrix larger than the weight buffer and an odd row count.
    let cfg = ALL_CONFIGS[0];
    assert!(g.gemv.plain_with(&g.stream, &cfg, 8, 160, &packed, &up.act, &up.sum, &out).is_err());
    assert!(g.gemv.plain(&g.stream, Role::Output, 16, 160, &packed, &up.act, &up.sum, &out).is_err());
    assert!(g.gemv.plain(&g.stream, Role::Output, 7, 160, &packed, &up.act, &up.sum, &out).is_err());
}

fn silu(x: f64) -> f64 {
    x / (1.0 + (-x).exp())
}

fn close(got: f32, want: f64, rel: f64, abs: f64) -> bool {
    (got as f64 - want).abs() <= rel * want.abs() + abs
}

#[test]
fn gated_epilogue_matches_the_reference_activation() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(23);
    for &(inter, cols, activation) in &[(64usize, 3072usize, Activation::Silu), (48, 2560, Activation::Relu2), (136, 160, Activation::Silu)] {
        let gate = weights("random", inter * cols, &mut rng);
        let up_w = weights("random", inter * cols, &mut rng);
        let act = activations("random", cols, &mut rng);
        let rb = row_bytes(cols);
        let packed = g.device.upload(&interleave_rows(&pack_planar(&gate, inter, cols), &pack_planar(&up_w, inter, cols), rb)).unwrap();
        let up = upload_activations(&g, &act);
        let (act_scale, ws_g, ws_u) = (0.037f32, 0.5f32, 0.25f32);
        let scale_buf = g.device.upload(&[act_scale]).unwrap();
        let v = g.device.alloc_zeroed::<f32>(inter).unwrap();
        // The cells start dirty to check that the launch resets the cell of the other parity and the activation sum.
        let cells = g.device.upload(&[0i32, 12345]).unwrap();
        let sum_out = g.device.upload(&[777i32]).unwrap();
        g.gemv.gated(&g.stream, 2 * inter, cols, &packed, &up.act, &up.sum, &scale_buf, (ws_g, ws_u), activation, &v, &cells, 0, &sum_out).unwrap();
        g.stream.synchronize().unwrap();
        let got = v.to_vec().unwrap();
        let rg = reference_gemv(&gate, inter, cols, &act);
        let ru = reference_gemv(&up_w, inter, cols, &act);
        let mut max_abs = 0f64;
        for i in 0..inter {
            let gv = rg[i] as f32 * (ws_g / act_scale);
            let uv = ru[i] as f32 * (ws_u / act_scale);
            let want = match activation {
                Activation::Silu => silu(gv as f64) * uv as f64,
                Activation::Relu2 => {
                    let p = gv.max(0.0);
                    ((p * p) * uv) as f64
                }
            };
            assert!(close(got[i], want, 2e-5, 1e-6), "{activation:?} row {i}: gpu {} want {want}", got[i]);
            max_abs = max_abs.max((got[i] as f64).abs());
        }
        let cells_after = cells.to_vec().unwrap();
        assert_eq!(f32::from_bits(cells_after[0] as u32) as f64, max_abs, "block maximum");
        assert_eq!(cells_after[1], 0, "the other parity cell is reset");
        assert_eq!(sum_out.to_vec().unwrap(), vec![0], "the activation sum is reset");
    }
}

fn f16_to_f32(h: u16) -> f64 {
    let sign = if h & 0x8000 != 0 { -1.0 } else { 1.0 };
    let exp = ((h >> 10) & 0x1f) as i32;
    let frac = (h & 0x3ff) as f64;
    match exp {
        0 => sign * frac * 2f64.powi(-24),
        31 => sign * f64::INFINITY,
        _ => sign * (1.0 + frac / 1024.0) * 2f64.powi(exp - 15),
    }
}

#[test]
fn rope_epilogue_rotates_and_appends_to_the_cache() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(31);
    // (query heads, kv heads, head_dim, rope_dims, cols): Falcon3-like, a 128-wide head and a partial rotation.
    for &(qh, kvh, hd, rd, cols) in &[(4usize, 2usize, 64usize, 64usize, 3072usize), (8, 2, 128, 128, 2048), (4, 2, 64, 32, 2560)] {
        let (q_rows, kv_rows) = (qh * hd, kvh * hd);
        let rows = q_rows + 2 * kv_rows;
        let (qw, kw, vw) = (weights("random", q_rows * cols, &mut rng), weights("random", kv_rows * cols, &mut rng), weights("random", kv_rows * cols, &mut rng));
        let packed = g
            .device
            .upload(&stack_rows(&[&pack_planar(&qw, q_rows, cols), &pack_planar(&kw, kv_rows, cols), &pack_planar(&vw, kv_rows, cols)]))
            .unwrap();
        let act = activations("random", cols, &mut rng);
        let up = upload_activations(&g, &act);
        let cache_len = 8;
        let pos = 5usize;
        let angles: Vec<f64> = (0..cache_len * rd / 2).map(|i| (i as f64 * 0.137).sin() * 3.0).collect();
        let cs: Vec<f32> = angles.iter().flat_map(|a| [a.cos() as f32, a.sin() as f32]).collect();
        // Small key and value scales keep the cached values inside the F16 range (65504).
        let (act_scale, ws_q, ws_k, ws_v) = (0.021f32, 0.4f32, 0.01f32, 0.01f32);
        let scale_buf = g.device.upload(&[act_scale]).unwrap();
        let cs_buf = g.device.upload(&cs).unwrap();
        let pos_buf = g.device.upload(&[pos as i32]).unwrap();
        let q_out = g.device.alloc_zeroed::<f32>(q_rows).unwrap();
        let k_cache = g.device.alloc_zeroed::<u16>(kvh * cache_len * hd).unwrap();
        let v_cache = g.device.alloc_zeroed::<u16>(kvh * cache_len * hd).unwrap();
        let layout = RopeLayout { q_rows, kv_rows, head_dim: hd, rope_dims: rd, cache_len };
        g.gemv
            .qkv_rope(&g.stream, layout, cols, &packed, &up.act, &up.sum, &scale_buf, (ws_q, ws_k, ws_v), &cs_buf, &pos_buf, &q_out, &k_cache, &v_cache)
            .unwrap();
        g.stream.synchronize().unwrap();
        let (rq, rk, rv) = (reference_gemv(&qw, q_rows, cols, &act), reference_gemv(&kw, kv_rows, cols, &act), reference_gemv(&vw, kv_rows, cols, &act));
        // Returns the rotated pair and the magnitude of its inputs: the error of the single-precision rotation scales with the inputs
        // (cancellation), not with the result.
        let rotate = |raw: &[i64], ws: f32, row_pair: usize| -> (f64, f64, f64) {
            let ps = ws / act_scale;
            let (x0, x1) = ((raw[row_pair] as f32 * ps) as f64, (raw[row_pair + 1] as f32 * ps) as f64);
            let e = row_pair % hd;
            let mag = x0.abs() + x1.abs();
            if e < rd {
                let (c, s) = (cs[pos * rd + e] as f64, cs[pos * rd + e + 1] as f64);
                (x0 * c - x1 * s, x0 * s + x1 * c, mag)
            } else {
                (x0, x1, mag)
            }
        };
        let got_q = q_out.to_vec().unwrap();
        for p in (0..q_rows).step_by(2) {
            let (o0, o1, mag) = rotate(&rq, ws_q, p);
            let tol = 1e-6 * mag + 1e-6;
            assert!((got_q[p] as f64 - o0).abs() <= tol && (got_q[p + 1] as f64 - o1).abs() <= tol, "q pair at row {p}");
        }
        let (got_k, got_v) = (k_cache.to_vec().unwrap(), v_cache.to_vec().unwrap());
        for p in (0..kv_rows).step_by(2) {
            let (head, e) = (p / hd, p % hd);
            let at = (head * cache_len + pos) * hd + e;
            let (k0, k1, kmag) = rotate(&rk, ws_k, p);
            let ps = ws_v / act_scale;
            let (v0, v1) = ((rv[p] as f32 * ps) as f64, (rv[p + 1] as f32 * ps) as f64);
            // F16 has an 11-bit significand: relative error up to 2^-11, plus the subnormal range.
            let tol = |x: f64, mag: f64| x.abs() * 1.0e-3 + 1e-6 * mag + 1e-6;
            assert!(
                (f16_to_f32(got_k[at]) - k0).abs() <= tol(k0, kmag) && (f16_to_f32(got_k[at + 1]) - k1).abs() <= tol(k1, kmag),
                "k pair at row {p}"
            );
            assert!((f16_to_f32(got_v[at]) - v0).abs() <= tol(v0, 0.0) && (f16_to_f32(got_v[at + 1]) - v1).abs() <= tol(v1, 0.0), "v pair at row {p}");
        }
        // Only position `pos` was written.
        for head in 0..kvh {
            for l in 0..cache_len {
                if l != pos {
                    let at = (head * cache_len + l) * hd;
                    assert!(got_k[at..at + hd].iter().chain(&got_v[at..at + hd]).all(|&h| h == 0), "position {l} of head {head} must stay untouched");
                }
            }
        }
        assert_eq!(rows, q_rows + 2 * kv_rows);
    }
}

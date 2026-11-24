//! The GPU kernels of the Nemotron-H decoder against the CPU runtime: every result must be bit-identical. No model is needed; the inputs are random. The tests skip
//! themselves (and print why) when the machine has no supported NVIDIA GPU or the kernels were not built. The functions that depend on the C library are compared
//! with the library of the machine, so on Windows they would differ and are skipped there.
#![cfg(feature = "cuda")]

use quintrit_core::block_scaled::gemv_block_scaled_1xn;
use quintrit_core::pack::pack_matrix;
use quintrit_cuda::{kernels, BlockScaledExperts, BlockScaledKernels, Device, Error, MambaLayout, NemotronKernels, Q8Matrix, Stream};
use quintrit_runtime::hybrid::dot_q8_0;
use quintrit_runtime::ops::{f16_to_f32, f32_to_f16, ggml_vec_dot_f32, quantize_q8_0_scalar, sigmoid_f32, silu};
use quintrit_runtime::refsort::reference_sort;

struct Gpu {
    device: Device,
    stream: Stream,
    nem: NemotronKernels,
    q2: BlockScaledKernels,
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
    let nem = NemotronKernels::new(&device).unwrap();
    let q2 = BlockScaledKernels::new(&device).unwrap();
    Some(Gpu { device, stream, nem, q2 })
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

fn same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()), "{what}: value {i}: got {g} ({:#010x}), want {w} ({:#010x})", g.to_bits(), w.to_bits());
    }
}

fn run_logf(g: &Gpu, x: &[f32]) -> Vec<f32> {
    let (xb, out) = (g.device.upload(x).unwrap(), g.device.alloc_zeroed::<f32>(x.len()).unwrap());
    g.nem.probe_logf(&g.stream, &xb, &out, x.len()).unwrap();
    g.stream.synchronize().unwrap();
    out.to_vec().unwrap()
}

#[cfg(not(windows))]
#[test]
fn logf_matches_the_c_library() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(5);
    let mut x: Vec<f32> = (0..4_000_000).map(|i| 0.5 + 24.0 * i as f32 / 4_000_000.0).collect(); // the arguments of the softplus: 1 + exp(x)
    x.extend((0..4_000_000).map(|_| f32::from_bits(((rng.next() << 20) ^ rng.next()) as u32)));
    x.extend([0.0, -0.0, 1.0, 2.0, 0.5, 1e-40, 1e-45, f32::MAX, f32::INFINITY, -1.0, f32::NAN]);
    let got = run_logf(&g, &x);
    let want: Vec<f32> = x.iter().map(|v| v.ln()).collect();
    same(&got, &want, "logf");
}

/// Every one of the 2^32 bit patterns (about two minutes): `cargo test --release --features cuda --test gpu_nemotron -- --ignored`.
#[cfg(not(windows))]
#[test]
#[ignore]
fn logf_matches_the_c_library_for_every_f32() {
    let Some(g) = gpu() else { return };
    let chunk = 1usize << 24;
    let (xb, out) = (g.device.alloc_zeroed::<f32>(chunk).unwrap(), g.device.alloc_zeroed::<f32>(chunk).unwrap());
    let mut bad = 0u64;
    for start in (0..(1u64 << 32)).step_by(chunk) {
        let x: Vec<f32> = (0..chunk as u64).map(|i| f32::from_bits((start + i) as u32)).collect();
        xb.copy_from_host(&x).unwrap();
        g.nem.probe_logf(&g.stream, &xb, &out, chunk).unwrap();
        g.stream.synchronize().unwrap();
        let got = out.to_vec().unwrap();
        for (i, (&v, &r)) in x.iter().zip(&got).enumerate() {
            let w = v.ln();
            if !(r.to_bits() == w.to_bits() || (r.is_nan() && w.is_nan())) {
                bad += 1;
                if bad < 10 {
                    eprintln!("logf({v}) [{:#010x}]: got {r} ({:#010x}), want {w} ({:#010x})", start + i as u64, r.to_bits(), w.to_bits());
                }
            }
        }
    }
    assert_eq!(bad, 0, "{bad} of 2^32 inputs differ");
}

#[test]
fn q8_0_projection_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(11);
    for &(rows, cols) in &[(7usize, 64usize), (33, 2688), (131072 / 64, 2688), (100, 4096)] {
        let qs: Vec<i8> = (0..rows * cols).map(|_| (rng.next() % 256) as u8 as i8).collect();
        let d_f32: Vec<f32> = (0..rows * cols / 32).map(|_| f16_to_f32(f32_to_f16(rng.unit().abs() * 0.05))).collect();
        let d16: Vec<u16> = d_f32.iter().map(|&v| f32_to_f16(v)).collect();
        let x: Vec<f32> = (0..cols).map(|i| rng.unit() * if i % 97 == 3 { 30.0 } else { 2.0 }).collect();
        let (mut aq, mut ad) = (vec![0i8; cols], vec![0f32; cols / 32]);
        quantize_q8_0_scalar(&x, &mut aq, &mut ad);
        let want: Vec<f32> = (0..rows).map(|r| dot_q8_0(&qs[r * cols..(r + 1) * cols], &d_f32[r * cols / 32..(r + 1) * cols / 32], &aq, &ad)).collect();
        let m = Q8Matrix::new(&g.device, &qs, &d16, rows, cols).unwrap();
        let (xb, aqb, adb) = (g.device.upload(&x).unwrap(), g.device.alloc_zeroed::<i8>(cols).unwrap(), g.device.alloc_zeroed::<f32>(cols / 32).unwrap());
        let out = g.device.alloc_zeroed::<f32>(rows).unwrap();
        g.q2.quantize(&g.stream, &xb, &aqb, &adb, cols).unwrap();
        g.nem.gemv_q8(&g.stream, &m, &aqb, &adb, &out).unwrap();
        g.stream.synchronize().unwrap();
        same(&out.to_vec().unwrap(), &want, &format!("Q8_0 projection {rows} x {cols}"));
    }
}

fn softplus_scan(x: f32) -> f32 {
    if x > 20.0 { x } else { (1.0 + x.exp()).ln() }
}

/// The Mamba-2 mixer of the CPU runtime (`Mamba2::forward` of `hybrid.rs`) from the output of the input projection on.
struct CpuMamba {
    layout: MambaLayout,
    conv_w: Vec<f32>,
    conv_b: Vec<f32>,
    dt_bias: Vec<f32>,
    a: Vec<f32>,
    d: Vec<f32>,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

impl CpuMamba {
    /// `a`: the output of the input projection (z | channels | dt); returns the convolution output and y (before the gate).
    fn step(&mut self, a: &[f32]) -> (Vec<f32>, Vec<f32>) {
        let l = self.layout;
        let (di, ds, ng, nh) = (l.d_inner, l.d_state, l.n_group, l.n_head);
        let hd = di / nh;
        let ch = l.channels();
        let km1 = l.kernel - 1;
        let mut c = vec![0f32; ch];
        for ci in 0..ch {
            let input = a[di + ci];
            let st = &mut self.conv_state[ci * km1..(ci + 1) * km1];
            let w = &self.conv_w[ci * l.kernel..(ci + 1) * l.kernel];
            let mut sum = 0.0f32;
            for (i, &wi) in w.iter().enumerate() {
                let v = if i < km1 { st[i] } else { input };
                sum = v.mul_add(wi, sum);
            }
            c[ci] = sum + self.conv_b[ci];
            st.copy_within(1.., 0);
            st[km1 - 1] = input;
        }
        let c_in = c.clone();
        silu(&c_in, &mut c);
        let conv_out = c.clone();
        let (xs, rest) = c.split_at(di);
        let (bm, cm) = rest.split_at(ng * ds);
        let dt = &a[2 * di + 2 * ng * ds..2 * di + 2 * ng * ds + nh];
        let mut y = vec![0f32; di];
        for h in 0..nh {
            let sp = softplus_scan(dt[h] + self.dt_bias[h]);
            let da = (sp * self.a[h]).exp();
            let g = h / (nh / ng);
            let b = &bm[g * ds..(g + 1) * ds];
            let cc = &cm[g * ds..(g + 1) * ds];
            for i1 in 0..hd {
                let ii = i1 + h * hd;
                let x_dt = xs[ii] * sp;
                let st = &mut self.ssm_state[ii * ds..(ii + 1) * ds];
                for (e, &bv) in st.iter_mut().zip(b) {
                    *e = *e * da + bv * x_dt;
                }
                y[ii] = ggml_vec_dot_f32(st, cc);
            }
        }
        for (ii, v) in y.iter_mut().enumerate() {
            *v += xs[ii] * self.d[ii / hd];
        }
        (conv_out, y)
    }
}

#[cfg(not(windows))]
#[test]
fn mamba_layer_matches_the_cpu_over_several_tokens() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(19);
    // The layer of Nemotron 3.5 Lightning, and a small one.
    for &(d_inner, d_state, n_group, n_head) in &[(4096usize, 128usize, 8usize, 64usize), (512, 64, 2, 8)] {
        let layout = MambaLayout { d_inner, d_state, n_group, n_head, kernel: 4 };
        let (ch, n_in) = (layout.channels(), layout.in_proj_rows());
        let conv_w: Vec<f32> = (0..ch * layout.kernel).map(|_| rng.unit() * 0.5).collect();
        let conv_b: Vec<f32> = (0..ch).map(|_| rng.unit() * 0.1).collect();
        let dt_bias: Vec<f32> = (0..n_head).map(|_| rng.unit() * 0.5).collect();
        let a_param: Vec<f32> = (0..n_head).map(|_| -rng.unit().abs() * 6.0 - 0.05).collect();
        let d_param: Vec<f32> = (0..n_head).map(|_| rng.unit()).collect();
        let mut cpu = CpuMamba {
            layout,
            conv_w: conv_w.clone(),
            conv_b: conv_b.clone(),
            dt_bias: dt_bias.clone(),
            a: a_param.clone(),
            d: d_param.clone(),
            conv_state: vec![0.0; layout.conv_state_len()],
            ssm_state: vec![0.0; layout.ssm_state_len()],
        };
        let up = |v: &[f32]| g.device.upload(v).unwrap();
        let (cw, cb, dtb, ap, dp) = (up(&conv_w), up(&conv_b), up(&dt_bias), up(&a_param), up(&d_param));
        // The device states start with garbage to prove that position 0 resets them.
        let conv_state = up(&vec![9.0f32; layout.conv_state_len()]);
        let state = up(&vec![-4.0f32; layout.ssm_state_len()]);
        let conv_out = g.device.alloc_zeroed::<f32>(ch).unwrap();
        let y = g.device.alloc_zeroed::<f32>(d_inner).unwrap();
        let pos_dev = g.device.alloc_zeroed::<i32>(1).unwrap();
        for token in 0..4 {
            let a: Vec<f32> = (0..n_in).map(|i| rng.unit() * if i >= n_in - n_head { 3.0 } else { 2.0 }).collect();
            let (want_conv, want_y) = cpu.step(&a);
            pos_dev.copy_from_host(&[token as i32]).unwrap();
            let ab = up(&a);
            g.nem.conv(&g.stream, layout, &ab, &conv_state, &cw, &cb, &conv_out, &pos_dev).unwrap();
            g.stream.synchronize().unwrap();
            same(&conv_out.to_vec().unwrap(), &want_conv, &format!("convolution, {d_inner}/{d_state}, token {token}"));
            g.nem.scan(&g.stream, layout, &state, &conv_out, &ab, &dtb, &ap, &dp, &y, &pos_dev).unwrap();
            g.stream.synchronize().unwrap();
            same(&y.to_vec().unwrap(), &want_y, &format!("scan, {d_inner}/{d_state}, token {token}"));
            same(&state.to_vec().unwrap(), &cpu.ssm_state, &format!("state, {d_inner}/{d_state}, token {token}"));
            same(&conv_state.to_vec().unwrap(), &cpu.conv_state, &format!("convolution state, {d_inner}/{d_state}, token {token}"));
        }
    }
}

#[test]
fn mamba_gate_and_group_norm_match_the_cpu() {
    use quintrit_runtime::ops::{rms_norm, swiglu};
    let Some(g) = gpu() else { return };
    let mut rng = Rng(29);
    let (di, ng) = (4096usize, 8usize);
    let z: Vec<f32> = (0..di).map(|_| rng.unit() * 3.0).collect();
    let y: Vec<f32> = (0..di).map(|_| rng.unit() * 2.0).collect();
    let mut gated = vec![0f32; di];
    swiglu(&z, &y, &mut gated);
    let (zb, yb, out) = (g.device.upload(&z).unwrap(), g.device.upload(&y).unwrap(), g.device.alloc_zeroed::<f32>(di).unwrap());
    g.nem.swiglu(&g.stream, &zb, &yb, &out, di).unwrap();
    g.stream.synchronize().unwrap();
    same(&out.to_vec().unwrap(), &gated, "swiglu of the Mamba gate");
    // The grouped norm: every group has its own weights.
    let gsz = di / ng;
    let norm: Vec<f32> = (0..di).map(|_| 0.5 + rng.unit().abs()).collect();
    let mut want = vec![0f32; di];
    for gi in 0..ng {
        rms_norm(&gated[gi * gsz..(gi + 1) * gsz], &norm[gi * gsz..(gi + 1) * gsz], 1e-5, &mut want[gi * gsz..(gi + 1) * gsz]);
    }
    let (nb, res) = (g.device.upload(&norm).unwrap(), g.device.alloc_zeroed::<f32>(di).unwrap());
    g.q2.rms_norm_grouped(&g.stream, &out, 0, gsz, &nb, gsz, &res, 0, gsz, ng, gsz, 1e-5).unwrap();
    g.stream.synchronize().unwrap();
    same(&res.to_vec().unwrap(), &want, "grouped RMSNorm");
}

#[test]
fn router_matches_the_cpu_including_ties() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(31);
    let (ne, n_used) = (128usize, 6usize);
    for round in 0..400 {
        // Few distinct values make ties likely, many make them rare; some rounds have a bias that creates ties between different probabilities.
        let levels = [3u64, 8, 40, 1 << 20][round % 4];
        let logits: Vec<f32> = (0..ne).map(|_| (rng.next() % levels) as f32 / levels as f32 * 6.0 - 3.0).collect();
        let bias: Vec<f32> = (0..ne).map(|_| if round % 3 == 0 { 0.0 } else { (rng.next() % 4) as f32 * 0.25 }).collect();
        let probs: Vec<f32> = logits.iter().map(|&l| sigmoid_f32(l)).collect();
        let sel: Vec<f32> = probs.iter().zip(&bias).map(|(p, b)| p + b).collect();
        let mut order: Vec<usize> = (0..ne).collect();
        reference_sort(&mut order, &|a: usize, b: usize| sel[a] > sel[b]);
        let want_chosen: Vec<i32> = order[..n_used].iter().map(|&e| e as i32).collect();
        let mut w: Vec<f32> = order[..n_used].iter().map(|&e| probs[e]).collect();
        let sum = w.iter().map(|&v| v as f64).sum::<f64>() as f32;
        let sum = sum.max(6.103515625e-5);
        for v in w.iter_mut() {
            *v /= sum;
        }
        for v in w.iter_mut() {
            *v *= 2.5;
        }
        let (lb, bb) = (g.device.upload(&logits).unwrap(), g.device.upload(&bias).unwrap());
        let (chosen, weights) = (g.device.alloc_zeroed::<i32>(n_used).unwrap(), g.device.alloc_zeroed::<f32>(n_used).unwrap());
        g.nem.route(&g.stream, &lb, &bb, &chosen, &weights, ne, n_used, true, 2.5).unwrap();
        g.stream.synchronize().unwrap();
        assert_eq!(chosen.to_vec().unwrap(), want_chosen, "round {round}: chosen experts");
        same(&weights.to_vec().unwrap(), &w, &format!("round {round}: expert weights"));
    }
}

#[test]
fn expert_projections_and_combination_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(37);
    // Experts of Nemotron 3.5 Lightning, scaled down in the number of experts: up (1856 x 2688) and down (2688 x 1856).
    let (n_experts, n_used) = (9usize, 6usize);
    for &(rows, cols) in &[(1856usize, 2688usize), (2688, 1856), (130, 192)] {
        let mut packed = Vec::new();
        let mut scales = Vec::new();
        let mut mats = Vec::new();
        for _ in 0..n_experts {
            let trits: Vec<i8> = (0..rows * cols).map(|_| rng.trit()).collect();
            let sc: Vec<f32> = (0..rows * cols / 64).map(|_| f16_to_f32(f32_to_f16(rng.unit() * 0.1))).collect();
            packed.extend_from_slice(&pack_matrix(&trits, rows, cols));
            scales.extend(sc.iter().map(|&s| f32_to_f16(s)));
            mats.push((pack_matrix(&trits, rows, cols), sc));
        }
        let experts = BlockScaledExperts::new(&g.device, &packed, &scales, n_experts, rows, cols).unwrap();
        let chosen_host = [7i32, 0, 3, 8, 2, 5];
        let chosen = g.device.upload(&chosen_host).unwrap();
        // Each slot has its own input (as the down projections do) of the quantized form.
        let mut qs_all = Vec::new();
        let mut ad_all = Vec::new();
        let mut want = Vec::new();
        for slot in 0..n_used {
            let x: Vec<f32> = (0..cols).map(|i| rng.unit() * if i % 71 == 5 { 20.0 } else { 2.0 }).collect();
            let (mut aq, mut ad) = (vec![0i8; cols], vec![0f32; cols / 32]);
            quantize_q8_0_scalar(&x, &mut aq, &mut ad);
            let (p, s) = &mats[chosen_host[slot] as usize];
            let mut out = vec![0f32; rows];
            gemv_block_scaled_1xn(p, s, &aq, &ad, rows, cols, &mut out);
            want.extend(out);
            qs_all.extend(aq);
            ad_all.extend(ad);
        }
        let (qb, db) = (g.device.upload(&qs_all).unwrap(), g.device.upload(&ad_all).unwrap());
        let out = g.device.alloc_zeroed::<f32>(n_used * rows).unwrap();
        g.nem.expert_gemv(&g.stream, &experts, &chosen, n_used, &qb, cols, &db, cols / 32, &out, rows).unwrap();
        g.stream.synchronize().unwrap();
        same(&out.to_vec().unwrap(), &want, &format!("expert projections {rows} x {cols}"));
        // The same input for every slot (the up projections).
        let x: Vec<f32> = (0..cols).map(|_| rng.unit() * 2.0).collect();
        let (mut aq, mut ad) = (vec![0i8; cols], vec![0f32; cols / 32]);
        quantize_q8_0_scalar(&x, &mut aq, &mut ad);
        let mut want = Vec::new();
        for slot in 0..n_used {
            let (p, s) = &mats[chosen_host[slot] as usize];
            let mut o = vec![0f32; rows];
            gemv_block_scaled_1xn(p, s, &aq, &ad, rows, cols, &mut o);
            want.extend(o);
        }
        let (qb, db) = (g.device.upload(&aq).unwrap(), g.device.upload(&ad).unwrap());
        g.nem.expert_gemv(&g.stream, &experts, &chosen, n_used, &qb, 0, &db, 0, &out, rows).unwrap();
        g.stream.synchronize().unwrap();
        same(&out.to_vec().unwrap(), &want, &format!("expert projections with a shared input {rows} x {cols}"));
    }

    // relu_sqr and the combination.
    let n = 2688usize;
    let heads: Vec<f32> = (0..n_used * n).map(|_| rng.unit() * 3.0).collect();
    let weights: Vec<f32> = (0..n_used).map(|_| rng.unit().abs() + 0.1).collect();
    let shared: Vec<f32> = (0..n).map(|_| rng.unit() * 2.0).collect();
    let mut want = vec![0f32; n];
    for o in 0..n {
        let mut routed = 0.0f32;
        for i in 0..n_used {
            let v = heads[i * n + o] * weights[i];
            routed = if i == 0 { v } else { routed + v };
        }
        want[o] = routed + shared[o];
    }
    let (hb, wb, sb, ob) = (g.device.upload(&heads).unwrap(), g.device.upload(&weights).unwrap(), g.device.upload(&shared).unwrap(), g.device.alloc_zeroed::<f32>(n).unwrap());
    g.nem.combine(&g.stream, &hb, &wb, &sb, &ob, n, n_used).unwrap();
    g.stream.synchronize().unwrap();
    same(&ob.to_vec().unwrap(), &want, "combination of the experts");
    let mut v: Vec<f32> = (0..1856).map(|_| rng.unit() * 3.0).collect();
    let vb = g.device.upload(&v).unwrap();
    g.nem.relu_sqr(&g.stream, &vb, v.len()).unwrap();
    g.stream.synchronize().unwrap();
    for x in v.iter_mut() {
        let r = if *x > 0.0 { *x } else { 0.0 };
        *x = r * r;
    }
    same(&vb.to_vec().unwrap(), &v, "relu_sqr");
}

#[test]
fn key_value_append_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(41);
    let (nq, nkv, hd, cache_len) = (32usize, 2usize, 128usize, 64usize);
    let (q_dim, kv_dim) = (nq * hd, nkv * hd);
    let qkv: Vec<f32> = (0..q_dim + 2 * kv_dim).map(|_| rng.unit() * 4.0).collect();
    let pos = 17usize;
    let (qb, pb) = (g.device.upload(&qkv).unwrap(), g.device.upload(&[pos as i32]).unwrap());
    let kc = g.device.alloc_zeroed::<u16>(nkv * cache_len * hd).unwrap();
    let vc = g.device.alloc_zeroed::<u16>(nkv * cache_len * hd).unwrap();
    g.nem.kv_append(&g.stream, &qb, &pb, &kc, &vc, nkv, hd, cache_len, q_dim).unwrap();
    g.stream.synchronize().unwrap();
    let (gk, gv) = (kc.to_vec().unwrap(), vc.to_vec().unwrap());
    for h in 0..nkv {
        for e in 0..hd {
            let at = (h * cache_len + pos) * hd + e;
            assert_eq!(gk[at], f32_to_f16(qkv[q_dim + h * hd + e]), "k head {h} element {e}");
            assert_eq!(gv[at], f32_to_f16(qkv[q_dim + kv_dim + h * hd + e]), "v head {h} element {e}");
        }
    }
}

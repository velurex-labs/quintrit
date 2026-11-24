//! The GPU kernels of the hybrid (Qwen3.5) decoders against the CPU runtime: every result must be bit-identical. No model is needed; the inputs are random. The
//! tests skip themselves (and print why) when the machine has no supported NVIDIA GPU or the kernels were not built. The scalar functions are compared with the
//! C library of the machine, so on Windows (another `expf`) they would differ and are skipped there.
#![cfg(feature = "cuda")]

use quintrit_cuda::{kernels, DeltaNetLayout, Device, Error, HybridKernels, Stream, ValueHeadPermutation};
use quintrit_runtime::hybrid::Hadamard;
use quintrit_runtime::ops::{
    bf16_to_f32, f32_to_bf16, ggml_vec_dot_bf16, ggml_vec_dot_f32, log1p_f32, rms_norm, sigmoid_f32, silu, softplus_f32,
};
use std::sync::Arc;

struct Gpu {
    device: Device,
    stream: Stream,
    hy: HybridKernels,
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
    let hy = HybridKernels::new(&device).unwrap();
    Some(Gpu { device, stream, hy })
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
}

fn bits(v: &[f32]) -> Vec<u32> {
    v.iter().map(|x| x.to_bits()).collect()
}

fn same(got: &[f32], want: &[f32], what: &str) {
    assert_eq!(got.len(), want.len(), "{what}: length");
    for (i, (g, w)) in got.iter().zip(want).enumerate() {
        assert!(g.to_bits() == w.to_bits() || (g.is_nan() && w.is_nan()), "{what}: value {i}: got {g} ({:#010x}), want {w} ({:#010x})", g.to_bits(), w.to_bits());
    }
}

fn run_unary(g: &Gpu, x: &[f32], which: i32) -> Vec<f32> {
    let (xb, out) = (g.device.upload(x).unwrap(), g.device.alloc_zeroed::<f32>(x.len()).unwrap());
    g.hy.probe(&g.stream, &xb, &out, x.len(), which).unwrap();
    g.stream.synchronize().unwrap();
    out.to_vec().unwrap()
}

/// Test inputs of a scalar function: a dense grid of the range the model uses, random finite bit patterns of every magnitude and the special values.
fn scalar_inputs(rng: &mut Rng, lo: f32, hi: f32, grid: usize, random: usize) -> Vec<f32> {
    let mut v: Vec<f32> = (0..grid).map(|i| lo + (hi - lo) * i as f32 / grid as f32).collect();
    v.extend((0..random).map(|_| {
        let b = ((rng.next() << 20) ^ rng.next()) as u32;
        let f = f32::from_bits(b);
        if f.is_finite() { f } else { 0.5 }
    }));
    v.extend([0.0, -0.0, 1.0, -1.0, 20.0, 20.000002, 19.999998, 88.0, -88.0, 87.5, -103.0, 1e-30, -1e-30, 5.9e-8, -5.9e-8, 0.41422, -0.2929, 1e9, -1e9]);
    while v.len() % 8 != 0 {
        v.push(0.25);
    }
    v
}

#[cfg(not(windows))]
#[test]
fn scalar_functions_match_the_c_library() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(7);
    // expf: the decay of the Gated DeltaNet and the weights of the softmax.
    let x = scalar_inputs(&mut rng, -100.0, 100.0, 2_000_000, 2_000_000);
    let got = run_unary(&g, &x, 0);
    let want: Vec<f32> = x.iter().map(|v| v.exp()).collect();
    same(&got, &want, "expf");
    // sigmoid.
    let got = run_unary(&g, &x, 1);
    let want: Vec<f32> = x.iter().map(|&v| sigmoid_f32(v)).collect();
    same(&got, &want, "sigmoid");
    // softplus: the argument of the gate is a small number; the grid covers [-30, 30] densely.
    let x = scalar_inputs(&mut rng, -30.0, 30.0, 3_000_000, 1_000_000);
    let got = run_unary(&g, &x, 2);
    let want: Vec<f32> = x.iter().map(|&v| softplus_f32(v)).collect();
    same(&got, &want, "softplus");
    // log1pf over the whole range of arguments (the inputs of softplus are exp(x), so positive).
    let x = scalar_inputs(&mut rng, -1.0, 3.0, 3_000_000, 3_000_000);
    let got = run_unary(&g, &x, 3);
    let want: Vec<f32> = x.iter().map(|&v| log1p_f32(v)).collect();
    same(&got, &want, "log1pf");
    // SiLU of ggml (the vector routine, so in chunks of 8).
    let x = scalar_inputs(&mut rng, -30.0, 30.0, 1_000_000, 0);
    let got = run_unary(&g, &x, 4);
    let mut want = vec![0f32; x.len()];
    silu(&x, &mut want);
    same(&got, &want, "silu");
}

#[test]
fn hadamard_transform_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(13);
    // (width, block, signs, permutation): widths and blocks of the Bonsai 2 models, with and without signs and the value-head permutation.
    let cases: [(usize, usize, bool, Option<(usize, usize, usize)>); 8] = [
        (2048, 512, true, None),
        (6144, 512, true, None),
        (2560, 512, true, Some((128, 16, 2))),
        (4096, 4096, true, Some((128, 16, 2))),
        (12288, 4096, true, None),
        (5120, 1024, false, None),
        (6144, 1024, true, Some((128, 16, 3))),
        (4096, 512, false, Some((128, 16, 2))),
    ];
    for &(width, block, with_signs, perm) in &cases {
        // The permutation covers the whole input: its head count has to match the width.
        let perm = perm.filter(|&(hd, nk, rep)| hd * nk * rep == width);
        let x: Vec<f32> = (0..width).map(|i| rng.unit() * if i % 53 == 0 { 20.0 } else { 3.0 }).collect();
        let signs: Vec<f32> = (0..width).map(|_| if rng.next() & 1 == 0 { 1.0 } else { -1.0 }).collect();
        let h = Hadamard { block, signs: with_signs.then(|| Arc::new(signs.clone())), v_perm: perm };
        let mut want = vec![0f32; width];
        h.apply_input(&x, &mut want);
        let (xb, out) = (g.device.upload(&x).unwrap(), g.device.alloc_zeroed::<f32>(width).unwrap());
        let sb = g.device.upload(&signs).unwrap();
        let perm_dev = perm.map(|(head_dim, k_heads, repeats)| ValueHeadPermutation { head_dim, k_heads, repeats });
        g.hy.hadamard(&g.stream, &xb, &out, width, block, with_signs.then_some(&sb), perm_dev).unwrap();
        g.stream.synchronize().unwrap();
        same(&out.to_vec().unwrap(), &want, &format!("Hadamard width {width}, block {block}, signs {with_signs}, permutation {perm:?}"));
    }
}

#[test]
fn f32_and_bf16_projections_match_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(17);
    for &(rows, cols) in &[(16usize, 2048usize), (32, 2560), (32, 4096), (48, 5120), (5, 96)] {
        let w: Vec<f32> = (0..rows * cols).map(|_| rng.unit() * 0.1).collect();
        let x: Vec<f32> = (0..cols).map(|i| rng.unit() * if i % 61 == 0 { 30.0 } else { 2.0 }).collect();
        let want: Vec<f32> = (0..rows).map(|r| ggml_vec_dot_f32(&w[r * cols..(r + 1) * cols], &x)).collect();
        let (wb, xb) = (g.device.upload(&w).unwrap(), g.device.upload(&x).unwrap());
        let out = g.device.alloc_zeroed::<f32>(rows + 3).unwrap();
        g.hy.gemv_f32(&g.stream, &wb, &xb, &out, 3, rows, cols).unwrap();
        g.stream.synchronize().unwrap();
        same(&out.to_vec().unwrap()[3..], &want, &format!("F32 projection {rows} x {cols}"));

        let w16: Vec<u16> = w.iter().map(|&v| f32_to_bf16(v)).collect();
        let x16: Vec<u16> = x.iter().map(|&v| f32_to_bf16(v)).collect();
        let want: Vec<f32> = (0..rows).map(|r| ggml_vec_dot_bf16(&w16[r * cols..(r + 1) * cols], &x16)).collect();
        let w16b = g.device.upload(&w16).unwrap();
        g.hy.gemv_bf16(&g.stream, &w16b, &xb, &out, 3, rows, cols).unwrap();
        g.stream.synchronize().unwrap();
        same(&out.to_vec().unwrap()[3..], &want, &format!("BF16 projection {rows} x {cols}"));
        // The conversion of the input rounds to nearest even; the widening of BF16 is exact.
        assert_eq!(bf16_to_f32(f32_to_bf16(1.0)), 1.0);
    }
}

/// The Gated DeltaNet of the CPU runtime (`GatedDeltaNet::forward` of `hybrid.rs`) from the projected values on: the gates, the convolution, the L2
/// normalization, the recurrence and the gated norm.
struct CpuDeltaNet {
    layout: DeltaNetLayout,
    conv_w: Vec<f32>,
    dt_bias: Vec<f32>,
    a: Vec<f32>,
    norm: Vec<f32>,
    eps: f32,
    conv_state: Vec<f32>,
    ssm_state: Vec<f32>,
}

impl CpuDeltaNet {
    /// `qkv`: the projection of the new token (channels), `z`, `beta_raw`, `alpha_raw`; returns the output before the output projection and the intermediate results.
    fn step(&mut self, qkv: &[f32], z: &[f32], beta_raw: &[f32], alpha_raw: &[f32]) -> (Vec<f32>, Vec<f32>, Vec<f32>, Vec<f32>) {
        let l = self.layout;
        let (hd, nv, nk, ch) = (l.head_dim, l.v_heads, l.k_heads, l.channels());
        let (key_dim, value_dim) = (nk * hd, nv * hd);
        let beta: Vec<f32> = beta_raw.iter().map(|&b| sigmoid_f32(b)).collect();
        let gate: Vec<f32> = alpha_raw.iter().enumerate().map(|(h, &a)| softplus_f32(a + self.dt_bias[h]) * self.a[h]).collect();
        let km1 = l.kernel - 1;
        let mut c = vec![0f32; ch];
        for ci in 0..ch {
            let st = &mut self.conv_state[ci * km1..(ci + 1) * km1];
            let w = &self.conv_w[ci * l.kernel..(ci + 1) * l.kernel];
            let mut sum = 0.0f32;
            for (i, &wi) in w.iter().enumerate() {
                let v = if i < km1 { st[i] } else { qkv[ci] };
                sum = v.mul_add(wi, sum);
            }
            c[ci] = sum;
            st.copy_within(1.., 0);
            st[km1 - 1] = qkv[ci];
        }
        let c_in = c.clone();
        silu(&c_in, &mut c);
        let conv_out = c.clone();
        let n = hd as f32;
        let eps_l2 = self.eps / n;
        let inv_sqrt = 1.0 / n.sqrt();
        let ones = vec![1.0f32; hd];
        for h in 0..2 * nk {
            let head = c[h * hd..(h + 1) * hd].to_vec();
            let v = &mut c[h * hd..(h + 1) * hd];
            rms_norm(&head, &ones, eps_l2, v);
            for x in v.iter_mut() {
                *x *= inv_sqrt;
            }
        }
        let normalized = c.clone();
        let scale = 1.0 / (hd as f32).sqrt();
        let mut d = vec![0f32; value_dim];
        let mut delta = vec![0f32; hd];
        for h in 0..nv {
            let kh = h % nk;
            let q = &c[kh * hd..(kh + 1) * hd];
            let k = &c[key_dim + kh * hd..key_dim + (kh + 1) * hd];
            let v = &c[2 * key_dim + h * hd..2 * key_dim + (h + 1) * hd];
            let st = &mut self.ssm_state[h * hd * hd..(h + 1) * hd * hd];
            let decay = gate[h].exp();
            for e in st.iter_mut() {
                *e *= decay;
            }
            for j in 0..hd {
                let sum = ggml_vec_dot_f32(&st[j * hd..(j + 1) * hd], k);
                delta[j] = (v[j] - sum) * beta[h];
            }
            for j in 0..hd {
                let dj = delta[j];
                for (e, &ki) in st[j * hd..(j + 1) * hd].iter_mut().zip(k) {
                    *e = ki.mul_add(dj, *e);
                }
            }
            for j in 0..hd {
                d[h * hd + j] = ggml_vec_dot_f32(&st[j * hd..(j + 1) * hd], q) * scale;
            }
        }
        let recur = d.clone();
        let mut zs = vec![0f32; value_dim];
        silu(z, &mut zs);
        for h in 0..nv {
            let head = d[h * hd..(h + 1) * hd].to_vec();
            let o = &mut d[h * hd..(h + 1) * hd];
            rms_norm(&head, &self.norm, self.eps, o);
            for (x, &gv) in o.iter_mut().zip(&zs[h * hd..(h + 1) * hd]) {
                *x *= gv;
            }
        }
        (conv_out, normalized, recur, d)
    }
}

#[cfg(not(windows))]
#[test]
fn gated_delta_net_matches_the_cpu_over_several_tokens() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(19);
    // (key heads, value heads, head dim): Bonsai 2 2B (16/16), 4B and 9B (16/32), 27B (16/48); a small odd one.
    for &(nk, nv, hd) in &[(16usize, 16usize, 128usize), (16, 32, 128), (16, 48, 128), (4, 12, 64)] {
        let layout = DeltaNetLayout { k_heads: nk, v_heads: nv, head_dim: hd, kernel: 4 };
        let (ch, value_dim) = (layout.channels(), nv * hd);
        let eps = 1e-6f32;
        let conv_w: Vec<f32> = (0..ch * layout.kernel).map(|_| rng.unit() * 0.6).collect();
        let dt_bias: Vec<f32> = (0..nv).map(|_| rng.unit() * 0.5).collect();
        let a_param: Vec<f32> = (0..nv).map(|_| -rng.unit().abs() * 5.0 - 0.1).collect();
        let norm: Vec<f32> = (0..hd).map(|_| 0.5 + rng.unit().abs()).collect();
        let mut cpu = CpuDeltaNet {
            layout,
            conv_w: conv_w.clone(),
            dt_bias: dt_bias.clone(),
            a: a_param.clone(),
            norm: norm.clone(),
            eps,
            conv_state: vec![0.0; layout.conv_state_len()],
            ssm_state: vec![0.0; layout.ssm_state_len()],
        };
        let up = |v: &[f32]| g.device.upload(v).unwrap();
        let (conv_w_b, dt_b, a_b, norm_b) = (up(&conv_w), up(&dt_bias), up(&a_param), up(&norm));
        // The device states start with garbage to prove that position 0 resets them.
        let conv_state = up(&vec![7.0f32; layout.conv_state_len()]);
        let ssm_state = up(&vec![-3.0f32; layout.ssm_state_len()]);
        let (beta_d, gate_d) = (g.device.alloc_zeroed::<f32>(nv).unwrap(), g.device.alloc_zeroed::<f32>(nv).unwrap());
        let conv_out = g.device.alloc_zeroed::<f32>(ch).unwrap();
        let recur_out = g.device.alloc_zeroed::<f32>(value_dim).unwrap();
        let gated = g.device.alloc_zeroed::<f32>(value_dim).unwrap();
        let pos_dev = g.device.alloc_zeroed::<i32>(1).unwrap();
        for token in 0..4 {
            let qkv: Vec<f32> = (0..ch).map(|_| rng.unit() * 2.0).collect();
            let z: Vec<f32> = (0..value_dim).map(|_| rng.unit() * 2.0).collect();
            let beta_raw: Vec<f32> = (0..nv).map(|_| rng.unit() * 3.0).collect();
            let alpha_raw: Vec<f32> = (0..nv).map(|_| rng.unit() * 3.0).collect();
            let (want_conv, want_norm, want_recur, want_out) = cpu.step(&qkv, &z, &beta_raw, &alpha_raw);

            pos_dev.copy_from_host(&[token as i32]).unwrap();
            let (qkv_b, z_b, br_b, ar_b) = (up(&qkv), up(&z), up(&beta_raw), up(&alpha_raw));
            g.hy.gates(&g.stream, &br_b, &ar_b, &dt_b, &a_b, &beta_d, &gate_d, nv).unwrap();
            g.hy.conv(&g.stream, layout, &qkv_b, &conv_state, &conv_w_b, &conv_out, &pos_dev).unwrap();
            g.stream.synchronize().unwrap();
            same(&conv_out.to_vec().unwrap(), &want_conv, &format!("convolution, {nk}/{nv}/{hd}, token {token}"));
            g.hy.l2_norm(&g.stream, layout, &conv_out, eps).unwrap();
            g.stream.synchronize().unwrap();
            same(&conv_out.to_vec().unwrap(), &want_norm, &format!("L2 norm, {nk}/{nv}/{hd}, token {token}"));
            g.hy.recurrence(&g.stream, layout, &ssm_state, &conv_out, &beta_d, &gate_d, &recur_out, &pos_dev).unwrap();
            g.stream.synchronize().unwrap();
            same(&recur_out.to_vec().unwrap(), &want_recur, &format!("recurrence, {nk}/{nv}/{hd}, token {token}"));
            g.hy.gated_norm(&g.stream, layout, &recur_out, &norm_b, &z_b, &gated, eps).unwrap();
            g.stream.synchronize().unwrap();
            same(&gated.to_vec().unwrap(), &want_out, &format!("gated norm, {nk}/{nv}/{hd}, token {token}"));
            same(&ssm_state.to_vec().unwrap(), &cpu.ssm_state, &format!("recurrent state, {nk}/{nv}/{hd}, token {token}"));
            same(&conv_state.to_vec().unwrap(), &cpu.conv_state, &format!("convolution state, {nk}/{nv}/{hd}, token {token}"));
        }
    }
}

#[cfg(not(windows))]
#[test]
fn attention_gate_matches_the_cpu() {
    let Some(g) = gpu() else { return };
    let mut rng = Rng(23);
    for &(heads, hd) in &[(8usize, 256usize), (16, 256), (24, 256), (4, 64)] {
        let n = heads * hd;
        let att: Vec<f32> = (0..n).map(|_| rng.unit() * 2.0).collect();
        let joint: Vec<f32> = (0..2 * n).map(|_| rng.unit() * 6.0).collect();
        let want: Vec<f32> = (0..n).map(|i| att[i] * sigmoid_f32(joint[2 * (i / hd) * hd + hd + i % hd])).collect();
        let (ab, jb) = (g.device.upload(&att).unwrap(), g.device.upload(&joint).unwrap());
        g.hy.attention_gate(&g.stream, &ab, &jb, heads, hd).unwrap();
        g.stream.synchronize().unwrap();
        same(&ab.to_vec().unwrap(), &want, &format!("attention gate, {heads} x {hd}"));
    }
}

#[test]
fn invalid_hybrid_launches_are_rejected() {
    let Some(g) = gpu() else { return };
    let x = g.device.alloc_zeroed::<f32>(100).unwrap();
    assert!(g.hy.hadamard(&g.stream, &x, &x, 100, 64, None, None).is_err(), "100 values are not a multiple of 64");
    assert!(g.hy.hadamard(&g.stream, &x, &x, 96, 48, None, None).is_err(), "48 is not a power of two");
    assert!(g.hy.gemv_f32(&g.stream, &x, &x, &x, 0, 4, 20).is_err(), "20 columns are not a multiple of 32");
    let layout = DeltaNetLayout { k_heads: 2, v_heads: 3, head_dim: 64, kernel: 4 };
    let pos = g.device.alloc_zeroed::<i32>(1).unwrap();
    assert!(g.hy.recurrence(&g.stream, layout, &x, &x, &x, &x, &x, &pos).is_err(), "3 value heads are not a multiple of 2 key heads");
}

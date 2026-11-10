//! Unified layer abstractions and multi-paradigm ternary GEMV dispatchers.
//!
//! Wraps the Quintrit velurex encoding kernels (1.60 bpw / 5 trits per byte) behind a standardized
//! interface and transcodes GGUF ternary weights (for example `I2_S`) into their in-memory layout.

use serde::{Deserialize, Serialize};

/// Supported ternary GEMV kernel implementations.
///
/// Both kernels use the velurex encoding (5 trits per byte, 1.60 bpw) and differ in the instruction
/// set and the tile height of the weight layout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TernaryKernelType {
    /// Quintrit activation lookup-table kernel: 5 trits per byte (1.60 bpw, velurex encoding),
    /// 32-row tiles, AVX2 / NEON.
    QuintritLut,
    /// Quintrit 64-row tiled activation lookup-table kernel with AVX-512 VNNI / VBMI: 5 trits per
    /// byte (1.60 bpw, velurex encoding).
    QuintritLutAVX512,
    /// Microsoft BitNet planar multiply-add kernel: 4 trits per byte (2.00 bpw), AVX2 / NEON. A comparison
    /// baseline from the crate `quintrit-bitnet`; available with the cargo feature `bitnet-baseline`.
    #[cfg(feature = "bitnet-baseline")]
    BitnetMad,
    /// The BitNet multiply-add kernel widened to AVX-512 (Microsoft ships no AVX-512 kernel): 4 trits per byte
    /// (2.00 bpw). Available with the cargo feature `bitnet-baseline`.
    #[cfg(feature = "bitnet-baseline")]
    BitnetMad512,
}

impl TernaryKernelType {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::QuintritLut => "QuintritLut",
            Self::QuintritLutAVX512 => "QuintritLutAVX512",
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad => "BitnetMad",
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad512 => "BitnetMad512",
        }
    }

    /// Largest positive code the activation quantizer may emit for this kernel.
    ///
    /// The Quintrit LUT kernels run on a hi/lo split of the activations, so they are exact for the
    /// full int8 range and use 127 levels.
    pub fn activation_quant_levels(&self) -> f32 {
        match self {
            Self::QuintritLut | Self::QuintritLutAVX512 => 127.0,
            // BitNet's reference runtime quantizes the activations of its multiply-add kernel to 127 levels.
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad | Self::BitnetMad512 => 127.0,
        }
    }

    pub fn bpw(&self) -> f32 {
        match self {
            Self::QuintritLut | Self::QuintritLutAVX512 => 1.6,
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad | Self::BitnetMad512 => 2.0,
        }
    }
}

/// Common trait for all ternary Matrix-Vector (GEMV) projection layers.
pub trait TernaryGemvLayer: Send + Sync {
    /// Computes $y = \alpha \beta \cdot (W \cdot x)$, where $x$ is 8-bit quantized activation.
    fn forward(&self, activations: &[i8], act_scale: f32, out: &mut [f32]);

    /// Returns the kernel architecture used by this layer.
    fn kernel_type(&self) -> TernaryKernelType;

    /// Memory footprint of packed weights in bytes.
    fn memory_bytes(&self) -> usize;

    /// Returns matrix dimensions: `(rows, cols)` = `(N, K)`.
    fn shape(&self) -> (usize, usize);

    /// Floating-point weight scale factor $\alpha$.
    fn weight_scale(&self) -> f32;
}

// ============================================================================
// Quintrit Activation-LUT Layer (1.60 bpw, 32-row tiled velurex encoding)
// ============================================================================

/// Projection layer using Quintrit's high-performance 1.60 bpw GEMV kernel.
#[derive(Clone)]
pub struct QuintritLayer {
    transformed_weights: Vec<u8>,
    rows: usize,
    cols: usize,
    weight_scale: f32,
    num_threads: usize,
}

impl QuintritLayer {
    pub fn new(transformed_weights: Vec<u8>, rows: usize, cols: usize, weight_scale: f32, num_threads: usize) -> Self {
        Self {
            transformed_weights,
            rows,
            cols,
            weight_scale,
            num_threads: num_threads.max(1),
        }
    }

    pub fn set_num_threads(&mut self, threads: usize) {
        self.num_threads = threads.max(1);
    }

    /// Computes the scaled GEMV from a wide activation table (`[lo | hi]`, see
    /// [`quintrit_core::lut::build_activation_lut_wide_into`]).
    ///
    /// The kernel runs once per half; every table entry then stays tiny, so the result is exact for
    /// the full int8 activation range: `W x = W lo + 16 * (W hi)`.
    pub fn forward_with_lut(&self, wide_qlut: &[i8], act_scale: f32, out: &mut [f32]) {
        assert_eq!(out.len(), self.rows);
        assert_eq!(
            wide_qlut.len(),
            quintrit_core::lut::wide_activation_lut_len(self.cols),
            "expected a wide (lo|hi) activation table"
        );
        if self.num_threads > 1 && self.rows >= 32 {
            quintrit_core::lut::gemv_lut_wide_parallel(
                &self.transformed_weights,
                wide_qlut,
                self.rows,
                self.cols,
                out,
                self.num_threads,
            );
        } else {
            quintrit_core::lut::gemv_lut_wide_1xn(&self.transformed_weights, wide_qlut, self.rows, self.cols, out);
        }

        if act_scale > 0.0 {
            let post_scale = self.weight_scale / act_scale;
            for v in out.iter_mut() {
                *v *= post_scale;
            }
        } else {
            out.fill(0.0);
        }
    }
}

impl TernaryGemvLayer for QuintritLayer {
    fn forward(&self, activations: &[i8], act_scale: f32, out: &mut [f32]) {
        assert_eq!(activations.len(), self.cols);
        let qlut = quintrit_core::lut::build_activation_lut_wide(activations, self.cols);
        self.forward_with_lut(&qlut, act_scale, out);
    }


    fn kernel_type(&self) -> TernaryKernelType {
        TernaryKernelType::QuintritLut
    }

    fn memory_bytes(&self) -> usize {
        self.transformed_weights.len()
    }

    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn weight_scale(&self) -> f32 {
        self.weight_scale
    }
}

// ============================================================================
// Quintrit 64-Row Tiled AVX-512 Activation-LUT Layer (1.60 bpw, velurex encoding)
// ============================================================================

/// Projection layer using Quintrit's 64-row tiled AVX-512 Activation-LUT kernel with VNNI / VBMI.
#[derive(Clone)]
pub struct QuintritLutAVX512Layer {
    transformed_weights: Vec<u8>,
    rows: usize,
    cols: usize,
    weight_scale: f32,
    num_threads: usize,
}

impl QuintritLutAVX512Layer {
    pub fn new(transformed_weights: Vec<u8>, rows: usize, cols: usize, weight_scale: f32, num_threads: usize) -> Self {
        Self {
            transformed_weights,
            rows,
            cols,
            weight_scale,
            num_threads: num_threads.max(1),
        }
    }

    pub fn set_num_threads(&mut self, threads: usize) {
        self.num_threads = threads.max(1);
    }

    /// Computes the scaled GEMV from a wide activation table (`[lo | hi]`): built by
    /// `build_activation_lut_wide_avx512` on x86-64 and by `build_activation_lut_wide` elsewhere.
    pub fn forward_with_lut(&self, wide_qlut: &[i8], act_scale: f32, out: &mut [f32]) {
        assert_eq!(out.len(), self.rows);

        #[cfg(target_arch = "x86_64")]
        {
            assert_eq!(wide_qlut.len(), quintrit_core::lut::wide_activation_lut_len_avx512(self.cols));
            if self.num_threads > 1 && self.rows >= 64 {
                quintrit_core::lut::gemv_lut_64_wide_parallel_avx512(
                    &self.transformed_weights,
                    wide_qlut,
                    self.rows,
                    self.cols,
                    out,
                    self.num_threads,
                );
            } else {
                quintrit_core::lut::gemv_lut_64_wide_1xn_avx512(&self.transformed_weights, wide_qlut, self.rows, self.cols, out);
            }
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            assert_eq!(wide_qlut.len(), quintrit_core::lut::wide_activation_lut_len(self.cols));
            if self.num_threads > 1 && self.rows >= 32 {
                quintrit_core::lut::gemv_lut_wide_parallel(
                    &self.transformed_weights,
                    wide_qlut,
                    self.rows,
                    self.cols,
                    out,
                    self.num_threads,
                );
            } else {
                quintrit_core::lut::gemv_lut_wide_1xn(&self.transformed_weights, wide_qlut, self.rows, self.cols, out);
            }
        }

        if act_scale > 0.0 {
            let post_scale = self.weight_scale / act_scale;
            for v in out.iter_mut() {
                *v *= post_scale;
            }
        } else {
            out.fill(0.0);
        }
    }
}

impl TernaryGemvLayer for QuintritLutAVX512Layer {
    fn forward(&self, activations: &[i8], act_scale: f32, out: &mut [f32]) {
        assert_eq!(activations.len(), self.cols);

        #[cfg(target_arch = "x86_64")]
        {
            thread_local! {
                // Reused across calls so that building the table needs no allocation (and no page faults).
                static QLUT_SCRATCH: std::cell::RefCell<Vec<i8>> = const { std::cell::RefCell::new(Vec::new()) };
            }
            QLUT_SCRATCH.with(|cell| {
                // A rayon worker that waits inside `forward_with_lut` may steal another `forward` call and re-enter
                // this closure on the same thread: the scratch buffer is then busy and a private table is used.
                match cell.try_borrow_mut() {
                    Ok(mut buf) => {
                        let need = quintrit_core::lut::wide_activation_lut_len_avx512(self.cols);
                        if buf.len() < need {
                            buf.resize(need, 0);
                        }
                        let len = quintrit_core::lut::build_activation_lut_wide_avx512_into(activations, self.cols, &mut buf);
                        self.forward_with_lut(&buf[..len], act_scale, out);
                    }
                    Err(_) => {
                        let qlut = quintrit_core::lut::build_activation_lut_wide_avx512(activations, self.cols);
                        self.forward_with_lut(&qlut, act_scale, out);
                    }
                }
            });
        }
        #[cfg(not(target_arch = "x86_64"))]
        {
            let qlut = quintrit_core::lut::build_activation_lut_wide(activations, self.cols);
            self.forward_with_lut(&qlut, act_scale, out);
        }
    }


    fn kernel_type(&self) -> TernaryKernelType {
        TernaryKernelType::QuintritLutAVX512
    }

    fn memory_bytes(&self) -> usize {
        self.transformed_weights.len()
    }

    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn weight_scale(&self) -> f32 {
        self.weight_scale
    }
}

// ============================================================================
// BitNet multiply-add layer (2.00 bpw, comparison baseline)
// ============================================================================

/// Projection layer using the planar multiply-add kernel of Microsoft BitNet (4 trits per byte, 2.00 bpw).
///
/// It exists to compare the 4-trit encoding with the 5-trit encoding on a whole model. `avx512` selects the
/// AVX-512 widening of the kernel; otherwise the AVX2 or NEON kernel of BitNet runs.
#[cfg(feature = "bitnet-baseline")]
#[derive(Clone)]
pub struct BitnetMadLayer {
    packed_weights: Vec<u8>,
    rows: usize,
    cols: usize,
    weight_scale: f32,
    num_threads: usize,
    avx512: bool,
}

#[cfg(feature = "bitnet-baseline")]
impl BitnetMadLayer {
    pub fn new(
        packed_weights: Vec<u8>,
        rows: usize,
        cols: usize,
        weight_scale: f32,
        num_threads: usize,
        avx512: bool,
    ) -> Self {
        assert_eq!(packed_weights.len(), rows * cols / 4);
        Self { packed_weights, rows, cols, weight_scale, num_threads: num_threads.max(1), avx512 }
    }

    pub fn set_num_threads(&mut self, threads: usize) {
        self.num_threads = threads.max(1);
    }
}

#[cfg(feature = "bitnet-baseline")]
impl TernaryGemvLayer for BitnetMadLayer {
    fn forward(&self, activations: &[i8], act_scale: f32, out: &mut [f32]) {
        assert_eq!(activations.len(), self.cols);
        assert_eq!(out.len(), self.rows);
        if act_scale <= 0.0 {
            out.fill(0.0);
            return;
        }

        // The kernel returns sum((trit + 1) * x); the reference runtime removes the offset with the sum of the
        // activations and applies one post scale per projection: (acc - sum_x) * (weight_scale / act_scale).
        #[cfg(target_arch = "x86_64")]
        if self.avx512 {
            if self.num_threads > 1 {
                quintrit_bitnet::gemv_bitnet_mad_avx512_parallel(
                    &self.packed_weights, activations, self.rows, self.cols, out, self.num_threads,
                );
            } else {
                quintrit_bitnet::gemv_bitnet_mad_avx512_1xn(&self.packed_weights, activations, self.rows, self.cols, out);
            }
        } else if self.num_threads > 1 {
            quintrit_bitnet::gemv_bitnet_mad_parallel(
                &self.packed_weights, activations, self.rows, self.cols, out, self.num_threads,
            );
        } else {
            quintrit_bitnet::gemv_bitnet_mad_1xn(&self.packed_weights, activations, self.rows, self.cols, out);
        }
        #[cfg(not(target_arch = "x86_64"))]
        if self.num_threads > 1 {
            quintrit_bitnet::gemv_bitnet_mad_parallel(
                &self.packed_weights, activations, self.rows, self.cols, out, self.num_threads,
            );
        } else {
            quintrit_bitnet::gemv_bitnet_mad_1xn(&self.packed_weights, activations, self.rows, self.cols, out);
        }

        let post_scale = self.weight_scale / act_scale;
        let sum_x: f32 = activations.iter().map(|&x| x as f32).sum();
        for v in out.iter_mut() {
            *v = (*v - sum_x) * post_scale;
        }
    }

    fn kernel_type(&self) -> TernaryKernelType {
        if self.avx512 { TernaryKernelType::BitnetMad512 } else { TernaryKernelType::BitnetMad }
    }

    fn memory_bytes(&self) -> usize {
        self.packed_weights.len()
    }

    fn shape(&self) -> (usize, usize) {
        (self.rows, self.cols)
    }

    fn weight_scale(&self) -> f32 {
        self.weight_scale
    }
}

// ============================================================================
// Unified Linear Layer Container
// ============================================================================

/// Polymorphic container representing any ternary linear GEMV projection layer.
#[derive(Clone)]
pub enum LinearLayer {
    /// Quintrit 32-row tiled velurex encoding (1.60 bpw).
    Quintrit(QuintritLayer),
    /// Quintrit 64-Row Tiled Dual-Hash Activation-LUT with AVX-512 VNNI/VBMI (1.60 bpw).
    QuintritAVX512(QuintritLutAVX512Layer),
    /// BitNet multiply-add kernel (4 trits per byte, 2.00 bpw), comparison baseline.
    #[cfg(feature = "bitnet-baseline")]
    BitnetMad(BitnetMadLayer),
}

impl LinearLayer {
    /// Computes $y = \alpha \beta \cdot (W \cdot x)$.
    #[inline]
    pub fn forward(&self, activations: &[i8], act_scale: f32, out: &mut [f32]) {
        match self {
            Self::Quintrit(layer) => layer.forward(activations, act_scale, out),
            Self::QuintritAVX512(layer) => layer.forward(activations, act_scale, out),
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad(layer) => layer.forward(activations, act_scale, out),
        }
    }

    /// Returns the kernel architecture used by this layer.
    #[inline]
    pub fn kernel_type(&self) -> TernaryKernelType {
        match self {
            Self::Quintrit(layer) => layer.kernel_type(),
            Self::QuintritAVX512(layer) => layer.kernel_type(),
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad(layer) => layer.kernel_type(),
        }
    }

    /// Memory footprint of packed weights in bytes.
    #[inline]
    pub fn memory_bytes(&self) -> usize {
        match self {
            Self::Quintrit(layer) => layer.memory_bytes(),
            Self::QuintritAVX512(layer) => layer.memory_bytes(),
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad(layer) => layer.memory_bytes(),
        }
    }

    /// Matrix dimensions: `(rows, cols)` = `(N, K)`.
    #[inline]
    pub fn shape(&self) -> (usize, usize) {
        match self {
            Self::Quintrit(layer) => layer.shape(),
            Self::QuintritAVX512(layer) => layer.shape(),
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad(layer) => layer.shape(),
        }
    }

    /// Floating-point weight scale factor $\alpha$.
    #[inline]
    pub fn weight_scale(&self) -> f32 {
        match self {
            Self::Quintrit(layer) => layer.weight_scale(),
            Self::QuintritAVX512(layer) => layer.weight_scale(),
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad(layer) => layer.weight_scale(),
        }
    }

    /// Returns the configured thread parallelism for this layer.
    #[inline]
    pub fn num_threads(&self) -> usize {
        match self {
            Self::Quintrit(l) => l.num_threads,
            Self::QuintritAVX512(l) => l.num_threads,
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad(l) => l.num_threads,
        }
    }

    /// Computes $y = \alpha \beta \cdot (W \cdot x)$, optionally reusing a precomputed QLUT.
    #[inline]
    pub fn forward_with_lut(&self, activations: &[i8], act_scale: f32, qlut: Option<&[i8]>, out: &mut [f32]) {
        match self {
            Self::Quintrit(l) => {
                if let Some(qlut_slice) = qlut {
                    l.forward_with_lut(qlut_slice, act_scale, out);
                } else {
                    l.forward(activations, act_scale, out);
                }
            }
            Self::QuintritAVX512(l) => {
                if let Some(qlut_slice) = qlut {
                    l.forward_with_lut(qlut_slice, act_scale, out);
                } else {
                    l.forward(activations, act_scale, out);
                }
            }
            // The BitNet kernel builds no activation table.
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad(l) => l.forward(activations, act_scale, out),
        }
    }

    /// Updates execution thread parallelism for the underlying GEMV kernel.
    #[inline]
    pub fn set_num_threads(&mut self, threads: usize) {
        let t = threads.max(1);
        match self {
            Self::Quintrit(l) => l.set_num_threads(t),
            Self::QuintritAVX512(l) => l.set_num_threads(t),
            #[cfg(feature = "bitnet-baseline")]
            Self::BitnetMad(l) => l.set_num_threads(t),
        }
    }
}

/// Concurrently evaluates two linear layers sharing the exact same input activations.
///
/// Reduces synchronization overhead and halves barrier count in FFN blocks (Gate and Up projections).
pub fn forward_linear_pair(
    layer1: &LinearLayer,
    out1: &mut [f32],
    layer2: &LinearLayer,
    out2: &mut [f32],
    activations: &[i8],
    act_scale: f32,
    qlut: Option<&[i8]>,
    num_threads: usize,
) {
    if num_threads <= 1 {
        layer1.forward_with_lut(activations, act_scale, qlut, out1);
        layer2.forward_with_lut(activations, act_scale, qlut, out2);
        return;
    }

    let pool = quintrit_core::threading::get_thread_pool(num_threads);
    pool.install(|| {
        rayon::join(
            || layer1.forward_with_lut(activations, act_scale, qlut, out1),
            || layer2.forward_with_lut(activations, act_scale, qlut, out2),
        );
    });
}

/// Concurrently evaluates three linear layers sharing the exact same input activations.
///
/// Reduces synchronization overhead and prevents thread starvation on small KV heads in Attention (Q, K, V projections).
pub fn forward_linear_triple(
    layer_q: &LinearLayer,
    out_q: &mut [f32],
    layer_k: &LinearLayer,
    out_k: &mut [f32],
    layer_v: &LinearLayer,
    out_v: &mut [f32],
    activations: &[i8],
    act_scale: f32,
    qlut: Option<&[i8]>,
    num_threads: usize,
) {
    if num_threads <= 1 {
        layer_q.forward_with_lut(activations, act_scale, qlut, out_q);
        layer_k.forward_with_lut(activations, act_scale, qlut, out_k);
        layer_v.forward_with_lut(activations, act_scale, qlut, out_v);
        return;
    }

    let pool = quintrit_core::threading::get_thread_pool(num_threads);
    pool.install(|| {
        rayon::join(
            || layer_q.forward_with_lut(activations, act_scale, qlut, out_q),
            || {
                rayon::join(
                    || layer_k.forward_with_lut(activations, act_scale, qlut, out_k),
                    || layer_v.forward_with_lut(activations, act_scale, qlut, out_v),
                );
            },
        );
    });
}


/// Polymorphic layer dispatcher converting any supported GGUF ternary format
/// (`GGML_TYPE_I2_S`, `GGML_TYPE_TQ1_0`, `GGML_TYPE_TQ2_0`, etc.) into the target execution kernel.
pub fn dispatch_ternary_layer(
    tensor_type: u32,
    raw_data: &[u8],
    rows: usize,
    cols: usize,
    weight_scale: f32,
    target_kernel: TernaryKernelType,
    num_threads: usize,
) -> Result<LinearLayer, String> {
    let decoder = quintrit_format::decoders::get_ternary_decoder(tensor_type)?;

    // The layers run whole tiles (32 rows, 64 for the AVX-512 layout); a projection of another height would be
    // transcoded and then fail an assertion in the first forward pass, so it is rejected here.
    let tile_rows = match target_kernel {
        TernaryKernelType::QuintritLut => 32,
        TernaryKernelType::QuintritLutAVX512 if cfg!(target_arch = "x86_64") => 64,
        TernaryKernelType::QuintritLutAVX512 => 32,
        #[cfg(feature = "bitnet-baseline")]
        TernaryKernelType::BitnetMad | TernaryKernelType::BitnetMad512 => 4,
    };
    if rows == 0 || cols == 0 || !rows.is_multiple_of(tile_rows) {
        return Err(format!(
            "a {rows} x {cols} projection is not supported by the {} kernel: the row count must be a positive multiple of {tile_rows}",
            target_kernel.as_str()
        ));
    }
    if rows.checked_mul(cols).is_none() {
        return Err(format!("a {rows} x {cols} projection is too large"));
    }

    match target_kernel {
        TernaryKernelType::QuintritLut => {
            // Dynamic in-memory transcoding directly into 1.60 bpw Velurex 32-row tiled format in RAM
            let transformed = decoder.transcode_to_velurex_tiled(raw_data, rows, cols)?;
            Ok(LinearLayer::Quintrit(QuintritLayer::new(
                transformed,
                rows,
                cols,
                weight_scale,
                num_threads,
            )))
        }
        TernaryKernelType::QuintritLutAVX512 => {
            let trits = decoder.unpack_trits(raw_data, rows, cols)?;
            let quintrit_packed = quintrit_core::pack::pack_matrix(&trits, rows, cols);
            #[cfg(target_arch = "x86_64")]
            let transformed = quintrit_core::lut::transform_weights_lut_64(&quintrit_packed, rows, cols);
            #[cfg(not(target_arch = "x86_64"))]
            let transformed = quintrit_core::lut::transform_weights_lut(&quintrit_packed, rows, cols);
            Ok(LinearLayer::QuintritAVX512(QuintritLutAVX512Layer::new(
                transformed,
                rows,
                cols,
                weight_scale,
                num_threads,
            )))
        }
        #[cfg(feature = "bitnet-baseline")]
        TernaryKernelType::BitnetMad | TernaryKernelType::BitnetMad512 => {
            let avx512 = target_kernel == TernaryKernelType::BitnetMad512;
            if avx512 {
                #[cfg(target_arch = "x86_64")]
                let supported = is_x86_feature_detected!("avx512f") && is_x86_feature_detected!("avx512bw");
                #[cfg(not(target_arch = "x86_64"))]
                let supported = false;
                if !supported {
                    return Err("the BitnetMad512 kernel needs a CPU with AVX-512 (F and BW)".to_string());
                }
            }
            if !cols.is_multiple_of(quintrit_bitnet::mad::QK_I2_S) {
                return Err(format!(
                    "a {rows} x {cols} projection is not supported by the {} kernel: the column count must be a multiple of {}",
                    target_kernel.as_str(),
                    quintrit_bitnet::mad::QK_I2_S
                ));
            }
            let trits = decoder.unpack_trits(raw_data, rows, cols)?;
            let packed = quintrit_bitnet::pack_trits_planar(&trits, rows, cols);
            Ok(LinearLayer::BitnetMad(BitnetMadLayer::new(packed, rows, cols, weight_scale, num_threads, avx512)))
        }
    }
}

/// Dispatches raw GGUF `GGML_TYPE_I2_S` weights into the selected target kernel representation.
#[inline]
pub fn dispatch_weight_from_i2_s(
    i2_s_weights: &[u8],
    rows: usize,
    cols: usize,
    weight_scale: f32,
    target_kernel: TernaryKernelType,
    num_threads: usize,
) -> LinearLayer {
    dispatch_ternary_layer(
        quintrit_format::GGML_TYPE_I2_S,
        i2_s_weights,
        rows,
        cols,
        weight_scale,
        target_kernel,
        num_threads,
    )
    .expect("Failed to dispatch I2_S layer")
}


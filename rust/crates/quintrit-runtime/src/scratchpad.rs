//! Zero-allocation inference scratchpad for high-throughput forward execution.

use crate::config::ModelConfig;

/// Pre-allocated reusable buffers for autoregressive token generation.
/// Completely eliminates dynamic memory allocations (`malloc`/`free`) in the hot forward loop.
#[derive(Clone, Debug)]
pub struct InferenceScratchpad {
    // Model-level activations
    pub x: Vec<f32>,
    pub x_final: Vec<f32>,
    pub x_final_f16: Vec<u16>,
    pub logits: Vec<f32>,

    // Block-level activations
    pub x_norm: Vec<f32>,
    pub attn_out: Vec<f32>,
    pub ffn_out: Vec<f32>,
    pub x_int8: Vec<i8>,

    // Attention projections and buffers
    pub q: Vec<f32>,
    pub k: Vec<f32>,
    pub v: Vec<f32>,
    pub k_f16: Vec<u16>,
    pub v_f16: Vec<u16>,
    pub q_slice_f16: Vec<u16>,
    pub scores: Vec<f32>,
    pub head_out_f16: Vec<u16>,
    pub attn_sub_buf: Vec<f32>,
    pub o_int8: Vec<i8>,

    // FFN projections and buffers
    pub ffn_int8: Vec<i8>,
    pub gate: Vec<f32>,
    pub up: Vec<f32>,
    pub inter: Vec<f32>,
    pub act_gate: Vec<f32>,
    pub ffn_sub_buf: Vec<f32>,
    pub down_int8: Vec<i8>,

    // Reusable QLUT lookup table buffer for 1.60 bpw velurex encoding
    pub qlut_buf: Vec<i8>,
    /// Scratch for the hi/lo activation split used while building wide QLUTs.
    pub act_split: Vec<i8>,
}

impl InferenceScratchpad {
    /// Pre-allocates all inference scratchpad buffers from raw model topology dimensions.
    pub fn from_dims(
        hidden_size: usize,
        intermediate_size: usize,
        vocab_size: usize,
        num_q_heads: usize,
        num_kv_heads: usize,
        head_dim: usize,
        max_seq_len: usize,
    ) -> Self {
        let q_dim = num_q_heads * head_dim;
        let kv_dim = num_kv_heads * head_dim;
        let effective_max_seq = max_seq_len.max(2048);

        // QLUT buffer size: maximum of hidden_size or intermediate_size columns
        let max_cols = hidden_size.max(intermediate_size);
        let blocks_per_row = max_cols.div_ceil(quintrit_core::pack::QK_5);
        let bytes_per_row = blocks_per_row * quintrit_core::pack::BLOCK_BYTES_5;
        let a2_offset = bytes_per_row * 64;
        // Wide tables: a `[lo | hi]` pair of single tables.
        let qlut_len = 2 * (a2_offset + bytes_per_row.next_multiple_of(32));

        Self {
            x: vec![0.0f32; hidden_size],
            x_final: vec![0.0f32; hidden_size],
            x_final_f16: vec![0u16; hidden_size],
            logits: vec![0.0f32; vocab_size],

            x_norm: vec![0.0f32; hidden_size],
            attn_out: vec![0.0f32; hidden_size],
            ffn_out: vec![0.0f32; hidden_size],
            x_int8: vec![0i8; hidden_size],

            q: vec![0.0f32; q_dim],
            k: vec![0.0f32; kv_dim],
            v: vec![0.0f32; kv_dim],
            k_f16: vec![0u16; kv_dim],
            v_f16: vec![0u16; kv_dim],
            q_slice_f16: vec![0u16; head_dim],
            scores: vec![0.0f32; effective_max_seq],
            head_out_f16: vec![0u16; head_dim],
            attn_sub_buf: vec![0.0f32; q_dim],
            o_int8: vec![0i8; q_dim],

            ffn_int8: vec![0i8; hidden_size],
            gate: vec![0.0f32; intermediate_size],
            up: vec![0.0f32; intermediate_size],
            inter: vec![0.0f32; intermediate_size],
            act_gate: vec![0.0f32; intermediate_size],
            ffn_sub_buf: vec![0.0f32; intermediate_size],
            down_int8: vec![0i8; intermediate_size],

            qlut_buf: vec![0i8; qlut_len],
            act_split: vec![0i8; 2 * max_cols],
        }
    }

    /// Pre-allocates all inference scratchpad buffers according to the model topology.
    pub fn new(config: &ModelConfig, max_seq_len: usize) -> Self {
        Self::from_dims(
            config.hidden_size,
            config.intermediate_size,
            config.vocab_size,
            config.num_attention_heads,
            config.num_key_value_heads,
            config.head_dim,
            max_seq_len.max(config.max_position_embeddings),
        )
    }
}


//! Writes tiny synthetic GGUF files (random ternary weights, a byte-level vocabulary) so that the format and runtime
//! level of the C API can be tested end to end without any model download.
#![allow(dead_code)]

use super::Rng;
use quintrit_format::decoders::i2_s::pack_gguf_i2_s;
use quintrit_format::{GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_I2_S, GGML_TYPE_TQ2_0};
use quintrit_runtime::ops::f32_to_f16;
use std::collections::HashMap;
use std::path::{Path, PathBuf};

pub const HIDDEN: usize = 128;
pub const INTER: usize = 256;
pub const LAYERS: usize = 2;
pub const HEADS: usize = 4;
pub const KV_HEADS: usize = 2;
pub const HEAD_DIM: usize = HIDDEN / HEADS;
pub const VOCAB: usize = 264;
pub const CONTEXT: usize = 48;
pub const BOS: u32 = 256;
pub const EOS: u32 = 257;
pub const I2S_SCALE: f32 = 0.05;

/// Knobs to damage the model in a controlled way.
#[derive(Clone, Default)]
pub struct Flaws {
    pub zero_heads: bool,
    pub drop_tokens: bool,
    pub short_norm: bool,
    pub vocab_mismatch: bool,
    pub odd_i2s_cols: bool,
}

/// Original trits of every ternary tensor, by name: `(rows, cols, trits)`.
pub type TritMap = HashMap<String, (usize, usize, Vec<i8>)>;

struct Tensor {
    name: String,
    shape: Vec<u64>,
    ggml_type: u32,
    data: Vec<u8>,
}

#[derive(Default)]
struct Gguf {
    meta: Vec<u8>,
    kv_count: u64,
    tensors: Vec<Tensor>,
}

fn put_str(buf: &mut Vec<u8>, s: &str) {
    buf.extend_from_slice(&(s.len() as u64).to_le_bytes());
    buf.extend_from_slice(s.as_bytes());
}

impl Gguf {
    fn key(&mut self, key: &str, vtype: u32) {
        put_str(&mut self.meta, key);
        self.meta.extend_from_slice(&vtype.to_le_bytes());
        self.kv_count += 1;
    }
    fn string(&mut self, key: &str, v: &str) {
        self.key(key, 8);
        put_str(&mut self.meta, v);
    }
    fn u32(&mut self, key: &str, v: u32) {
        self.key(key, 4);
        self.meta.extend_from_slice(&v.to_le_bytes());
    }
    fn f32(&mut self, key: &str, v: f32) {
        self.key(key, 6);
        self.meta.extend_from_slice(&v.to_le_bytes());
    }
    fn boolean(&mut self, key: &str, v: bool) {
        self.key(key, 7);
        self.meta.push(v as u8);
    }
    fn string_array(&mut self, key: &str, items: &[String]) {
        self.key(key, 9);
        self.meta.extend_from_slice(&8u32.to_le_bytes());
        self.meta.extend_from_slice(&(items.len() as u64).to_le_bytes());
        for s in items {
            put_str(&mut self.meta, s);
        }
    }
    fn tensor(&mut self, name: &str, shape: &[u64], ggml_type: u32, data: Vec<u8>) {
        self.tensors.push(Tensor { name: name.into(), shape: shape.to_vec(), ggml_type, data });
    }

    fn finish(&self) -> Vec<u8> {
        let align = 32usize;
        let mut out = Vec::new();
        out.extend_from_slice(b"GGUF");
        out.extend_from_slice(&3u32.to_le_bytes());
        out.extend_from_slice(&(self.tensors.len() as u64).to_le_bytes());
        out.extend_from_slice(&self.kv_count.to_le_bytes());
        out.extend_from_slice(&self.meta);
        let mut offset = 0usize;
        for t in &self.tensors {
            put_str(&mut out, &t.name);
            out.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for d in &t.shape {
                out.extend_from_slice(&d.to_le_bytes());
            }
            out.extend_from_slice(&t.ggml_type.to_le_bytes());
            out.extend_from_slice(&(offset as u64).to_le_bytes());
            offset = (offset + t.data.len()).next_multiple_of(align);
        }
        out.resize(out.len().next_multiple_of(align), 0);
        for t in &self.tensors {
            out.extend_from_slice(&t.data);
            out.resize(out.len().next_multiple_of(align), 0);
        }
        out
    }
}

/// GPT-2 byte-to-unicode table: the vocabulary of a byte-level BPE tokenizer.
fn byte_chars() -> Vec<char> {
    let mut bs: Vec<u32> = (33..=126).chain(161..=172).chain(174..=255).collect();
    let mut cs = bs.clone();
    let mut n = 0;
    for b in 0..256u32 {
        if !bs.contains(&b) {
            bs.push(b);
            cs.push(256 + n);
            n += 1;
        }
    }
    let mut table = vec!['\0'; 256];
    for (b, c) in bs.iter().zip(&cs) {
        table[*b as usize] = char::from_u32(*c).unwrap();
    }
    table
}

fn f16_bytes(rng: &mut Rng, n: usize, amplitude: f32) -> Vec<u8> {
    let mut out = Vec::with_capacity(n * 2);
    for _ in 0..n {
        let v = ((rng.next() % 2001) as f32 / 1000.0 - 1.0) * amplitude;
        out.extend_from_slice(&f32_to_f16(v).to_le_bytes());
    }
    out
}

fn f32_bytes(values: impl Iterator<Item = f32>) -> Vec<u8> {
    values.flat_map(|v| v.to_le_bytes()).collect()
}

fn i2s_tensor(g: &mut Gguf, trits: &mut TritMap, rng: &mut Rng, name: &str, rows: usize, cols: usize) {
    i2s_tensor_declared(g, trits, rng, name, rows, cols, cols);
}

/// Like [`i2s_tensor`], but the shape in the tensor table claims `declared_cols` columns (to build damaged files).
fn i2s_tensor_declared(g: &mut Gguf, trits: &mut TritMap, rng: &mut Rng, name: &str, rows: usize, cols: usize, declared_cols: usize) {
    let t = rng.trits(rows * cols);
    let mut data = pack_gguf_i2_s(&t, rows, cols);
    data.extend_from_slice(&I2S_SCALE.to_le_bytes());
    g.tensor(name, &[declared_cols as u64, rows as u64], GGML_TYPE_I2_S, data);
    trits.insert(name.to_string(), (rows, cols, t));
}

/// `TQ2_0` payload (66-byte blocks of 256 weights, scale 0.5 as half precision) for the given trits.
fn tq2_0_bytes(trits: &[i8]) -> Vec<u8> {
    assert_eq!(trits.len() % 256, 0);
    let mut out = Vec::new();
    for block in trits.chunks(256) {
        let mut qs = [0u8; 64];
        for j in 0..2 {
            for l in 0..4 {
                for m in 0..32 {
                    let w = block[j * 128 + l * 32 + m];
                    let code = match w {
                        -1 => 0u8,
                        1 => 2u8,
                        _ => 1u8,
                    };
                    qs[j * 32 + m] |= code << (2 * l);
                }
            }
        }
        out.extend_from_slice(&qs);
        out.extend_from_slice(&0x3800u16.to_le_bytes()); // 0.5
    }
    out
}

/// Builds a model file at `path` and returns the original ternary weights.
pub fn write_model(path: &Path, seed: u64, flaws: &Flaws) -> TritMap {
    let mut rng = Rng::new(seed);
    let mut g = Gguf::default();
    let mut trits = TritMap::new();

    let chars = byte_chars();
    let mut tokens: Vec<String> = chars.iter().map(|c| c.to_string()).collect();
    tokens.push("<s>".into());
    tokens.push("</s>".into());
    for i in 0..(VOCAB - 258) {
        tokens.push(format!("<f{i}>"));
    }
    if flaws.drop_tokens {
        // keep the key but make the array empty by writing it under another name below
    }

    g.string("general.architecture", "llama");
    g.string("general.name", "synthetic");
    g.u32("llama.vocab_size", if flaws.vocab_mismatch { VOCAB as u32 - 8 } else { VOCAB as u32 });
    g.u32("llama.embedding_length", HIDDEN as u32);
    g.u32("llama.feed_forward_length", INTER as u32);
    g.u32("llama.block_count", LAYERS as u32);
    g.u32("llama.attention.head_count", if flaws.zero_heads { 0 } else { HEADS as u32 });
    g.u32("llama.attention.head_count_kv", KV_HEADS as u32);
    g.f32("llama.attention.layer_norm_rms_epsilon", 1e-5);
    g.f32("llama.rope.freq_base", 10000.0);
    g.u32("llama.context_length", CONTEXT as u32);
    g.string("tokenizer.ggml.model", "gpt2");
    g.u32("tokenizer.ggml.bos_token_id", BOS);
    g.u32("tokenizer.ggml.eos_token_id", EOS);
    g.boolean("tokenizer.ggml.add_bos_token", true);
    if !flaws.drop_tokens {
        g.string_array("tokenizer.ggml.tokens", &tokens);
    }
    g.string_array("tokenizer.ggml.merges", &[]);

    g.tensor("token_embd.weight", &[HIDDEN as u64, VOCAB as u64], GGML_TYPE_F16, f16_bytes(&mut rng, HIDDEN * VOCAB, 0.5));
    g.tensor("output_norm.weight", &[HIDDEN as u64], GGML_TYPE_F32, f32_bytes((0..HIDDEN).map(|_| 1.0)));
    g.tensor("output.weight", &[HIDDEN as u64, VOCAB as u64], GGML_TYPE_F16, f16_bytes(&mut rng, HIDDEN * VOCAB, 0.5));
    for l in 0..LAYERS {
        let norm_len = if flaws.short_norm && l == 1 { HIDDEN - 1 } else { HIDDEN };
        g.tensor(&format!("blk.{l}.attn_norm.weight"), &[norm_len as u64], GGML_TYPE_F32, f32_bytes((0..norm_len).map(|_| 1.0)));
        g.tensor(&format!("blk.{l}.ffn_norm.weight"), &[HIDDEN as u64], GGML_TYPE_F32, f32_bytes((0..HIDDEN).map(|_| 1.0)));
        let declared = if flaws.odd_i2s_cols { HIDDEN - 32 } else { HIDDEN };
        i2s_tensor_declared(&mut g, &mut trits, &mut rng, &format!("blk.{l}.attn_q.weight"), HEADS * HEAD_DIM, HIDDEN, declared);
        i2s_tensor(&mut g, &mut trits, &mut rng, &format!("blk.{l}.attn_k.weight"), KV_HEADS * HEAD_DIM, HIDDEN);
        i2s_tensor(&mut g, &mut trits, &mut rng, &format!("blk.{l}.attn_v.weight"), KV_HEADS * HEAD_DIM, HIDDEN);
        i2s_tensor(&mut g, &mut trits, &mut rng, &format!("blk.{l}.attn_output.weight"), HIDDEN, HEADS * HEAD_DIM);
        i2s_tensor(&mut g, &mut trits, &mut rng, &format!("blk.{l}.ffn_gate.weight"), INTER, HIDDEN);
        i2s_tensor(&mut g, &mut trits, &mut rng, &format!("blk.{l}.ffn_up.weight"), INTER, HIDDEN);
        i2s_tensor(&mut g, &mut trits, &mut rng, &format!("blk.{l}.ffn_down.weight"), HIDDEN, INTER);
    }
    // A tensor of another ternary format, not used by the model: 5 rows of 256 columns.
    let tq = rng.trits(5 * 256);
    g.tensor("extra.tq2.weight", &[256, 5], GGML_TYPE_TQ2_0, tq2_0_bytes(&tq));
    trits.insert("extra.tq2.weight".into(), (5, 256, tq));

    std::fs::write(path, g.finish()).unwrap();
    trits
}

/// A scratch directory below the target directory, removed on drop.
pub struct TempDir(pub PathBuf);

impl TempDir {
    pub fn new(tag: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("quintrit-c-{tag}-{}-{:?}", std::process::id(), std::thread::current().id()));
        let _ = std::fs::remove_dir_all(&dir);
        std::fs::create_dir_all(&dir).unwrap();
        Self(dir)
    }
    pub fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.0);
    }
}

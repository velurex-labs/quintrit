//! Tests for the GGUF reader accessors, the tensor decoders and the transcoder statistics.

use byteorder::{LittleEndian, WriteBytesExt};
use quintrit_format::decoders::i2_s::{pack_gguf_i2_s, unpack_gguf_i2_s};
use quintrit_format::*;
use std::io::Write;

// ----------------------------------------------------------------------------
// Minimal GGUF v3 writer
// ----------------------------------------------------------------------------

enum Kv {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    F32(f32),
    Bool(bool),
    Str(&'static str),
    StrArray(Vec<&'static str>),
    U64(u64),
    I64(i64),
    F64(f64),
}

fn put_str(out: &mut Vec<u8>, s: &str) {
    out.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    out.extend_from_slice(s.as_bytes());
}

fn put_kv(out: &mut Vec<u8>, key: &str, value: &Kv) {
    put_str(out, key);
    match value {
        Kv::U8(v) => {
            out.write_u32::<LittleEndian>(0).unwrap();
            out.push(*v);
        }
        Kv::I8(v) => {
            out.write_u32::<LittleEndian>(1).unwrap();
            out.push(*v as u8);
        }
        Kv::U16(v) => {
            out.write_u32::<LittleEndian>(2).unwrap();
            out.write_u16::<LittleEndian>(*v).unwrap();
        }
        Kv::I16(v) => {
            out.write_u32::<LittleEndian>(3).unwrap();
            out.write_i16::<LittleEndian>(*v).unwrap();
        }
        Kv::U32(v) => {
            out.write_u32::<LittleEndian>(4).unwrap();
            out.write_u32::<LittleEndian>(*v).unwrap();
        }
        Kv::I32(v) => {
            out.write_u32::<LittleEndian>(5).unwrap();
            out.write_i32::<LittleEndian>(*v).unwrap();
        }
        Kv::F32(v) => {
            out.write_u32::<LittleEndian>(6).unwrap();
            out.write_f32::<LittleEndian>(*v).unwrap();
        }
        Kv::Bool(v) => {
            out.write_u32::<LittleEndian>(7).unwrap();
            out.push(*v as u8);
        }
        Kv::Str(v) => {
            out.write_u32::<LittleEndian>(8).unwrap();
            put_str(out, v);
        }
        Kv::StrArray(items) => {
            out.write_u32::<LittleEndian>(9).unwrap();
            out.write_u32::<LittleEndian>(8).unwrap();
            out.write_u64::<LittleEndian>(items.len() as u64).unwrap();
            for s in items {
                put_str(out, s);
            }
        }
        Kv::U64(v) => {
            out.write_u32::<LittleEndian>(10).unwrap();
            out.write_u64::<LittleEndian>(*v).unwrap();
        }
        Kv::I64(v) => {
            out.write_u32::<LittleEndian>(11).unwrap();
            out.write_i64::<LittleEndian>(*v).unwrap();
        }
        Kv::F64(v) => {
            out.write_u32::<LittleEndian>(12).unwrap();
            out.write_f64::<LittleEndian>(*v).unwrap();
        }
    }
}

struct TensorSpec {
    name: &'static str,
    shape: Vec<u64>,
    tensor_type: u32,
    payload: Vec<u8>,
}

/// Serializes a GGUF file; tensor payloads are placed at 32-byte aligned offsets.
fn build_gguf(kvs: &[(&str, Kv)], tensors: &[TensorSpec], alignment: usize) -> Vec<u8> {
    let mut offsets = Vec::new();
    let mut cursor = 0usize;
    for t in tensors {
        offsets.push(cursor);
        cursor = (cursor + t.payload.len() + alignment - 1) & !(alignment - 1);
    }

    let mut out = Vec::new();
    out.extend_from_slice(b"GGUF");
    out.write_u32::<LittleEndian>(3).unwrap();
    out.write_u64::<LittleEndian>(tensors.len() as u64).unwrap();
    out.write_u64::<LittleEndian>(kvs.len() as u64).unwrap();
    for (key, value) in kvs {
        put_kv(&mut out, key, value);
    }
    for (t, off) in tensors.iter().zip(&offsets) {
        put_str(&mut out, t.name);
        out.write_u32::<LittleEndian>(t.shape.len() as u32).unwrap();
        for &d in &t.shape {
            out.write_u64::<LittleEndian>(d).unwrap();
        }
        out.write_u32::<LittleEndian>(t.tensor_type).unwrap();
        out.write_u64::<LittleEndian>(*off as u64).unwrap();
    }
    while out.len() % alignment != 0 {
        out.push(0);
    }
    for t in tensors {
        out.write_all(&t.payload).unwrap();
        while out.len() % alignment != 0 {
            out.push(0);
        }
    }
    out
}

fn write_temp(name: &str, bytes: &[u8]) -> std::path::PathBuf {
    let path = std::env::temp_dir().join(format!("quintrit_{}_{}.gguf", name, std::process::id()));
    std::fs::write(&path, bytes).unwrap();
    path
}

fn f32_bytes(values: &[f32]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

fn f16_bytes(values: &[u16]) -> Vec<u8> {
    values.iter().flat_map(|v| v.to_le_bytes()).collect()
}

// ----------------------------------------------------------------------------
// Metadata value conversions
// ----------------------------------------------------------------------------

#[test]
fn metadata_value_conversions() {
    use GgufMetadataValue as V;

    assert_eq!(V::String("x".into()).as_str(), Some("x"));
    assert_eq!(V::Uint32(1).as_str(), None);

    assert_eq!(V::Uint8(7).as_u64(), Some(7));
    assert_eq!(V::Uint16(700).as_u64(), Some(700));
    assert_eq!(V::Uint32(70_000).as_u64(), Some(70_000));
    assert_eq!(V::Uint64(u64::MAX).as_u64(), Some(u64::MAX));
    assert_eq!(V::Int32(5).as_u64(), Some(5));
    assert_eq!(V::Int64(9).as_u64(), Some(9));
    assert_eq!(V::Int32(-1).as_u64(), None, "negative integers are not unsigned");
    assert_eq!(V::Int64(-1).as_u64(), None);
    assert_eq!(V::Float32(1.0).as_u64(), None);
    assert_eq!(V::Uint32(42).as_usize(), Some(42));

    assert_eq!(V::Float32(0.5).as_f32(), Some(0.5));
    assert_eq!(V::Float64(0.25).as_f32(), Some(0.25));
    assert_eq!(V::Uint32(1).as_f32(), None);

    assert_eq!(V::Bool(true).as_bool(), Some(true));
    assert_eq!(V::Uint8(1).as_bool(), None);

    let arr = V::Array(vec![V::String("a".into()), V::String("b".into())]);
    assert_eq!(arr.as_string_array(), Some(vec!["a".to_string(), "b".to_string()]));
    let mixed = V::Array(vec![V::String("a".into()), V::Uint32(1)]);
    assert_eq!(mixed.as_string_array(), None);
    assert_eq!(V::Uint32(1).as_string_array(), None);
}

// ----------------------------------------------------------------------------
// File parsing, typed accessors and tensor access
// ----------------------------------------------------------------------------

#[test]
fn parses_all_metadata_types_and_architecture_accessors() {
    let kvs: Vec<(&str, Kv)> = vec![
        ("general.architecture", Kv::Str("llama")),
        ("llama.context_length", Kv::U32(4096)),
        ("llama.embedding_length", Kv::U32(3072)),
        ("llama.block_count", Kv::U32(40)),
        ("llama.feed_forward_length", Kv::U64(23040)),
        ("llama.attention.head_count", Kv::U32(12)),
        ("llama.attention.head_count_kv", Kv::U32(4)),
        ("llama.attention.layer_norm_rms_epsilon", Kv::F32(1e-6)),
        ("llama.rope.freq_base", Kv::F32(1_000_042.0)),
        ("llama.vocab_size", Kv::U32(131072)),
        ("t.u8", Kv::U8(200)),
        ("t.i8", Kv::I8(-5)),
        ("t.u16", Kv::U16(60_000)),
        ("t.i16", Kv::I16(-300)),
        ("t.i32", Kv::I32(-7)),
        ("t.i64", Kv::I64(-9)),
        ("t.f64", Kv::F64(0.125)),
        ("t.bool", Kv::Bool(true)),
        ("t.tokens", Kv::StrArray(vec!["<s>", "hello", "world"])),
    ];
    let path = write_temp("meta", &build_gguf(&kvs, &[], 32));
    let g = GgufFile::open(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(g.version, 3);
    assert_eq!(g.architecture(), Some("llama"));
    assert_eq!(g.context_length(), Some(4096));
    assert_eq!(g.embedding_length(), Some(3072));
    assert_eq!(g.block_count(), Some(40));
    assert_eq!(g.feed_forward_length(), Some(23040));
    assert_eq!(g.head_count(), Some(12));
    assert_eq!(g.head_count_kv(), Some(4));
    assert_eq!(g.rms_norm_eps(), Some(1e-6));
    assert_eq!(g.rope_freq_base(), Some(1_000_042.0));
    assert_eq!(g.vocab_size(), Some(131072));

    assert_eq!(g.metadata["t.u8"].as_u64(), Some(200));
    assert_eq!(g.metadata["t.u16"].as_u64(), Some(60_000));
    assert_eq!(g.metadata["t.i32"].as_u64(), None);
    assert_eq!(g.metadata["t.f64"].as_f32(), Some(0.125));
    assert_eq!(g.metadata["t.bool"].as_bool(), Some(true));
    assert_eq!(
        g.metadata["t.tokens"].as_string_array(),
        Some(vec!["<s>".to_string(), "hello".to_string(), "world".to_string()])
    );
    assert!(matches!(g.metadata["t.i8"], GgufMetadataValue::Int8(-5)));
    assert!(matches!(g.metadata["t.i16"], GgufMetadataValue::Int16(-300)));
    assert!(matches!(g.metadata["t.i64"], GgufMetadataValue::Int64(-9)));
}

#[test]
fn missing_architecture_keys_yield_none() {
    let path = write_temp("noarch", &build_gguf(&[("x", Kv::U32(1))], &[], 32));
    let g = GgufFile::open(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(g.architecture(), None);
    assert_eq!(g.context_length(), None);
    assert_eq!(g.block_count(), None);
    assert_eq!(g.vocab_size(), None);
}

#[test]
fn tensor_accessors_return_typed_payloads() {
    let f32_values = [1.5f32, -2.0, 0.0, 3.25, 4.0, -5.5, 6.0, 7.75];
    let f16_values: Vec<u16> = (0..16u16).map(|i| 0x3C00 + i).collect();
    let rows = 32usize;
    let cols = 128usize;
    let trits: Vec<i8> = (0..rows * cols).map(|i| ((i * 7 + 1) % 3) as i8 - 1).collect();
    let planar = pack_gguf_i2_s(&trits, rows, cols);
    let mut i2s_payload = planar.clone();
    i2s_payload.extend_from_slice(&0.0625f32.to_le_bytes());

    let tensors = vec![
        TensorSpec { name: "norm.weight", shape: vec![8], tensor_type: GGML_TYPE_F32, payload: f32_bytes(&f32_values) },
        TensorSpec { name: "embd.weight", shape: vec![4, 4], tensor_type: GGML_TYPE_F16, payload: f16_bytes(&f16_values) },
        TensorSpec { name: "blk.0.ffn.weight", shape: vec![cols as u64, rows as u64], tensor_type: GGML_TYPE_I2_S, payload: i2s_payload },
    ];
    let path = write_temp("tensors", &build_gguf(&[("general.architecture", Kv::Str("bitnet"))], &tensors, 32));
    let g = GgufFile::open(&path).unwrap();
    let _ = std::fs::remove_file(&path);

    assert_eq!(g.data_start() % 32, 0);
    assert_eq!(g.tensors["embd.weight"].num_elements(), 16);
    assert_eq!(g.tensors["blk.0.ffn.weight"].num_elements(), rows * cols);

    assert_eq!(g.get_tensor_f32("norm.weight").unwrap(), f32_values.to_vec());
    assert_eq!(g.get_tensor_data("norm.weight").unwrap().len(), 8 * 4);
    assert_eq!(g.get_tensor_f16("embd.weight").unwrap(), &f16_values[..]);
    assert_eq!(g.get_tensor_data("embd.weight").unwrap().len(), 16 * 2);

    let raw = g.get_tensor_data("blk.0.ffn.weight").unwrap();
    assert_eq!(raw, &planar[..]);
    assert_eq!(g.get_tensor_scale("blk.0.ffn.weight"), Some(0.0625));
    assert_eq!(g.get_tensor_scale("norm.weight"), None, "only I2_S carries a trailing scale");

    assert!(g.get_tensor_data("does.not.exist").is_none());
    assert!(g.get_tensor_f32("does.not.exist").is_none());
    assert!(g.get_tensor_scale("does.not.exist").is_none());

    // load_and_transcode_gguf_tensor wires reader, decoder and scale together.
    let matrix = load_and_transcode_gguf_tensor(&g, "blk.0.ffn.weight", rows, cols).unwrap();
    assert_eq!(matrix.scale, 0.0625);
    assert!(verify_transcoded_parity(&planar, &matrix.packed_tiled, rows, cols));
    assert!(load_and_transcode_gguf_tensor(&g, "norm.weight", rows, cols).is_err(), "F32 is not a ternary type");
    assert!(load_and_transcode_gguf_tensor(&g, "missing", rows, cols).is_err());
}

#[test]
fn custom_alignment_moves_data_start() {
    let tensors = vec![TensorSpec {
        name: "t",
        shape: vec![4],
        tensor_type: GGML_TYPE_F32,
        payload: f32_bytes(&[1.0, 2.0, 3.0, 4.0]),
    }];
    let kvs = [("general.alignment", Kv::U32(64))];
    let path = write_temp("align", &build_gguf(&kvs, &tensors, 64));
    let g = GgufFile::open(&path).unwrap();
    let _ = std::fs::remove_file(&path);
    assert_eq!(g.data_start() % 64, 0);
    assert_eq!(g.get_tensor_f32("t").unwrap(), vec![1.0, 2.0, 3.0, 4.0]);
}

#[test]
fn rejects_invalid_and_truncated_files() {
    let path = write_temp("badmagic", b"NOPE\x03\x00\x00\x00");
    assert!(GgufFile::open(&path).is_err(), "bad magic must be rejected");
    let _ = std::fs::remove_file(&path);

    let valid = build_gguf(&[("general.architecture", Kv::Str("llama"))], &[], 32);
    let path = write_temp("trunc", &valid[..valid.len() / 2]);
    assert!(GgufFile::open(&path).is_err(), "truncated header must be rejected");
    let _ = std::fs::remove_file(&path);

    assert!(GgufFile::open(std::env::temp_dir().join("quintrit_definitely_missing.gguf")).is_err());

    // A tensor whose payload lies beyond the end of the file is rejected when the file is opened.
    let tensors = vec![TensorSpec { name: "t", shape: vec![8], tensor_type: GGML_TYPE_F32, payload: f32_bytes(&[0.0; 8]) }];
    let bytes = build_gguf(&[], &tensors, 32);
    let path = write_temp("cut_payload", &bytes[..bytes.len() - 16]);
    assert!(GgufFile::open(&path).is_err(), "a payload beyond the end of the file must be rejected");
    let _ = std::fs::remove_file(&path);
}

// ----------------------------------------------------------------------------
// Decoders and transcoder
// ----------------------------------------------------------------------------

#[test]
fn i2_s_decoder_unpacks_and_transcodes_consistently() {
    let decoder = get_ternary_decoder(GGML_TYPE_I2_S).unwrap();
    assert_eq!(decoder.nominal_bpw(), 2.0);
    for &(rows, cols) in &[(1usize, 128usize), (4, 256), (32, 384)] {
        let trits: Vec<i8> = (0..rows * cols).map(|i| ((i * 5 + 2) % 3) as i8 - 1).collect();
        let packed = pack_gguf_i2_s(&trits, rows, cols);
        assert_eq!(decoder.unpack_trits(&packed, rows, cols).unwrap(), trits);
        assert_eq!(decoder.unpack_trits(&packed, rows, cols).unwrap(), unpack_gguf_i2_s(&packed, rows, cols));
        if rows % 32 == 0 {
            let direct = transcode_trits_to_velurex_tiled(&trits, rows, cols);
            assert_eq!(decoder.transcode_to_velurex_tiled(&packed, rows, cols).unwrap(), direct);
            assert_eq!(transcode_gguf_i2_s_to_velurex_tiled(&packed, rows, cols), direct);
            assert!(verify_transcoded_parity(&packed, &direct, rows, cols));
        }
    }
    // Truncated input is an error, not a panic.
    assert!(decoder.unpack_trits(&[0u8; 8], 4, 128).is_err());
}

#[test]
fn parity_check_detects_corruption() {
    let (rows, cols) = (32usize, 128usize);
    let trits: Vec<i8> = (0..rows * cols).map(|i| ((i * 11 + 4) % 3) as i8 - 1).collect();
    let packed = pack_gguf_i2_s(&trits, rows, cols);
    let mut tiled = transcode_trits_to_velurex_tiled(&trits, rows, cols);
    assert!(verify_transcoded_parity(&packed, &tiled, rows, cols));
    tiled[3] = tiled[3].wrapping_add(1);
    assert!(!verify_transcoded_parity(&packed, &tiled, rows, cols));
}

#[test]
fn transcoded_matrix_reports_memory_statistics() {
    let (rows, cols) = (32usize, 160usize);
    let trits: Vec<i8> = (0..rows * cols).map(|i| ((i * 13 + 5) % 3) as i8 - 1).collect();
    let tiled = transcode_trits_to_velurex_tiled(&trits, rows, cols);
    let bytes_per_row = cols.div_ceil(5 * 32) * 32;
    let matrix = TranscodedTernaryMatrix {
        packed_tiled: tiled.clone(),
        rows,
        cols,
        scale: 1.0,
        bytes_per_row,
    };
    assert_eq!(matrix.dram_bytes(), tiled.len());
    assert_eq!(matrix.baseline_2bpw_bytes(), rows * cols / 4);
    // 5 trits per byte against 4 trits per byte: exactly 20% smaller for aligned widths.
    assert!((matrix.dram_savings_pct() - 20.0).abs() < 1e-9, "got {}", matrix.dram_savings_pct());
}

#[test]
fn ternary_decoder_registry_and_error_paths() {
    assert!(get_ternary_decoder(0).is_err());
    assert!(!is_supported_ternary_type(1));
    let formats = supported_ternary_formats();
    for (name, ty, bpw) in &formats {
        let d = get_ternary_decoder(*ty).unwrap();
        assert_eq!(d.format_name(), *name);
        assert_eq!(d.nominal_bpw(), *bpw);
        assert_eq!(d.tensor_type(), *ty);
    }

    // TQ decoders: shape and size validation and transcode == pack(unpack) consistency.
    for ty in [GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0] {
        let d = get_ternary_decoder(ty).unwrap();
        let block_bytes = if ty == GGML_TYPE_TQ1_0 { 54 } else { 66 };
        assert!(d.unpack_trits(&vec![0u8; block_bytes], 1, 255).is_err(), "not a multiple of 256");
        assert!(d.unpack_trits(&vec![0u8; block_bytes - 1], 1, 256).is_err(), "short buffer");

        let mut data = vec![0u8; block_bytes * 4];
        for (i, b) in data.iter_mut().enumerate() {
            *b = (i * 37 + 11) as u8;
        }
        let trits = d.unpack_trits(&data, 32, 32).unwrap();
        assert_eq!(trits.len(), 32 * 32);
        assert!(trits.iter().all(|t| (-1..=1).contains(t)));
        let tiled = d.transcode_to_velurex_tiled(&data, 32, 32).unwrap();
        assert_eq!(tiled, transcode_trits_to_velurex_tiled(&trits, 32, 32));
    }
}

// ----------------------------------------------------------------------------
// Layout conformance with the reference ggml dequantizers
// ----------------------------------------------------------------------------

/// Re-implementation of ggml `dequantize_row_tq2_0` (block of 256 values, scale ignored).
fn ggml_tq2_0_trits(block: &[u8]) -> Vec<i8> {
    let mut out = Vec::new();
    for j in (0..64).step_by(32) {
        for l in 0..4 {
            for m in 0..32 {
                // Code 3 is never produced by the encoder (ggml would yield +2); the decoder treats it as zero.
                out.push(match (block[j + m] >> (l * 2)) & 3 { 0 => -1, 2 => 1, _ => 0 });
            }
        }
    }
    out
}

/// Re-implementation of ggml `dequantize_row_tq1_0` (block of 256 values, scale ignored).
fn ggml_tq1_0_trits(block: &[u8]) -> Vec<i8> {
    let pow3 = [1u32, 3, 9, 27, 81, 243];
    let digit = |byte: u8, n: usize| -> i8 {
        let q = (byte as u32 * pow3[n]) as u8 as u32;
        (((q * 3) >> 8) as i16 - 1) as i8
    };
    let qs = &block[..48];
    let qh = &block[48..52];
    let mut out = Vec::new();
    for j in (0..32).step_by(32) {
        for n in 0..5 {
            for m in 0..32 {
                out.push(digit(qs[j + m], n));
            }
        }
    }
    for n in 0..5 {
        for m in 0..16 {
            out.push(digit(qs[32 + m], n));
        }
    }
    for n in 0..4 {
        for j in 0..4 {
            out.push(digit(qh[j], n));
        }
    }
    out
}

fn test_block(len: usize) -> Vec<u8> {
    (0..len).map(|i| ((i * 73 + 29) % 251) as u8).collect()
}

#[test]
fn tq2_0_decoder_matches_ggml_reference_layout() {
    let block = test_block(66);
    let got = get_ternary_decoder(GGML_TYPE_TQ2_0).unwrap().unpack_trits(&block, 1, 256).unwrap();
    assert_eq!(got, ggml_tq2_0_trits(&block));
}

#[test]
fn tq1_0_decoder_matches_ggml_reference_layout() {
    // Arbitrary bytes: the reference dequantizer is defined for every byte value.
    let block = test_block(54);
    let got = get_ternary_decoder(GGML_TYPE_TQ1_0).unwrap().unpack_trits(&block, 1, 256).unwrap();
    assert_eq!(got, ggml_tq1_0_trits(&block));
}

// ----------------------------------------------------------------------------
// Encoders following ggml `quantize_row_tq1_0_ref` / `quantize_row_tq2_0_ref`
// ----------------------------------------------------------------------------

fn encode_tq2_0(trits: &[i8], d_bits: u16) -> Vec<u8> {
    assert_eq!(trits.len(), 256);
    let mut block = vec![0u8; 66];
    for j in (0..64).step_by(32) {
        let base = j / 32 * 128;
        for m in 0..32 {
            let mut q = 0u8;
            for n in 0..4 {
                q += (((trits[base + m + n * 32] + 1) as u8) & 3) << (2 * n);
            }
            block[j + m] = q;
        }
    }
    block[64..66].copy_from_slice(&d_bits.to_le_bytes());
    block
}

fn encode_tq1_0(trits: &[i8], d_bits: u16) -> Vec<u8> {
    assert_eq!(trits.len(), 256);
    let ceil = |q: u32| ((q * 256 + 242) / 243) as u8;
    let mut block = vec![0u8; 54];
    let mut off = 0;
    for m in 0..32 {
        let mut q = 0u32;
        for n in 0..5 {
            q = q * 3 + (trits[off + m + n * 32] + 1) as u32;
        }
        block[m] = ceil(q);
    }
    off += 160;
    for m in 0..16 {
        let mut q = 0u32;
        for n in 0..5 {
            q = q * 3 + (trits[off + m + n * 16] + 1) as u32;
        }
        block[32 + m] = ceil(q);
    }
    off += 80;
    for j in 0..4 {
        let mut q = 0u32;
        for m in 0..4 {
            q = q * 3 + (trits[off + j + m * 4] + 1) as u32;
        }
        block[48 + j] = ceil(q * 3);
    }
    block[52..54].copy_from_slice(&d_bits.to_le_bytes());
    block
}

fn pseudo_trits(n: usize, seed: u64) -> Vec<i8> {
    let mut s = seed | 1;
    (0..n)
        .map(|_| {
            s ^= s << 13;
            s ^= s >> 7;
            s ^= s << 17;
            (s % 3) as i8 - 1
        })
        .collect()
}

const D_HALF: u16 = 0x3800; // 0.5

fn encode_matrix(ty: u32, trits: &[i8], d_bits: &[u16]) -> Vec<u8> {
    trits
        .chunks(256)
        .zip(d_bits.iter().cycle())
        .flat_map(|(chunk, &d)| if ty == GGML_TYPE_TQ1_0 { encode_tq1_0(chunk, d) } else { encode_tq2_0(chunk, d) })
        .collect()
}

#[test]
fn tq_decoders_roundtrip_encoder_output_for_every_trit_pattern() {
    for ty in [GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0] {
        let decoder = get_ternary_decoder(ty).unwrap();
        for &(rows, cols) in &[(1usize, 256usize), (4, 256), (3, 512), (32, 1024)] {
            let trits = pseudo_trits(rows * cols, 77 + rows as u64);
            let data = encode_matrix(ty, &trits, &[D_HALF]);
            assert_eq!(decoder.unpack_trits(&data, rows, cols).unwrap(), trits, "type {ty} ({rows}x{cols})");
        }
        // Extremes: all -1, all 0, all +1 (exercise the ceiling-division digit packing).
        for v in [-1i8, 0, 1] {
            let trits = vec![v; 256];
            assert_eq!(decoder.unpack_trits(&encode_matrix(ty, &trits, &[D_HALF]), 1, 256).unwrap(), trits);
        }
    }
}

#[test]
fn tq_block_scale_is_exposed_only_when_uniform() {
    for ty in [GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0] {
        let trits = pseudo_trits(4 * 256, 5);
        assert_eq!(uniform_block_scale(ty, &encode_matrix(ty, &trits, &[D_HALF])), Some(0.5));
        assert_eq!(uniform_block_scale(ty, &encode_matrix(ty, &trits, &[0x3C00])), Some(1.0));
        assert_eq!(uniform_block_scale(ty, &encode_matrix(ty, &trits, &[0xC000])), Some(-2.0));
        assert_eq!(
            uniform_block_scale(ty, &encode_matrix(ty, &trits, &[0x0001])),
            Some(2.0f32.powi(-24)),
            "subnormal half"
        );
        assert_eq!(uniform_block_scale(ty, &encode_matrix(ty, &trits, &[0x0000])), Some(0.0));
        assert_eq!(
            uniform_block_scale(ty, &encode_matrix(ty, &trits, &[D_HALF, D_HALF, 0x3C00, D_HALF])),
            None,
            "differing block scales cannot be represented"
        );
        let good = encode_matrix(ty, &trits, &[D_HALF]);
        assert_eq!(uniform_block_scale(ty, &good[..good.len() - 1]), None, "truncated data");
        assert_eq!(uniform_block_scale(ty, &[]), None);
    }
    assert_eq!(uniform_block_scale(GGML_TYPE_I2_S, &[0u8; 64]), None);
}

#[test]
fn tq_tensors_load_through_gguf_with_scale_and_transcode_to_the_same_trits() {
    use quintrit_core::lut::untransform_weights_lut;
    use quintrit_core::pack::unpack_matrix;
    let (rows, cols) = (32usize, 512usize);
    for ty in [GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0] {
        let trits = pseudo_trits(rows * cols, 9);
        let payload = encode_matrix(ty, &trits, &[D_HALF]);
        let tensors = vec![TensorSpec {
            name: "blk.0.ffn.weight",
            shape: vec![cols as u64, rows as u64],
            tensor_type: ty,
            payload: payload.clone(),
        }];
        let path = write_temp("tq_load", &build_gguf(&[("general.architecture", Kv::Str("llama"))], &tensors, 32));
        let g = GgufFile::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);

        assert_eq!(g.get_tensor_data("blk.0.ffn.weight").unwrap(), &payload[..], "type {ty}");
        assert_eq!(g.get_tensor_scale("blk.0.ffn.weight"), Some(0.5));

        let matrix = load_and_transcode_gguf_tensor(&g, "blk.0.ffn.weight", rows, cols).unwrap();
        assert_eq!(matrix.scale, 0.5);
        let restored = unpack_matrix(&untransform_weights_lut(&matrix.packed_tiled, rows, cols), rows, cols);
        assert_eq!(&restored[..rows * cols], &trits[..], "type {ty}");
    }
}

#[test]
fn tq_tensor_with_differing_block_scales_is_rejected() {
    let (rows, cols) = (32usize, 256usize);
    for ty in [GGML_TYPE_TQ1_0, GGML_TYPE_TQ2_0] {
        let trits = pseudo_trits(rows * cols, 11);
        let payload = encode_matrix(ty, &trits, &[D_HALF, 0x3C00]);
        let tensors = vec![TensorSpec { name: "w", shape: vec![cols as u64, rows as u64], tensor_type: ty, payload }];
        let path = write_temp("tq_mixed", &build_gguf(&[], &tensors, 32));
        let g = GgufFile::open(&path).unwrap();
        let _ = std::fs::remove_file(&path);
        assert_eq!(g.get_tensor_scale("w"), None);
        assert!(load_and_transcode_gguf_tensor(&g, "w", rows, cols).is_err(), "type {ty}");
    }
}

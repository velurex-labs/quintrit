//! Integration and verification tests for `quintrit-format`.

use byteorder::{LittleEndian, WriteBytesExt};
use std::fs::File;
use std::io::{Seek, Write};
use quintrit_core::lut::{build_activation_lut, gemv_lut_1xn};
use quintrit_format::decoders::i2_s::pack_gguf_i2_s;
use quintrit_format::*;

#[test]
fn test_in_memory_transcode_parity_and_savings() {
    let test_shapes = [(32, 128), (64, 256), (32, 640), (96, 512)];

    for (rows, cols) in test_shapes {
        // Generate pseudo-random ternary trits in {-1, 0, +1}
        let trits: Vec<i8> = (0..rows * cols)
            .map(|i| ((i * 17 + 5) % 3) as i8 - 1)
            .collect();

        // 1. Pack into the 2.00 bpw planar GGUF GGML_TYPE_I2_S layout
        let packed_planar = pack_gguf_i2_s(&trits, rows, cols);
        assert_eq!(packed_planar.len(), (rows * cols) / 4);

        // 2. Transcode on-the-fly into 1.60 bpw velurex encoding 32-row tiled layout
        let transcoded_tiled = transcode_gguf_i2_s_to_velurex_tiled(&packed_planar, rows, cols);

        // 3. Verify 100% bit-for-bit parity
        let is_identical = verify_transcoded_parity(&packed_planar, &transcoded_tiled, rows, cols);
        assert!(
            is_identical,
            "Transcoded Velurex tiled matrix must be bit-for-bit identical to source trits for shape ({}, {})",
            rows, cols
        );

        // 4. Verify DRAM savings for aligned dimensions (cols multiple of 160)
        if cols % 160 == 0 {
            let base_bytes = (rows * cols) / 4;
            let actual_bytes = transcoded_tiled.len();
            let savings = (1.0 - (actual_bytes as f64 / base_bytes as f64)) * 100.0;
            assert!(
                (savings - 20.0).abs() < 1e-4,
                "Expected exact 20.0% DRAM reduction, got {:.2}% for shape ({}, {})",
                savings, rows, cols
            );
        }
    }
}

#[test]
fn test_gemv_execution_on_transcoded_matrix() {
    let rows = 64;
    let cols = 640;

    let trits: Vec<i8> = (0..rows * cols)
        .map(|i| ((i * 31 + 7) % 3) as i8 - 1)
        .collect();

    let activations: Vec<i8> = (0..cols)
        .map(|i| (((i * 13 + 3) % 51) as i16 - 25) as i8)
        .collect();

    // 1. Compute ground-truth naive dot product
    let mut expected = vec![0.0f32; rows];
    for r in 0..rows {
        let mut sum = 0i32;
        for c in 0..cols {
            sum += (trits[r * cols + c] as i32) * (activations[c] as i32);
        }
        expected[r] = sum as f32;
    }

    // 2. Pack to the planar GGUF I2_S layout
    let packed_planar = pack_gguf_i2_s(&trits, rows, cols);

    // 3. Transcode to Velurex 32-row tiled format in RAM
    let transcoded_tiled = transcode_gguf_i2_s_to_velurex_tiled(&packed_planar, rows, cols);

    // 4. Execute GEMV via SIMD lookup tables
    let qlut = build_activation_lut(&activations, cols);
    let mut actual = vec![0.0f32; rows];
    gemv_lut_1xn(&transcoded_tiled, &qlut, rows, cols, &mut actual);

    // 5. Verify results match ground truth exactly
    for r in 0..rows {
        assert_eq!(
            actual[r], expected[r],
            "GEMV output mismatch at row {}: got {}, expected {}",
            r, actual[r], expected[r]
        );
    }
}

#[test]
fn test_synthetic_gguf_parsing_and_transcoding() {
    let temp_dir = std::env::temp_dir();
    let temp_path = temp_dir.join(format!("test_model_{}.gguf", std::process::id()));

    // Construct a minimal valid GGUF v3 file
    let rows = 32;
    let cols = 128;
    let trits: Vec<i8> = (0..rows * cols)
        .map(|i| ((i * 19 + 11) % 3) as i8 - 1)
        .collect();
    let planar_data = pack_gguf_i2_s(&trits, rows, cols);
    let tensor_scale = 0.054321f32;

    {
        let mut f = File::create(&temp_path).expect("failed to create temp gguf file");

        // Magic "GGUF"
        f.write_all(b"GGUF").unwrap();
        // Version 3
        f.write_u32::<LittleEndian>(3).unwrap();
        // Tensor count = 1
        f.write_u64::<LittleEndian>(1).unwrap();
        // Metadata KV count = 2
        f.write_u64::<LittleEndian>(2).unwrap();

        // KV 1: "general.architecture" -> "bitnet"
        write_test_string(&mut f, "general.architecture");
        f.write_u32::<LittleEndian>(8).unwrap(); // type 8 = string
        write_test_string(&mut f, "bitnet");

        // KV 2: "bitnet.block_count" -> 4 (uint32)
        write_test_string(&mut f, "bitnet.block_count");
        f.write_u32::<LittleEndian>(4).unwrap(); // type 4 = uint32
        f.write_u32::<LittleEndian>(4).unwrap();

        // Tensor 1: "blk.0.attn_q.weight", 2D [128, 32], type 36 (I2_S), offset 0
        write_test_string(&mut f, "blk.0.attn_q.weight");
        f.write_u32::<LittleEndian>(2).unwrap(); // 2 dims
        f.write_u64::<LittleEndian>(cols as u64).unwrap();
        f.write_u64::<LittleEndian>(rows as u64).unwrap();
        f.write_u32::<LittleEndian>(GGML_TYPE_I2_S).unwrap();
        f.write_u64::<LittleEndian>(0).unwrap(); // offset 0 from data start

        // Align to 32 bytes
        let current_pos = f.stream_position().unwrap() as usize;
        let alignment = 32;
        let pad = ((current_pos + alignment - 1) & !(alignment - 1)) - current_pos;
        f.write_all(&vec![0u8; pad]).unwrap();

        // Write tensor payload: packed planar bytes + trailing f32 scale
        f.write_all(&planar_data).unwrap();
        f.write_f32::<LittleEndian>(tensor_scale).unwrap();
        f.flush().unwrap();
    }

    // Now open and parse with GgufFile
    let gguf = GgufFile::open(&temp_path).expect("failed to open test gguf");
    assert_eq!(gguf.architecture(), Some("bitnet"));
    assert_eq!(gguf.block_count(), Some(4));

    // Verify tensor info
    assert!(gguf.tensors.contains_key("blk.0.attn_q.weight"));
    let tensor = &gguf.tensors["blk.0.attn_q.weight"];
    assert_eq!(tensor.tensor_type, GGML_TYPE_I2_S);
    assert_eq!(tensor.shape, vec![cols, rows]);

    // Verify scale factor extraction
    let read_scale = gguf.get_tensor_scale("blk.0.attn_q.weight").unwrap();
    assert!((read_scale - tensor_scale).abs() < 1e-6);

    // Transcode into 1.60 bpw Velurex tiled matrix
    let matrix = load_and_transcode_gguf_tensor(&gguf, "blk.0.attn_q.weight", rows, cols)
        .expect("transcode failed");

    assert_eq!(matrix.rows, rows);
    assert_eq!(matrix.cols, cols);
    assert_eq!(matrix.scale, tensor_scale);

    // Verify parity against original planar data
    assert!(verify_transcoded_parity(
        &planar_data,
        &matrix.packed_tiled,
        rows,
        cols
    ));

    let _ = std::fs::remove_file(temp_path);
}

#[test]
fn test_modular_decoders_registry_and_formats() {
    // 1. Verify format registry
    let formats = supported_ternary_formats();
    assert_eq!(formats.len(), 3);
    assert!(is_supported_ternary_type(GGML_TYPE_I2_S));
    assert!(is_supported_ternary_type(GGML_TYPE_TQ1_0));
    assert!(is_supported_ternary_type(GGML_TYPE_TQ2_0));
    assert!(!is_supported_ternary_type(9999));

    // 2. Test TQ1_0 decoder
    let tq1_decoder = get_ternary_decoder(GGML_TYPE_TQ1_0).unwrap();
    assert_eq!(tq1_decoder.tensor_type(), GGML_TYPE_TQ1_0);
    assert_eq!(tq1_decoder.nominal_bpw(), 1.6875);

    // Synthetic block of 256 weights: 54 bytes
    let mut synthetic_tq1_block = vec![0u8; 54];
    // ggml stores the first of five trits as the most significant base-3 digit, scaled by 256/243.
    synthetic_tq1_block[0] = 196;
    let decoded_trits = tq1_decoder.unpack_trits(&synthetic_tq1_block, 1, 256).unwrap();
    assert_eq!(decoded_trits.len(), 256);
    // trit 0 of byte 0: (196 * 3) >> 8 = 2 -> +1; trit 1 (at element 32): ((196 * 3) % 256 * 3) >> 8 = 0 -> -1
    assert_eq!(decoded_trits[0], 1);
    assert_eq!(decoded_trits[32], -1);

    // 3. Test TQ2_0 decoder
    let tq2_decoder = get_ternary_decoder(GGML_TYPE_TQ2_0).unwrap();
    assert_eq!(tq2_decoder.tensor_type(), GGML_TYPE_TQ2_0);
    assert_eq!(tq2_decoder.nominal_bpw(), 2.0625);

    let synthetic_tq2_block = vec![0u8; 66];
    let decoded_tq2_trits = tq2_decoder.unpack_trits(&synthetic_tq2_block, 1, 256).unwrap();
    assert_eq!(decoded_tq2_trits.len(), 256);
}

fn write_test_string<W: Write>(w: &mut W, s: &str) {
    w.write_u64::<LittleEndian>(s.len() as u64).unwrap();
    w.write_all(s.as_bytes()).unwrap();
}

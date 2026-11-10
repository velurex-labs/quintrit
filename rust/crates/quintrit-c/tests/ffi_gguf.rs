//! Format-level tests: GGUF open, listing, metadata, tensor conversion, and robustness against damaged files.

mod common;

use common::synth::*;
use common::*;
use quintrit_c::*;
use quintrit_format::TernaryDecoder;
use std::ffi::c_char;
use std::ptr::{null, null_mut};

fn open(path: &std::path::Path) -> Result<*mut GgufHandle, i32> {
    let p = cstr(path.to_str().unwrap());
    let mut h: *mut GgufHandle = std::ptr::dangling_mut::<GgufHandle>(); // must be reset to NULL by the call
    let st = unsafe { quintrit_gguf_open(p.as_ptr(), &mut h) };
    if st == QUINTRIT_OK {
        assert!(!h.is_null());
        Ok(h)
    } else {
        assert!(h.is_null(), "*out_gguf must be NULL after a failure");
        Err(st)
    }
}

fn tensor_info(h: *const GgufHandle, name: &str) -> TensorInfo {
    let mut info: TensorInfo = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<TensorInfo>() as u32;
    expect_status(unsafe { quintrit_gguf_tensor_find(h, cstr(name).as_ptr(), &mut info) }, QUINTRIT_OK);
    info
}

fn layout_of() -> TiledLayout {
    TiledLayout { struct_size: size_of::<TiledLayout>() as u32, ..Default::default() }
}

#[test]
fn open_list_and_read_metadata() {
    let dir = TempDir::new("gguf-list");
    let file = dir.path("m.gguf");
    let trits = write_model(&file, 1, &Flaws::default());
    let h = open(&file).unwrap();

    let mut info = GgufInfo { struct_size: size_of::<GgufInfo>() as u32, ..Default::default() };
    expect_status(unsafe { quintrit_gguf_info(h, &mut info) }, QUINTRIT_OK);
    assert_eq!(info.version, 3);
    let expected_tensors = 3 + LAYERS * 9 + 1;
    assert_eq!(info.tensor_count as usize, expected_tensors);
    assert_eq!(info.data_offset % 32, 0);

    // Every tensor can be listed by index, names are unique, offsets are ascending.
    let mut names = std::collections::HashSet::new();
    let mut last_offset = 0u64;
    for i in 0..expected_tensors {
        let mut t: TensorInfo = unsafe { std::mem::zeroed() };
        t.struct_size = size_of::<TensorInfo>() as u32;
        expect_status(unsafe { quintrit_gguf_tensor_at(h, i, &mut t) }, QUINTRIT_OK);
        let name = from_c(t.name);
        assert!(names.insert(name.clone()), "duplicate {name}");
        assert!(t.offset >= last_offset);
        last_offset = t.offset;
        assert_eq!(t.num_elements, t.shape[..t.n_dims as usize].iter().product::<u64>());
        let ternary = matches!(from_c(t.type_name).as_str(), "I2_S" | "TQ2_0");
        assert_eq!(t.is_ternary == 1, ternary, "{name}");
        assert_eq!(t.is_ternary == 1, trits.contains_key(&name), "{name}");
    }
    let mut t: TensorInfo = unsafe { std::mem::zeroed() };
    t.struct_size = size_of::<TensorInfo>() as u32;
    expect_status(unsafe { quintrit_gguf_tensor_at(h, expected_tensors, &mut t) }, QUINTRIT_ERR_INVALID_ARGUMENT);

    let q = tensor_info(h, "blk.1.attn_k.weight");
    assert_eq!((q.n_dims, q.shape[0], q.shape[1]), (2, HIDDEN as u64, (KV_HEADS * HEAD_DIM) as u64));
    assert_eq!(from_c(q.type_name), "I2_S");
    assert_eq!(q.ggml_type, QUINTRIT_GGML_I2_S);
    assert_eq!(q.data_bytes, q.num_elements / 4);
    assert_eq!(from_c(tensor_info(h, "token_embd.weight").type_name), "F16");
    assert_eq!(from_c(tensor_info(h, "output_norm.weight").type_name), "F32");
    assert_eq!(from_c(tensor_info(h, "extra.tq2.weight").type_name), "TQ2_0");
    expect_status(unsafe { quintrit_gguf_tensor_find(h, cstr("nope").as_ptr(), &mut t) }, QUINTRIT_ERR_NOT_FOUND);

    // Metadata.
    let mut len = 0usize;
    let key = cstr("general.name");
    assert_eq!(unsafe { quintrit_gguf_metadata_string(h, key.as_ptr(), null_mut(), 0, &mut len) }, QUINTRIT_ERR_BUFFER_TOO_SMALL);
    assert_eq!(len, "synthetic".len());
    let mut buf = vec![0 as c_char; len + 1];
    expect_status(unsafe { quintrit_gguf_metadata_string(h, key.as_ptr(), buf.as_mut_ptr(), buf.len(), &mut len) }, QUINTRIT_OK);
    assert_eq!(from_c(buf.as_ptr()), "synthetic");
    let mut small = vec![0x7f as c_char; len];
    expect_status(unsafe { quintrit_gguf_metadata_string(h, key.as_ptr(), small.as_mut_ptr(), small.len(), &mut len) }, QUINTRIT_ERR_BUFFER_TOO_SMALL);
    assert!(small.iter().all(|&c| c == 0x7f), "nothing may be written on failure");
    expect_status(unsafe { quintrit_gguf_metadata_string(h, cstr("llama.context_length").as_ptr(), buf.as_mut_ptr(), buf.len(), &mut len) }, QUINTRIT_ERR_FORMAT);
    expect_status(unsafe { quintrit_gguf_metadata_string(h, cstr("missing").as_ptr(), buf.as_mut_ptr(), buf.len(), &mut len) }, QUINTRIT_ERR_NOT_FOUND);
    expect_status(unsafe { quintrit_gguf_metadata_string(h, null(), buf.as_mut_ptr(), buf.len(), &mut len) }, QUINTRIT_ERR_NULL_POINTER);

    let mut v = 0u64;
    expect_status(unsafe { quintrit_gguf_metadata_u64(h, cstr("llama.context_length").as_ptr(), &mut v) }, QUINTRIT_OK);
    assert_eq!(v as usize, CONTEXT);
    expect_status(unsafe { quintrit_gguf_metadata_u64(h, cstr("general.name").as_ptr(), &mut v) }, QUINTRIT_ERR_FORMAT);
    expect_status(unsafe { quintrit_gguf_metadata_u64(h, cstr("missing").as_ptr(), &mut v) }, QUINTRIT_ERR_NOT_FOUND);
    expect_status(unsafe { quintrit_gguf_metadata_u64(h, cstr("llama.context_length").as_ptr(), null_mut()) }, QUINTRIT_ERR_NULL_POINTER);

    // Raw data.
    let mut ptr: *const u8 = null();
    let mut n = 0usize;
    expect_status(unsafe { quintrit_gguf_tensor_data(h, cstr("blk.0.attn_q.weight").as_ptr(), &mut ptr, &mut n) }, QUINTRIT_OK);
    assert_eq!(n, HIDDEN * HIDDEN / 4);
    assert!(!ptr.is_null());
    expect_status(unsafe { quintrit_gguf_tensor_data(h, cstr("nope").as_ptr(), &mut ptr, &mut n) }, QUINTRIT_ERR_NOT_FOUND);

    // Wrong struct sizes.
    let mut bad: TensorInfo = unsafe { std::mem::zeroed() };
    expect_status(unsafe { quintrit_gguf_tensor_at(h, 0, &mut bad) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    expect_status(unsafe { quintrit_gguf_tensor_at(h, 0, null_mut()) }, QUINTRIT_ERR_NULL_POINTER);
    expect_status(unsafe { quintrit_gguf_info(h, null_mut()) }, QUINTRIT_ERR_NULL_POINTER);

    expect_status(unsafe { quintrit_gguf_close(h) }, QUINTRIT_OK);
    expect_status(unsafe { quintrit_gguf_close(null_mut()) }, QUINTRIT_OK);
}

#[test]
fn convert_ternary_tensors_to_every_layout() {
    let dir = TempDir::new("gguf-convert");
    let file = dir.path("m.gguf");
    let trits = write_model(&file, 2, &Flaws::default());
    let h = open(&file).unwrap();
    let mut rng = Rng::new(5);

    for name in ["blk.0.attn_q.weight", "blk.1.attn_k.weight", "blk.1.ffn_down.weight", "extra.tq2.weight"] {
        let (rows, cols, original) = &trits[name];
        for kernel in [QUINTRIT_KERNEL_SCALAR, QUINTRIT_KERNEL_SIMD32, QUINTRIT_KERNEL_AVX512, QUINTRIT_KERNEL_AUTO] {
            let mut layout = layout_of();
            let mut scale = 0.0f32;
            expect_status(unsafe { quintrit_gguf_tensor_layout(h, cstr(name).as_ptr(), kernel, &mut layout, &mut scale) }, QUINTRIT_OK);
            assert_eq!((layout.rows as usize, layout.cols as usize), (*rows, *cols), "{name}");
            let want_scale = if name.starts_with("extra") { 0.5 } else { I2S_SCALE };
            assert_eq!(scale, want_scale, "{name}");

            let mut out = vec![0u8; layout.total_bytes as usize];
            let mut layout2 = layout_of();
            let mut scale2 = 0.0f32;
            expect_status(
                unsafe { quintrit_gguf_convert_tensor(h, cstr(name).as_ptr(), kernel, out.as_mut_ptr(), out.len(), &mut layout2, &mut scale2) },
                QUINTRIT_OK,
            );
            assert_eq!(layout2, layout);
            assert_eq!(scale2, scale);

            // Identical to packing the original trits through the kernel-level API.
            let mut direct = vec![0u8; layout.total_bytes as usize];
            expect_status(
                unsafe { quintrit_pack_matrix(kernel, original.as_ptr(), original.len(), *rows, *cols, direct.as_mut_ptr(), direct.len()) },
                QUINTRIT_OK,
            );
            assert_eq!(out, direct, "{name} kernel {kernel}");

            // NULL layout / scale pointers are accepted.
            expect_status(
                unsafe { quintrit_gguf_convert_tensor(h, cstr(name).as_ptr(), kernel, out.as_mut_ptr(), out.len(), null_mut(), null_mut()) },
                QUINTRIT_OK,
            );
        }

        // The 32-row layout equals what the engine's decoder transcodes (it pads to whole tiles the same way).
        if rows % 32 == 0 && name.starts_with("blk") {
            let mut p: *const u8 = null();
            let mut n = 0usize;
            expect_status(unsafe { quintrit_gguf_tensor_data(h, cstr(name).as_ptr(), &mut p, &mut n) }, QUINTRIT_OK);
            let raw = unsafe { std::slice::from_raw_parts(p, n) };
            let expected = quintrit_format::I2_S_DECODER.transcode_to_velurex_tiled(raw, *rows, *cols).unwrap();
            let mut layout = layout_of();
            expect_status(unsafe { quintrit_gguf_tensor_layout(h, cstr(name).as_ptr(), QUINTRIT_KERNEL_SIMD32, &mut layout, null_mut()) }, QUINTRIT_OK);
            let mut out = vec![0u8; layout.total_bytes as usize];
            expect_status(unsafe { quintrit_gguf_convert_tensor(h, cstr(name).as_ptr(), QUINTRIT_KERNEL_SIMD32, out.as_mut_ptr(), out.len(), null_mut(), null_mut()) }, QUINTRIT_OK);
            assert_eq!(out, expected, "{name} against the format crate's transcoder");
        }

        // GEMV on the converted tensor (best supported kernel) equals the brute-force product with the original trits.
        let mut resolved = 0i32;
        expect_status(unsafe { quintrit_kernel_resolve(QUINTRIT_KERNEL_AUTO, &mut resolved) }, QUINTRIT_OK);
        let mut layout = layout_of();
        expect_status(unsafe { quintrit_gguf_tensor_layout(h, cstr(name).as_ptr(), resolved, &mut layout, null_mut()) }, QUINTRIT_OK);
        let mut tiled = vec![0u8; layout.total_bytes as usize];
        expect_status(unsafe { quintrit_gguf_convert_tensor(h, cstr(name).as_ptr(), resolved, tiled.as_mut_ptr(), tiled.len(), null_mut(), null_mut()) }, QUINTRIT_OK);
        let acts = rng.acts(*cols);
        let mut out = vec![0i32; *rows];
        expect_status(
            unsafe { quintrit_gemv_i32(resolved, tiled.as_ptr(), tiled.len(), *rows, *cols, acts.as_ptr(), acts.len(), out.as_mut_ptr(), out.len(), 1) },
            QUINTRIT_OK,
        );
        assert_eq!(out.iter().map(|&v| v as i64).collect::<Vec<_>>(), truth(original, &acts, *rows, *cols), "{name}");
    }

    // Errors.
    let name = cstr("blk.0.attn_q.weight");
    let mut layout = layout_of();
    expect_status(unsafe { quintrit_gguf_tensor_layout(h, cstr("token_embd.weight").as_ptr(), 1, &mut layout, null_mut()) }, QUINTRIT_ERR_UNSUPPORTED);
    expect_status(unsafe { quintrit_gguf_tensor_layout(h, cstr("output_norm.weight").as_ptr(), 1, &mut layout, null_mut()) }, QUINTRIT_ERR_UNSUPPORTED);
    expect_status(unsafe { quintrit_gguf_tensor_layout(h, cstr("nope").as_ptr(), 1, &mut layout, null_mut()) }, QUINTRIT_ERR_NOT_FOUND);
    expect_status(unsafe { quintrit_gguf_tensor_layout(h, name.as_ptr(), 17, &mut layout, null_mut()) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    expect_status(unsafe { quintrit_gguf_tensor_layout(h, null(), 1, &mut layout, null_mut()) }, QUINTRIT_ERR_NULL_POINTER);
    let mut small = vec![0u8; 10];
    expect_status(unsafe { quintrit_gguf_convert_tensor(h, name.as_ptr(), 1, small.as_mut_ptr(), small.len(), null_mut(), null_mut()) }, QUINTRIT_ERR_BUFFER_TOO_SMALL);
    expect_status(unsafe { quintrit_gguf_convert_tensor(h, name.as_ptr(), 1, null_mut(), 1 << 20, null_mut(), null_mut()) }, QUINTRIT_ERR_NULL_POINTER);
    let mut bad_layout = TiledLayout::default();
    let mut big = vec![0u8; 1 << 16];
    expect_status(unsafe { quintrit_gguf_convert_tensor(h, name.as_ptr(), 1, big.as_mut_ptr(), big.len(), &mut bad_layout, null_mut()) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    expect_status(unsafe { quintrit_gguf_close(h) }, QUINTRIT_OK);
}

#[test]
fn open_rejects_bad_files_and_arguments() {
    let dir = TempDir::new("gguf-bad");
    let good = dir.path("good.gguf");
    write_model(&good, 3, &Flaws::default());
    let bytes = std::fs::read(&good).unwrap();

    assert_eq!(open(&dir.path("missing.gguf")).unwrap_err(), QUINTRIT_ERR_IO);
    assert_eq!(open(&dir.0).unwrap_err(), QUINTRIT_ERR_IO, "a directory is not a model");

    let write = |name: &str, data: &[u8]| {
        let p = dir.path(name);
        std::fs::write(&p, data).unwrap();
        p
    };
    assert_eq!(open(&write("empty.gguf", b"")).unwrap_err(), QUINTRIT_ERR_FORMAT);
    assert_eq!(open(&write("short.gguf", b"GGU")).unwrap_err(), QUINTRIT_ERR_FORMAT);
    assert_eq!(open(&write("magic.gguf", &[b"GGUX".as_slice(), &bytes[4..]].concat())).unwrap_err(), QUINTRIT_ERR_FORMAT);
    assert_eq!(open(&write("text.gguf", &vec![b'a'; 4096])).unwrap_err(), QUINTRIT_ERR_FORMAT);
    let mut v1 = bytes.clone();
    v1[4..8].copy_from_slice(&1u32.to_le_bytes());
    assert_eq!(open(&write("v1.gguf", &v1)).unwrap_err(), QUINTRIT_ERR_FORMAT);
    // Truncated in the header, in the tensor table and inside the data.
    for cut in [10usize, 24, 100, 600, 3000, bytes.len() / 2, bytes.len() - 40] {
        let st = open(&write("cut.gguf", &bytes[..cut]));
        assert_eq!(st.unwrap_err(), QUINTRIT_ERR_FORMAT, "cut at {cut}");
    }
    // Absurd counts and lengths.
    for (offset, what) in [(8usize, "tensor count"), (16, "metadata count"), (24, "first key length")] {
        let mut b = bytes.clone();
        b[offset..offset + 8].copy_from_slice(&u64::MAX.to_le_bytes());
        assert_eq!(open(&write("huge.gguf", &b)).unwrap_err(), QUINTRIT_ERR_FORMAT, "{what}");
        let mut b = bytes.clone();
        b[offset..offset + 8].copy_from_slice(&(1u64 << 60).to_le_bytes());
        assert_eq!(open(&write("huge.gguf", &b)).unwrap_err(), QUINTRIT_ERR_FORMAT, "{what}");
    }

    // Arguments.
    let mut h: *mut GgufHandle = null_mut();
    expect_status(unsafe { quintrit_gguf_open(null(), &mut h) }, QUINTRIT_ERR_NULL_POINTER);
    let p = cstr(good.to_str().unwrap());
    expect_status(unsafe { quintrit_gguf_open(p.as_ptr(), null_mut()) }, QUINTRIT_ERR_NULL_POINTER);
    let bad_utf8 = [0xffu8, 0xfe, 0];
    expect_status(unsafe { quintrit_gguf_open(bad_utf8.as_ptr() as *const c_char, &mut h) }, QUINTRIT_ERR_INVALID_ARGUMENT);

    // Handles of the wrong kind or garbage are rejected, not dereferenced blindly.
    let fake = [0u64; 16];
    let fake_ptr = fake.as_ptr() as *const GgufHandle;
    let mut info = GgufInfo { struct_size: size_of::<GgufInfo>() as u32, ..Default::default() };
    expect_status(unsafe { quintrit_gguf_info(fake_ptr, &mut info) }, QUINTRIT_ERR_INVALID_HANDLE);
    expect_status(unsafe { quintrit_gguf_info(null(), &mut info) }, QUINTRIT_ERR_INVALID_HANDLE);
    expect_status(unsafe { quintrit_gguf_close(fake_ptr as *mut GgufHandle) }, QUINTRIT_ERR_INVALID_HANDLE);
}

/// Damages the header and tensor table of a valid file in many ways: opening it must never panic (INTERNAL) and, when
/// it succeeds, every tensor must still be listable and convertible without a crash.
#[test]
fn damaged_files_never_panic() {
    let dir = TempDir::new("gguf-fuzz");
    let good = dir.path("good.gguf");
    write_model(&good, 4, &Flaws::default());
    let bytes = std::fs::read(&good).unwrap();
    let h = open(&good).unwrap();
    let mut info = GgufInfo { struct_size: size_of::<GgufInfo>() as u32, ..Default::default() };
    expect_status(unsafe { quintrit_gguf_info(h, &mut info) }, QUINTRIT_OK);
    let header = info.data_offset as usize;
    unsafe { quintrit_gguf_close(h) };

    let mut rng = Rng::new(77);
    let victim = dir.path("fuzz.gguf");
    let (mut opened, mut rejected) = (0, 0);
    for round in 0..400 {
        let mut b = bytes.clone();
        let patches = 1 + rng.next() as usize % 3;
        for _ in 0..patches {
            let pos = rng.next() as usize % header;
            match rng.next() % 4 {
                0 => b[pos] = rng.next() as u8,
                1 => b[pos] = 0xFF,
                2 => {
                    let end = (pos + 8).min(b.len());
                    b[pos..end].fill(0xFF);
                }
                _ => {
                    let end = (pos + 8).min(b.len());
                    b[pos..end].fill(0);
                }
            }
        }
        std::fs::write(&victim, &b).unwrap();
        match open(&victim) {
            Ok(h) => {
                opened += 1;
                let mut info = GgufInfo { struct_size: size_of::<GgufInfo>() as u32, ..Default::default() };
                expect_status(unsafe { quintrit_gguf_info(h, &mut info) }, QUINTRIT_OK);
                for i in 0..info.tensor_count as usize {
                    let mut t: TensorInfo = unsafe { std::mem::zeroed() };
                    t.struct_size = size_of::<TensorInfo>() as u32;
                    expect_status(unsafe { quintrit_gguf_tensor_at(h, i, &mut t) }, QUINTRIT_OK);
                    if t.is_ternary == 1 {
                        let mut layout = layout_of();
                        let st = unsafe { quintrit_gguf_tensor_layout(h, t.name, QUINTRIT_KERNEL_SIMD32, &mut layout, null_mut()) };
                        assert_ne!(st, QUINTRIT_ERR_INTERNAL, "round {round}: {:?}", last_error());
                        if st == QUINTRIT_OK && layout.total_bytes < (1 << 26) {
                            let mut out = vec![0u8; layout.total_bytes as usize];
                            let st = unsafe { quintrit_gguf_convert_tensor(h, t.name, QUINTRIT_KERNEL_SIMD32, out.as_mut_ptr(), out.len(), null_mut(), null_mut()) };
                            assert_ne!(st, QUINTRIT_ERR_INTERNAL, "round {round}: {:?}", last_error());
                        }
                    }
                }
                unsafe { quintrit_gguf_close(h) };
            }
            Err(st) => {
                rejected += 1;
                assert_ne!(st, QUINTRIT_ERR_INTERNAL, "round {round}: {:?}", last_error());
            }
        }
    }
    assert!(opened > 0 && rejected > 0, "the fuzz mix should hit both outcomes (opened {opened}, rejected {rejected})");
}

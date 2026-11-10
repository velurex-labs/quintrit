//! The GGUF reader against crafted malformed and hostile files: huge counts and lengths, truncation at every byte,
//! overflowing shapes and offsets, overlapping and misplaced tensors, bad alignment. Every case must return an error
//! (or, for the benign mutations, parse) without panicking and without allocating on the word of the file. The
//! files are parsed from anonymous read-only memory maps, which is what `GgufFile::open` hands to the parser too.

use quintrit_format::{GgufFile, GGML_TYPE_F16, GGML_TYPE_F32, GGML_TYPE_I2_S, GGML_TYPE_TQ2_0};
use std::io::ErrorKind;

// ----------------------------------------------------------------------------
// Raw GGUF builder (independent of the reader under test)
// ----------------------------------------------------------------------------

fn u32b(v: u32) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

fn u64b(v: u64) -> Vec<u8> {
    v.to_le_bytes().to_vec()
}

fn string(s: &str) -> Vec<u8> {
    [u64b(s.len() as u64), s.as_bytes().to_vec()].concat()
}

fn kv_u32(key: &str, v: u32) -> Vec<u8> {
    [string(key), u32b(4), u32b(v)].concat()
}

fn kv_str(key: &str, v: &str) -> Vec<u8> {
    [string(key), u32b(8), string(v)].concat()
}

/// A tensor descriptor with an arbitrary dimension list.
fn tensor(name: &str, dims: &[u64], ty: u32, offset: u64) -> Vec<u8> {
    let mut out = string(name);
    out.extend(u32b(dims.len() as u32));
    for &d in dims {
        out.extend(u64b(d));
    }
    out.extend(u32b(ty));
    out.extend(u64b(offset));
    out
}

/// Header with explicit (possibly lying) counts followed by `body`.
fn raw(version: u32, tensor_count: u64, kv_count: u64, body: &[u8]) -> Vec<u8> {
    [b"GGUF".to_vec(), u32b(version), u64b(tensor_count), u64b(kv_count), body.to_vec()].concat()
}

/// Pads `bytes` with zeros to the default alignment of 32, so that an empty data section starts inside the file.
fn padded(mut bytes: Vec<u8>) -> Vec<u8> {
    while !bytes.len().is_multiple_of(32) {
        bytes.push(0);
    }
    bytes
}

/// A consistent file: `kvs` and `tensors` are already encoded; the data section is padded to `alignment`.
fn file(kvs: &[Vec<u8>], tensors: &[Vec<u8>], data: &[u8], alignment: usize) -> Vec<u8> {
    let body: Vec<u8> = kvs.iter().chain(tensors).flatten().copied().collect();
    let mut out = raw(3, tensors.len() as u64, kvs.len() as u64, &body);
    while !out.len().is_multiple_of(alignment) {
        out.push(0);
    }
    out.extend_from_slice(data);
    out
}

/// Parses `bytes` through a read-only anonymous memory map, hands the result to `check` and unmaps afterwards.
fn with_file<R>(bytes: &[u8], check: impl FnOnce(std::io::Result<GgufFile>) -> R) -> R {
    assert!(!bytes.is_empty(), "an empty mapping is not possible");
    let mut map = memmap2::MmapMut::map_anon(bytes.len()).unwrap();
    map.copy_from_slice(bytes);
    check(GgufFile::from_mmap(map.make_read_only().unwrap()))
}

fn rejects(what: &str, bytes: &[u8]) -> std::io::Error {
    with_file(bytes, |r| match r {
        Ok(_) => panic!("{what}: the file must be rejected"),
        Err(e) => e,
    })
}

fn accepts(what: &str, bytes: &[u8]) {
    with_file(bytes, |r| {
        if let Err(e) = r {
            panic!("{what}: the file must be accepted, got {e}");
        }
    });
}

fn f32_data(n: usize) -> Vec<u8> {
    (0..n).flat_map(|i| (i as f32).to_le_bytes()).collect()
}

fn valid_two_tensor_file() -> Vec<u8> {
    let mut data = f32_data(8);
    data.extend(f32_data(8));
    file(
        &[kv_str("general.architecture", "llama"), kv_u32("llama.block_count", 2)],
        &[tensor("a", &[8], GGML_TYPE_F32, 0), tensor("b", &[2, 4], GGML_TYPE_F32, 32)],
        &data,
        32,
    )
}

// ----------------------------------------------------------------------------
// Baseline
// ----------------------------------------------------------------------------

#[test]
fn the_builder_produces_a_file_the_reader_accepts() {
    with_file(&valid_two_tensor_file(), |r| {
        let g = r.unwrap();
        assert_eq!(g.version, 3);
        assert_eq!(g.architecture(), Some("llama"));
        assert_eq!(g.tensors["a"].shape, vec![8]);
        assert_eq!(g.tensors["b"].shape, vec![2, 4]);
        assert_eq!(g.get_tensor_f32("b").unwrap()[3], 3.0);
        assert_eq!(g.get_tensor_data("a").unwrap().len(), 32);
    });
}

// ----------------------------------------------------------------------------
// Counts and lengths
// ----------------------------------------------------------------------------

#[test]
fn huge_counts_are_rejected_without_allocating() {
    for &n in &[u64::MAX, 1 << 63, 1 << 40, 1 << 32, 1_000_000] {
        rejects(&format!("tensor count {n}"), &raw(3, n, 0, &[]));
        rejects(&format!("metadata count {n}"), &raw(3, 0, n, &[]));
        rejects(&format!("both counts {n}"), &raw(3, n, n, &kv_u32("k", 1)));
    }
    // A count that is plausible for the declared entry size but larger than the file.
    let one = kv_u32("llama.block_count", 2);
    let e = rejects("count larger than the file", &raw(3, 0, 1000, &one));
    assert_eq!(e.kind(), ErrorKind::InvalidData);
}

#[test]
fn huge_string_lengths_are_rejected() {
    // Key length claims more than the file holds, or more than the u64 range of usize.
    for &len in &[u64::MAX, 1 << 62, 1 << 32, 70_000, 100] {
        let body = [u64b(len), b"abc".to_vec(), u32b(4), u32b(1)].concat();
        rejects(&format!("key length {len}"), &raw(3, 0, 1, &body));
        // The same for a string value.
        let body = [string("k"), u32b(8), u64b(len), b"abc".to_vec()].concat();
        rejects(&format!("value length {len}"), &raw(3, 0, 1, &body));
        // And for a tensor name.
        let body = [u64b(len), b"abc".to_vec(), vec![0u8; 64]].concat();
        rejects(&format!("tensor name length {len}"), &raw(3, 1, 0, &body));
    }
}

#[test]
fn a_key_longer_than_the_name_limit_is_rejected_even_if_it_fits_the_file() {
    let long = "k".repeat(70_000);
    rejects("long key", &file(&[kv_u32(&long, 1)], &[], &[], 32));
    rejects("long tensor name", &file(&[], &[tensor(&long, &[4], GGML_TYPE_F32, 0)], &f32_data(4), 32));
}

#[test]
fn invalid_utf8_is_rejected() {
    let body = [u64b(2), vec![0xFF, 0xFE], u32b(4), u32b(1)].concat();
    assert_eq!(rejects("key", &raw(3, 0, 1, &body)).kind(), ErrorKind::InvalidData);
    let body = [string("k"), u32b(8), u64b(2), vec![0xC0, 0x80]].concat();
    assert_eq!(rejects("value", &raw(3, 0, 1, &body)).kind(), ErrorKind::InvalidData);
}

#[test]
fn metadata_arrays_are_bounded_by_the_file() {
    // Element count far beyond the file for every element type.
    for item_type in [0u32, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12] {
        for &count in &[u64::MAX, 1 << 60, 1 << 40, 1_000_000] {
            let body = [string("arr"), u32b(9), u32b(item_type), u64b(count), vec![0u8; 16]].concat();
            rejects(&format!("array type {item_type} count {count}"), &raw(3, 0, 1, &body));
        }
    }
    // Unknown element type, with a count of zero (still rejected: the layout would be unknown).
    let body = [string("arr"), u32b(9), u32b(77), u64b(0)].concat();
    rejects("unknown array element type", &raw(3, 0, 1, &body));
    // Unknown value type.
    let body = [string("v"), u32b(99), vec![0u8; 8]].concat();
    rejects("unknown value type", &raw(3, 0, 1, &body));
}

#[test]
fn deeply_nested_arrays_are_rejected() {
    // An array of arrays of arrays ...: every level holds one element of the next level.
    fn nested(levels: usize) -> Vec<u8> {
        // innermost: array of u8 with 1 element
        let mut v = [u32b(0), u64b(1), vec![7u8]].concat();
        for _ in 0..levels {
            v = [u32b(9), u64b(1), v].concat();
        }
        v
    }
    // `nested(n)` is the body of an array value that contains `n` further levels of arrays inside it.
    let ok = [string("arr"), u32b(9), nested(2)].concat();
    accepts("shallow nesting", &padded(raw(3, 0, 1, &ok)));
    let deep = [string("arr"), u32b(9), nested(40)].concat();
    assert!(rejects("40 levels of nesting", &raw(3, 0, 1, &deep)).to_string().contains("nested"));
    let very_deep = [string("arr"), u32b(9), nested(2_000)].concat();
    assert!(rejects("2000 levels of nesting", &raw(3, 0, 1, &very_deep)).to_string().contains("nested"));
}

// ----------------------------------------------------------------------------
// Version, magic, truncation
// ----------------------------------------------------------------------------

#[test]
fn bad_magic_and_versions_are_rejected() {
    let good = valid_two_tensor_file();
    let mut bad = good.clone();
    bad[0] = b'X';
    assert_eq!(rejects("magic", &bad).kind(), ErrorKind::InvalidData);
    for version in [0u32, 1, 4, 5, 0x0300_0000, u32::MAX] {
        let mut b = good.clone();
        b[4..8].copy_from_slice(&version.to_le_bytes());
        rejects(&format!("version {version}"), &b);
    }
    for version in [2u32, 3] {
        let mut b = good.clone();
        b[4..8].copy_from_slice(&version.to_le_bytes());
        accepts(&format!("version {version}"), &b);
    }
}

#[test]
fn truncation_at_every_length_never_panics() {
    let good = valid_two_tensor_file();
    // Everything before the end of the last payload must be an error; only the alignment padding may be missing.
    let payload_end = good.len();
    for cut in 0..payload_end {
        // An empty file cannot be memory-mapped; the reader is only given non-empty ones.
        if cut == 0 {
            continue;
        }
        let e = rejects(&format!("cut at {cut}"), &good[..cut]);
        assert!(matches!(e.kind(), ErrorKind::InvalidData | ErrorKind::UnexpectedEof), "cut at {cut}: {e:?}");
    }
    accepts("complete file", &good);
}

// ----------------------------------------------------------------------------
// Tensor descriptors
// ----------------------------------------------------------------------------

#[test]
fn absurd_dimensions_are_rejected() {
    let data = f32_data(16);
    let cases: Vec<(&str, Vec<u64>)> = vec![
        ("dimension u64::MAX", vec![u64::MAX]),
        ("dimension 2^63", vec![1 << 63]),
        ("product overflows u64", vec![1 << 33, 1 << 33]),
        ("product overflows u128-safe wrap", vec![u64::MAX, u64::MAX, u64::MAX]),
        ("more than 2^40 elements", vec![(1 << 40) + 1]),
        ("zero in a huge product", vec![u64::MAX, 0, u64::MAX]),
        ("nine dimensions", vec![1; 9]),
        ("no dimensions", vec![]),
    ];
    for (what, dims) in cases {
        let f = file(&[], &[tensor("t", &dims, GGML_TYPE_F32, 0)], &data, 32);
        // "zero in a huge product" overflows on the first multiplication before reaching the zero.
        rejects(what, &f);
    }
    // The dimension count itself can lie.
    for &n_dims in &[0u32, 9, 1000, u32::MAX] {
        let body = [string("t"), u32b(n_dims), u64b(4), u32b(GGML_TYPE_F32), u64b(0)].concat();
        rejects(&format!("dimension count {n_dims}"), &raw(3, 1, 0, &body));
    }
}

#[test]
fn a_shape_that_is_valid_but_larger_than_the_file_is_rejected_for_known_types() {
    let data = f32_data(8);
    for (ty, dims) in [
        (GGML_TYPE_F32, vec![1_000_000u64]),
        (GGML_TYPE_F16, vec![1 << 30]),
        (GGML_TYPE_I2_S, vec![1 << 36]),
        (GGML_TYPE_TQ2_0, vec![256 * 1000]),
    ] {
        let f = file(&[], &[tensor("t", &dims, ty, 0)], &data, 32);
        rejects(&format!("type {ty} shape {dims:?}"), &f);
    }
}

#[test]
fn tensor_offsets_outside_the_data_section_are_rejected() {
    let data = f32_data(8);
    for &offset in &[32u64, 64, 1 << 20, 1 << 40, u64::MAX, u64::MAX - 31, u64::MAX - 7, (1 << 63) + 32] {
        let f = file(&[], &[tensor("t", &[8], GGML_TYPE_F32, offset)], &data, 32);
        rejects(&format!("offset {offset}"), &f);
    }
    // The largest offset inside the section with a zero-length payload (a zero-element tensor) is fine.
    accepts("empty tensor at the end", &file(&[], &[tensor("t", &[0], GGML_TYPE_F32, 32)], &data, 32));
    // One element past the end is not.
    rejects("payload one element past the end", &file(&[], &[tensor("t", &[8], GGML_TYPE_F32, 32)], &data, 32));
}

#[test]
fn overlapping_tensors_are_rejected() {
    let data = f32_data(16);
    let overlap = file(
        &[],
        &[tensor("a", &[16], GGML_TYPE_F32, 0), tensor("b", &[8], GGML_TYPE_F32, 32)],
        &data,
        32,
    );
    let e = rejects("overlap", &overlap);
    assert!(e.to_string().contains("overlaps"), "{e}");
    // The same tensor twice under different names.
    rejects(
        "identical ranges",
        &file(&[], &[tensor("a", &[8], GGML_TYPE_F32, 0), tensor("b", &[8], GGML_TYPE_F32, 0)], &data, 32),
    );
    // Adjacent tensors are not overlapping, whatever the order of the descriptors.
    accepts(
        "adjacent, reversed order",
        &file(&[], &[tensor("b", &[8], GGML_TYPE_F32, 32), tensor("a", &[8], GGML_TYPE_F32, 0)], &data, 32),
    );
}

#[test]
fn misaligned_tensor_offsets_are_rejected() {
    let data = f32_data(16);
    for &offset in &[1u64, 2, 4, 16, 31, 33] {
        rejects(
            &format!("offset {offset}"),
            &file(&[], &[tensor("t", &[4], GGML_TYPE_F32, offset)], &data, 32),
        );
    }
}

#[test]
fn duplicate_names_and_keys_are_rejected() {
    let data = f32_data(16);
    rejects(
        "duplicate tensor",
        &file(&[], &[tensor("t", &[4], GGML_TYPE_F32, 0), tensor("t", &[4], GGML_TYPE_F32, 32)], &data, 32),
    );
    rejects("duplicate key", &file(&[kv_u32("k", 1), kv_u32("k", 2)], &[], &[], 32));
}

// ----------------------------------------------------------------------------
// Alignment
// ----------------------------------------------------------------------------

#[test]
fn bad_alignment_is_rejected() {
    let data = f32_data(8);
    let t = [tensor("t", &[8], GGML_TYPE_F32, 0)];
    for &a in &[0u32, 3, 6, 33, 100, 1000, u32::MAX] {
        rejects(&format!("alignment {a}"), &file(&[kv_u32("general.alignment", a)], &t, &data, 32));
    }
    // A power of two so large that the data section would start beyond the file.
    rejects("alignment 2^31", &file(&[kv_u32("general.alignment", 1 << 31)], &t, &data, 32));
    let huge = [string("general.alignment"), u32b(10), u64b(1 << 62)].concat();
    rejects("alignment 2^62 as u64", &file(&[huge], &t, &data, 32));
    let max = [string("general.alignment"), u32b(10), u64b(u64::MAX)].concat();
    rejects("alignment u64::MAX", &file(&[max], &t, &data, 32));
    // Wrong value type.
    rejects("alignment as string", &file(&[kv_str("general.alignment", "32")], &t, &data, 32));
    let neg = [string("general.alignment"), u32b(5), (-32i32).to_le_bytes().to_vec()].concat();
    rejects("negative alignment", &file(&[neg], &t, &data, 32));
}

#[test]
fn valid_alignments_are_honoured() {
    let data = f32_data(8);
    for a in [1u32, 2, 4, 8, 16, 32, 64, 256, 4096] {
        let f = file(&[kv_u32("general.alignment", a)], &[tensor("t", &[8], GGML_TYPE_F32, 0)], &data, a as usize);
        with_file(&f, |r| {
            let g = r.unwrap_or_else(|e| panic!("alignment {a}: {e}"));
            assert_eq!(g.data_start() % a as usize, 0);
            assert_eq!(g.get_tensor_f32("t").unwrap().len(), 8);
        });
    }
}

#[test]
fn the_data_section_must_start_inside_the_file() {
    // Only descriptors, padded to the default alignment: the data section is empty but starts at the end of the file.
    accepts("empty data section", &file(&[kv_u32("k", 1)], &[], &[], 32));
    // A larger alignment than the builder used pushes the start of the data section past the end of the file.
    rejects("padding beyond the end", &file(&[kv_u32("general.alignment", 4096)], &[], &[], 1));
}

// ----------------------------------------------------------------------------
// Accessors on odd but valid files
// ----------------------------------------------------------------------------

#[test]
fn accessors_never_read_out_of_bounds_or_misaligned() {
    // Alignment 1 lets an F16 tensor start at an odd file offset: it is served as bytes or None, never read unaligned.
    let mut data = vec![0u8]; // one byte of padding in front of the F16 tensor
    data.extend_from_slice(&[1, 0, 2, 0, 3, 0, 4, 0]);
    let f = file(
        &[kv_u32("general.alignment", 1)],
        &[tensor("h", &[4], GGML_TYPE_F16, 1), tensor("scalar", &[1], GGML_TYPE_I2_S, 0)],
        &data,
        1,
    );
    with_file(&f, |r| {
        let g = r.unwrap();
        assert_eq!(g.get_tensor_data("h").unwrap().len(), 8);
        // Either aligned (and then correct) or None.
        if let Some(h) = g.get_tensor_f16("h") {
            assert_eq!(h, &[1, 2, 3, 4]);
        }
        // Wrong accessor for the type.
        assert!(g.get_tensor_f32("h").is_none());
        assert!(g.get_tensor_f16("scalar").is_none());
        // The scale of an I2_S tensor that is not followed by four bytes is absent, not a panic.
        let _ = g.get_tensor_scale("scalar");
        assert!(g.get_tensor_scale("missing").is_none());
    });
}

// ----------------------------------------------------------------------------
// Mutation sweep
// ----------------------------------------------------------------------------

#[test]
fn single_byte_mutations_of_a_valid_header_never_panic() {
    let good = valid_two_tensor_file();
    let header_len = good.len() - 64; // descriptors, metadata and padding: everything before the payload
    for pos in 0..header_len {
        for &value in &[0x00u8, 0x01, 0x7F, 0x80, 0xFF] {
            let mut b = good.clone();
            b[pos] = value;
            with_file(&b, |r| {
                if let Ok(g) = r {
                    // Whatever parsed must be internally consistent and safe to read.
                    for name in g.tensors.keys() {
                        let _ = g.get_tensor_data(name);
                        let _ = g.get_tensor_f32(name);
                        let _ = g.get_tensor_f16(name);
                        let _ = g.get_tensor_scale(name);
                    }
                }
            });
        }
    }
}

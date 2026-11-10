//! Memory-mapped GGUF v2/v3 model parser.
//!
//! Provides fast, safe reading of metadata, model architecture hyperparameters,
//! and tensor data layouts from standard GGUF files (including Microsoft BitNet b1.58
//! and community ternary models with `GGML_TYPE_I2_S` tensors).
//!
//! ## Hostile and damaged files
//!
//! The parser does not trust anything it reads. Every count and length is bounded by the bytes that remain in the
//! file before anything is allocated, all offset and size arithmetic is checked, and a file that is inconsistent is
//! rejected with an [`std::io::Error`] (`InvalidData`, or `UnexpectedEof` for truncation) instead of panicking or
//! allocating on the word of the file. The checks, in file order:
//!
//! - the magic bytes and the version (2 or 3);
//! - the metadata and tensor counts must fit into the remaining bytes (each entry needs a minimum number of bytes);
//! - keys and tensor names are at most 65536 bytes and unique; strings must fit into the remaining bytes and be UTF-8;
//! - metadata arrays nest at most four levels deep and their element count must fit into the remaining bytes;
//! - `general.alignment` must be an unsigned integer and a power of two;
//! - a tensor has 1 to 8 dimensions, at most 2^40 elements, an offset that is a multiple of the alignment and that
//!   lies inside the data section; for the types whose size is known (`F32`, `F16`, `BF16`, `Q8_0`, `I2_S`, `TQ1_0`,
//!   `TQ2_0`) the whole payload must lie inside the file, and the payloads of different tensors must not overlap.
//!
//! The parser is still not a substitute for trusting the source of a model: a valid file can describe a model that is
//! larger than your memory.

use memmap2::Mmap;
use std::collections::{HashMap, HashSet};
use std::fs::File;
use std::io::{Error, ErrorKind, Result as IoResult};
use std::mem::size_of;
use std::path::Path;

/// Longest accepted metadata key or tensor name, in bytes.
const MAX_NAME_LEN: usize = 1 << 16;
/// Deepest nesting of metadata arrays accepted.
const MAX_ARRAY_DEPTH: u32 = 4;
/// Largest accepted number of dimensions of a tensor (ggml itself supports four).
const MAX_TENSOR_DIMS: u32 = 8;
/// Largest accepted number of elements of a tensor (2^40, far beyond any real model).
const MAX_TENSOR_ELEMENTS: usize = 1 << 40;
/// Smallest encoded metadata entry: key length (8), value type (4) and a one-byte value.
const MIN_METADATA_ENTRY_BYTES: usize = 13;
/// Smallest encoded tensor entry: name length (8), dimension count (4), one dimension (8), type (4) and offset (8).
const MIN_TENSOR_ENTRY_BYTES: usize = 32;

/// Supported GGML tensor datatypes in GGUF.
pub const GGML_TYPE_F32: u32 = 0;
pub const GGML_TYPE_F16: u32 = 1;
pub const GGML_TYPE_Q8_0: u32 = 8;
pub const GGML_TYPE_BF16: u32 = 30;
/// llama.cpp TQ1_0 ternary quantization format (1.6875 bpw, 256 weights in 54 bytes).
pub const GGML_TYPE_TQ1_0: u32 = 34;
/// llama.cpp TQ2_0 ternary quantization format (2.0625 bpw, 256 weights in 66 bytes).
pub const GGML_TYPE_TQ2_0: u32 = 35;
/// Microsoft BitNet 2.00 bpw (4 trits per byte) planar block-interleaved format.
pub const GGML_TYPE_I2_S: u32 = 36;

/// Metadata value stored in a GGUF header.
#[derive(Debug, Clone)]
pub enum GgufMetadataValue {
    Uint8(u8),
    Int8(i8),
    Uint16(u16),
    Int16(i16),
    Uint32(u32),
    Int32(i32),
    Float32(f32),
    Bool(bool),
    String(String),
    Array(Vec<GgufMetadataValue>),
    Uint64(u64),
    Int64(i64),
    Float64(f64),
}

impl GgufMetadataValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Self::String(s) => Some(s.as_str()),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Self::Uint8(v) => Some(*v as u64),
            Self::Uint16(v) => Some(*v as u64),
            Self::Uint32(v) => Some(*v as u64),
            Self::Uint64(v) => Some(*v),
            Self::Int32(v) if *v >= 0 => Some(*v as u64),
            Self::Int64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_usize(&self) -> Option<usize> {
        self.as_u64().map(|v| v as usize)
    }

    pub fn as_f32(&self) -> Option<f32> {
        match self {
            Self::Float32(v) => Some(*v),
            Self::Float64(v) => Some(*v as f32),
            _ => None,
        }
    }

    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Self::Bool(b) => Some(*b),
            _ => None,
        }
    }

    pub fn as_string_array(&self) -> Option<Vec<String>> {
        match self {
            Self::Array(arr) => {
                let mut res = Vec::with_capacity(arr.len());
                for item in arr {
                    if let Self::String(s) = item {
                        res.push(s.clone());
                    } else {
                        return None;
                    }
                }
                Some(res)
            }
            _ => None,
        }
    }
}

/// Tensor descriptor in a GGUF file.
#[derive(Debug, Clone)]
pub struct GgufTensorInfo {
    pub name: String,
    pub shape: Vec<usize>,
    pub tensor_type: u32,
    pub offset: u64,
}

impl GgufTensorInfo {
    /// Total element count of the tensor (saturating, so a hand-built descriptor with absurd dimensions cannot
    /// overflow; descriptors produced by [`GgufFile`] are always within bounds).
    pub fn num_elements(&self) -> usize {
        self.shape.iter().fold(1usize, |acc, &d| acc.saturating_mul(d))
    }

    /// Size of the tensor payload in bytes, or `None` when the type is not one whose layout is known here or the
    /// element count is not a multiple of the block size of the type. For `I2_S` this is the packed trits only; the
    /// per-tensor scale that follows the payload is not included.
    pub fn byte_len(&self) -> Option<usize> {
        let n = self.num_elements();
        match self.tensor_type {
            GGML_TYPE_F32 => n.checked_mul(size_of::<f32>()),
            GGML_TYPE_F16 | GGML_TYPE_BF16 => n.checked_mul(size_of::<u16>()),
            GGML_TYPE_Q8_0 if n.is_multiple_of(32) => (n / 32).checked_mul(34),
            GGML_TYPE_I2_S => Some(n.div_ceil(4)),
            GGML_TYPE_TQ1_0 if n.is_multiple_of(256) => (n / 256).checked_mul(54),
            GGML_TYPE_TQ2_0 if n.is_multiple_of(256) => (n / 256).checked_mul(66),
            _ => None,
        }
    }
}

/// Memory-mapped GGUF model file reader.
pub struct GgufFile {
    mmap: Mmap,
    pub version: u32,
    pub metadata: HashMap<String, GgufMetadataValue>,
    pub tensors: HashMap<String, GgufTensorInfo>,
    data_start: usize,
}

impl GgufFile {
    /// Opens and memory-maps a GGUF model file from disk.
    pub fn open<P: AsRef<Path>>(path: P) -> IoResult<Self> {
        let file = File::open(path)?;
        let mmap = unsafe { Mmap::map(&file)? };
        Self::from_mmap(mmap)
    }

    /// Parses a GGUF file from an existing memory map.
    ///
    /// Returns an error (never panics) for a truncated, inconsistent or hostile file; see the module documentation
    /// for the checks.
    pub fn from_mmap(mmap: Mmap) -> IoResult<Self> {
        let file_len = mmap.len();
        let mut r = Reader { buf: &mmap[..], pos: 0 };

        let magic = r.take(4, "magic bytes")?;
        if magic != b"GGUF" {
            return Err(invalid(format!("Invalid GGUF magic header bytes: {magic:?}")));
        }

        let version = r.u32("version")?;
        if !(2..=3).contains(&version) {
            return Err(invalid(format!("unsupported GGUF version {version} (supported: 2 and 3)")));
        }
        let tensor_count = r.count("tensor count", MIN_TENSOR_ENTRY_BYTES)?;
        let metadata_kv_count = r.count("metadata count", MIN_METADATA_ENTRY_BYTES)?;

        let mut metadata = HashMap::with_capacity(metadata_kv_count);
        for _ in 0..metadata_kv_count {
            let key = r.string("metadata key", MAX_NAME_LEN)?;
            let vtype = r.u32("metadata value type")?;
            let val = read_gguf_value(&mut r, vtype, 0)?;
            if metadata.insert(key.clone(), val).is_some() {
                return Err(invalid(format!("duplicate metadata key '{key}'")));
            }
        }

        let mut infos = Vec::with_capacity(tensor_count);
        for _ in 0..tensor_count {
            let name = r.string("tensor name", MAX_NAME_LEN)?;
            let n_dims = r.u32("tensor dimension count")?;
            if !(1..=MAX_TENSOR_DIMS).contains(&n_dims) {
                return Err(invalid(format!("tensor '{name}' has {n_dims} dimensions (supported: 1 to {MAX_TENSOR_DIMS})")));
            }
            let mut shape = Vec::with_capacity(n_dims as usize);
            let mut elements: usize = 1;
            for _ in 0..n_dims {
                let d = r.u64("tensor dimension")?;
                let d = usize::try_from(d).map_err(|_| invalid(format!("tensor '{name}': dimension {d} is too large")))?;
                elements = elements
                    .checked_mul(d)
                    .filter(|&e| e <= MAX_TENSOR_ELEMENTS)
                    .ok_or_else(|| invalid(format!("tensor '{name}' has more than {MAX_TENSOR_ELEMENTS} elements")))?;
                shape.push(d);
            }
            let tensor_type = r.u32("tensor type")?;
            let offset = r.u64("tensor offset")?;
            infos.push(GgufTensorInfo { name, shape, tensor_type, offset });
        }

        let alignment = match metadata.get("general.alignment") {
            None => 32,
            Some(v) => v
                .as_u64()
                .and_then(|a| usize::try_from(a).ok())
                .filter(|a| a.is_power_of_two())
                .ok_or_else(|| invalid("general.alignment must be an unsigned integer that is a power of two".to_string()))?,
        };

        // The data section starts at the next multiple of the alignment after the tensor descriptors.
        let data_start = r
            .pos
            .checked_next_multiple_of(alignment)
            .filter(|&d| d <= file_len)
            .ok_or_else(|| invalid(format!("the tensor data section would start beyond the end of the file ({file_len} bytes)")))?;
        let data_len = file_len - data_start;

        // (start, end, name) of every payload whose size is known, for the overlap check.
        let mut ranges: Vec<(usize, usize, &str)> = Vec::new();
        let mut names = HashSet::with_capacity(infos.len());
        for info in &infos {
            if !names.insert(info.name.as_str()) {
                return Err(invalid(format!("duplicate tensor name '{}'", info.name)));
            }
            let offset = usize::try_from(info.offset)
                .ok()
                .filter(|&o| o <= data_len)
                .ok_or_else(|| invalid(format!("tensor '{}' starts beyond the end of the file", info.name)))?;
            if offset % alignment != 0 {
                return Err(invalid(format!(
                    "tensor '{}' has offset {offset}, which is not a multiple of the alignment {alignment}",
                    info.name
                )));
            }
            if let Some(len) = info.byte_len() {
                if len > data_len - offset {
                    return Err(invalid(format!("tensor '{}' ({len} bytes) extends beyond the end of the file", info.name)));
                }
                if len > 0 {
                    ranges.push((offset, offset + len, info.name.as_str()));
                }
            }
        }
        ranges.sort_unstable();
        for pair in ranges.windows(2) {
            if pair[0].1 > pair[1].0 {
                return Err(invalid(format!("the data of tensors '{}' and '{}' overlaps", pair[0].2, pair[1].2)));
            }
        }
        drop(names);
        drop(ranges);

        let tensors = infos.into_iter().map(|t| (t.name.clone(), t)).collect();

        Ok(Self {
            mmap,
            version,
            metadata,
            tensors,
            data_start,
        })
    }

    /// Returns a slice to the raw bytes of a tensor, or `None` for an unknown tensor, a type this reader cannot
    /// serve (`Q8_0`, `BF16` and the unknown types) or a payload that does not lie inside the file.
    pub fn get_tensor_data(&self, name: &str) -> Option<&[u8]> {
        let tensor = self.tensors.get(name)?;
        if !matches!(
            tensor.tensor_type,
            GGML_TYPE_F32 | GGML_TYPE_F16 | GGML_TYPE_I2_S | GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0
        ) {
            return None;
        }
        let start = self.data_start.checked_add(usize::try_from(tensor.offset).ok()?)?;
        let end = start.checked_add(tensor.byte_len()?)?;
        self.mmap.get(start..end)
    }

    /// Stores a 32-bit float scale factor immediately after the packed weight payload:
    /// `offset + packed_byte_length`.
    pub fn get_tensor_scale(&self, name: &str) -> Option<f32> {
        let tensor = self.tensors.get(name)?;
        if matches!(tensor.tensor_type, GGML_TYPE_TQ1_0 | GGML_TYPE_TQ2_0) {
            // TQ formats carry a scale in every block instead of one trailing tensor scale.
            return crate::decoders::uniform_block_scale(tensor.tensor_type, self.get_tensor_data(name)?);
        }
        if tensor.tensor_type != GGML_TYPE_I2_S {
            return None;
        }
        let scale_offset = self
            .data_start
            .checked_add(usize::try_from(tensor.offset).ok()?)?
            .checked_add(tensor.byte_len()?)?;
        let bytes: [u8; 4] = self.mmap.get(scale_offset..scale_offset.checked_add(4)?)?.try_into().ok()?;
        Some(f32::from_le_bytes(bytes))
    }

    /// Loads an F32 tensor into a vector. Returns `None` if the tensor is missing, not F32 or outside the file.
    pub fn get_tensor_f32(&self, name: &str) -> Option<Vec<f32>> {
        if self.tensors.get(name)?.tensor_type != GGML_TYPE_F32 {
            return None;
        }
        let bytes = self.get_tensor_data(name)?;
        Some(
            bytes
                .chunks_exact(size_of::<f32>())
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect(),
        )
    }

    /// Returns an F16 tensor as a slice of 16-bit half-precision floats (raw bits). Returns `None` if the tensor is
    /// missing, not F16, outside the file or not 2-byte aligned inside the mapping.
    pub fn get_tensor_f16(&self, name: &str) -> Option<&[u16]> {
        if self.tensors.get(name)?.tensor_type != GGML_TYPE_F16 {
            return None;
        }
        let bytes = self.get_tensor_data(name)?;
        // SAFETY: every bit pattern is a valid u16; `align_to` only yields the correctly aligned middle part.
        let (head, mid, tail) = unsafe { bytes.align_to::<u16>() };
        (head.is_empty() && tail.is_empty()).then_some(mid)
    }

    /// Returns the architecture name (e.g., "bitnet", "falcon3", "llama").
    pub fn architecture(&self) -> Option<&str> {
        self.metadata.get("general.architecture").and_then(|v| v.as_str())
    }

    /// Returns the context length / maximum position embeddings.
    pub fn context_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{}.context_length", arch)).and_then(|v| v.as_usize())
    }

    /// Returns the hidden embedding dimension size.
    pub fn embedding_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{}.embedding_length", arch)).and_then(|v| v.as_usize())
    }

    /// Returns the number of transformer block layers.
    pub fn block_count(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{}.block_count", arch)).and_then(|v| v.as_usize())
    }

    /// Returns the intermediate feed-forward network dimension.
    pub fn feed_forward_length(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{}.feed_forward_length", arch)).and_then(|v| v.as_usize())
    }

    /// Returns the number of attention heads.
    pub fn head_count(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{}.attention.head_count", arch)).and_then(|v| v.as_usize())
    }

    /// Returns the number of key-value attention heads (for GQA).
    pub fn head_count_kv(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{}.attention.head_count_kv", arch)).and_then(|v| v.as_usize())
    }

    /// Returns the RMSNorm epsilon parameter.
    pub fn rms_norm_eps(&self) -> Option<f32> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{}.attention.layer_norm_rms_epsilon", arch)).and_then(|v| v.as_f32())
    }

    /// Returns the RoPE base frequency theta.
    pub fn rope_freq_base(&self) -> Option<f32> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{}.rope.freq_base", arch)).and_then(|v| v.as_f32())
    }

    /// Returns the vocabulary size.
    pub fn vocab_size(&self) -> Option<usize> {
        let arch = self.architecture()?;
        self.metadata.get(&format!("{}.vocab_size", arch)).and_then(|v| v.as_usize())
    }

    /// Returns the byte offset in the file where tensor data begins.
    pub fn data_start(&self) -> usize {
        self.data_start
    }
}

fn invalid(msg: String) -> Error {
    Error::new(ErrorKind::InvalidData, msg)
}

/// Bounds-checked little-endian cursor over the mapped file.
struct Reader<'a> {
    buf: &'a [u8],
    pos: usize,
}

impl<'a> Reader<'a> {
    fn remaining(&self) -> usize {
        self.buf.len() - self.pos
    }

    /// Takes the next `n` bytes, or fails with `UnexpectedEof` if fewer remain.
    fn take(&mut self, n: usize, what: &str) -> IoResult<&'a [u8]> {
        if n > self.remaining() {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                format!("truncated GGUF file: {what} needs {n} bytes at offset {} but only {} remain", self.pos, self.remaining()),
            ));
        }
        let out = &self.buf[self.pos..self.pos + n];
        self.pos += n;
        Ok(out)
    }

    fn u8(&mut self, what: &str) -> IoResult<u8> {
        Ok(self.take(1, what)?[0])
    }

    fn u16(&mut self, what: &str) -> IoResult<u16> {
        Ok(u16::from_le_bytes(self.take(2, what)?.try_into().unwrap()))
    }

    fn u32(&mut self, what: &str) -> IoResult<u32> {
        Ok(u32::from_le_bytes(self.take(4, what)?.try_into().unwrap()))
    }

    fn u64(&mut self, what: &str) -> IoResult<u64> {
        Ok(u64::from_le_bytes(self.take(8, what)?.try_into().unwrap()))
    }

    /// Reads an element count whose elements need at least `min_entry_bytes` each; the count must fit into the
    /// bytes that remain, so it is safe to reserve memory for it.
    fn count(&mut self, what: &str, min_entry_bytes: usize) -> IoResult<usize> {
        let n = self.u64(what)?;
        usize::try_from(n)
            .ok()
            .filter(|&n| n.checked_mul(min_entry_bytes).is_some_and(|bytes| bytes <= self.remaining()))
            .ok_or_else(|| invalid(format!("{what} {n} does not fit into the {} remaining bytes of the file", self.remaining())))
    }

    /// Reads a length-prefixed UTF-8 string of at most `max_len` bytes.
    fn string(&mut self, what: &str, max_len: usize) -> IoResult<String> {
        let len = self.u64(what)?;
        let len = usize::try_from(len)
            .ok()
            .filter(|&l| l <= max_len && l <= self.remaining())
            .ok_or_else(|| invalid(format!("{what} of {len} bytes is longer than the limit of {max_len} bytes or the rest of the file")))?;
        let bytes = self.take(len, what)?;
        String::from_utf8(bytes.to_vec()).map_err(|e| Error::new(ErrorKind::InvalidData, e))
    }
}

fn read_gguf_value(r: &mut Reader<'_>, vtype: u32, depth: u32) -> IoResult<GgufMetadataValue> {
    const W: &str = "metadata value";
    match vtype {
        0 => Ok(GgufMetadataValue::Uint8(r.u8(W)?)),
        1 => Ok(GgufMetadataValue::Int8(r.u8(W)? as i8)),
        2 => Ok(GgufMetadataValue::Uint16(r.u16(W)?)),
        3 => Ok(GgufMetadataValue::Int16(r.u16(W)? as i16)),
        4 => Ok(GgufMetadataValue::Uint32(r.u32(W)?)),
        5 => Ok(GgufMetadataValue::Int32(r.u32(W)? as i32)),
        6 => Ok(GgufMetadataValue::Float32(f32::from_bits(r.u32(W)?))),
        7 => Ok(GgufMetadataValue::Bool(r.u8(W)? != 0)),
        8 => Ok(GgufMetadataValue::String(r.string("metadata string", usize::MAX)?)),
        9 => {
            if depth >= MAX_ARRAY_DEPTH {
                return Err(invalid(format!("metadata arrays are nested deeper than {MAX_ARRAY_DEPTH} levels")));
            }
            let item_type = r.u32("array element type")?;
            // Smallest encoded size of one element, which bounds the element count by the remaining bytes.
            let min_item: usize = match item_type {
                0 | 1 | 7 => 1,
                2 | 3 => 2,
                4..=6 => 4,
                8 | 10..=12 => 8,
                9 => 12,
                other => return Err(invalid(format!("unsupported metadata array element type {other}"))),
            };
            let len = r.count("metadata array length", min_item)?;
            let mut items = Vec::with_capacity(len);
            for _ in 0..len {
                items.push(read_gguf_value(r, item_type, depth + 1)?);
            }
            Ok(GgufMetadataValue::Array(items))
        }
        10 => Ok(GgufMetadataValue::Uint64(r.u64(W)?)),
        11 => Ok(GgufMetadataValue::Int64(r.u64(W)? as i64)),
        12 => Ok(GgufMetadataValue::Float64(f64::from_bits(r.u64(W)?))),
        other => Err(invalid(format!("Unsupported GGUF metadata type code: {other}"))),
    }
}

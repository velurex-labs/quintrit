//! Contiguous preallocated Key-Value (KV) cache for autoregressive transformer inference.

/// Allocates `n` zeroed `u16` values, or returns `None` if the size overflows or the allocator fails (instead of the
/// abort that `vec![0; n]` performs on a failed allocation).
fn try_zeroed_u16(n: usize) -> Option<Vec<u16>> {
    if n == 0 {
        return Some(Vec::new());
    }
    let layout = std::alloc::Layout::array::<u16>(n).ok()?;
    // SAFETY: the layout has a non-zero size. A non-null pointer from `alloc_zeroed` with the layout of `[u16; n]`
    // points to `n` initialized (zero) values, which is exactly what `Vec::from_raw_parts` with length and capacity `n`
    // requires; the vector frees it with the same global allocator and layout.
    unsafe {
        let ptr = std::alloc::alloc_zeroed(layout) as *mut u16;
        if ptr.is_null() {
            return None;
        }
        Some(Vec::from_raw_parts(ptr, n, n))
    }
}

/// Multi-layer Key-Value cache storing intermediate attention keys and values.
#[derive(Debug, Clone)]
pub struct KVCache {
    num_layers: usize,
    max_seq_len: usize,
    num_kv_heads: usize,
    head_dim: usize,
    layer_stride: usize,
    token_stride: usize,
    head_stride: usize,
    /// Contiguous buffer for cached Keys in FP16 format: [layers, max_seq_len, num_kv_heads, head_dim]
    k_buf: Vec<u16>,
    /// Contiguous buffer for cached Values in FP16 format: [layers, max_seq_len, num_kv_heads, head_dim]
    v_buf: Vec<u16>,
    /// Current number of stored tokens in the cache.
    current_len: usize,
    /// Number of calls to [`KVCache::reset`], so that a holder of cached positions can tell that someone else has
    /// started a new sequence in the meantime.
    reset_count: u64,
}

impl KVCache {
    /// Preallocates contiguous Key and Value memory buffers for all layers.
    ///
    /// Panics if the size overflows or the memory cannot be reserved; use [`KVCache::try_new`] for dimensions that come
    /// from an untrusted source.
    pub fn new(num_layers: usize, max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Self {
        Self::try_new(num_layers, max_seq_len, num_kv_heads, head_dim).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Preallocates the cache like [`KVCache::new`], but returns an error instead of panicking or aborting when the
    /// size overflows or the allocator cannot reserve the buffers.
    ///
    /// The buffers are allocated zeroed, so the operating system commits their pages only when they are first written:
    /// a cache for a long context costs memory only for the positions that are used.
    pub fn try_new(num_layers: usize, max_seq_len: usize, num_kv_heads: usize, head_dim: usize) -> Result<Self, String> {
        let head_stride = head_dim;
        let too_large = || {
            format!("a KV cache of {num_layers} layers x {max_seq_len} positions x {num_kv_heads} heads x {head_dim} is too large")
        };
        let token_stride = num_kv_heads.checked_mul(head_stride).ok_or_else(too_large)?;
        let layer_stride = max_seq_len.checked_mul(token_stride).ok_or_else(too_large)?;
        let total_elements = num_layers.checked_mul(layer_stride).ok_or_else(too_large)?;
        let alloc_failed = || format!("{} (could not allocate {} bytes)", too_large(), total_elements.saturating_mul(2 * size_of::<u16>()));
        let k_buf = try_zeroed_u16(total_elements).ok_or_else(alloc_failed)?;
        let v_buf = try_zeroed_u16(total_elements).ok_or_else(alloc_failed)?;

        Ok(Self {
            num_layers,
            max_seq_len,
            num_kv_heads,
            head_dim,
            layer_stride,
            token_stride,
            head_stride,
            k_buf,
            v_buf,
            current_len: 0,
            reset_count: 0,
        })
    }

    /// Stores the key and value vectors for a given layer at generation step `pos`.
    pub fn append(&mut self, layer: usize, pos: usize, k: &[u16], v: &[u16]) {
        assert!(layer < self.num_layers);
        assert!(pos < self.max_seq_len);
        let expected_len = self.num_kv_heads * self.head_dim;
        assert_eq!(k.len(), expected_len);
        assert_eq!(v.len(), expected_len);

        let offset = layer * self.layer_stride + pos * self.token_stride;
        self.k_buf[offset..offset + expected_len].copy_from_slice(k);
        self.v_buf[offset..offset + expected_len].copy_from_slice(v);

        if pos + 1 > self.current_len {
            self.current_len = pos + 1;
        }
    }

    /// Returns a slice to the cached Key vector for `(layer, pos, kv_head)`.
    #[inline]
    pub fn get_key(&self, layer: usize, pos: usize, kv_head: usize) -> &[u16] {
        let offset = layer * self.layer_stride + pos * self.token_stride + kv_head * self.head_stride;
        &self.k_buf[offset..offset + self.head_dim]
    }

    /// Returns a slice to the cached Value vector for `(layer, pos, kv_head)`.
    #[inline]
    pub fn get_val(&self, layer: usize, pos: usize, kv_head: usize) -> &[u16] {
        let offset = layer * self.layer_stride + pos * self.token_stride + kv_head * self.head_stride;
        &self.v_buf[offset..offset + self.head_dim]
    }

    /// Resets the active cache length to zero without deallocating backing memory buffers.
    pub fn reset(&mut self) {
        self.current_len = 0;
        self.reset_count = self.reset_count.wrapping_add(1);
    }

    /// Keeps the first `len` positions and drops the rest, so that the next token can be appended at position `len`.
    ///
    /// Unlike [`KVCache::reset`], this continues the current sequence: positions below `len` stay valid.
    pub fn truncate(&mut self, len: usize) {
        self.current_len = self.current_len.min(len);
    }

    /// Number of times [`KVCache::reset`] has been called on this cache.
    #[inline]
    pub fn reset_count(&self) -> u64 {
        self.reset_count
    }

    /// Number of active tokens currently stored in the cache.
    #[inline]
    pub fn len(&self) -> usize {
        self.current_len
    }

    /// Whether the cache is currently empty.
    #[inline]
    pub fn is_empty(&self) -> bool {
        self.current_len == 0
    }

    /// Maximum supported sequence length.
    #[inline]
    pub fn max_seq_len(&self) -> usize {
        self.max_seq_len
    }

    /// Total memory consumption of both Key and Value buffers in bytes.
    pub fn memory_bytes(&self) -> usize {
        (self.k_buf.len() + self.v_buf.len()) * size_of::<u16>()
    }
}

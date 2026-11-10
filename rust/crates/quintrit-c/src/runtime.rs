//! Runtime level: model and session handles, tokenizer access and text generation.
//!
//! A `quintrit_model_t` owns the transcoded weights and the tokenizer and is immutable once sessions exist. A
//! `quintrit_session_t` owns the mutable per-conversation state (KV cache, scratch buffers) and shares the model.

use crate::error::{
    fail, guard, Failure, Res, QUINTRIT_ERR_BUSY, QUINTRIT_ERR_BUFFER_TOO_SMALL, QUINTRIT_ERR_CONTEXT_OVERFLOW,
    QUINTRIT_ERR_INVALID_ARGUMENT, QUINTRIT_ERR_INVALID_VALUE, QUINTRIT_ERR_MODEL,
    QUINTRIT_ERR_UNSUPPORTED,
};
use crate::kernel::{resolve, resolve_threads, Kernel, QUINTRIT_KERNEL_AUTO};
use crate::util::{
    c_str, check_handle, check_out_struct, in_slice, out_ptr, out_slice, read_versioned, text, write_out, HasMagic,
    MAGIC_MODEL, MAGIC_SESSION,
};
use quintrit_format::GgufFile;
use quintrit_runtime::engine::find_safe_flush_index;
use quintrit_runtime::scratchpad::InferenceScratchpad;
use quintrit_runtime::{
    format_chat_prompt, ChatMessage, InferenceEngine, KVCache, ModelArchitecture, QuintritModel, QuintritTokenizer, Sampler,
    SamplingConfig, StreamDecoder, TernaryKernelType, VerboseScope,
};
use std::ffi::{c_char, c_int, c_void};
use std::path::Path;
use std::sync::{Arc, Mutex, MutexGuard, TryLockError};

/// `quintrit_generate_params_t::flags`: continue the conversation in the KV cache instead of starting a new one.
pub const QUINTRIT_GENERATE_KEEP_CONTEXT: u32 = 1;
/// `quintrit_model_tokenize` flag: prepend the BOS token (as the generation functions do).
pub const QUINTRIT_TOKENIZE_ADD_BOS: u32 = 1;

/// `quintrit_model_options_t::flags`: print loading progress to the standard output (silent by default).
pub const QUINTRIT_MODEL_OPEN_VERBOSE: u32 = 1;

/// Why a generation stopped (`quintrit_generate_result_t::finish_reason`).
pub const QUINTRIT_FINISH_LENGTH: i32 = 0;
pub const QUINTRIT_FINISH_EOS: i32 = 1;
pub const QUINTRIT_FINISH_STOP_SEQUENCE: i32 = 2;
pub const QUINTRIT_FINISH_CALLBACK: i32 = 3;
pub const QUINTRIT_FINISH_CONTEXT_FULL: i32 = 4;

/// Largest number of stop sequences accepted per call.
const MAX_STOP_SEQUENCES: usize = 64;

// ---------------------------------------------------------------------------------------------------------------------
// Structs shared with C
// ---------------------------------------------------------------------------------------------------------------------

/// `quintrit_model_options_t`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ModelOptions {
    pub struct_size: u32,
    /// Worker threads; 0 = automatic.
    pub num_threads: u32,
    /// `QUINTRIT_KERNEL_AUTO`, `QUINTRIT_KERNEL_SIMD32` or `QUINTRIT_KERNEL_AVX512`.
    pub kernel: i32,
    /// `QUINTRIT_MODEL_OPEN_VERBOSE` or 0.
    pub flags: u32,
}

/// `quintrit_model_info_t`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ModelInfo {
    pub struct_size: u32,
    /// Concrete kernel in use.
    pub kernel: i32,
    pub num_threads: u32,
    pub num_layers: u32,
    pub hidden_size: u32,
    pub num_heads: u32,
    pub num_kv_heads: u32,
    pub head_dim: u32,
    pub vocab_size: u64,
    pub context_length: u64,
    /// -1 if the model has no BOS / EOS token.
    pub bos_token_id: i64,
    pub eos_token_id: i64,
    /// Architecture family: "bitnet", "falcon3" or "llama" (NUL-terminated).
    pub architecture: [c_char; 16],
}

/// `quintrit_session_options_t`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SessionOptions {
    pub struct_size: u32,
    /// Context window of the session in tokens; 0 = the model's maximum.
    pub context_length: u32,
}

/// `quintrit_session_info_t`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct SessionInfo {
    pub struct_size: u32,
    pub reserved: u32,
    pub context_length: u64,
    /// Number of tokens currently held in the KV cache.
    pub position: u64,
}

/// `quintrit_generate_params_t`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct GenerateParams {
    pub struct_size: u32,
    pub flags: u32,
    pub max_tokens: u32,
    /// Top-k truncation; 0 disables it, 1 is greedy.
    pub top_k: u32,
    /// 0 = greedy.
    pub temperature: f32,
    /// Nucleus threshold in (0, 1]; 1 disables it.
    pub top_p: f32,
    pub seed: u64,
    /// Array of `stop_count` NUL-terminated UTF-8 strings, or NULL.
    pub stop_sequences: *const *const c_char,
    pub stop_count: usize,
}

/// `quintrit_generate_result_t`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct GenerateResult {
    pub struct_size: u32,
    pub finish_reason: i32,
    pub prompt_tokens: u32,
    pub generated_tokens: u32,
    /// Length in bytes of the generated text (without NUL).
    pub text_bytes: u64,
}

/// `quintrit_chat_message_t`.
#[repr(C)]
#[derive(Copy, Clone)]
pub struct ChatMessageC {
    pub role: *const c_char,
    pub content: *const c_char,
}

/// Streaming callback: receives UTF-8 text (not NUL-terminated); returns 0 to continue, non-zero to stop.
pub type TokenCallback = Option<unsafe extern "C" fn(text: *const c_char, len: usize, user_data: *mut c_void) -> c_int>;

// ---------------------------------------------------------------------------------------------------------------------
// Handles
// ---------------------------------------------------------------------------------------------------------------------

pub struct ModelInner {
    model: QuintritModel,
    tokenizer: QuintritTokenizer,
    kernel: Kernel,
    threads: usize,
}

/// Opaque handle `quintrit_model_t`.
#[repr(C)]
pub struct ModelHandle {
    magic: u32,
    inner: Mutex<Arc<ModelInner>>,
}

struct SessionState {
    model: Arc<ModelInner>,
    kv: KVCache,
    scratch: InferenceScratchpad,
    logits: Vec<f32>,
    ctx_len: usize,
}

/// Opaque handle `quintrit_session_t`.
#[repr(C)]
pub struct SessionHandle {
    magic: u32,
    state: Mutex<SessionState>,
}

impl HasMagic for ModelHandle {
    const MAGIC: u32 = MAGIC_MODEL;
    fn magic(&self) -> u32 {
        self.magic
    }
}

impl HasMagic for SessionHandle {
    const MAGIC: u32 = MAGIC_SESSION;
    fn magic(&self) -> u32 {
        self.magic
    }
}

// The C API promises that a model may be used from several threads; fail the build if that stops being true.
const _: fn() = || {
    fn is_send_sync<T: Send + Sync>() {}
    is_send_sync::<ModelInner>();
    is_send_sync::<SessionState>();
};

fn model_arc(h: &ModelHandle) -> Arc<ModelInner> {
    h.inner.lock().unwrap_or_else(|e| e.into_inner()).clone()
}

fn lock_session(h: &SessionHandle) -> Res<MutexGuard<'_, SessionState>> {
    match h.state.try_lock() {
        Ok(g) => Ok(g),
        Err(TryLockError::Poisoned(e)) => Ok(e.into_inner()),
        Err(TryLockError::WouldBlock) => fail!(QUINTRIT_ERR_BUSY, "the session is in use by another thread"),
    }
}

fn arch_name(a: ModelArchitecture) -> &'static str {
    match a {
        ModelArchitecture::BitNet => "bitnet",
        ModelArchitecture::Falcon3 => "falcon3",
        ModelArchitecture::Llama => "llama",
    }
}

// ---------------------------------------------------------------------------------------------------------------------
// Model validation and loading
// ---------------------------------------------------------------------------------------------------------------------

/// Checks the hyper-parameters and the tensors the loader indexes without bounds checks of its own, so that a damaged
/// or foreign model file is rejected with an error instead of panicking inside the forward pass. The checks live in
/// `quintrit-runtime` (`validate_gguf_model`), which the engine runs as well; this maps the error to the C error code.
fn validate_model(g: &GgufFile) -> Res<()> {
    quintrit_runtime::validate_gguf_model(g).map(|_| ()).map_err(|m| Failure::new(QUINTRIT_ERR_MODEL, m))
}

fn to_runtime_kernel(k: Kernel) -> Res<TernaryKernelType> {
    match k {
        Kernel::Simd32 => Ok(TernaryKernelType::QuintritLut),
        Kernel::Avx512 => Ok(TernaryKernelType::QuintritLutAVX512),
        Kernel::Scalar => fail!(
            QUINTRIT_ERR_UNSUPPORTED,
            "the scalar kernel is a kernel-level reference only; models run on 'simd32' or 'avx512' (use QUINTRIT_KERNEL_AUTO)"
        ),
    }
}

fn open_model(path: &str, opts: Option<ModelOptions>) -> Res<ModelHandle> {
    let (threads, kernel_id, flags) = opts.map_or((0, 0, 0), |o| (o.num_threads, o.kernel, o.flags));
    if flags & !QUINTRIT_MODEL_OPEN_VERBOSE != 0 {
        fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "options.flags has unknown bits set: {flags:#x}");
    }
    let kernel = if kernel_id == QUINTRIT_KERNEL_AUTO {
        // The 32-row layer also runs on CPUs without AVX2 through the portable fallback of the core crate.
        if Kernel::Avx512.is_supported() { Kernel::Avx512 } else { Kernel::Simd32 }
    } else {
        let k = resolve(kernel_id)?;
        k.require_supported()?;
        k
    };
    let rt_kernel = to_runtime_kernel(kernel)?;
    let threads = resolve_threads(threads)?;

    let gguf = crate::gguf_check::open_checked(Path::new(path))?;
    validate_model(&gguf)?;
    // The library is silent unless the caller asked for progress output; the previous setting is restored afterwards.
    let engine = {
        let _verbose = VerboseScope::new(flags & QUINTRIT_MODEL_OPEN_VERBOSE != 0);
        InferenceEngine::load_from_parsed_gguf(&gguf, threads, Some(rt_kernel), Some(path))
            .map_err(|m| Failure::new(QUINTRIT_ERR_MODEL, m))?
    };
    let InferenceEngine { model, tokenizer, .. } = engine;
    Ok(ModelHandle {
        magic: MAGIC_MODEL,
        inner: Mutex::new(Arc::new(ModelInner { model, tokenizer, kernel, threads })),
    })
}

// ---------------------------------------------------------------------------------------------------------------------
// Generation
// ---------------------------------------------------------------------------------------------------------------------

struct Emitter {
    callback: TokenCallback,
    user_data: *mut c_void,
    stopped: bool,
}

impl Emitter {
    /// Hands text to the callback; returns true once the callback has asked to stop.
    fn emit(&mut self, s: &str) -> bool {
        if self.stopped || s.is_empty() {
            return self.stopped;
        }
        if let Some(cb) = self.callback {
            // SAFETY: the callback contract is documented in the header; the text is valid UTF-8 for `len` bytes.
            if unsafe { cb(s.as_ptr() as *const c_char, s.len(), self.user_data) } != 0 {
                self.stopped = true;
            }
        }
        self.stopped
    }
}

struct Generated {
    text: String,
    result: GenerateResult,
}

fn generate(
    st: &mut SessionState,
    prompt: &str,
    params: &GenerateParams,
    stops: &[&str],
    emitter: &mut Emitter,
) -> Res<Generated> {
    let model = st.model.clone();
    let keep = params.flags & QUINTRIT_GENERATE_KEEP_CONTEXT != 0;
    let vocab = model.model.config.vocab_size;

    let mut tokens = model.tokenizer.encode(prompt);
    if !keep
        && let Some(bos) = model.tokenizer.bos_token_id()
        && tokens.first() != Some(&bos)
    {
        tokens.insert(0, bos);
    }
    if tokens.is_empty() {
        fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "the prompt produced no tokens");
    }
    if let Some(bad) = tokens.iter().find(|&&t| t as usize >= vocab) {
        fail!(QUINTRIT_ERR_MODEL, "tokenizer produced token {bad} outside the vocabulary of {vocab}");
    }
    let pos0 = if keep { st.kv.len() } else { 0 };
    if pos0 + tokens.len() > st.ctx_len {
        fail!(
            QUINTRIT_ERR_CONTEXT_OVERFLOW,
            "{} prompt tokens do not fit: the context window holds {} tokens and {pos0} are already used",
            tokens.len(),
            st.ctx_len
        );
    }

    if !keep {
        st.kv.reset();
    }
    let mut sampler = Sampler::new(SamplingConfig {
        temperature: params.temperature,
        top_k: params.top_k as usize,
        top_p: params.top_p,
        seed: params.seed,
    });
    let eos = model.tokenizer.eos_token_id();
    let llama_like = matches!(model.model.config.architecture, ModelArchitecture::BitNet | ModelArchitecture::Llama);

    for (i, &tok) in tokens.iter().enumerate() {
        model.model.forward_token_with_scratchpad(tok, pos0 + i, &mut st.kv, &mut st.scratch, &mut st.logits);
    }
    let mut pos = pos0 + tokens.len();

    let mut full = String::new();
    let mut pending = String::new();
    let mut generated = 0u32;
    let mut reason = QUINTRIT_FINISH_LENGTH;
    // Decodes without splitting a multi-byte UTF-8 character across two pieces.
    let mut decoder = StreamDecoder::new();

    #[allow(clippy::explicit_counter_loop)]
    for _ in 0..params.max_tokens {
        let next = sampler.sample(&st.logits);
        if Some(next) == eos || (llama_like && next == 128009) {
            reason = QUINTRIT_FINISH_EOS;
            break;
        }
        let piece = decoder.push(&model.tokenizer, next);
        full.push_str(&piece);
        generated += 1;

        let mut halt = false;
        if stops.is_empty() {
            if emitter.emit(&piece) {
                reason = QUINTRIT_FINISH_CALLBACK;
                halt = true;
            }
        } else {
            pending.push_str(&piece);
            let earliest = stops.iter().filter_map(|s| pending.find(s)).min();
            if let Some(idx) = earliest {
                let cb_stop = emitter.emit(&pending[..idx]);
                full.truncate(full.len() - (pending.len() - idx));
                pending.clear();
                reason = if cb_stop { QUINTRIT_FINISH_CALLBACK } else { QUINTRIT_FINISH_STOP_SEQUENCE };
                break;
            }
            let safe = find_safe_flush_index(&pending, stops);
            if safe > 0 {
                if emitter.emit(&pending[..safe]) {
                    reason = QUINTRIT_FINISH_CALLBACK;
                    halt = true;
                }
                pending.drain(..safe);
            }
        }

        if pos >= st.ctx_len {
            if !halt {
                reason = QUINTRIT_FINISH_CONTEXT_FULL;
            }
            break;
        }
        // Feed the token (also after a callback stop, so that the cache matches the text the caller has seen).
        model.model.forward_token_with_scratchpad(next, pos, &mut st.kv, &mut st.scratch, &mut st.logits);
        pos += 1;
        if halt {
            break;
        }
    }
    if reason != QUINTRIT_FINISH_STOP_SEQUENCE && reason != QUINTRIT_FINISH_CALLBACK {
        // Text held back by the decoder (an incomplete character at the very end).
        let rest = decoder.finish(&model.tokenizer);
        full.push_str(&rest);
        if stops.is_empty() {
            emitter.emit(&rest);
        } else {
            pending.push_str(&rest);
            if let Some(idx) = stops.iter().filter_map(|s| pending.find(s)).min() {
                emitter.emit(&pending[..idx]);
                full.truncate(full.len() - (pending.len() - idx));
                pending.clear();
                reason = QUINTRIT_FINISH_STOP_SEQUENCE;
            }
        }
    }
    if !pending.is_empty() {
        emitter.emit(&pending);
    }
    let text_bytes = full.len() as u64;
    Ok(Generated {
        text: full,
        result: GenerateResult {
            struct_size: size_of::<GenerateResult>() as u32,
            finish_reason: reason,
            prompt_tokens: tokens.len() as u32,
            generated_tokens: generated,
            text_bytes,
        },
    })
}

/// Copies `s` into a caller buffer snprintf-style: truncated at a character boundary, NUL-terminated.
unsafe fn write_text(buf: *mut c_char, cap: usize, s: &str) -> Res<()> {
    // SAFETY: validated by out_slice.
    let dst = unsafe { out_slice(buf as *mut u8, cap, "out")? };
    if dst.is_empty() {
        return Ok(());
    }
    let mut n = s.len().min(dst.len() - 1);
    while !s.is_char_boundary(n) {
        n -= 1;
    }
    dst[..n].copy_from_slice(&s.as_bytes()[..n]);
    dst[n] = 0;
    Ok(())
}

// ---------------------------------------------------------------------------------------------------------------------
// Exported functions
// ---------------------------------------------------------------------------------------------------------------------

/// Loads a GGUF model and transcodes its ternary weights into the quintrit layout (this takes seconds to minutes for
/// large models). Nothing is printed unless `options.flags` contains `QUINTRIT_MODEL_OPEN_VERBOSE`, which prints
/// progress lines to the standard output. `options` may be NULL for the defaults (automatic threads, best kernel,
/// silent).
///
/// # Safety
/// `path` is a NUL-terminated UTF-8 path; `options` is NULL or a `quintrit_model_options_t` with `struct_size` set;
/// `out_model` points to a handle pointer, which is set to NULL on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_model_open(
    path: *const c_char,
    options: *const ModelOptions,
    out_model: *mut *mut ModelHandle,
) -> c_int {
    guard(|| {
        // SAFETY: validated by c_str / read_versioned / out_ptr.
        let path = unsafe { c_str(path, "path")? };
        let opts = unsafe { read_versioned(options, "options")? };
        out_ptr(out_model, "out_model")?;
        unsafe { out_model.write(std::ptr::null_mut()) };
        let handle = Box::new(open_model(path, opts)?);
        unsafe { out_model.write(Box::into_raw(handle)) };
        Ok(())
    })
}

/// Frees a model. NULL is allowed. Sessions created from the model keep it alive until they are freed too.
///
/// # Safety
/// `model` must be NULL or a live handle that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_model_free(model: *mut ModelHandle) -> c_int {
    guard(|| {
        if model.is_null() {
            return Ok(());
        }
        // SAFETY: the tag is validated before the Box is rebuilt.
        unsafe { check_handle(model, "model")? };
        unsafe {
            (*model).magic = 0;
            drop(Box::from_raw(model));
        }
        Ok(())
    })
}

/// Describes a loaded model.
///
/// # Safety
/// `model` is a live handle; `out_info` points to a `quintrit_model_info_t` with `struct_size` set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_model_info(model: *const ModelHandle, out_info: *mut ModelInfo) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / check_out_struct.
        let h = unsafe { check_handle(model, "model")? };
        let out = unsafe { check_out_struct(out_info, "out_info")? };
        let m = model_arc(h);
        let c = &m.model.config;
        let mut architecture = [0 as c_char; 16];
        for (dst, b) in architecture.iter_mut().zip(arch_name(c.architecture).bytes()) {
            *dst = b as c_char;
        }
        let info = ModelInfo {
            struct_size: size_of::<ModelInfo>() as u32,
            kernel: m.kernel.id(),
            num_threads: m.threads as u32,
            num_layers: c.num_hidden_layers as u32,
            hidden_size: c.hidden_size as u32,
            num_heads: c.num_attention_heads as u32,
            num_kv_heads: c.num_key_value_heads as u32,
            head_dim: c.head_dim as u32,
            vocab_size: c.vocab_size as u64,
            context_length: c.max_position_embeddings as u64,
            bos_token_id: m.tokenizer.bos_token_id().map_or(-1, |v| v as i64),
            eos_token_id: m.tokenizer.eos_token_id().map_or(-1, |v| v as i64),
            architecture,
        };
        unsafe { out.write(info) };
        Ok(())
    })
}

/// Sets the number of worker threads of the model (0 = automatic). Fails with `QUINTRIT_ERR_BUSY` while any session
/// of the model exists or another call is using it: set the thread count before creating sessions.
///
/// # Safety
/// `model` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_model_set_threads(model: *mut ModelHandle, num_threads: u32) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle.
        let h = unsafe { check_handle(model, "model")? };
        let threads = resolve_threads(num_threads)?;
        let mut guard = h.inner.lock().unwrap_or_else(|e| e.into_inner());
        match Arc::get_mut(&mut guard) {
            Some(inner) => {
                inner.model.set_num_threads(threads);
                inner.threads = threads;
                Ok(())
            }
            None => fail!(QUINTRIT_ERR_BUSY, "the model has live sessions; free them before changing the thread count"),
        }
    })
}

/// Converts UTF-8 text into token ids. Call with `out = NULL, capacity = 0` to learn the count: `*out_count` always
/// receives the number of tokens, and `QUINTRIT_ERR_BUFFER_TOO_SMALL` is returned if `capacity` is smaller.
/// `flags`: `QUINTRIT_TOKENIZE_ADD_BOS` or 0.
///
/// # Safety
/// `text` points to `text_len` bytes (not necessarily NUL-terminated); `out` to `capacity` `uint32_t`; `out_count` to a
/// `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_model_tokenize(
    model: *const ModelHandle,
    text_ptr: *const c_char,
    text_len: usize,
    flags: u32,
    out: *mut u32,
    capacity: usize,
    out_count: *mut usize,
) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / text / out_slice / write_out.
        let h = unsafe { check_handle(model, "model")? };
        let s = unsafe { text(text_ptr, text_len, "text")? };
        if flags & !QUINTRIT_TOKENIZE_ADD_BOS != 0 {
            fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "unknown flags 0x{flags:x}");
        }
        let m = model_arc(h);
        let mut ids = m.tokenizer.encode(s);
        if flags & QUINTRIT_TOKENIZE_ADD_BOS != 0
            && let Some(bos) = m.tokenizer.bos_token_id()
            && ids.first() != Some(&bos)
        {
            ids.insert(0, bos);
        }
        unsafe { write_out(out_count, ids.len(), "out_count")? };
        let dst = unsafe { out_slice(out, capacity, "out")? };
        if dst.len() < ids.len() {
            fail!(QUINTRIT_ERR_BUFFER_TOO_SMALL, "{} tokens do not fit into a buffer of {capacity}", ids.len());
        }
        dst[..ids.len()].copy_from_slice(&ids);
        Ok(())
    })
}

/// Converts token ids back into UTF-8 text, NUL-terminated. `*out_len` receives the text length without the NUL;
/// `QUINTRIT_ERR_BUFFER_TOO_SMALL` if `capacity <= *out_len` (call with `out = NULL, capacity = 0` for the size).
///
/// # Safety
/// `tokens` points to `count` `uint32_t`; `out` to `capacity` bytes; `out_len` to a `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_model_detokenize(
    model: *const ModelHandle,
    tokens: *const u32,
    count: usize,
    out: *mut c_char,
    capacity: usize,
    out_len: *mut usize,
) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / in_slice / out_slice / write_out.
        let h = unsafe { check_handle(model, "model")? };
        let ids = unsafe { in_slice(tokens, count, "tokens")? };
        let m = model_arc(h);
        let vocab = m.model.config.vocab_size;
        if let Some(bad) = ids.iter().find(|&&t| t as usize >= vocab) {
            fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "token id {bad} is outside the vocabulary of {vocab}");
        }
        let s = m.tokenizer.decode(ids);
        unsafe { write_out(out_len, s.len(), "out_len")? };
        let dst = unsafe { out_slice(out as *mut u8, capacity, "out")? };
        if dst.len() <= s.len() {
            fail!(QUINTRIT_ERR_BUFFER_TOO_SMALL, "the text needs {} bytes including the NUL, buffer holds {capacity}", s.len() + 1);
        }
        dst[..s.len()].copy_from_slice(s.as_bytes());
        dst[s.len()] = 0;
        Ok(())
    })
}

/// Formats a chat conversation into the prompt format of the model's architecture and appends the assistant prefix.
/// Roles: "system", "user", "assistant" (anything else is treated as "user"). Output as `quintrit_model_detokenize`.
///
/// # Safety
/// `messages` points to `count` messages with NUL-terminated strings; `out` to `capacity` bytes; `out_len` to a `size_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_model_format_chat(
    model: *const ModelHandle,
    messages: *const ChatMessageC,
    count: usize,
    out: *mut c_char,
    capacity: usize,
    out_len: *mut usize,
) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / in_slice / c_str / out_slice / write_out.
        let h = unsafe { check_handle(model, "model")? };
        let raw = unsafe { in_slice(messages, count, "messages")? };
        let mut msgs = Vec::with_capacity(raw.len());
        for (i, m) in raw.iter().enumerate() {
            let role = unsafe { c_str(m.role, &format!("messages[{i}].role"))? };
            let content = unsafe { c_str(m.content, &format!("messages[{i}].content"))? };
            msgs.push(ChatMessage { role: role.to_string(), content: content.to_string(), name: None });
        }
        let m = model_arc(h);
        let s = format_chat_prompt(&msgs, m.model.config.architecture);
        unsafe { write_out(out_len, s.len(), "out_len")? };
        let dst = unsafe { out_slice(out as *mut u8, capacity, "out")? };
        if dst.len() <= s.len() {
            fail!(QUINTRIT_ERR_BUFFER_TOO_SMALL, "the prompt needs {} bytes including the NUL, buffer holds {capacity}", s.len() + 1);
        }
        dst[..s.len()].copy_from_slice(s.as_bytes());
        dst[s.len()] = 0;
        Ok(())
    })
}

/// Creates a session (KV cache and scratch buffers) for a model. `options` may be NULL.
///
/// # Safety
/// `model` is a live handle; `options` is NULL or a `quintrit_session_options_t`; `out_session` points to a handle
/// pointer, set to NULL on failure.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_session_create(
    model: *const ModelHandle,
    options: *const SessionOptions,
    out_session: *mut *mut SessionHandle,
) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / read_versioned / out_ptr.
        let h = unsafe { check_handle(model, "model")? };
        let opts = unsafe { read_versioned(options, "options")? };
        out_ptr(out_session, "out_session")?;
        unsafe { out_session.write(std::ptr::null_mut()) };
        let m = model_arc(h);
        let c = &m.model.config;
        let max_ctx = c.max_position_embeddings;
        let ctx = match opts.map_or(0, |o| o.context_length as usize) {
            0 => max_ctx,
            n if n <= max_ctx => n,
            n => fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "context_length {n} exceeds the model's maximum of {max_ctx}"),
        };
        let kv_bytes = [c.num_hidden_layers, ctx, c.num_key_value_heads, c.head_dim, 4]
            .iter()
            .try_fold(1usize, |a, &b| a.checked_mul(b));
        if kv_bytes.is_none_or(|b| b > (1usize << 42)) {
            fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "a context of {ctx} tokens needs an unreasonably large KV cache");
        }
        let kv = KVCache::try_new(c.num_hidden_layers, ctx, c.num_key_value_heads, c.head_dim)
            .map_err(|m| Failure::new(QUINTRIT_ERR_INVALID_ARGUMENT, m))?;
        let state = SessionState {
            kv,
            scratch: InferenceScratchpad::new(c, ctx),
            logits: vec![0.0; c.vocab_size],
            ctx_len: ctx,
            model: m.clone(),
        };
        let handle = Box::new(SessionHandle { magic: MAGIC_SESSION, state: Mutex::new(state) });
        unsafe { out_session.write(Box::into_raw(handle)) };
        Ok(())
    })
}

/// Frees a session. NULL is allowed.
///
/// # Safety
/// `session` must be NULL or a live handle that no other thread is using.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_session_free(session: *mut SessionHandle) -> c_int {
    guard(|| {
        if session.is_null() {
            return Ok(());
        }
        // SAFETY: the tag is validated before the Box is rebuilt.
        unsafe { check_handle(session, "session")? };
        unsafe {
            (*session).magic = 0;
            drop(Box::from_raw(session));
        }
        Ok(())
    })
}

/// Forgets the conversation: empties the KV cache.
///
/// # Safety
/// `session` must be a live handle.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_session_reset(session: *mut SessionHandle) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle.
        let h = unsafe { check_handle(session, "session")? };
        lock_session(h)?.kv.reset();
        Ok(())
    })
}

/// Reports the context window and the number of tokens currently in the KV cache.
///
/// # Safety
/// `session` is a live handle; `out_info` points to a `quintrit_session_info_t` with `struct_size` set.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_session_info(session: *const SessionHandle, out_info: *mut SessionInfo) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / check_out_struct.
        let h = unsafe { check_handle(session, "session")? };
        let out = unsafe { check_out_struct(out_info, "out_info")? };
        let st = lock_session(h)?;
        let info = SessionInfo {
            struct_size: size_of::<SessionInfo>() as u32,
            reserved: 0,
            context_length: st.ctx_len as u64,
            position: st.kv.len() as u64,
        };
        unsafe { out.write(info) };
        Ok(())
    })
}

/// Fills `params` with the defaults: greedy decoding, 128 new tokens, no stop sequences.
///
/// # Safety
/// `params` points to a `quintrit_generate_params_t`.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_generate_params_default(params: *mut GenerateParams) -> c_int {
    guard(|| {
        let p = out_ptr(params, "params")?;
        // SAFETY: non-NULL and aligned.
        unsafe {
            p.write(GenerateParams {
                struct_size: size_of::<GenerateParams>() as u32,
                flags: 0,
                max_tokens: 128,
                top_k: 1,
                temperature: 0.0,
                top_p: 1.0,
                seed: 0,
                stop_sequences: std::ptr::null(),
                stop_count: 0,
            });
        }
        Ok(())
    })
}

/// Generates text for a prompt.
///
/// Unless `QUINTRIT_GENERATE_KEEP_CONTEXT` is set the KV cache is reset first and the BOS token is prepended. The text
/// is streamed to `callback` (may be NULL) in UTF-8 pieces and also collected into `out` (may be NULL with
/// `out_capacity == 0`), truncated like `snprintf` and always NUL-terminated if `out_capacity > 0`; `*out_len` (may be
/// NULL) receives the full length without the NUL, so a value `>= out_capacity` means truncation. The callback returns
/// non-zero to stop. `params == NULL` selects the defaults of `quintrit_generate_params_default`; `result` may be NULL.
///
/// A session must not be used by two threads at the same time (`QUINTRIT_ERR_BUSY`). The callback runs on the calling
/// thread, must not call back into the same session and must not throw or `longjmp`.
///
/// # Safety
/// `prompt` points to `prompt_len` bytes; `params` is NULL or a `quintrit_generate_params_t` whose `stop_sequences`
/// array holds `stop_count` NUL-terminated strings; `out` to `out_capacity` bytes.
#[unsafe(no_mangle)]
pub unsafe extern "C" fn quintrit_session_generate(
    session: *mut SessionHandle,
    prompt: *const c_char,
    prompt_len: usize,
    params: *const GenerateParams,
    callback: TokenCallback,
    user_data: *mut c_void,
    out: *mut c_char,
    out_capacity: usize,
    out_len: *mut usize,
    result: *mut GenerateResult,
) -> c_int {
    guard(|| {
        // SAFETY: validated by check_handle / text / read_versioned / in_slice / c_str / check_out_struct.
        let h = unsafe { check_handle(session, "session")? };
        let prompt = unsafe { text(prompt, prompt_len, "prompt")? };
        let params = match unsafe { read_versioned(params, "params")? } {
            Some(p) => p,
            None => GenerateParams {
                struct_size: size_of::<GenerateParams>() as u32,
                flags: 0,
                max_tokens: 128,
                top_k: 1,
                temperature: 0.0,
                top_p: 1.0,
                seed: 0,
                stop_sequences: std::ptr::null(),
                stop_count: 0,
            },
        };
        if params.flags & !QUINTRIT_GENERATE_KEEP_CONTEXT != 0 {
            fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "unknown generate flags 0x{:x}", params.flags);
        }
        if !params.temperature.is_finite() || params.temperature < 0.0 {
            fail!(QUINTRIT_ERR_INVALID_VALUE, "temperature must be a finite number >= 0");
        }
        if !(params.top_p > 0.0 && params.top_p <= 1.0) {
            fail!(QUINTRIT_ERR_INVALID_VALUE, "top_p must be in (0, 1]");
        }
        if params.stop_count > MAX_STOP_SEQUENCES {
            fail!(QUINTRIT_ERR_INVALID_ARGUMENT, "at most {MAX_STOP_SEQUENCES} stop sequences are supported");
        }
        let raw_stops = unsafe { in_slice(params.stop_sequences, params.stop_count, "stop_sequences")? };
        let mut stops = Vec::with_capacity(raw_stops.len());
        for (i, &p) in raw_stops.iter().enumerate() {
            let s = unsafe { c_str(p, &format!("stop_sequences[{i}]"))? };
            if !s.is_empty() {
                stops.push(s);
            }
        }
        let result_ptr = if result.is_null() { None } else { Some(unsafe { check_out_struct(result, "result")? }) };
        if !out_len.is_null() {
            out_ptr(out_len, "out_len")?;
        }
        if out.is_null() != (out_capacity == 0) {
            fail!(
                crate::error::QUINTRIT_ERR_NULL_POINTER,
                "out and out_capacity must be given together (NULL with capacity 0, or a buffer with a capacity)"
            );
        }

        let mut st = lock_session(h)?;
        let mut emitter = Emitter { callback, user_data, stopped: false };
        let g = generate(&mut st, prompt, &params, &stops, &mut emitter)?;
        drop(st);

        unsafe { write_text(out, out_capacity, &g.text)? };
        if !out_len.is_null() {
            unsafe { out_len.write(g.text.len()) };
        }
        if let Some(r) = result_ptr {
            unsafe { r.write(g.result) };
        }
        Ok(())
    })
}

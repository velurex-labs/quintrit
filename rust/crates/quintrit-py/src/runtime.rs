//! Runtime level: model loading, tokenizer, text generation and token streaming.
//!
//! The generation loop is implemented here, step by step, on top of the public pieces of the runtime crate
//! (tokenizer, model forward pass, KV cache, sampler). That keeps token streaming pull based (a Python iterator
//! instead of a callback), makes stop sequences and `KeyboardInterrupt` work, and runs the same code for
//! `generate` and `stream`. Its behaviour follows `InferenceEngine::generate_streaming_with_stops`.

use crate::error::{guard, Fail, Kind, QError, QResult};
use crate::kernels::{resolve_threads, Kernel};
use numpy::PyArray1;
use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList};
use quintrit_runtime::engine::find_safe_flush_index;
use quintrit_runtime::StreamDecoder;
use quintrit_runtime::VerboseScope;
use quintrit_runtime::{
    format_chat_prompt, ActivationType, ChatMessage, InferenceEngine, ModelArchitecture, SamplingConfig, Sampler,
};
use std::path::PathBuf;
use std::sync::atomic::{AtomicU64, AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};

/// State shared between a `Model`, its `Tokenizer` and its token streams.
struct Shared {
    engine: Mutex<InferenceEngine>,
    /// Incremented whenever the KV cache is taken over (new generation, `reset`, `logits`); a stream whose epoch
    /// is older has lost its context.
    epoch: AtomicU64,
}

impl Shared {
    fn lock(&self) -> MutexGuard<'_, InferenceEngine> {
        self.engine.lock().unwrap_or_else(|p| p.into_inner())
    }
}

fn architecture_name(a: ModelArchitecture) -> &'static str {
    match a {
        ModelArchitecture::BitNet => "bitnet",
        ModelArchitecture::Falcon3 => "falcon3",
        ModelArchitecture::Llama => "llama",
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Generation state machine
// ----------------------------------------------------------------------------------------------------------------

/// Validated generation parameters.
struct GenParams {
    max_tokens: usize,
    sampling: SamplingConfig,
    stops: Vec<String>,
}

impl GenParams {
    fn new(
        max_tokens: i64,
        temperature: f64,
        top_k: i64,
        top_p: f64,
        seed: u64,
        stop: Option<&Bound<'_, PyAny>>,
    ) -> QResult<Self> {
        let bad = |m: String| QError::new(Kind::InvalidArgument, m);
        if max_tokens < 0 {
            return Err(bad(format!("max_tokens must be >= 0, got {max_tokens}")));
        }
        if !temperature.is_finite() || temperature < 0.0 {
            return Err(bad(format!("temperature must be a finite number >= 0, got {temperature}")));
        }
        if top_k < 0 {
            return Err(bad(format!("top_k must be >= 0 (0 disables it), got {top_k}")));
        }
        if !(top_p > 0.0 && top_p <= 1.0) {
            return Err(bad(format!("top_p must be in (0, 1], got {top_p}")));
        }
        let stops: Vec<String> = match stop {
            None => Vec::new(),
            Some(obj) if obj.is_none() => Vec::new(),
            Some(obj) => {
                if let Ok(s) = obj.extract::<String>() {
                    vec![s]
                } else if let Ok(v) = obj.extract::<Vec<String>>() {
                    v
                } else {
                    return Err(QError::new(Kind::DType, "stop must be a string or a list of strings"));
                }
            }
        }
        .into_iter()
        .filter(|s| !s.is_empty())
        .collect();
        let sampling = if temperature == 0.0 {
            SamplingConfig::greedy()
        } else {
            SamplingConfig { temperature: temperature as f32, top_k: top_k as usize, top_p: top_p as f32, seed }
        };
        Ok(Self { max_tokens: max_tokens as usize, sampling, stops })
    }
}

/// Why a generation ended.
#[derive(Clone, Copy, PartialEq, Eq, Debug)]
enum Finish {
    /// `max_tokens` reached.
    Length,
    /// End-of-sequence token.
    Eos,
    /// A stop sequence matched.
    Stop,
    /// The context window is full.
    Context,
    /// The prompt produced no tokens.
    Empty,
    /// Closed by the caller or aborted by an error.
    Closed,
}

impl Finish {
    fn as_str(self) -> &'static str {
        match self {
            Finish::Length => "length",
            Finish::Eos => "eos",
            Finish::Stop => "stop",
            Finish::Context => "context",
            Finish::Empty => "empty",
            Finish::Closed => "closed",
        }
    }
}

struct GenState {
    epoch: u64,
    prompt: Vec<u32>,
    prefilled: bool,
    sampler: Sampler,
    logits: Vec<f32>,
    /// Next position to fill in the KV cache.
    pos: usize,
    max_tokens: usize,
    produced: usize,
    stops: Vec<String>,
    /// Text held back because it may be the beginning of a stop sequence.
    pending: String,
    /// Everything yielded so far.
    text: String,
    finish: Option<Finish>,
    detok: StreamDecoder,
    detok_flushed: bool,
    eos_id: Option<u32>,
    stop_at_eot: bool,
    max_position: usize,
}

impl GenState {
    fn new(eng: &InferenceEngine, prompt: &str, params: GenParams) -> QResult<Self> {
        let mut tokens = eng.tokenizer.encode(prompt);
        if let Some(bos) = eng.tokenizer.bos_token_id() {
            if tokens.first() != Some(&bos) {
                tokens.insert(0, bos);
            }
        }
        let cfg = &eng.model.config;
        if tokens.len() > cfg.max_position_embeddings {
            return Err(QError::new(
                Kind::ContextOverflow,
                format!(
                    "the prompt has {} tokens but the context window holds {}",
                    tokens.len(),
                    cfg.max_position_embeddings
                ),
            ));
        }
        let finish = if tokens.is_empty() { Some(Finish::Empty) } else { None };
        Ok(Self {
            epoch: 0,
            prompt: tokens,
            prefilled: false,
            sampler: Sampler::new(params.sampling),
            logits: vec![0.0; cfg.vocab_size],
            pos: 0,
            max_tokens: params.max_tokens,
            produced: 0,
            stops: params.stops,
            pending: String::new(),
            text: String::new(),
            finish,
            detok: StreamDecoder::new(),
            detok_flushed: false,
            eos_id: eng.tokenizer.eos_token_id(),
            stop_at_eot: matches!(cfg.architecture, ModelArchitecture::BitNet | ModelArchitecture::Llama),
            max_position: cfg.max_position_embeddings,
        })
    }

    fn forward(&mut self, eng: &mut InferenceEngine, token: u32, pos: usize) -> Result<(), Fail> {
        guard("forward pass", || {
            let InferenceEngine { model, kv_cache, scratchpad, .. } = eng;
            model.forward_token_with_scratchpad(token, pos, kv_cache, scratchpad, &mut self.logits);
        })?;
        Ok(())
    }

    /// Produces the next chunk of text, or `None` when the generation is over.
    fn next_chunk(&mut self, eng: &mut InferenceEngine) -> Result<Option<String>, Fail> {
        let result = self.next_chunk_inner(eng);
        if result.is_err() {
            self.finish.get_or_insert(Finish::Closed);
        }
        result
    }

    fn next_chunk_inner(&mut self, eng: &mut InferenceEngine) -> Result<Option<String>, Fail> {
        if !self.prefilled && self.finish.is_none() {
            eng.kv_cache.reset();
            let prompt = std::mem::take(&mut self.prompt);
            for (pos, &id) in prompt.iter().enumerate() {
                self.forward(eng, id, pos)?;
                Python::attach(|py| py.check_signals())?;
            }
            self.pos = prompt.len();
            self.prompt = prompt;
            self.prefilled = true;
            if self.max_tokens == 0 {
                self.finish = Some(Finish::Length);
            }
        }
        loop {
            if self.finish.is_some() {
                if !self.detok_flushed {
                    self.detok_flushed = true;
                    if matches!(self.finish, Some(Finish::Length | Finish::Eos | Finish::Context)) {
                        let rest = guard("decode", || self.detok.finish(&eng.tokenizer))?;
                        self.pending.push_str(&rest);
                    }
                }
                if self.pending.is_empty() {
                    return Ok(None);
                }
                let rest = std::mem::take(&mut self.pending);
                self.text.push_str(&rest);
                return Ok(Some(rest));
            }
            let token = self.sampler.sample(&self.logits);
            if Some(token) == self.eos_id || (self.stop_at_eot && token == 128009) {
                self.finish = Some(Finish::Eos);
                continue;
            }
            let piece = guard("decode", || self.detok.push(&eng.tokenizer, token))?;
            self.produced += 1;

            let mut chunk: Option<String> = None;
            if self.stops.is_empty() {
                if !piece.is_empty() {
                    chunk = Some(piece);
                }
            } else {
                self.pending.push_str(&piece);
                let stops: Vec<&str> = self.stops.iter().map(String::as_str).collect();
                let earliest = stops.iter().filter_map(|s| self.pending.find(s)).min();
                if let Some(idx) = earliest {
                    let safe = self.pending[..idx].to_string();
                    self.pending.clear();
                    self.finish = Some(Finish::Stop);
                    if !safe.is_empty() {
                        chunk = Some(safe);
                    }
                } else {
                    let idx = find_safe_flush_index(&self.pending, &stops);
                    if idx > 0 {
                        chunk = Some(self.pending[..idx].to_string());
                        self.pending.drain(..idx);
                    }
                }
            }

            if self.finish.is_none() {
                if self.produced >= self.max_tokens {
                    self.finish = Some(Finish::Length);
                } else if self.pos >= self.max_position {
                    self.finish = Some(Finish::Context);
                } else {
                    let pos = self.pos;
                    self.forward(eng, token, pos)?;
                    self.pos += 1;
                    Python::attach(|py| py.check_signals())?;
                }
            }
            if let Some(c) = chunk {
                self.text.push_str(&c);
                return Ok(Some(c));
            }
        }
    }
}

fn new_stream(py: Python<'_>, shared: &Arc<Shared>, prompt: &str, params: GenParams) -> PyResult<TokenStream> {
    let mut state = py
        .detach(|| {
            let eng = shared.lock();
            GenState::new(&eng, prompt, params)
        })
        .map_err(PyErr::from)?;
    state.epoch = shared.epoch.fetch_add(1, Ordering::SeqCst) + 1;
    Ok(TokenStream { shared: shared.clone(), state: Mutex::new(state) })
}

/// Runs one step of a stream with the GIL released.
fn step(py: Python<'_>, stream: &TokenStream) -> PyResult<Option<String>> {
    py.detach(|| -> Result<Option<String>, Fail> {
        let mut st = stream.state.lock().unwrap_or_else(|p| p.into_inner());
        if st.finish.is_some() && st.pending.is_empty() {
            return Ok(None);
        }
        let mut eng = stream.shared.lock();
        if stream.shared.epoch.load(Ordering::SeqCst) != st.epoch {
            st.finish = Some(Finish::Closed);
            return Err(QError::new(
                Kind::StaleStream,
                "this token stream was invalidated by a newer generation, logits() or reset() on the same model",
            )
            .into());
        }
        st.next_chunk(&mut eng)
    })
    .map_err(PyErr::from)
}

/// Iterator over the text pieces of one generation.
///
/// A model has a single context: starting another generation (or calling `reset()` / `logits()`) invalidates the
/// stream, and its next step raises `StaleStreamError`.
#[pyclass(frozen, name = "TokenStream", module = "quintrit")]
pub struct TokenStream {
    shared: Arc<Shared>,
    state: Mutex<GenState>,
}

#[pymethods]
impl TokenStream {
    fn __iter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    fn __next__(&self, py: Python<'_>) -> PyResult<Option<String>> {
        step(py, self)
    }

    /// Everything yielded so far.
    #[getter]
    fn text(&self) -> String {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).text.clone()
    }

    /// Number of tokens sampled so far.
    #[getter]
    fn n_tokens(&self) -> usize {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).produced
    }

    /// Ids of the tokens sampled so far (the end-of-sequence token is not included).
    #[getter]
    fn token_ids(&self) -> Vec<u32> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).detok.ids().to_vec()
    }

    /// `None` while running, then `"length"`, `"eos"`, `"stop"`, `"context"`, `"empty"` or `"closed"`.
    #[getter]
    fn finish_reason(&self) -> Option<&'static str> {
        self.state.lock().unwrap_or_else(|p| p.into_inner()).finish.map(Finish::as_str)
    }

    /// Ends the generation early.
    fn close(&self) {
        let mut st = self.state.lock().unwrap_or_else(|p| p.into_inner());
        st.finish.get_or_insert(Finish::Closed);
        st.pending.clear();
    }

    fn __enter__(slf: PyRef<'_, Self>) -> PyRef<'_, Self> {
        slf
    }

    #[pyo3(signature = (*_args))]
    fn __exit__(&self, _args: &Bound<'_, pyo3::types::PyTuple>) {
        self.close();
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Tokenizer
// ----------------------------------------------------------------------------------------------------------------

/// The model's BPE tokenizer.
#[pyclass(frozen, name = "Tokenizer", module = "quintrit")]
pub struct PyTokenizer {
    shared: Arc<Shared>,
}

#[pymethods]
impl PyTokenizer {
    /// Encodes text to token ids; `add_bos=True` prepends the BOS token if the model uses one.
    #[pyo3(signature = (text, *, add_bos = false))]
    fn encode(&self, py: Python<'_>, text: &str, add_bos: bool) -> PyResult<Vec<u32>> {
        py.detach(|| {
            let eng = self.shared.lock();
            guard("encode", || {
                let mut ids = eng.tokenizer.encode(text);
                if add_bos {
                    if let Some(bos) = eng.tokenizer.bos_token_id() {
                        if ids.first() != Some(&bos) {
                            ids.insert(0, bos);
                        }
                    }
                }
                ids
            })
        })
        .map_err(|e| PyErr::from(QError::new(Kind::Tokenizer, e.msg)))
    }

    /// Decodes token ids to text.
    fn decode(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<String> {
        py.detach(|| {
            let eng = self.shared.lock();
            guard("decode", || eng.tokenizer.decode(&ids))
        })
        .map_err(|e| PyErr::from(QError::new(Kind::Tokenizer, e.msg)))
    }

    /// The text pieces a token stream yields for these ids, one per id plus a final one for text held back at the end.
    ///
    /// A piece is empty while a multi-byte character is still incomplete; pieces never contain half characters,
    /// so `"".join(pieces) == decode(ids)` for valid UTF-8.
    fn decode_pieces(&self, py: Python<'_>, ids: Vec<u32>) -> PyResult<Vec<String>> {
        py.detach(|| {
            let eng = self.shared.lock();
            guard("decode", || {
                let mut detok = StreamDecoder::new();
                let mut out: Vec<String> = ids.iter().map(|&id| detok.push(&eng.tokenizer, id)).collect();
                out.push(detok.finish(&eng.tokenizer));
                out
            })
        })
        .map_err(|e| PyErr::from(QError::new(Kind::Tokenizer, e.msg)))
    }

    /// Decodes one token id to its text piece.
    fn decode_token(&self, id: u32) -> PyResult<String> {
        let eng = self.shared.lock();
        Ok(guard("decode_token", || eng.tokenizer.decode_token(id))
            .map_err(|e| QError::new(Kind::Tokenizer, e.msg))?)
    }

    /// Vocabulary entry of a token id (`None` if the id is out of range).
    fn id_to_token(&self, id: u32) -> Option<String> {
        self.shared.lock().tokenizer.inner().id_to_token(id)
    }

    #[getter]
    fn bos_token_id(&self) -> Option<u32> {
        self.shared.lock().tokenizer.bos_token_id()
    }

    #[getter]
    fn eos_token_id(&self) -> Option<u32> {
        self.shared.lock().tokenizer.eos_token_id()
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.shared.lock().tokenizer.vocab_size()
    }

    fn __len__(&self) -> usize {
        self.shared.lock().tokenizer.vocab_size()
    }
}

// ----------------------------------------------------------------------------------------------------------------
// Model
// ----------------------------------------------------------------------------------------------------------------

/// A loaded ternary language model (GGUF with `I2_S`, `TQ1_0` or `TQ2_0` weights).
#[pyclass(frozen, name = "Model", module = "quintrit")]
pub struct PyModel {
    shared: Arc<Shared>,
    path: PathBuf,
    kernel: Kernel,
    threads: AtomicUsize,
}

/// Resolves a model path: a file, or a directory holding `ggml-model-i2_s.gguf` (or exactly one `.gguf` file).
fn resolve_model_path(path: PathBuf) -> QResult<PathBuf> {
    if !path.exists() {
        return Err(QError::new(Kind::FileNotFound, format!("{}: no such file or directory", path.display())));
    }
    if !path.is_dir() {
        return Ok(path);
    }
    let preferred = path.join("ggml-model-i2_s.gguf");
    if preferred.is_file() {
        return Ok(preferred);
    }
    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(&path)
        .map_err(|e| QError::new(Kind::Io, format!("{}: {e}", path.display())))?
        .filter_map(|e| e.ok().map(|e| e.path()))
        .filter(|p| p.extension().is_some_and(|x| x.eq_ignore_ascii_case("gguf")))
        .collect();
    if ggufs.len() == 1 {
        return Ok(ggufs.remove(0));
    }
    Err(QError::new(
        Kind::ModelLoad,
        format!("{}: expected ggml-model-i2_s.gguf or exactly one .gguf file in the directory", path.display()),
    ))
}

#[pymethods]
impl PyModel {
    /// Loads a model and transcodes its ternary weights into the in-memory layout of the selected kernel.
    ///
    /// `path` is a GGUF file or a directory containing one. `threads` defaults to the number of logical CPUs;
    /// `kernel` is `"auto"` (default), `"quintrit"` or `"quintrit_512"`. Loading takes seconds to minutes for
    /// multi-gigabyte models and releases the GIL. Loading is silent; `verbose=True` prints progress lines to the
    /// standard output of the process (the C-level stream, not `sys.stdout`).
    #[staticmethod]
    #[pyo3(signature = (path, *, threads = None, kernel = None, verbose = false))]
    fn load(py: Python<'_>, path: PathBuf, threads: Option<usize>, kernel: Option<&str>, verbose: bool) -> PyResult<PyModel> {
        let threads = resolve_threads(threads)?;
        let kernel = Kernel::parse(kernel)?;
        let path = resolve_model_path(path)?;
        let engine = py
            .detach(|| {
                guard("model loading", || {
                    let _verbose = VerboseScope::new(verbose);
                    InferenceEngine::load_from_gguf(&path, threads, Some(kernel.runtime_kind()))
                })
            })
            .map_err(|e| PyErr::from(QError::new(Kind::ModelLoad, e.msg)))?
            .map_err(|e| QError::new(Kind::ModelLoad, format!("{}: {e}", path.display())))?;
        Ok(PyModel {
            shared: Arc::new(Shared { engine: Mutex::new(engine), epoch: AtomicU64::new(0) }),
            path,
            kernel,
            threads: AtomicUsize::new(threads),
        })
    }

    #[getter]
    fn path(&self) -> String {
        self.path.display().to_string()
    }

    /// Kernel the weights were transcoded for.
    #[getter]
    fn kernel(&self) -> &'static str {
        self.kernel.name()
    }

    /// Worker threads used by the matrix kernels (assignable).
    #[getter]
    fn threads(&self) -> usize {
        self.threads.load(Ordering::Relaxed)
    }

    #[setter]
    fn set_threads(&self, py: Python<'_>, threads: usize) -> PyResult<()> {
        let threads = resolve_threads(Some(threads))?;
        py.detach(|| self.shared.lock().set_num_threads(threads));
        self.threads.store(threads, Ordering::Relaxed);
        Ok(())
    }

    /// The model hyperparameters as a dict.
    #[getter]
    fn config<'py>(&self, py: Python<'py>) -> PyResult<Bound<'py, PyDict>> {
        let eng = self.shared.lock();
        let c = &eng.model.config;
        let d = PyDict::new(py);
        d.set_item("architecture", architecture_name(c.architecture))?;
        d.set_item("vocab_size", c.vocab_size)?;
        d.set_item("hidden_size", c.hidden_size)?;
        d.set_item("intermediate_size", c.intermediate_size)?;
        d.set_item("num_hidden_layers", c.num_hidden_layers)?;
        d.set_item("num_attention_heads", c.num_attention_heads)?;
        d.set_item("num_key_value_heads", c.num_key_value_heads)?;
        d.set_item("head_dim", c.head_dim)?;
        d.set_item("rms_norm_eps", c.rms_norm_eps)?;
        d.set_item("rope_theta", c.rope_theta)?;
        d.set_item("max_position_embeddings", c.max_position_embeddings)?;
        d.set_item("hidden_act", match c.hidden_act { ActivationType::Relu2 => "relu2", ActivationType::Silu => "silu" })?;
        d.set_item("has_sub_norm", c.has_sub_norm)?;
        d.set_item("tie_word_embeddings", c.tie_word_embeddings)?;
        d.set_item("bos_token_id", c.bos_token_id)?;
        d.set_item("eos_token_id", c.eos_token_id)?;
        Ok(d)
    }

    #[getter]
    fn architecture(&self) -> &'static str {
        architecture_name(self.shared.lock().model.config.architecture)
    }

    #[getter]
    fn vocab_size(&self) -> usize {
        self.shared.lock().model.config.vocab_size
    }

    /// Maximum number of tokens (prompt plus generated) the context window holds.
    #[getter]
    fn context_length(&self) -> usize {
        self.shared.lock().model.config.max_position_embeddings
    }

    #[getter]
    fn n_layers(&self) -> usize {
        self.shared.lock().model.config.num_hidden_layers
    }

    #[getter]
    fn hidden_size(&self) -> usize {
        self.shared.lock().model.config.hidden_size
    }

    /// The model's tokenizer.
    #[getter]
    fn tokenizer(&self) -> PyTokenizer {
        PyTokenizer { shared: self.shared.clone() }
    }

    /// Drops the KV cache contents and invalidates running token streams.
    fn reset(&self, py: Python<'_>) {
        self.shared.epoch.fetch_add(1, Ordering::SeqCst);
        py.detach(|| self.shared.lock().kv_cache.reset());
    }

    /// Generates text for `prompt` and returns it.
    ///
    /// Every call starts from an empty context. `temperature=0` (default) is greedy decoding and deterministic;
    /// otherwise tokens are sampled with `top_k` (0 disables it), `top_p` and `seed`. `stop` is a string or a list
    /// of strings; the output ends before the first match. Ctrl-C interrupts the generation.
    #[pyo3(signature = (prompt, max_tokens = 128, *, temperature = 0.0, top_k = 40, top_p = 0.9, seed = 42, stop = None))]
    fn generate(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: i64,
        temperature: f64,
        top_k: i64,
        top_p: f64,
        seed: u64,
        stop: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let params = GenParams::new(max_tokens, temperature, top_k, top_p, seed, stop)?;
        let stream = new_stream(py, &self.shared, prompt, params)?;
        let mut out = String::new();
        while let Some(piece) = step(py, &stream)? {
            out.push_str(&piece);
        }
        Ok(out)
    }

    /// Like `generate`, but returns an iterator over the text pieces (`TokenStream`).
    #[pyo3(signature = (prompt, max_tokens = 128, *, temperature = 0.0, top_k = 40, top_p = 0.9, seed = 42, stop = None))]
    fn stream(
        &self,
        py: Python<'_>,
        prompt: &str,
        max_tokens: i64,
        temperature: f64,
        top_k: i64,
        top_p: f64,
        seed: u64,
        stop: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<TokenStream> {
        let params = GenParams::new(max_tokens, temperature, top_k, top_p, seed, stop)?;
        new_stream(py, &self.shared, prompt, params)
    }

    /// Formats chat messages (dicts with `role` and `content`) into the prompt of the model's chat template.
    fn format_chat(&self, messages: &Bound<'_, PyAny>) -> PyResult<String> {
        let msgs = extract_messages(messages)?;
        let arch = self.shared.lock().model.config.architecture;
        Ok(format_chat_prompt(&msgs, arch))
    }

    /// Generates the assistant reply to a list of chat messages; the other arguments are those of `generate`.
    #[pyo3(signature = (messages, max_tokens = 128, *, temperature = 0.0, top_k = 40, top_p = 0.9, seed = 42, stop = None))]
    fn chat(
        &self,
        py: Python<'_>,
        messages: &Bound<'_, PyAny>,
        max_tokens: i64,
        temperature: f64,
        top_k: i64,
        top_p: f64,
        seed: u64,
        stop: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<String> {
        let prompt = self.format_chat(messages)?;
        self.generate(py, &prompt, max_tokens, temperature, top_k, top_p, seed, stop)
    }

    /// Next-token logits (float32, shape `(vocab_size,)`) after the prompt, given as text or as a list of token ids.
    /// Uses (and replaces) the model's context.
    fn logits<'py>(&self, py: Python<'py>, prompt: &Bound<'py, PyAny>) -> PyResult<Bound<'py, PyArray1<f32>>> {
        enum Input {
            Text(String),
            Ids(Vec<u32>),
        }
        let input = if let Ok(s) = prompt.extract::<String>() {
            Input::Text(s)
        } else if let Ok(ids) = prompt.extract::<Vec<u32>>() {
            Input::Ids(ids)
        } else {
            return Err(QError::new(Kind::DType, "prompt must be a string or a list of token ids").into());
        };
        let out = py
            .detach(|| -> QResult<Vec<f32>> {
                let mut eng = self.shared.lock();
                let (vocab, max_pos) = (eng.model.config.vocab_size, eng.model.config.max_position_embeddings);
                let ids = match input {
                    Input::Ids(ids) => ids,
                    Input::Text(text) => {
                        let mut ids = eng.tokenizer.encode(&text);
                        if let Some(bos) = eng.tokenizer.bos_token_id() {
                            if ids.first() != Some(&bos) {
                                ids.insert(0, bos);
                            }
                        }
                        ids
                    }
                };
                if ids.is_empty() {
                    return Err(QError::new(Kind::InvalidArgument, "the prompt has no tokens"));
                }
                if ids.len() > max_pos {
                    return Err(QError::new(
                        Kind::ContextOverflow,
                        format!("the prompt has {} tokens but the context window holds {max_pos}", ids.len()),
                    ));
                }
                if let Some(&bad) = ids.iter().find(|&&t| t as usize >= vocab) {
                    return Err(QError::new(
                        Kind::InvalidArgument,
                        format!("token id {bad} is outside the vocabulary of {vocab} tokens"),
                    ));
                }
                self.shared.epoch.fetch_add(1, Ordering::SeqCst);
                eng.kv_cache.reset();
                let mut logits = vec![0.0f32; vocab];
                guard("forward pass", || {
                    let InferenceEngine { model, kv_cache, scratchpad, .. } = &mut *eng;
                    for (pos, &id) in ids.iter().enumerate() {
                        model.forward_token_with_scratchpad(id, pos, kv_cache, scratchpad, &mut logits);
                    }
                })?;
                Ok(logits)
            })
            .map_err(PyErr::from)?;
        Ok(PyArray1::from_vec(py, out))
    }

    fn __repr__(&self) -> String {
        let eng = self.shared.lock();
        format!(
            "Model(path={:?}, architecture={:?}, layers={}, kernel={:?}, threads={})",
            self.path.display().to_string(),
            architecture_name(eng.model.config.architecture),
            eng.model.config.num_hidden_layers,
            self.kernel.name(),
            self.threads.load(Ordering::Relaxed)
        )
    }
}

fn extract_messages(obj: &Bound<'_, PyAny>) -> PyResult<Vec<ChatMessage>> {
    let list = obj
        .cast::<PyList>()
        .map_err(|_| QError::new(Kind::DType, "messages must be a list of dicts with 'role' and 'content'"))?;
    let mut out = Vec::with_capacity(list.len());
    for item in list.iter() {
        let dict = item
            .cast::<PyDict>()
            .map_err(|_| QError::new(Kind::DType, "each message must be a dict with 'role' and 'content'"))?;
        let get = |key: &str| -> PyResult<String> {
            match dict.get_item(key)? {
                Some(v) => v
                    .extract::<String>()
                    .map_err(|_| QError::new(Kind::DType, format!("message field {key:?} must be a string")).into()),
                None => Err(QError::new(Kind::InvalidArgument, format!("message is missing the field {key:?}")).into()),
            }
        };
        out.push(ChatMessage { role: get("role")?, content: get("content")?, name: None });
    }
    Ok(out)
}

pub fn register(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<PyModel>()?;
    m.add_class::<PyTokenizer>()?;
    m.add_class::<TokenStream>()?;
    Ok(())
}

//! Runtime-level tests on a tiny synthetic model: loading, tokenization, generation, sessions, error paths, and damaged
//! model files. No model download is needed; the weights are random.

mod common;

use common::synth::*;
use common::*;
use quintrit_c::*;
use quintrit_runtime::{InferenceEngine, SamplingConfig};
use std::ffi::{c_char, c_void};
use std::path::Path;
use std::ptr::{null, null_mut};
use std::sync::OnceLock;

struct Fixture {
    _dir: TempDir,
    path: std::path::PathBuf,
}

/// One model file shared by the tests of this binary.
fn fixture() -> &'static Fixture {
    static F: OnceLock<Fixture> = OnceLock::new();
    F.get_or_init(|| {
        let dir = TempDir::new("runtime");
        let path = dir.path("model.gguf");
        write_model(&path, 11, &Flaws::default());
        Fixture { _dir: dir, path }
    })
}

fn open_model_at(path: &Path, threads: u32, kernel: i32) -> Result<*mut ModelHandle, i32> {
    let opts = ModelOptions { struct_size: size_of::<ModelOptions>() as u32, num_threads: threads, kernel, flags: 0 };
    let p = cstr(path.to_str().unwrap());
    let mut m: *mut ModelHandle = std::ptr::dangling_mut::<ModelHandle>();
    let st = unsafe { quintrit_model_open(p.as_ptr(), &opts, &mut m) };
    if st == QUINTRIT_OK {
        assert!(!m.is_null());
        Ok(m)
    } else {
        assert!(m.is_null(), "*out_model must be NULL after a failure");
        Err(st)
    }
}

fn model(threads: u32) -> *mut ModelHandle {
    open_model_at(&fixture().path, threads, QUINTRIT_KERNEL_AUTO).unwrap()
}

fn session(m: *const ModelHandle, ctx: u32) -> *mut SessionHandle {
    let opts = SessionOptions { struct_size: size_of::<SessionOptions>() as u32, context_length: ctx };
    let mut s: *mut SessionHandle = null_mut();
    expect_status(unsafe { quintrit_session_create(m, &opts, &mut s) }, QUINTRIT_OK);
    assert!(!s.is_null());
    s
}

fn params() -> GenerateParams {
    let mut p: GenerateParams = unsafe { std::mem::zeroed() };
    expect_status(unsafe { quintrit_generate_params_default(&mut p) }, QUINTRIT_OK);
    p
}

fn result() -> GenerateResult {
    GenerateResult { struct_size: size_of::<GenerateResult>() as u32, finish_reason: -1, prompt_tokens: 0, generated_tokens: 0, text_bytes: 0 }
}

/// Runs a generation and returns the text from the output buffer.
fn run(s: *mut SessionHandle, prompt: &str, p: &GenerateParams) -> (String, GenerateResult) {
    let mut out = vec![0 as c_char; 8192];
    let mut len = 0usize;
    let mut r = result();
    expect_status(
        unsafe { quintrit_session_generate(s, prompt.as_ptr() as *const c_char, prompt.len(), p, None, null_mut(), out.as_mut_ptr(), out.len(), &mut len, &mut r) },
        QUINTRIT_OK,
    );
    assert!(len < out.len());
    assert_eq!(r.text_bytes as usize, len);
    (from_c(out.as_ptr()), r)
}

fn info(m: *const ModelHandle) -> ModelInfo {
    let mut i: ModelInfo = unsafe { std::mem::zeroed() };
    i.struct_size = size_of::<ModelInfo>() as u32;
    expect_status(unsafe { quintrit_model_info(m, &mut i) }, QUINTRIT_OK);
    i
}

fn sinfo(s: *const SessionHandle) -> SessionInfo {
    let mut i: SessionInfo = unsafe { std::mem::zeroed() };
    i.struct_size = size_of::<SessionInfo>() as u32;
    expect_status(unsafe { quintrit_session_info(s, &mut i) }, QUINTRIT_OK);
    i
}

const PROMPT: &str = "<|user|>\nHello there\n<|assistant|>\n";

#[test]
fn model_info_and_tokenizer() {
    let m = model(2);
    let i = info(m);
    assert_eq!((i.num_layers as usize, i.hidden_size as usize, i.num_heads as usize), (LAYERS, HIDDEN, HEADS));
    assert_eq!((i.num_kv_heads as usize, i.head_dim as usize), (KV_HEADS, HEAD_DIM));
    assert_eq!((i.vocab_size as usize, i.context_length as usize), (VOCAB, CONTEXT));
    assert_eq!((i.bos_token_id, i.eos_token_id), (BOS as i64, EOS as i64));
    assert_eq!(i.num_threads, 2);
    assert_eq!(from_c(i.architecture.as_ptr()), "llama");
    assert!(i.kernel == QUINTRIT_KERNEL_SIMD32 || i.kernel == QUINTRIT_KERNEL_AVX512);

    // Byte-level vocabulary: every byte is its own token, id = byte value.
    let text = "Hello, World! 123";
    let mut count = 0usize;
    assert_eq!(unsafe { quintrit_model_tokenize(m, text.as_ptr() as *const c_char, text.len(), 0, null_mut(), 0, &mut count) }, QUINTRIT_ERR_BUFFER_TOO_SMALL);
    assert_eq!(count, text.len());
    let mut ids = vec![0u32; count];
    expect_status(unsafe { quintrit_model_tokenize(m, text.as_ptr() as *const c_char, text.len(), 0, ids.as_mut_ptr(), ids.len(), &mut count) }, QUINTRIT_OK);
    assert_eq!(ids, text.bytes().map(|b| b as u32).collect::<Vec<_>>());

    let mut with_bos = vec![0u32; count + 1];
    let mut n = 0usize;
    expect_status(unsafe { quintrit_model_tokenize(m, text.as_ptr() as *const c_char, text.len(), QUINTRIT_TOKENIZE_ADD_BOS, with_bos.as_mut_ptr(), with_bos.len(), &mut n) }, QUINTRIT_OK);
    assert_eq!((n, with_bos[0]), (count + 1, BOS));
    // Not NUL-terminated input of exact length, and an empty text.
    let raw = *b"abcXYZ";
    let mut three = [0u32; 3];
    expect_status(unsafe { quintrit_model_tokenize(m, raw.as_ptr() as *const c_char, 3, 0, three.as_mut_ptr(), three.len(), &mut n) }, QUINTRIT_OK);
    assert_eq!((n, three), (3, [b'a' as u32, b'b' as u32, b'c' as u32]));
    expect_status(unsafe { quintrit_model_tokenize(m, null(), 0, 0, null_mut(), 0, &mut n) }, QUINTRIT_OK);
    assert_eq!(n, 0);

    // Detokenize with a size query.
    let mut len = 0usize;
    assert_eq!(unsafe { quintrit_model_detokenize(m, ids.as_ptr(), count, null_mut(), 0, &mut len) }, QUINTRIT_ERR_BUFFER_TOO_SMALL);
    assert_eq!(len, text.len());
    let mut buf = vec![0 as c_char; len + 1];
    expect_status(unsafe { quintrit_model_detokenize(m, ids.as_ptr(), count, buf.as_mut_ptr(), buf.len(), &mut len) }, QUINTRIT_OK);
    assert_eq!(from_c(buf.as_ptr()), text);
    let mut tight = vec![0 as c_char; len];
    assert_eq!(unsafe { quintrit_model_detokenize(m, ids.as_ptr(), count, tight.as_mut_ptr(), tight.len(), &mut len) }, QUINTRIT_ERR_BUFFER_TOO_SMALL);

    // Chat template of the llama family.
    let (role_u, content_u) = (cstr("user"), cstr("Hi"));
    let (role_s, content_s) = (cstr("system"), cstr("Be brief."));
    let msgs = [ChatMessageC { role: role_s.as_ptr(), content: content_s.as_ptr() }, ChatMessageC { role: role_u.as_ptr(), content: content_u.as_ptr() }];
    let mut chat = vec![0 as c_char; 256];
    expect_status(unsafe { quintrit_model_format_chat(m, msgs.as_ptr(), msgs.len(), chat.as_mut_ptr(), chat.len(), &mut len) }, QUINTRIT_OK);
    assert_eq!(from_c(chat.as_ptr()), "### System:\nBe brief.\n\n### User:\nHi\n\n### Assistant:\n");

    // Errors.
    let ok_text = "x";
    let bad = [0xC3u8, 0x28]; // invalid UTF-8
    expect_status(unsafe { quintrit_model_tokenize(m, bad.as_ptr() as *const c_char, 2, 0, null_mut(), 0, &mut n) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    expect_status(unsafe { quintrit_model_tokenize(m, ok_text.as_ptr() as *const c_char, 1, 0x80, null_mut(), 0, &mut n) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    expect_status(unsafe { quintrit_model_tokenize(m, null(), 5, 0, null_mut(), 0, &mut n) }, QUINTRIT_ERR_NULL_POINTER);
    expect_status(unsafe { quintrit_model_tokenize(m, ok_text.as_ptr() as *const c_char, 1, 0, null_mut(), 0, null_mut()) }, QUINTRIT_ERR_NULL_POINTER);
    expect_status(unsafe { quintrit_model_tokenize(null(), ok_text.as_ptr() as *const c_char, 1, 0, null_mut(), 0, &mut n) }, QUINTRIT_ERR_INVALID_HANDLE);
    let out_of_vocab = [VOCAB as u32];
    expect_status(unsafe { quintrit_model_detokenize(m, out_of_vocab.as_ptr(), 1, buf.as_mut_ptr(), buf.len(), &mut len) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    expect_status(unsafe { quintrit_model_detokenize(m, null(), 3, buf.as_mut_ptr(), buf.len(), &mut len) }, QUINTRIT_ERR_NULL_POINTER);
    let null_role = [ChatMessageC { role: null(), content: content_u.as_ptr() }];
    expect_status(unsafe { quintrit_model_format_chat(m, null_role.as_ptr(), 1, chat.as_mut_ptr(), chat.len(), &mut len) }, QUINTRIT_ERR_NULL_POINTER);
    let mut small_info: ModelInfo = unsafe { std::mem::zeroed() };
    expect_status(unsafe { quintrit_model_info(m, &mut small_info) }, QUINTRIT_ERR_INVALID_ARGUMENT);

    expect_status(unsafe { quintrit_model_free(m) }, QUINTRIT_OK);
}

#[test]
fn greedy_generation_is_deterministic_and_matches_the_engine() {
    let m = model(2);
    let s = session(m, 0);
    let mut p = params();
    p.max_tokens = 12;
    let (a, ra) = run(s, PROMPT, &p);
    let (b, rb) = run(s, PROMPT, &p);
    assert_eq!(a, b, "a second call restarts from an empty cache");
    assert_eq!(ra.generated_tokens, rb.generated_tokens);
    assert!(ra.generated_tokens > 0);
    assert_eq!(ra.prompt_tokens as usize, PROMPT.len() + 1, "byte tokens plus BOS");
    assert_eq!(sinfo(s).position as usize, ra.prompt_tokens as usize + ra.generated_tokens as usize);

    // The loop is a re-implementation of the engine's; both must produce the same greedy text.
    let mut engine = InferenceEngine::load_from_gguf(&fixture().path, 2, None).unwrap();
    let want = engine.generate_streaming(PROMPT, 12, &SamplingConfig::greedy(), |_| {}).unwrap();
    assert_eq!(a, want);

    // With a stop sequence taken from the middle of the text.
    assert!(a.chars().count() >= 6, "the synthetic model should produce a few characters, got {a:?}");
    {
        let chars: Vec<char> = a.chars().collect();
        let stop: String = chars[3..5].iter().collect();
        let cs = cstr(&stop);
        let stops = [cs.as_ptr()];
        let mut ps = params();
        ps.max_tokens = 12;
        ps.stop_sequences = stops.as_ptr();
        ps.stop_count = 1;
        let (t, r) = run(s, PROMPT, &ps);
        let at = a.find(&stop).unwrap();
        assert_eq!(t, a[..at], "stop sequence '{stop}' removes itself and everything after it");
        assert_eq!(r.finish_reason, QUINTRIT_FINISH_STOP_SEQUENCE);
        let want = engine.generate_streaming_with_stops(PROMPT, 12, &SamplingConfig::greedy(), &[stop.as_str()], |_| {}).unwrap();
        assert_eq!(t, want);
    }

    unsafe {
        quintrit_session_free(s);
        quintrit_model_free(m);
    }
}

unsafe extern "C" fn collect(text: *const c_char, len: usize, user: *mut c_void) -> i32 {
    let v = unsafe { &mut *(user as *mut Vec<u8>) };
    v.extend_from_slice(unsafe { std::slice::from_raw_parts(text as *const u8, len) });
    0
}

struct StopAfter {
    pieces: usize,
    limit: usize,
}

unsafe extern "C" fn stop_after(_: *const c_char, _: usize, user: *mut c_void) -> i32 {
    let st = unsafe { &mut *(user as *mut StopAfter) };
    st.pieces += 1;
    (st.pieces >= st.limit) as i32
}

#[test]
fn streaming_callback_buffer_truncation_and_stopping() {
    let m = model(1);
    let s = session(m, 0);
    let mut p = params();
    p.max_tokens = 10;
    let (full, _) = run(s, PROMPT, &p);

    // The streamed pieces concatenate to the returned text.
    let mut streamed: Vec<u8> = Vec::new();
    let mut len = 0usize;
    let mut r = result();
    expect_status(
        unsafe { quintrit_session_generate(s, PROMPT.as_ptr() as *const c_char, PROMPT.len(), &p, Some(collect), &mut streamed as *mut _ as *mut c_void, null_mut(), 0, &mut len, &mut r) },
        QUINTRIT_OK,
    );
    assert_eq!(String::from_utf8(streamed).unwrap(), full);
    assert_eq!(len, full.len());

    // snprintf-style truncation: NUL-terminated, at a character boundary, the full length is reported.
    for cap in [1usize, 2, 5] {
        let mut out = vec![0x55 as c_char; cap];
        expect_status(
            unsafe { quintrit_session_generate(s, PROMPT.as_ptr() as *const c_char, PROMPT.len(), &p, None, null_mut(), out.as_mut_ptr(), cap, &mut len, null_mut()) },
            QUINTRIT_OK,
        );
        assert_eq!(len, full.len());
        let got = from_c(out.as_ptr());
        assert!(full.starts_with(&got) && got.len() < cap, "cap {cap}: {got:?} of {full:?}");
    }

    // The callback can stop the generation; the cache then holds the tokens that were streamed.
    let mut st = StopAfter { pieces: 0, limit: 3 };
    expect_status(
        unsafe { quintrit_session_generate(s, PROMPT.as_ptr() as *const c_char, PROMPT.len(), &p, Some(stop_after), &mut st as *mut _ as *mut c_void, null_mut(), 0, null_mut(), &mut r) },
        QUINTRIT_OK,
    );
    assert_eq!((st.pieces, r.generated_tokens, r.finish_reason), (3, 3, QUINTRIT_FINISH_CALLBACK));
    assert_eq!(sinfo(s).position as usize, r.prompt_tokens as usize + 3);

    // max_tokens = 0 only ingests the prompt.
    let mut p0 = params();
    p0.max_tokens = 0;
    let (t, r) = run(s, PROMPT, &p0);
    assert_eq!((t.as_str(), r.generated_tokens, r.finish_reason), ("", 0, QUINTRIT_FINISH_LENGTH));
    assert_eq!(sinfo(s).position as usize, PROMPT.len() + 1);

    unsafe {
        quintrit_session_free(s);
        quintrit_model_free(m);
    }
}

#[test]
fn context_window_keep_context_and_reset() {
    let m = model(1);
    let s = session(m, 0);
    assert_eq!(sinfo(s).context_length as usize, CONTEXT);

    // The prompt must fit.
    let long = "x".repeat(CONTEXT);
    let p = params();
    let mut out = vec![0 as c_char; 16];
    expect_status(
        unsafe { quintrit_session_generate(s, long.as_ptr() as *const c_char, long.len(), &p, None, null_mut(), out.as_mut_ptr(), out.len(), null_mut(), null_mut()) },
        QUINTRIT_ERR_CONTEXT_OVERFLOW,
    );
    // CONTEXT - 1 characters plus BOS fill the window exactly; one token can still be sampled and is not fed back.
    let fits = "x".repeat(CONTEXT - 1);
    let mut r = result();
    let mut pf = params();
    pf.max_tokens = 100;
    expect_status(
        unsafe { quintrit_session_generate(s, fits.as_ptr() as *const c_char, fits.len(), &pf, None, null_mut(), null_mut(), 0, null_mut(), &mut r) },
        QUINTRIT_OK,
    );
    assert_eq!(r.prompt_tokens as usize, CONTEXT);
    assert!(
        (r.generated_tokens, r.finish_reason) == (1, QUINTRIT_FINISH_CONTEXT_FULL) || r.finish_reason == QUINTRIT_FINISH_EOS,
        "{} tokens, reason {}",
        r.generated_tokens,
        r.finish_reason
    );

    // A shorter window.
    let small = session(m, 8);
    assert_eq!(sinfo(small).context_length, 8);
    let ten = "y".repeat(10);
    expect_status(
        unsafe { quintrit_session_generate(small, ten.as_ptr() as *const c_char, ten.len(), &p, None, null_mut(), null_mut(), 0, null_mut(), null_mut()) },
        QUINTRIT_ERR_CONTEXT_OVERFLOW,
    );
    unsafe { quintrit_session_free(small) };
    let too_big = SessionOptions { struct_size: size_of::<SessionOptions>() as u32, context_length: CONTEXT as u32 + 1 };
    let mut bad: *mut SessionHandle = std::ptr::dangling_mut::<SessionHandle>();
    expect_status(unsafe { quintrit_session_create(m, &too_big, &mut bad) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    assert!(bad.is_null());

    // Keep-context turns accumulate; reset starts over and reproduces the first answer.
    expect_status(unsafe { quintrit_session_reset(s) }, QUINTRIT_OK);
    assert_eq!(sinfo(s).position, 0);
    let mut p1 = params();
    p1.max_tokens = 4;
    let (first, r1) = run(s, "abc", &p1);
    let pos1 = sinfo(s).position as usize;
    assert_eq!(pos1, r1.prompt_tokens as usize + r1.generated_tokens as usize);
    let mut p2 = p1;
    p2.flags = QUINTRIT_GENERATE_KEEP_CONTEXT;
    let (_, r2) = run(s, "de", &p2);
    assert_eq!(r2.prompt_tokens, 2, "no BOS when continuing");
    assert_eq!(sinfo(s).position as usize, pos1 + 2 + r2.generated_tokens as usize);
    // Continuing beyond the window fails and leaves the session intact.
    let before = sinfo(s).position;
    let filler = "z".repeat(CONTEXT);
    expect_status(
        unsafe { quintrit_session_generate(s, filler.as_ptr() as *const c_char, filler.len(), &p2, None, null_mut(), null_mut(), 0, null_mut(), null_mut()) },
        QUINTRIT_ERR_CONTEXT_OVERFLOW,
    );
    assert_eq!(sinfo(s).position, before);
    // Empty prompt: fine with BOS, rejected when continuing.
    expect_status(unsafe { quintrit_session_generate(s, null(), 0, &p2, None, null_mut(), null_mut(), 0, null_mut(), null_mut()) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    expect_status(unsafe { quintrit_session_reset(s) }, QUINTRIT_OK);
    let (again, _) = run(s, "abc", &p1);
    assert_eq!(again, first);
    expect_status(unsafe { quintrit_session_generate(s, null(), 0, &p1, None, null_mut(), null_mut(), 0, null_mut(), null_mut()) }, QUINTRIT_OK);

    unsafe {
        quintrit_session_free(s);
        quintrit_model_free(m);
    }
}

#[test]
fn sampling_is_reproducible_per_seed() {
    let m = model(1);
    let s = session(m, 0);
    let mut p = params();
    p.max_tokens = 10;
    p.temperature = 1.5;
    p.top_k = 20;
    p.top_p = 0.95;
    p.seed = 1234;
    let (a, _) = run(s, PROMPT, &p);
    let (b, _) = run(s, PROMPT, &p);
    assert_eq!(a, b);
    let mut texts = std::collections::HashSet::new();
    for seed in 1..=8u64 {
        p.seed = seed;
        texts.insert(run(s, PROMPT, &p).0);
    }
    assert!(texts.len() > 1, "different seeds should give different samples");
    unsafe {
        quintrit_session_free(s);
        quintrit_model_free(m);
    }
}

#[test]
fn generate_argument_validation() {
    let m = model(1);
    let s = session(m, 0);
    let call = |prompt: &str, p: *const GenerateParams, out: *mut c_char, cap: usize| unsafe {
        quintrit_session_generate(s, prompt.as_ptr() as *const c_char, prompt.len(), p, None, null_mut(), out, cap, null_mut(), null_mut())
    };
    let mut buf = vec![0 as c_char; 64];

    // Defaults are used for NULL params.
    expect_status(call("a", null(), buf.as_mut_ptr(), buf.len()), QUINTRIT_OK);

    let mut p = params();
    p.max_tokens = 2;
    for (name, mutate) in [
        ("temperature NaN", Box::new(|p: &mut GenerateParams| p.temperature = f32::NAN) as Box<dyn Fn(&mut GenerateParams)>),
        ("temperature negative", Box::new(|p| p.temperature = -1.0)),
        ("temperature infinite", Box::new(|p| p.temperature = f32::INFINITY)),
        ("top_p zero", Box::new(|p| p.top_p = 0.0)),
        ("top_p above one", Box::new(|p| p.top_p = 1.5)),
        ("top_p NaN", Box::new(|p| p.top_p = f32::NAN)),
    ] {
        let mut q = p;
        mutate(&mut q);
        expect_status(call("a", &q, buf.as_mut_ptr(), buf.len()), QUINTRIT_ERR_INVALID_VALUE);
        let _ = name;
    }
    let mut q = p;
    q.flags = 0x10;
    expect_status(call("a", &q, buf.as_mut_ptr(), buf.len()), QUINTRIT_ERR_INVALID_ARGUMENT);
    let mut q = p;
    q.struct_size = 8;
    expect_status(call("a", &q, buf.as_mut_ptr(), buf.len()), QUINTRIT_ERR_INVALID_ARGUMENT);
    let mut q = p;
    q.stop_count = 65;
    q.stop_sequences = buf.as_ptr() as *const *const c_char;
    expect_status(call("a", &q, buf.as_mut_ptr(), buf.len()), QUINTRIT_ERR_INVALID_ARGUMENT);
    let mut q = p;
    q.stop_count = 1;
    q.stop_sequences = null();
    expect_status(call("a", &q, buf.as_mut_ptr(), buf.len()), QUINTRIT_ERR_NULL_POINTER);
    let nulls: [*const c_char; 1] = [null()];
    q.stop_sequences = nulls.as_ptr();
    expect_status(call("a", &q, buf.as_mut_ptr(), buf.len()), QUINTRIT_ERR_NULL_POINTER);
    let bad_stop = [0xffu8, 0];
    let bad = [bad_stop.as_ptr() as *const c_char];
    q.stop_sequences = bad.as_ptr();
    expect_status(call("a", &q, buf.as_mut_ptr(), buf.len()), QUINTRIT_ERR_INVALID_ARGUMENT);

    // Output buffer and capacity must come together; invalid UTF-8 prompt; NULL prompt with a length.
    expect_status(call("a", &p, null_mut(), 5), QUINTRIT_ERR_NULL_POINTER);
    expect_status(call("a", &p, buf.as_mut_ptr(), 0), QUINTRIT_ERR_NULL_POINTER);
    let bad_prompt = [0xC3u8, 0x28];
    expect_status(
        unsafe { quintrit_session_generate(s, bad_prompt.as_ptr() as *const c_char, 2, &p, None, null_mut(), null_mut(), 0, null_mut(), null_mut()) },
        QUINTRIT_ERR_INVALID_ARGUMENT,
    );
    expect_status(unsafe { quintrit_session_generate(s, null(), 3, &p, None, null_mut(), null_mut(), 0, null_mut(), null_mut()) }, QUINTRIT_ERR_NULL_POINTER);
    // A result struct without struct_size.
    let mut r: GenerateResult = unsafe { std::mem::zeroed() };
    expect_status(
        unsafe { quintrit_session_generate(s, c"a".as_ptr(), 1, &p, None, null_mut(), null_mut(), 0, null_mut(), &mut r) },
        QUINTRIT_ERR_INVALID_ARGUMENT,
    );
    // Handles.
    expect_status(
        unsafe { quintrit_session_generate(null_mut(), c"a".as_ptr(), 1, &p, None, null_mut(), null_mut(), 0, null_mut(), null_mut()) },
        QUINTRIT_ERR_INVALID_HANDLE,
    );
    // A model handle is not a session handle.
    expect_status(
        unsafe { quintrit_session_generate(m as *mut SessionHandle, c"a".as_ptr(), 1, &p, None, null_mut(), null_mut(), 0, null_mut(), null_mut()) },
        QUINTRIT_ERR_INVALID_HANDLE,
    );
    expect_status(unsafe { quintrit_model_free(s as *mut ModelHandle) }, QUINTRIT_ERR_INVALID_HANDLE);
    expect_status(unsafe { quintrit_model_info(s as *const ModelHandle, null_mut()) }, QUINTRIT_ERR_INVALID_HANDLE);
    expect_status(unsafe { quintrit_session_free(null_mut()) }, QUINTRIT_OK);
    expect_status(unsafe { quintrit_model_free(null_mut()) }, QUINTRIT_OK);
    expect_status(unsafe { quintrit_session_reset(null_mut()) }, QUINTRIT_ERR_INVALID_HANDLE);

    // After all the rejected calls the session still works.
    expect_status(call("a", &p, buf.as_mut_ptr(), buf.len()), QUINTRIT_OK);
    unsafe {
        quintrit_session_free(s);
        quintrit_model_free(m);
    }
}

struct Rendezvous {
    entered: std::sync::mpsc::Sender<()>,
    proceed: std::sync::mpsc::Receiver<()>,
}

unsafe extern "C" fn block_first_piece(_: *const c_char, _: usize, user: *mut c_void) -> i32 {
    let r = unsafe { &mut *(user as *mut Rendezvous) };
    let _ = r.entered.send(());
    let _ = r.proceed.recv();
    1
}

#[test]
fn sessions_threads_and_handle_lifetimes() {
    let m = model(2);
    let s = session(m, 0);
    let mut p = params();
    p.max_tokens = 6;
    let (expected, _) = run(s, PROMPT, &p);

    // The thread count cannot change while a session exists.
    expect_status(unsafe { quintrit_model_set_threads(m, 3) }, QUINTRIT_ERR_BUSY);
    expect_status(unsafe { quintrit_model_set_threads(m, QUINTRIT_MAX_THREADS + 1) }, QUINTRIT_ERR_INVALID_ARGUMENT);

    // A session that is generating refuses a second caller.
    let (entered_tx, entered_rx) = std::sync::mpsc::channel();
    let (go_tx, go_rx) = std::sync::mpsc::channel();
    let addr = s as usize;
    let worker = std::thread::spawn(move || {
        let mut rv = Rendezvous { entered: entered_tx, proceed: go_rx };
        let p = params();
        let st = unsafe {
            quintrit_session_generate(addr as *mut SessionHandle, PROMPT.as_ptr() as *const c_char, PROMPT.len(), &p, Some(block_first_piece), &mut rv as *mut _ as *mut c_void, null_mut(), 0, null_mut(), null_mut())
        };
        assert_eq!(st, QUINTRIT_OK);
    });
    entered_rx.recv().unwrap();
    expect_status(
        unsafe { quintrit_session_generate(s, c"a".as_ptr(), 1, &p, None, null_mut(), null_mut(), 0, null_mut(), null_mut()) },
        QUINTRIT_ERR_BUSY,
    );
    expect_status(unsafe { quintrit_session_reset(s) }, QUINTRIT_ERR_BUSY);
    go_tx.send(()).unwrap();
    worker.join().unwrap();

    // Two sessions of one model on two threads give the text of a sequential run.
    let s2 = session(m, 0);
    std::thread::scope(|sc| {
        for sess in [s as usize, s2 as usize] {
            let expected = &expected;
            sc.spawn(move || {
                let mut p = params();
                p.max_tokens = 6;
                let (t, _) = run(sess as *mut SessionHandle, PROMPT, &p);
                assert_eq!(&t, expected);
            });
        }
    });

    // The model outlives its handle while sessions exist, and the thread count can be changed once they are gone.
    expect_status(unsafe { quintrit_model_free(m) }, QUINTRIT_OK);
    let (t, _) = run(s, PROMPT, &p);
    assert_eq!(t, expected);
    unsafe {
        quintrit_session_free(s);
        quintrit_session_free(s2);
    }
    let m = model(1);
    expect_status(unsafe { quintrit_model_set_threads(m, 3) }, QUINTRIT_OK);
    assert_eq!(info(m).num_threads, 3);
    expect_status(unsafe { quintrit_model_set_threads(m, 0) }, QUINTRIT_OK);
    unsafe { quintrit_model_free(m) };
}

#[test]
fn every_supported_kernel_generates_the_same_text() {
    let mut texts = Vec::new();
    for kernel in [QUINTRIT_KERNEL_SIMD32, QUINTRIT_KERNEL_AVX512, QUINTRIT_KERNEL_AUTO] {
        let mut supported = 0;
        expect_status(unsafe { quintrit_kernel_supported(kernel, &mut supported) }, QUINTRIT_OK);
        if supported == 0 {
            assert_eq!(open_model_at(&fixture().path, 1, kernel).unwrap_err(), QUINTRIT_ERR_UNSUPPORTED);
            continue;
        }
        let m = open_model_at(&fixture().path, 2, kernel).unwrap();
        let s = session(m, 0);
        let mut p = params();
        p.max_tokens = 10;
        texts.push((kernel, run(s, PROMPT, &p).0));
        unsafe {
            quintrit_session_free(s);
            quintrit_model_free(m);
        }
    }
    assert!(texts.len() >= 2);
    for (k, t) in &texts {
        assert_eq!(t, &texts[0].1, "kernel {k} differs from kernel {}", texts[0].0);
    }
}

#[test]
fn model_open_error_paths() {
    let dir = TempDir::new("runtime-bad");
    let good = &fixture().path;

    assert_eq!(open_model_at(&dir.path("missing.gguf"), 1, 0).unwrap_err(), QUINTRIT_ERR_IO);
    std::fs::write(dir.path("junk.gguf"), vec![7u8; 5000]).unwrap();
    assert_eq!(open_model_at(&dir.path("junk.gguf"), 1, 0).unwrap_err(), QUINTRIT_ERR_FORMAT);
    assert_eq!(open_model_at(good, 1, QUINTRIT_KERNEL_SCALAR).unwrap_err(), QUINTRIT_ERR_UNSUPPORTED);
    assert_eq!(open_model_at(good, 1, 99).unwrap_err(), QUINTRIT_ERR_INVALID_ARGUMENT);
    assert_eq!(open_model_at(good, QUINTRIT_MAX_THREADS + 1, 0).unwrap_err(), QUINTRIT_ERR_INVALID_ARGUMENT);

    let p = cstr(good.to_str().unwrap());
    let mut m: *mut ModelHandle = null_mut();
    expect_status(unsafe { quintrit_model_open(null(), null(), &mut m) }, QUINTRIT_ERR_NULL_POINTER);
    expect_status(unsafe { quintrit_model_open(p.as_ptr(), null(), null_mut()) }, QUINTRIT_ERR_NULL_POINTER);
    let bad_opts = ModelOptions { struct_size: 4, num_threads: 1, kernel: 0, flags: 0 };
    expect_status(unsafe { quintrit_model_open(p.as_ptr(), &bad_opts, &mut m) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    // NULL options select the defaults.
    expect_status(unsafe { quintrit_model_open(p.as_ptr(), null(), &mut m) }, QUINTRIT_OK);
    unsafe { quintrit_model_free(m) };

    // Models that are well-formed GGUF but inconsistent are reported as model errors, not crashes.
    for (name, flaws) in [
        ("zero_heads", Flaws { zero_heads: true, ..Default::default() }),
        ("no_tokens", Flaws { drop_tokens: true, ..Default::default() }),
        ("short_norm", Flaws { short_norm: true, ..Default::default() }),
        ("vocab_mismatch", Flaws { vocab_mismatch: true, ..Default::default() }),
        ("odd_cols", Flaws { odd_i2s_cols: true, ..Default::default() }),
    ] {
        let path = dir.path(&format!("{name}.gguf"));
        write_model(&path, 5, &flaws);
        let st = open_model_at(&path, 1, 0).unwrap_err();
        assert_eq!(st, QUINTRIT_ERR_MODEL, "{name}: {:?}", last_error());
    }
}

/// Damaged model files must produce an error or a working model, never a panic.
#[test]
fn damaged_model_files_never_panic() {
    let dir = TempDir::new("runtime-fuzz");
    let src = dir.path("src.gguf");
    write_model(&src, 6, &Flaws::default());
    let bytes = std::fs::read(&src).unwrap();
    let h = {
        let p = cstr(src.to_str().unwrap());
        let mut h: *mut GgufHandle = null_mut();
        expect_status(unsafe { quintrit_gguf_open(p.as_ptr(), &mut h) }, QUINTRIT_OK);
        h
    };
    let mut gi = GgufInfo { struct_size: size_of::<GgufInfo>() as u32, ..Default::default() };
    expect_status(unsafe { quintrit_gguf_info(h, &mut gi) }, QUINTRIT_OK);
    unsafe { quintrit_gguf_close(h) };
    let header = gi.data_offset as usize;

    let mut rng = Rng::new(404);
    let victim = dir.path("victim.gguf");
    let mut loaded = 0;
    for round in 0..60 {
        let mut b = bytes.clone();
        for _ in 0..1 + rng.next() % 3 {
            // Bias towards the metadata values and tensor table, where the interesting fields are.
            let pos = rng.next() as usize % header;
            let end = (pos + 1 + rng.next() as usize % 8).min(b.len());
            let fill = [0u8, 0xFF, 0x01, 0x80][rng.next() as usize % 4];
            b[pos..end].fill(fill);
        }
        std::fs::write(&victim, &b).unwrap();
        match open_model_at(&victim, 1, QUINTRIT_KERNEL_SIMD32) {
            Ok(m) => {
                loaded += 1;
                let s = session(m, 16);
                let mut p = params();
                p.max_tokens = 3;
                let mut out = vec![0 as c_char; 256];
                let st = unsafe { quintrit_session_generate(s, c"hi".as_ptr(), 2, &p, None, null_mut(), out.as_mut_ptr(), out.len(), null_mut(), null_mut()) };
                assert_ne!(st, QUINTRIT_ERR_INTERNAL, "round {round}: {:?}", last_error());
                unsafe {
                    quintrit_session_free(s);
                    quintrit_model_free(m);
                }
            }
            Err(st) => assert_ne!(st, QUINTRIT_ERR_INTERNAL, "round {round}: {:?}", last_error()),
        }
    }
    assert!(loaded > 0, "some damaged files only change weights and should still load");
}

//! The GPU part of the C API on the tiny synthetic model: a build without GPU support answers every GPU function with `QUINTRIT_ERR_UNSUPPORTED`; a build
//! with it must produce the same text on the GPU as on the CPU. No model download is needed. Without a supported GPU the cuda tests skip themselves.

mod common;

use common::synth::*;
use common::*;
use quintrit_c::*;
#[cfg(feature = "cuda")]
use std::ffi::c_char;
use std::ptr::null_mut;

#[cfg(feature = "cuda")]
const PROMPT: &str = "<|user|>\nHello there\n<|assistant|>\n";

fn open_cpu(path: &std::path::Path) -> *mut ModelHandle {
    let p = cstr(path.to_str().unwrap());
    let mut m: *mut ModelHandle = null_mut();
    expect_status(unsafe { quintrit_model_open(p.as_ptr(), std::ptr::null(), &mut m) }, QUINTRIT_OK);
    m
}

#[cfg(feature = "cuda")]
fn generate(m: *const ModelHandle, ctx: u32, max_tokens: u32, temperature: f32, seed: u64) -> String {
    let so = SessionOptions { struct_size: size_of::<SessionOptions>() as u32, context_length: ctx };
    let mut s: *mut SessionHandle = null_mut();
    expect_status(unsafe { quintrit_session_create(m, &so, &mut s) }, QUINTRIT_OK);
    let mut p: GenerateParams = unsafe { std::mem::zeroed() };
    expect_status(unsafe { quintrit_generate_params_default(&mut p) }, QUINTRIT_OK);
    p.max_tokens = max_tokens;
    p.temperature = temperature;
    p.top_k = if temperature == 0.0 { 1 } else { 40 };
    p.seed = seed;
    let mut out = vec![0 as c_char; 4096];
    let mut len = 0usize;
    expect_status(
        unsafe { quintrit_session_generate(s, PROMPT.as_ptr() as *const c_char, PROMPT.len(), &p, None, null_mut(), out.as_mut_ptr(), out.len(), &mut len, null_mut()) },
        QUINTRIT_OK,
    );
    expect_status(unsafe { quintrit_session_free(s) }, QUINTRIT_OK);
    from_c(out.as_ptr())
}

#[cfg(not(feature = "cuda"))]
#[test]
fn a_build_without_gpu_support_says_so() {
    let mut n = 7u32;
    expect_status(unsafe { quintrit_cuda_device_count(&mut n) }, QUINTRIT_ERR_UNSUPPORTED);
    let mut info: CudaDeviceInfo = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<CudaDeviceInfo>() as u32;
    expect_status(unsafe { quintrit_cuda_device_info(0, &mut info) }, QUINTRIT_ERR_UNSUPPORTED);
    let dir = TempDir::new("ffi-cuda-none");
    let path = dir.path("model.gguf");
    write_model(&path, 1, &Flaws::default());
    let p = cstr(path.to_str().unwrap());
    let mut m: *mut ModelHandle = std::ptr::dangling_mut::<ModelHandle>();
    expect_status(unsafe { quintrit_model_open_gpu(p.as_ptr(), std::ptr::null(), &mut m) }, QUINTRIT_ERR_UNSUPPORTED);
    assert!(m.is_null());
    // A CPU model reports the CPU.
    let cpu = open_cpu(&path);
    let (mut device, mut index) = (-1i32, 99u32);
    expect_status(unsafe { quintrit_model_device(cpu, &mut device, &mut index) }, QUINTRIT_OK);
    assert_eq!((device, index), (QUINTRIT_DEVICE_CPU, 0));
    unsafe { quintrit_model_free(cpu) };
}

#[cfg(feature = "cuda")]
fn first_supported_gpu() -> Option<u32> {
    let mut n = 0u32;
    expect_status(unsafe { quintrit_cuda_device_count(&mut n) }, QUINTRIT_OK);
    for i in 0..n {
        let mut info: CudaDeviceInfo = unsafe { std::mem::zeroed() };
        info.struct_size = size_of::<CudaDeviceInfo>() as u32;
        expect_status(unsafe { quintrit_cuda_device_info(i, &mut info) }, QUINTRIT_OK);
        assert_eq!(info.index, i);
        assert!(info.sm_count > 0 && info.total_memory > 0 && from_c(info.name.as_ptr()).len() > 3);
        if info.supported == 1 && quintrit_cuda::kernels::available() {
            return Some(i);
        }
    }
    eprintln!("skipped: no supported GPU or no GPU kernels in this build");
    None
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_model_generates_the_same_text_as_the_cpu_model() {
    let Some(index) = first_supported_gpu() else { return };
    let dir = TempDir::new("ffi-cuda");
    let path = dir.path("model.gguf");
    write_model(&path, 11, &Flaws::default());
    let cpu = open_cpu(&path);
    let opts = GpuOptions { struct_size: size_of::<GpuOptions>() as u32, device_index: index, context_length: 0, flags: 0 };
    let p = cstr(path.to_str().unwrap());
    let mut gpu: *mut ModelHandle = null_mut();
    expect_status(unsafe { quintrit_model_open_gpu(p.as_ptr(), &opts, &mut gpu) }, QUINTRIT_OK);

    let (mut device, mut idx) = (-1i32, 99u32);
    expect_status(unsafe { quintrit_model_device(gpu, &mut device, &mut idx) }, QUINTRIT_OK);
    assert_eq!((device, idx), (QUINTRIT_DEVICE_CUDA, index));
    let mut info: ModelInfo = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<ModelInfo>() as u32;
    expect_status(unsafe { quintrit_model_info(gpu, &mut info) }, QUINTRIT_OK);
    assert_eq!((info.num_layers as usize, info.hidden_size as usize, info.vocab_size as usize), (LAYERS, HIDDEN, VOCAB));
    assert_eq!(from_c(info.architecture.as_ptr()), "llama");

    // Greedy and sampled text, and a second session of the same model next to the first.
    for (temperature, seed) in [(0.0f32, 0u64), (0.8, 5)] {
        let want = generate(cpu, 0, 24, temperature, seed);
        assert_eq!(generate(gpu, 0, 24, temperature, seed), want, "temperature {temperature}");
    }
    // Context lengths are rounded and checked: the synthetic model has a window of 48 tokens.
    let so = SessionOptions { struct_size: size_of::<SessionOptions>() as u32, context_length: 4096 };
    let mut s: *mut SessionHandle = null_mut();
    expect_status(unsafe { quintrit_session_create(gpu, &so, &mut s) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    assert!(s.is_null());
    let (mut s1, mut s2): (*mut SessionHandle, *mut SessionHandle) = (null_mut(), null_mut());
    expect_status(unsafe { quintrit_session_create(gpu, std::ptr::null(), &mut s1) }, QUINTRIT_OK);
    expect_status(unsafe { quintrit_session_create(gpu, std::ptr::null(), &mut s2) }, QUINTRIT_OK);
    let mut sinfo: SessionInfo = unsafe { std::mem::zeroed() };
    sinfo.struct_size = size_of::<SessionInfo>() as u32;
    expect_status(unsafe { quintrit_session_info(s1, &mut sinfo) }, QUINTRIT_OK);
    assert_eq!(sinfo.context_length as usize, CONTEXT);
    unsafe {
        quintrit_session_free(s1);
        quintrit_session_free(s2);
        quintrit_model_free(gpu);
        quintrit_model_free(cpu);
    }
}

#[cfg(feature = "cuda")]
#[test]
fn gpu_errors_are_reported() {
    let Some(index) = first_supported_gpu() else { return };
    let dir = TempDir::new("ffi-cuda-err");
    let path = dir.path("model.gguf");
    write_model(&path, 2, &Flaws::default());
    let p = cstr(path.to_str().unwrap());
    let mut m: *mut ModelHandle = std::ptr::dangling_mut::<ModelHandle>();
    // A GPU that does not exist.
    let opts = GpuOptions { struct_size: size_of::<GpuOptions>() as u32, device_index: 999, context_length: 0, flags: 0 };
    assert_ne!(unsafe { quintrit_model_open_gpu(p.as_ptr(), &opts, &mut m) }, QUINTRIT_OK);
    assert!(m.is_null());
    // Unknown flag bits.
    let opts = GpuOptions { struct_size: size_of::<GpuOptions>() as u32, device_index: index, context_length: 0, flags: 0x10 };
    expect_status(unsafe { quintrit_model_open_gpu(p.as_ptr(), &opts, &mut m) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    // A struct_size that is too small, a missing file.
    let small = GpuOptions { struct_size: 8, device_index: 0, context_length: 0, flags: 0 };
    expect_status(unsafe { quintrit_model_open_gpu(p.as_ptr(), &small, &mut m) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    let missing = cstr("does-not-exist.gguf");
    assert_ne!(unsafe { quintrit_model_open_gpu(missing.as_ptr(), std::ptr::null(), &mut m) }, QUINTRIT_OK);
    // An index out of range for the device query.
    let mut info: CudaDeviceInfo = unsafe { std::mem::zeroed() };
    info.struct_size = size_of::<CudaDeviceInfo>() as u32;
    expect_status(unsafe { quintrit_cuda_device_info(999, &mut info) }, QUINTRIT_ERR_INVALID_ARGUMENT);
}

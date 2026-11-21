//! Raw bindings of the CUDA driver API functions this crate uses, loaded from the driver library at run time.
//!
//! Only the driver API is used (never the runtime API), so the crate links no NVIDIA library at build time and the
//! workspace builds on machines without CUDA. Function names with a `_v2` suffix are the versioned symbols that the
//! macros of `cuda.h` select.

use crate::error::{Error, Result};
use libloading::Library;
use std::ffi::{c_char, c_int, c_uint, c_void};
use std::sync::OnceLock;

pub type CuResult = c_int;
pub type CuDevice = c_int;
pub type CuDevicePtr = u64;
pub type CuContext = *mut c_void;
pub type CuModule = *mut c_void;
pub type CuFunction = *mut c_void;
pub type CuStream = *mut c_void;
pub type CuEvent = *mut c_void;
pub type CuGraph = *mut c_void;
pub type CuGraphExec = *mut c_void;

pub const CUDA_SUCCESS: CuResult = 0;

// CUdevice_attribute
pub const ATTR_MAX_THREADS_PER_BLOCK: c_int = 1;
pub const ATTR_MAX_SHARED_MEMORY_PER_BLOCK: c_int = 8;
pub const ATTR_WARP_SIZE: c_int = 10;
pub const ATTR_CLOCK_RATE: c_int = 13;
pub const ATTR_MULTIPROCESSOR_COUNT: c_int = 16;
pub const ATTR_MEMORY_CLOCK_RATE: c_int = 36;
pub const ATTR_GLOBAL_MEMORY_BUS_WIDTH: c_int = 37;
pub const ATTR_L2_CACHE_SIZE: c_int = 38;
pub const ATTR_COMPUTE_CAPABILITY_MAJOR: c_int = 75;
pub const ATTR_COMPUTE_CAPABILITY_MINOR: c_int = 76;
pub const ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN: c_int = 97;

// CUfunction_attribute
pub const FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES: c_int = 8;
pub const FUNC_ATTR_PREFERRED_SHARED_MEMORY_CARVEOUT: c_int = 9;

// CUstreamCaptureMode
pub const CAPTURE_MODE_THREAD_LOCAL: c_int = 1;

// cuEventCreate flags
pub const EVENT_DEFAULT: c_uint = 0;

macro_rules! driver_functions {
    ($($field:ident = $sym:literal ( $($arg:ty),* );)*) => {
        /// The loaded driver library and its function table.
        pub struct Api {
            _library: Library,
            $(pub $field: unsafe extern "C" fn($($arg),*) -> CuResult,)*
        }

        impl Api {
            fn load() -> Result<Api> {
                let library = open_library()?;
                $(
                    let $field = unsafe {
                        *library
                            .get::<unsafe extern "C" fn($($arg),*) -> CuResult>($sym.as_bytes())
                            .map_err(|e| Error::NoDriver(format!("symbol {} not found: {e}", $sym)))?
                    };
                )*
                Ok(Api { _library: library, $($field),* })
            }
        }
    };
}

driver_functions! {
    init = "cuInit"(c_uint);
    driver_get_version = "cuDriverGetVersion"(*mut c_int);
    device_get_count = "cuDeviceGetCount"(*mut c_int);
    device_get = "cuDeviceGet"(*mut CuDevice, c_int);
    device_get_name = "cuDeviceGetName"(*mut c_char, c_int, CuDevice);
    device_get_attribute = "cuDeviceGetAttribute"(*mut c_int, c_int, CuDevice);
    device_total_mem = "cuDeviceTotalMem_v2"(*mut usize, CuDevice);
    primary_ctx_retain = "cuDevicePrimaryCtxRetain"(*mut CuContext, CuDevice);
    primary_ctx_release = "cuDevicePrimaryCtxRelease_v2"(CuDevice);
    ctx_set_current = "cuCtxSetCurrent"(CuContext);
    ctx_synchronize = "cuCtxSynchronize"();
    mem_get_info = "cuMemGetInfo_v2"(*mut usize, *mut usize);
    mem_alloc = "cuMemAlloc_v2"(*mut CuDevicePtr, usize);
    mem_free = "cuMemFree_v2"(CuDevicePtr);
    mem_alloc_host = "cuMemAllocHost_v2"(*mut *mut c_void, usize);
    mem_free_host = "cuMemFreeHost"(*mut c_void);
    memcpy_htod = "cuMemcpyHtoD_v2"(CuDevicePtr, *const c_void, usize);
    memcpy_dtoh = "cuMemcpyDtoH_v2"(*mut c_void, CuDevicePtr, usize);
    memcpy_htod_async = "cuMemcpyHtoDAsync_v2"(CuDevicePtr, *const c_void, usize, CuStream);
    memcpy_dtoh_async = "cuMemcpyDtoHAsync_v2"(*mut c_void, CuDevicePtr, usize, CuStream);
    memcpy_dtod_async = "cuMemcpyDtoDAsync_v2"(CuDevicePtr, CuDevicePtr, usize, CuStream);
    memset_d8 = "cuMemsetD8_v2"(CuDevicePtr, u8, usize);
    memset_d8_async = "cuMemsetD8Async"(CuDevicePtr, u8, usize, CuStream);
    module_load_data = "cuModuleLoadData"(*mut CuModule, *const c_void);
    module_unload = "cuModuleUnload"(CuModule);
    module_get_function = "cuModuleGetFunction"(*mut CuFunction, CuModule, *const c_char);
    func_set_attribute = "cuFuncSetAttribute"(CuFunction, c_int, c_int);
    launch_kernel = "cuLaunchKernel"(CuFunction, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, c_uint, CuStream, *mut *mut c_void, *mut *mut c_void);
    stream_create = "cuStreamCreate"(*mut CuStream, c_uint);
    stream_destroy = "cuStreamDestroy_v2"(CuStream);
    stream_synchronize = "cuStreamSynchronize"(CuStream);
    stream_begin_capture = "cuStreamBeginCapture_v2"(CuStream, c_int);
    stream_end_capture = "cuStreamEndCapture"(CuStream, *mut CuGraph);
    graph_instantiate = "cuGraphInstantiateWithFlags"(*mut CuGraphExec, CuGraph, u64);
    graph_launch = "cuGraphLaunch"(CuGraphExec, CuStream);
    graph_exec_destroy = "cuGraphExecDestroy"(CuGraphExec);
    graph_destroy = "cuGraphDestroy"(CuGraph);
    event_create = "cuEventCreate"(*mut CuEvent, c_uint);
    event_record = "cuEventRecord"(CuEvent, CuStream);
    event_synchronize = "cuEventSynchronize"(CuEvent);
    event_elapsed_time = "cuEventElapsedTime"(*mut f32, CuEvent, CuEvent);
    event_destroy = "cuEventDestroy_v2"(CuEvent);
    get_error_name = "cuGetErrorName"(CuResult, *mut *const c_char);
    get_error_string = "cuGetErrorString"(CuResult, *mut *const c_char);
}

#[cfg(windows)]
const LIBRARY_NAMES: &[&str] = &["nvcuda.dll"];
#[cfg(all(unix, not(target_os = "macos")))]
const LIBRARY_NAMES: &[&str] = &["libcuda.so.1", "libcuda.so"];
#[cfg(not(any(windows, all(unix, not(target_os = "macos")))))]
const LIBRARY_NAMES: &[&str] = &[];

fn open_library() -> Result<Library> {
    let mut last = String::from("this platform has no CUDA driver");
    for name in LIBRARY_NAMES {
        // SAFETY: loading the NVIDIA driver library runs its initializers, which is what the caller asked for.
        match unsafe { Library::new(name) } {
            Ok(lib) => return Ok(lib),
            Err(e) => last = format!("{name}: {e}"),
        }
    }
    Err(Error::NoDriver(last))
}

static API: OnceLock<std::result::Result<Api, Error>> = OnceLock::new();

/// The driver function table; loads the driver library and runs `cuInit` on first use.
pub fn api() -> Result<&'static Api> {
    let loaded = API.get_or_init(|| {
        let api = Api::load()?;
        // SAFETY: `cuInit` takes only the flags argument, which must be 0.
        let rc = unsafe { (api.init)(0) };
        if rc != CUDA_SUCCESS {
            return Err(Error::NoDriver(format!("cuInit failed with code {rc} (no usable NVIDIA GPU or driver)")));
        }
        Ok(api)
    });
    match loaded {
        Ok(api) => Ok(api),
        Err(e) => Err(e.clone()),
    }
}

fn error_text(api: &Api, rc: CuResult) -> String {
    let read = |f: unsafe extern "C" fn(CuResult, *mut *const c_char) -> CuResult| -> String {
        let mut p: *const c_char = std::ptr::null();
        // SAFETY: the driver stores a pointer to a static NUL-terminated string.
        unsafe {
            if f(rc, &mut p) == CUDA_SUCCESS && !p.is_null() {
                std::ffi::CStr::from_ptr(p).to_string_lossy().into_owned()
            } else {
                String::new()
            }
        }
    };
    let name = read(api.get_error_name);
    let text = read(api.get_error_string);
    match (name.is_empty(), text.is_empty()) {
        (true, true) => String::from("unknown error"),
        (false, true) => name,
        (true, false) => text,
        (false, false) => format!("{name}: {text}"),
    }
}

/// Turns a `CUresult` into an error naming the failed call.
pub fn check(call: &'static str, rc: CuResult) -> Result<()> {
    if rc == CUDA_SUCCESS {
        return Ok(());
    }
    let message = api().map(|a| error_text(a, rc)).unwrap_or_default();
    Err(Error::Driver { call, code: rc, message })
}

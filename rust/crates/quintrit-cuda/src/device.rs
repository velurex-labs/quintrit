//! Devices, device memory, streams, modules, kernel launches and CUDA graphs: a small safe layer over [`crate::driver`].

use crate::driver::{self as drv, Api, CuContext, CuDevice, CuDevicePtr, CuEvent, CuFunction, CuGraph, CuGraphExec, CuModule, CuStream};
use crate::error::{Error, Result};
use std::collections::HashMap;
use std::ffi::{c_char, c_void, CString};
use std::marker::PhantomData;
use std::sync::{Arc, Mutex};

/// Lowest compute capability the kernels support (Ampere, `sm_80`).
pub const MIN_COMPUTE_CAPABILITY: (u32, u32) = (8, 0);
/// Opt-in shared memory per block the kernels need (the attention and projection kernels run with up to 96 KiB).
pub const MIN_SHARED_MEMORY_OPTIN: u32 = 96 * 1024;

/// Properties of a CUDA device that the kernel configuration depends on.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct DeviceInfo {
    /// Ordinal of the device (as in `nvidia-smi`).
    pub ordinal: usize,
    /// Marketing name reported by the driver. Informational only: kernels are never selected by name.
    pub name: String,
    /// Compute capability `(major, minor)`.
    pub compute_capability: (u32, u32),
    /// Number of streaming multiprocessors.
    pub sm_count: u32,
    /// Size of the L2 cache in bytes.
    pub l2_cache_bytes: u64,
    /// Maximum opt-in shared memory per block in bytes.
    pub max_shared_memory_optin: u32,
    /// Maximum number of threads per block.
    pub max_threads_per_block: u32,
    /// Total device memory in bytes.
    pub total_memory: u64,
    /// Width of the memory bus in bits.
    pub memory_bus_width_bits: u32,
    /// Memory clock in kHz.
    pub memory_clock_khz: u32,
    /// Version of the installed driver, as reported by `cuDriverGetVersion` (for example 13040 for CUDA 13.4).
    pub driver_version: u32,
}

impl DeviceInfo {
    /// Reads the properties of device `ordinal` without creating a context.
    pub fn query(ordinal: usize) -> Result<DeviceInfo> {
        let api = drv::api()?;
        let dev = get_device(api, ordinal)?;
        let attr = |a: i32, call: &'static str| -> Result<i32> {
            let mut v = 0;
            // SAFETY: `v` is a valid out pointer for one `int`.
            drv::check(call, unsafe { (api.device_get_attribute)(&mut v, a, dev) })?;
            Ok(v)
        };
        let mut name = [0 as c_char; 256];
        // SAFETY: the buffer holds 256 bytes, which is the length passed.
        drv::check("cuDeviceGetName", unsafe { (api.device_get_name)(name.as_mut_ptr(), 256, dev) })?;
        let name = unsafe { std::ffi::CStr::from_ptr(name.as_ptr()) }.to_string_lossy().into_owned();
        let mut total = 0usize;
        // SAFETY: `total` is a valid out pointer.
        drv::check("cuDeviceTotalMem", unsafe { (api.device_total_mem)(&mut total, dev) })?;
        let mut version = 0;
        // SAFETY: `version` is a valid out pointer.
        drv::check("cuDriverGetVersion", unsafe { (api.driver_get_version)(&mut version) })?;
        let u = |v: i32| v.max(0) as u32;
        Ok(DeviceInfo {
            ordinal,
            name,
            compute_capability: (
                u(attr(drv::ATTR_COMPUTE_CAPABILITY_MAJOR, "cuDeviceGetAttribute")?),
                u(attr(drv::ATTR_COMPUTE_CAPABILITY_MINOR, "cuDeviceGetAttribute")?),
            ),
            sm_count: u(attr(drv::ATTR_MULTIPROCESSOR_COUNT, "cuDeviceGetAttribute")?),
            l2_cache_bytes: u(attr(drv::ATTR_L2_CACHE_SIZE, "cuDeviceGetAttribute")?) as u64,
            max_shared_memory_optin: u(attr(drv::ATTR_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, "cuDeviceGetAttribute")?),
            max_threads_per_block: u(attr(drv::ATTR_MAX_THREADS_PER_BLOCK, "cuDeviceGetAttribute")?),
            total_memory: total as u64,
            memory_bus_width_bits: u(attr(drv::ATTR_GLOBAL_MEMORY_BUS_WIDTH, "cuDeviceGetAttribute")?),
            memory_clock_khz: u(attr(drv::ATTR_MEMORY_CLOCK_RATE, "cuDeviceGetAttribute")?),
            driver_version: u(version),
        })
    }

    /// Checks that the device can run the kernels: compute capability 8.0 or newer and enough opt-in shared memory.
    pub fn check_supported(&self) -> Result<()> {
        if self.compute_capability < MIN_COMPUTE_CAPABILITY {
            return Err(Error::Unsupported(format!(
                "{} has compute capability {}.{}; the kernels need {}.{} (Ampere) or newer",
                self.name, self.compute_capability.0, self.compute_capability.1, MIN_COMPUTE_CAPABILITY.0, MIN_COMPUTE_CAPABILITY.1
            )));
        }
        if self.max_shared_memory_optin < MIN_SHARED_MEMORY_OPTIN {
            return Err(Error::Unsupported(format!(
                "{} offers {} bytes of opt-in shared memory per block; the kernels need {}",
                self.name, self.max_shared_memory_optin, MIN_SHARED_MEMORY_OPTIN
            )));
        }
        Ok(())
    }

    /// Theoretical peak memory bandwidth in GB/s (double data rate); a measured read-only ceiling is lower.
    pub fn peak_bandwidth_gbs(&self) -> f64 {
        2.0 * self.memory_clock_khz as f64 * 1e3 * (self.memory_bus_width_bits as f64 / 8.0) / 1e9
    }
}

fn get_device(api: &Api, ordinal: usize) -> Result<CuDevice> {
    let mut n = 0;
    // SAFETY: `n` is a valid out pointer.
    drv::check("cuDeviceGetCount", unsafe { (api.device_get_count)(&mut n) })?;
    if ordinal >= n.max(0) as usize {
        return Err(Error::InvalidArgument(format!("device {ordinal} does not exist ({n} CUDA devices found)")));
    }
    let mut dev = 0;
    // SAFETY: `dev` is a valid out pointer and the ordinal was range-checked.
    drv::check("cuDeviceGet", unsafe { (api.device_get)(&mut dev, ordinal as i32) })?;
    Ok(dev)
}

/// Number of CUDA devices, or an error when no driver is installed.
pub fn device_count() -> Result<usize> {
    let api = drv::api()?;
    let mut n = 0;
    // SAFETY: `n` is a valid out pointer.
    drv::check("cuDeviceGetCount", unsafe { (api.device_get_count)(&mut n) })?;
    Ok(n.max(0) as usize)
}

struct ModuleInner {
    raw: CuModule,
    ctx: CuContext,
}

// SAFETY: a CUmodule handle may be used from any thread while its context is current.
unsafe impl Send for ModuleInner {}
unsafe impl Sync for ModuleInner {}

impl Drop for ModuleInner {
    fn drop(&mut self) {
        if let Ok(api) = drv::api() {
            // SAFETY: the context outlives every module because functions keep the device alive and the cache is cleared first.
            unsafe {
                (api.ctx_set_current)(self.ctx);
                (api.module_unload)(self.raw);
            }
        }
    }
}

struct DeviceInner {
    dev: CuDevice,
    ctx: CuContext,
    info: DeviceInfo,
    modules: Mutex<HashMap<&'static str, Arc<ModuleInner>>>,
}

// SAFETY: the primary context is shared by all threads; every operation binds it to the calling thread first.
unsafe impl Send for DeviceInner {}
unsafe impl Sync for DeviceInner {}

impl DeviceInner {
    fn bind(&self) -> Result<()> {
        let api = drv::api()?;
        // SAFETY: `ctx` is the retained primary context of this device.
        drv::check("cuCtxSetCurrent", unsafe { (api.ctx_set_current)(self.ctx) })
    }
    /// Waits for the synchronous copies and memsets of this thread, which run on the legacy default stream. A copy from
    /// pageable host memory may return before its DMA has finished, and streams of this crate are non-blocking, so
    /// nothing else would order a later kernel after it.
    fn sync_default_stream(&self) -> Result<()> {
        let api = drv::api()?;
        // SAFETY: a null stream is the legacy default stream.
        drv::check("cuStreamSynchronize", unsafe { (api.stream_synchronize)(std::ptr::null_mut()) })
    }
}

impl Drop for DeviceInner {
    fn drop(&mut self) {
        self.modules.lock().map(|mut m| m.clear()).ok();
        if let Ok(api) = drv::api() {
            // SAFETY: balances the `cuDevicePrimaryCtxRetain` of `Device::open`.
            unsafe { (api.primary_ctx_release)(self.dev) };
        }
    }
}

/// A CUDA device with its primary context. Cloning is cheap and shares the context.
#[derive(Clone)]
pub struct Device {
    inner: Arc<DeviceInner>,
}

impl Device {
    /// Opens device `ordinal` and checks that it is supported (compute capability 8.0 or newer).
    pub fn open(ordinal: usize) -> Result<Device> {
        let device = Device::open_unchecked(ordinal)?;
        device.inner.info.check_supported()?;
        Ok(device)
    }

    /// Opens device `ordinal` without checking the compute capability (for diagnostics only; kernels may fail to load).
    pub fn open_unchecked(ordinal: usize) -> Result<Device> {
        let info = DeviceInfo::query(ordinal)?;
        let api = drv::api()?;
        let dev = get_device(api, ordinal)?;
        let mut ctx: CuContext = std::ptr::null_mut();
        // SAFETY: `ctx` is a valid out pointer; the retain is released in `DeviceInner::drop`.
        drv::check("cuDevicePrimaryCtxRetain", unsafe { (api.primary_ctx_retain)(&mut ctx, dev) })?;
        let inner = Arc::new(DeviceInner { dev, ctx, info, modules: Mutex::new(HashMap::new()) });
        inner.bind()?;
        Ok(Device { inner })
    }

    /// Properties of the device.
    pub fn info(&self) -> &DeviceInfo {
        &self.inner.info
    }

    /// Makes the context of this device current on the calling thread. Every operation of this crate does it itself;
    /// it is only needed before calling the CUDA driver directly.
    pub fn bind(&self) -> Result<()> {
        self.inner.bind()
    }

    /// Free and total device memory in bytes.
    pub fn memory_info(&self) -> Result<(u64, u64)> {
        self.inner.bind()?;
        let api = drv::api()?;
        let (mut free, mut total) = (0usize, 0usize);
        // SAFETY: both are valid out pointers.
        drv::check("cuMemGetInfo", unsafe { (api.mem_get_info)(&mut free, &mut total) })?;
        Ok((free as u64, total as u64))
    }

    /// Waits for all work on the device.
    pub fn synchronize(&self) -> Result<()> {
        self.inner.bind()?;
        let api = drv::api()?;
        // SAFETY: plain driver call on the current context.
        drv::check("cuCtxSynchronize", unsafe { (api.ctx_synchronize)() })
    }

    /// Creates a stream.
    pub fn create_stream(&self) -> Result<Stream> {
        self.inner.bind()?;
        let api = drv::api()?;
        let mut raw: CuStream = std::ptr::null_mut();
        // A non-blocking stream does not synchronize implicitly with the legacy default stream (the one synchronous copies and
        // allocations of other threads use), which a capture into a graph must not do.
        // SAFETY: `raw` is a valid out pointer; flag 1 is CU_STREAM_NON_BLOCKING.
        drv::check("cuStreamCreate", unsafe { (api.stream_create)(&mut raw, 1) })?;
        Ok(Stream { raw, dev: self.inner.clone() })
    }

    /// Creates an event for timing.
    pub fn create_event(&self) -> Result<Event> {
        self.inner.bind()?;
        let api = drv::api()?;
        let mut raw: CuEvent = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer.
        drv::check("cuEventCreate", unsafe { (api.event_create)(&mut raw, drv::EVENT_DEFAULT) })?;
        Ok(Event { raw, dev: self.inner.clone() })
    }

    /// Allocates `len` uninitialized elements of device memory.
    pub fn alloc<T: Copy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        self.inner.bind()?;
        let api = drv::api()?;
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::InvalidArgument("allocation size overflows".into()))?;
        let mut ptr: CuDevicePtr = 0;
        // SAFETY: `ptr` is a valid out pointer; a zero-sized request is rounded up to one byte because the driver rejects it.
        drv::check("cuMemAlloc", unsafe { (api.mem_alloc)(&mut ptr, bytes.max(1)) })?;
        Ok(DeviceBuffer { ptr, len, dev: self.inner.clone(), _elem: PhantomData })
    }

    /// Allocates `len` elements and sets every byte to zero.
    pub fn alloc_zeroed<T: Copy>(&self, len: usize) -> Result<DeviceBuffer<T>> {
        let buf = self.alloc::<T>(len)?;
        let api = drv::api()?;
        // SAFETY: the allocation holds at least `byte_len` bytes.
        drv::check("cuMemsetD8", unsafe { (api.memset_d8)(buf.ptr, 0, buf.byte_len().max(1)) })?;
        self.inner.sync_default_stream()?;
        Ok(buf)
    }

    /// Allocates a buffer and copies `data` into it.
    pub fn upload<T: Copy>(&self, data: &[T]) -> Result<DeviceBuffer<T>> {
        let buf = self.alloc::<T>(data.len())?;
        buf.copy_from_host(data)?;
        Ok(buf)
    }

    /// Allocates page-locked host memory (needed for asynchronous copies that overlap with kernels).
    pub fn alloc_pinned<T: Copy + Default>(&self, len: usize) -> Result<PinnedBuffer<T>> {
        self.inner.bind()?;
        let api = drv::api()?;
        let bytes = len
            .checked_mul(std::mem::size_of::<T>())
            .ok_or_else(|| Error::InvalidArgument("allocation size overflows".into()))?;
        let mut ptr: *mut c_void = std::ptr::null_mut();
        // SAFETY: `ptr` is a valid out pointer.
        drv::check("cuMemAllocHost", unsafe { (api.mem_alloc_host)(&mut ptr, bytes.max(1)) })?;
        let p = ptr as *mut T;
        for i in 0..len {
            // SAFETY: `i < len` and the allocation holds `len` elements.
            unsafe { p.add(i).write(T::default()) };
        }
        Ok(PinnedBuffer { ptr: p, len, dev: self.inner.clone() })
    }

    /// Loads (once) and returns the module with the given name from the kernel images embedded at build time.
    pub fn module(&self, name: &'static str) -> Result<Module> {
        let image = crate::kernels::image(name)?;
        let mut cache = self.inner.modules.lock().map_err(|_| Error::InvalidArgument("module cache poisoned".into()))?;
        if let Some(m) = cache.get(name) {
            return Ok(Module { raw: m.clone(), dev: self.inner.clone() });
        }
        self.inner.bind()?;
        let api = drv::api()?;
        let mut raw: CuModule = std::ptr::null_mut();
        // SAFETY: `image` is a fat binary produced by nvcc; the driver copies it.
        drv::check("cuModuleLoadData", unsafe { (api.module_load_data)(&mut raw, image.as_ptr() as *const c_void) })?;
        let inner = Arc::new(ModuleInner { raw, ctx: self.inner.ctx });
        cache.insert(name, inner.clone());
        Ok(Module { raw: inner, dev: self.inner.clone() })
    }
}

/// A pointer into device memory, as passed to kernels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct DevicePtr(pub u64);

impl DevicePtr {
    /// The pointer `bytes` further on.
    pub fn offset_bytes(self, bytes: usize) -> DevicePtr {
        DevicePtr(self.0 + bytes as u64)
    }
}

/// An allocation of device memory holding `len` elements of `T`.
pub struct DeviceBuffer<T: Copy> {
    ptr: CuDevicePtr,
    len: usize,
    dev: Arc<DeviceInner>,
    _elem: PhantomData<T>,
}

// SAFETY: the buffer owns device memory; access is through driver calls that bind the context.
unsafe impl<T: Copy> Send for DeviceBuffer<T> {}
unsafe impl<T: Copy> Sync for DeviceBuffer<T> {}

impl<T: Copy> DeviceBuffer<T> {
    /// Number of elements.
    pub fn len(&self) -> usize {
        self.len
    }
    /// True when the buffer holds no elements.
    pub fn is_empty(&self) -> bool {
        self.len == 0
    }
    /// Size in bytes.
    pub fn byte_len(&self) -> usize {
        self.len * std::mem::size_of::<T>()
    }
    /// The device pointer, for kernel arguments.
    pub fn ptr(&self) -> DevicePtr {
        DevicePtr(self.ptr)
    }

    /// Copies `data` (exactly `len` elements) from the host and waits until the data has arrived on the device. Streams of
    /// this crate are non-blocking: a kernel launched afterwards on any of them sees the data, but a synchronous copy in the
    /// other direction does not wait for kernels that are still running; synchronize the stream first.
    pub fn copy_from_host(&self, data: &[T]) -> Result<()> {
        if data.len() != self.len {
            return Err(Error::InvalidArgument(format!("copy of {} elements into a buffer of {}", data.len(), self.len)));
        }
        self.dev.bind()?;
        let api = drv::api()?;
        // SAFETY: both ranges hold `byte_len` bytes.
        drv::check("cuMemcpyHtoD", unsafe { (api.memcpy_htod)(self.ptr, data.as_ptr() as *const c_void, self.byte_len()) })?;
        self.dev.sync_default_stream()
    }

    /// Copies the buffer into `out` (exactly `len` elements) and waits for the copy.
    pub fn copy_to_host(&self, out: &mut [T]) -> Result<()> {
        if out.len() != self.len {
            return Err(Error::InvalidArgument(format!("copy of a buffer of {} elements into {}", self.len, out.len())));
        }
        self.dev.bind()?;
        let api = drv::api()?;
        // SAFETY: both ranges hold `byte_len` bytes.
        drv::check("cuMemcpyDtoH", unsafe { (api.memcpy_dtoh)(out.as_mut_ptr() as *mut c_void, self.ptr, self.byte_len()) })
    }

    /// The buffer as a vector (synchronous).
    pub fn to_vec(&self) -> Result<Vec<T>>
    where
        T: Default,
    {
        let mut v = vec![T::default(); self.len];
        self.copy_to_host(&mut v)?;
        Ok(v)
    }

    /// Sets every byte of the buffer to `value`, asynchronously on `stream`.
    pub fn fill_bytes_async(&self, value: u8, stream: &Stream) -> Result<()> {
        self.dev.bind()?;
        let api = drv::api()?;
        // SAFETY: the range lies inside the allocation.
        drv::check("cuMemsetD8Async", unsafe { (api.memset_d8_async)(self.ptr, value, self.byte_len().max(1), stream.raw) })
    }
}

impl<T: Copy> Drop for DeviceBuffer<T> {
    fn drop(&mut self) {
        if self.dev.bind().is_ok() {
            if let Ok(api) = drv::api() {
                // SAFETY: the pointer came from `cuMemAlloc` and is freed once.
                unsafe { (api.mem_free)(self.ptr) };
            }
        }
    }
}

/// Page-locked host memory, usable for asynchronous copies.
pub struct PinnedBuffer<T: Copy> {
    ptr: *mut T,
    len: usize,
    dev: Arc<DeviceInner>,
}

// SAFETY: the buffer owns its allocation; the caller synchronizes access with the streams that use it.
unsafe impl<T: Copy> Send for PinnedBuffer<T> {}

impl<T: Copy> PinnedBuffer<T> {
    /// The memory as a slice.
    pub fn as_slice(&self) -> &[T] {
        // SAFETY: `ptr` points to `len` initialized elements.
        unsafe { std::slice::from_raw_parts(self.ptr, self.len) }
    }
    /// The memory as a mutable slice. Do not modify it while a copy from it is in flight.
    pub fn as_mut_slice(&mut self) -> &mut [T] {
        // SAFETY: `ptr` points to `len` initialized elements and `&mut self` is exclusive.
        unsafe { std::slice::from_raw_parts_mut(self.ptr, self.len) }
    }

    /// Starts an asynchronous copy of the first `n` elements into `dst` on `stream`. The host memory must stay
    /// untouched until the stream has passed the copy.
    pub fn copy_to_device_async(&self, dst: &DeviceBuffer<T>, n: usize, stream: &Stream) -> Result<()> {
        if n > self.len || n > dst.len {
            return Err(Error::InvalidArgument("asynchronous copy exceeds a buffer".into()));
        }
        self.dev.bind()?;
        let api = drv::api()?;
        // SAFETY: both ranges hold at least `n` elements.
        drv::check("cuMemcpyHtoDAsync", unsafe {
            (api.memcpy_htod_async)(dst.ptr, self.ptr as *const c_void, n * std::mem::size_of::<T>(), stream.raw)
        })
    }

    /// Starts an asynchronous copy of the first `n` elements of `src` into this buffer on `stream`.
    pub fn copy_from_device_async(&mut self, src: &DeviceBuffer<T>, n: usize, stream: &Stream) -> Result<()> {
        if n > self.len || n > src.len {
            return Err(Error::InvalidArgument("asynchronous copy exceeds a buffer".into()));
        }
        self.dev.bind()?;
        let api = drv::api()?;
        // SAFETY: both ranges hold at least `n` elements.
        drv::check("cuMemcpyDtoHAsync", unsafe {
            (api.memcpy_dtoh_async)(self.ptr as *mut c_void, src.ptr, n * std::mem::size_of::<T>(), stream.raw)
        })
    }
}

impl<T: Copy> Drop for PinnedBuffer<T> {
    fn drop(&mut self) {
        if self.dev.bind().is_ok() {
            if let Ok(api) = drv::api() {
                // SAFETY: the pointer came from `cuMemAllocHost` and is freed once.
                unsafe { (api.mem_free_host)(self.ptr as *mut c_void) };
            }
        }
    }
}

/// A CUDA stream.
pub struct Stream {
    raw: CuStream,
    dev: Arc<DeviceInner>,
}

// SAFETY: a stream handle may be used from any thread while the context is bound.
unsafe impl Send for Stream {}
unsafe impl Sync for Stream {}

impl Stream {
    /// Waits for all work queued on the stream.
    pub fn synchronize(&self) -> Result<()> {
        self.dev.bind()?;
        let api = drv::api()?;
        // SAFETY: `raw` is a live stream.
        drv::check("cuStreamSynchronize", unsafe { (api.stream_synchronize)(self.raw) })
    }

    /// Records `event` on the stream.
    pub fn record(&self, event: &Event) -> Result<()> {
        self.dev.bind()?;
        let api = drv::api()?;
        // SAFETY: both handles are live.
        drv::check("cuEventRecord", unsafe { (api.event_record)(event.raw, self.raw) })
    }

    /// Starts recording the launches and copies queued on this stream into a graph instead of running them.
    pub fn begin_capture(&self) -> Result<()> {
        self.dev.bind()?;
        let api = drv::api()?;
        // SAFETY: `raw` is a live stream that is not capturing.
        drv::check("cuStreamBeginCapture", unsafe { (api.stream_begin_capture)(self.raw, drv::CAPTURE_MODE_THREAD_LOCAL) })
    }

    /// Ends the capture and returns the recorded graph.
    pub fn end_capture(&self) -> Result<Graph> {
        self.dev.bind()?;
        let api = drv::api()?;
        let mut raw: CuGraph = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer; the stream is capturing.
        drv::check("cuStreamEndCapture", unsafe { (api.stream_end_capture)(self.raw, &mut raw) })?;
        Ok(Graph { raw, dev: self.dev.clone() })
    }

    /// Raw handle, for passing to the CUDA driver directly.
    pub fn raw(&self) -> *mut c_void {
        self.raw
    }
}

impl Drop for Stream {
    fn drop(&mut self) {
        if self.dev.bind().is_ok() {
            if let Ok(api) = drv::api() {
                // SAFETY: the stream was created by `cuStreamCreate` and is destroyed once.
                unsafe { (api.stream_destroy)(self.raw) };
            }
        }
    }
}

/// A CUDA event for timing.
pub struct Event {
    raw: CuEvent,
    dev: Arc<DeviceInner>,
}

// SAFETY: an event handle may be used from any thread while the context is bound.
unsafe impl Send for Event {}
unsafe impl Sync for Event {}

impl Event {
    /// Waits until the event has been reached.
    pub fn synchronize(&self) -> Result<()> {
        self.dev.bind()?;
        let api = drv::api()?;
        // SAFETY: `raw` is a live event.
        drv::check("cuEventSynchronize", unsafe { (api.event_synchronize)(self.raw) })
    }

    /// Milliseconds between `start` and `self` (both must have been recorded and reached).
    pub fn elapsed_ms_since(&self, start: &Event) -> Result<f32> {
        let api = drv::api()?;
        let mut ms = 0f32;
        // SAFETY: both handles are live; `ms` is a valid out pointer.
        drv::check("cuEventElapsedTime", unsafe { (api.event_elapsed_time)(&mut ms, start.raw, self.raw) })?;
        Ok(ms)
    }
}

impl Drop for Event {
    fn drop(&mut self) {
        if self.dev.bind().is_ok() {
            if let Ok(api) = drv::api() {
                // SAFETY: the event was created by `cuEventCreate` and is destroyed once.
                unsafe { (api.event_destroy)(self.raw) };
            }
        }
    }
}

/// A graph recorded by [`Stream::begin_capture`] / [`Stream::end_capture`].
pub struct Graph {
    raw: CuGraph,
    dev: Arc<DeviceInner>,
}

// SAFETY: see `Stream`.
unsafe impl Send for Graph {}

impl Graph {
    /// Turns the graph into an executable that can be launched repeatedly.
    pub fn instantiate(&self) -> Result<GraphExec> {
        self.dev.bind()?;
        let api = drv::api()?;
        let mut raw: CuGraphExec = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer; flags 0 are the default.
        drv::check("cuGraphInstantiateWithFlags", unsafe { (api.graph_instantiate)(&mut raw, self.raw, 0) })?;
        Ok(GraphExec { raw, dev: self.dev.clone() })
    }
}

impl Drop for Graph {
    fn drop(&mut self) {
        if self.dev.bind().is_ok() {
            if let Ok(api) = drv::api() {
                // SAFETY: the graph came from `cuStreamEndCapture` and is destroyed once.
                unsafe { (api.graph_destroy)(self.raw) };
            }
        }
    }
}

/// An instantiated graph.
pub struct GraphExec {
    raw: CuGraphExec,
    dev: Arc<DeviceInner>,
}

// SAFETY: see `Stream`.
unsafe impl Send for GraphExec {}

impl GraphExec {
    /// Launches the whole graph on `stream`.
    pub fn launch(&self, stream: &Stream) -> Result<()> {
        self.dev.bind()?;
        let api = drv::api()?;
        // SAFETY: both handles are live.
        drv::check("cuGraphLaunch", unsafe { (api.graph_launch)(self.raw, stream.raw) })
    }
}

impl Drop for GraphExec {
    fn drop(&mut self) {
        if self.dev.bind().is_ok() {
            if let Ok(api) = drv::api() {
                // SAFETY: the executable came from `cuGraphInstantiateWithFlags` and is destroyed once.
                unsafe { (api.graph_exec_destroy)(self.raw) };
            }
        }
    }
}

/// A loaded module (the kernels of one `.cu` file).
pub struct Module {
    raw: Arc<ModuleInner>,
    dev: Arc<DeviceInner>,
}

impl Module {
    /// Looks up a kernel by its `extern "C"` name.
    pub fn function(&self, name: &str) -> Result<Function> {
        self.dev.bind()?;
        let api = drv::api()?;
        let cname = CString::new(name).map_err(|_| Error::InvalidArgument("kernel name contains a NUL byte".into()))?;
        let mut raw: CuFunction = std::ptr::null_mut();
        // SAFETY: `raw` is a valid out pointer; the name is NUL terminated.
        drv::check("cuModuleGetFunction", unsafe { (api.module_get_function)(&mut raw, self.raw.raw, cname.as_ptr()) })?;
        Ok(Function { raw, _module: self.raw.clone(), dev: self.dev.clone() })
    }
}

/// A kernel.
pub struct Function {
    raw: CuFunction,
    // Declared before `dev`: the module is unloaded while the device (context) is still alive.
    _module: Arc<ModuleInner>,
    dev: Arc<DeviceInner>,
}

// SAFETY: a CUfunction handle may be used from any thread while the context is bound.
unsafe impl Send for Function {}
unsafe impl Sync for Function {}

impl Function {
    /// Allows `bytes` of dynamic shared memory per block (the opt-in needed above 48 KiB) and asks for the largest
    /// shared-memory carve-out of the L1 cache.
    pub fn set_max_dynamic_shared(&self, bytes: u32) -> Result<()> {
        self.dev.bind()?;
        let api = drv::api()?;
        // SAFETY: `raw` is a live function.
        drv::check("cuFuncSetAttribute", unsafe {
            (api.func_set_attribute)(self.raw, drv::FUNC_ATTR_MAX_DYNAMIC_SHARED_SIZE_BYTES, bytes as i32)
        })?;
        // The carve-out is a hint; a failure to set it does not change results.
        unsafe { (api.func_set_attribute)(self.raw, drv::FUNC_ATTR_PREFERRED_SHARED_MEMORY_CARVEOUT, 100) };
        Ok(())
    }

    /// Launches the kernel.
    ///
    /// # Safety
    /// `args` must match the parameter list of the kernel exactly (types, order and sizes), pointers must refer to live
    /// device memory of the right size, and the grid must not read or write outside it.
    pub unsafe fn launch(&self, stream: &Stream, grid: (u32, u32, u32), block: (u32, u32, u32), shared_bytes: u32, args: &Args) -> Result<()> {
        self.dev.bind()?;
        let api = drv::api()?;
        let mut ptrs = args.pointers();
        // SAFETY: guaranteed by the caller; the pointer array and the values it points to live until the call returns.
        drv::check("cuLaunchKernel", unsafe {
            (api.launch_kernel)(
                self.raw,
                grid.0,
                grid.1,
                grid.2,
                block.0,
                block.1,
                block.2,
                shared_bytes,
                stream.raw,
                ptrs.as_mut_ptr(),
                std::ptr::null_mut(),
            )
        })
    }
}

const MAX_ARGS: usize = 40;
const ARG_WORDS: usize = 96;

/// The argument list of a kernel launch. Values are stored by value in 8-byte aligned slots, so no allocation happens
/// per launch. Every value must have the size and layout of the corresponding kernel parameter.
#[derive(Clone)]
pub struct Args {
    words: [u64; ARG_WORDS],
    offsets: [u8; MAX_ARGS],
    count: usize,
    used: usize,
}

impl Default for Args {
    fn default() -> Self {
        Args::new()
    }
}

impl Args {
    /// An empty list.
    pub fn new() -> Args {
        Args { words: [0; ARG_WORDS], offsets: [0; MAX_ARGS], count: 0, used: 0 }
    }

    /// Appends a plain value (`i32`, `u32`, `f32`, `u64`, a `#[repr(C)]` struct of such values, ...).
    pub fn push<T: Copy>(&mut self, value: T) -> &mut Self {
        let size = std::mem::size_of::<T>();
        assert!(std::mem::align_of::<T>() <= 8, "kernel argument alignment above 8 bytes is not supported");
        let n = size.div_ceil(8).max(1);
        assert!(self.count < MAX_ARGS && self.used + n <= ARG_WORDS, "too many kernel arguments");
        // SAFETY: the slot has room for `size` bytes; `T: Copy` values are copied bytewise.
        unsafe { std::ptr::copy_nonoverlapping(&value as *const T as *const u8, self.words.as_mut_ptr().add(self.used) as *mut u8, size) };
        self.offsets[self.count] = self.used as u8;
        self.count += 1;
        self.used += n;
        self
    }

    /// Appends a device pointer.
    pub fn ptr(&mut self, p: DevicePtr) -> &mut Self {
        self.push(p.0)
    }

    /// Appends the pointer of a buffer.
    pub fn buf<T: Copy>(&mut self, b: &DeviceBuffer<T>) -> &mut Self {
        self.push(b.ptr().0)
    }

    /// A null pointer argument (an optional buffer that is not used).
    pub fn null(&mut self) -> &mut Self {
        self.push(0u64)
    }

    fn pointers(&self) -> [*mut c_void; MAX_ARGS] {
        let mut out = [std::ptr::null_mut(); MAX_ARGS];
        for (i, slot) in out.iter_mut().enumerate().take(self.count) {
            *slot = &self.words[self.offsets[i] as usize] as *const u64 as *mut c_void;
        }
        out
    }
}

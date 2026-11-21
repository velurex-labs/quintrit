//! Device and module tests. They skip themselves (and print why) when the machine has no usable NVIDIA GPU or the
//! crate was built without CUDA kernels.

use quintrit_cuda::{kernels, Args, Device, Error};

/// The first supported device, or `None` (with a message) when the test cannot run here.
fn device() -> Option<Device> {
    match Device::open(0) {
        Ok(d) => Some(d),
        Err(e @ (Error::NoDriver(_) | Error::Unsupported(_))) => {
            eprintln!("skipped: {e}");
            None
        }
        Err(e) => panic!("unexpected error opening device 0: {e}"),
    }
}

#[test]
fn device_properties_are_plausible() {
    let Some(device) = device() else { return };
    let info = device.info();
    assert!(info.compute_capability >= (8, 0));
    assert!(info.sm_count > 0);
    assert!(info.l2_cache_bytes > 0);
    assert!(info.max_shared_memory_optin >= quintrit_cuda::MIN_SHARED_MEMORY_OPTIN);
    assert!(info.total_memory > 1 << 30);
    assert!(info.peak_bandwidth_gbs() > 50.0);
    let (free, total) = device.memory_info().unwrap();
    assert!(free <= total);
}

#[test]
fn buffers_round_trip() {
    let Some(device) = device() else { return };
    let data: Vec<i32> = (0..1000).map(|i| i * 3 - 7).collect();
    let buf = device.upload(&data).unwrap();
    assert_eq!(buf.to_vec().unwrap(), data);
    let zeros = device.alloc_zeroed::<u8>(4096).unwrap();
    assert!(zeros.to_vec().unwrap().iter().all(|&b| b == 0));
    // Wrong sizes are rejected, not silently truncated.
    assert!(buf.copy_from_host(&data[..10]).is_err());
}

#[test]
fn probe_kernel_reports_a_supported_architecture() {
    let Some(device) = device() else { return };
    if !kernels::available() {
        eprintln!("skipped: built without CUDA kernels");
        assert_eq!(device.module("probe").err(), Some(Error::NoKernels));
        return;
    }
    let module = device.module("probe").unwrap();
    let probe = module.function("quintrit_probe").unwrap();
    let out = device.alloc_zeroed::<i32>(2).unwrap();
    let stream = device.create_stream().unwrap();
    let mut args = Args::new();
    args.buf(&out);
    // SAFETY: the kernel takes one pointer to two ints.
    unsafe { probe.launch(&stream, (3, 1, 1), (32, 1, 1), 0, &args).unwrap() };
    stream.synchronize().unwrap();
    let result = out.to_vec().unwrap();
    assert!(matches!(result[0], 800 | 860 | 890), "unexpected architecture {}", result[0]);
    assert_eq!(result[1], 96);
    eprintln!("{} runs the code compiled for sm_{}", device.info().name, result[0] / 10);
}

#[test]
fn graph_capture_replays_a_launch() {
    let Some(device) = device() else { return };
    if !kernels::available() {
        return;
    }
    let probe = device.module("probe").unwrap().function("quintrit_probe").unwrap();
    let out = device.alloc_zeroed::<i32>(2).unwrap();
    let stream = device.create_stream().unwrap();
    let mut args = Args::new();
    args.buf(&out);
    stream.begin_capture().unwrap();
    // SAFETY: as above.
    unsafe { probe.launch(&stream, (2, 1, 1), (64, 1, 1), 0, &args).unwrap() };
    let graph = stream.end_capture().unwrap();
    let exec = graph.instantiate().unwrap();
    // Nothing ran during the capture.
    assert_eq!(out.to_vec().unwrap(), vec![0, 0]);
    exec.launch(&stream).unwrap();
    stream.synchronize().unwrap();
    assert_eq!(out.to_vec().unwrap()[1], 128);
}

#[test]
fn unknown_device_is_an_error() {
    match Device::open(10_000) {
        Err(Error::NoDriver(_)) | Err(Error::InvalidArgument(_)) => {}
        other => panic!("expected an error, got {:?}", other.map(|_| ())),
    }
}

fn synthetic(cc: (u32, u32), shared: u32) -> quintrit_cuda::DeviceInfo {
    quintrit_cuda::DeviceInfo {
        ordinal: 0,
        name: "synthetic".into(),
        compute_capability: cc,
        sm_count: 40,
        l2_cache_bytes: 8 << 20,
        max_shared_memory_optin: shared,
        max_threads_per_block: 1024,
        total_memory: 8 << 30,
        memory_bus_width_bits: 256,
        memory_clock_khz: 7_000_000,
        driver_version: 13_040,
    }
}

#[test]
fn support_check_needs_ampere_and_shared_memory() {
    // Runs without a GPU: the check only looks at the properties.
    assert!(synthetic((8, 0), 163 * 1024).check_supported().is_ok());
    assert!(synthetic((8, 6), 99 * 1024).check_supported().is_ok());
    assert!(synthetic((9, 0), 227 * 1024).check_supported().is_ok());
    assert!(matches!(synthetic((7, 5), 64 * 1024).check_supported(), Err(Error::Unsupported(_))));
    assert!(matches!(synthetic((8, 6), 48 * 1024).check_supported(), Err(Error::Unsupported(_))));
    // 7 GHz x 256 bit double data rate = 448 GB/s.
    assert!((synthetic((8, 6), 99 * 1024).peak_bandwidth_gbs() - 448.0).abs() < 1.0);
}

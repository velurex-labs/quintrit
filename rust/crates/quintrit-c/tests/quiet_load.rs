//! A library must not write to the standard output of its host unasked: loading a model through the C API is silent
//! unless `QUINTRIT_MODEL_OPEN_VERBOSE` is set. The loads run in a child process (this test binary started again) so
//! that its real standard output can be captured.

mod common;

use common::synth::*;
use common::*;
use quintrit_c::*;
use std::process::Command;
use std::ptr::null_mut;

const CHILD_ENV: &str = "QUINTRIT_QUIET_LOAD_CHILD";

/// Child side: loads the model in `CHILD_ENV` with the flags named by the variable value and exits.
fn child(spec: &str) {
    let (flags, path) = spec.split_once(':').unwrap();
    let flags: u32 = flags.parse().unwrap();
    let opts = ModelOptions { struct_size: size_of::<ModelOptions>() as u32, num_threads: 1, kernel: QUINTRIT_KERNEL_AUTO, flags };
    let p = cstr(path);
    let mut m: *mut ModelHandle = null_mut();
    expect_status(unsafe { quintrit_model_open(p.as_ptr(), &opts, &mut m) }, QUINTRIT_OK);
    expect_status(unsafe { quintrit_model_free(m) }, QUINTRIT_OK);
}

/// Runs this test binary again for the given flags and returns what it wrote to the standard output.
fn stdout_of_load(flags: u32, model: &std::path::Path) -> String {
    let out = Command::new(std::env::current_exe().unwrap())
        .args(["--exact", "load_child", "--nocapture", "--test-threads=1"])
        .env(CHILD_ENV, format!("{flags}:{}", model.to_str().unwrap()))
        .output()
        .unwrap();
    assert!(out.status.success(), "child failed: {}", String::from_utf8_lossy(&out.stderr));
    String::from_utf8_lossy(&out.stdout).into_owned()
}

/// Entry point of the child process; does nothing when the test binary runs normally.
#[test]
fn load_child() {
    if let Ok(spec) = std::env::var(CHILD_ENV) {
        child(&spec);
    }
}

#[test]
fn default_load_prints_nothing_and_verbose_flag_prints_progress() {
    let dir = TempDir::new("quiet_load");
    let path = dir.path("model.gguf");
    write_model(&path, 5, &Flaws::default());

    let quiet = stdout_of_load(0, &path);
    assert!(!quiet.contains("==>"), "a default load must not print progress, got:\n{quiet}");
    assert!(!quiet.contains("Arch:"), "a default load must not print the model summary, got:\n{quiet}");

    let loud = stdout_of_load(QUINTRIT_MODEL_OPEN_VERBOSE, &path);
    assert!(loud.contains("==> Loading & Transcoding"), "QUINTRIT_MODEL_OPEN_VERBOSE must print progress, got:\n{loud}");
    assert!(loud.contains("==> Model successfully loaded"), "got:\n{loud}");
}

#[test]
fn unknown_option_flags_are_rejected() {
    let dir = TempDir::new("quiet_flags");
    let path = dir.path("model.gguf");
    write_model(&path, 5, &Flaws::default());
    let opts = ModelOptions { struct_size: size_of::<ModelOptions>() as u32, num_threads: 1, kernel: QUINTRIT_KERNEL_AUTO, flags: 2 };
    let p = cstr(path.to_str().unwrap());
    let mut m: *mut ModelHandle = std::ptr::dangling_mut::<ModelHandle>();
    expect_status(unsafe { quintrit_model_open(p.as_ptr(), &opts, &mut m) }, QUINTRIT_ERR_INVALID_ARGUMENT);
    assert!(m.is_null());
}

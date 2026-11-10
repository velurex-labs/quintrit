//! Builds and runs the C and C++ tests under `tests/c` and `tests/cpp` against the real shared library.
//!
//! A C / C++ compiler is located with the `cc` crate (MSVC through vswhere, `cc` / `c++` / `clang` / `gcc` on Unix). If
//! none is found the tests print `SKIPPED` and pass, unless the environment variable `QUINTRIT_REQUIRE_C_COMPILER=1`
//! is set (use it in CI), in which case a missing compiler is a failure. The shared library is built by a nested
//! `cargo build -p quintrit-c`, which is a no-op when `cargo test` has already built it.

mod common;

use common::synth::*;
use common::*;
use quintrit_c::*;
use std::ffi::c_char;
use std::path::{Path, PathBuf};
use std::process::Command;
use std::ptr::null_mut;

fn crate_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
}

/// `target/<profile>` of the running test executable.
fn profile_dir() -> PathBuf {
    let exe = std::env::current_exe().unwrap();
    exe.parent().unwrap().parent().unwrap().to_path_buf()
}

fn required() -> bool {
    std::env::var("QUINTRIT_REQUIRE_C_COMPILER").is_ok_and(|v| v == "1")
}

fn skip_or_fail(why: &str) {
    assert!(!required(), "QUINTRIT_REQUIRE_C_COMPILER=1 but: {why}");
    eprintln!("SKIPPED: {why}");
}

fn compiler(cpp: bool) -> Option<cc::Tool> {
    let triple = env!("QUINTRIT_C_TARGET");
    cc::Build::new()
        .cpp(cpp)
        .target(triple)
        .host(triple)
        .opt_level(1)
        .debug(false)
        .out_dir(std::env::temp_dir())
        .cargo_metadata(false)
        .try_get_compiler()
        .ok()
}

struct Library {
    dir: PathBuf,
    /// What the linker takes: the import library on Windows, `-lquintrit_c` elsewhere.
    implib: Option<PathBuf>,
}

/// Makes sure the shared library exists and returns where it is (built once per test process).
fn library() -> Option<&'static Library> {
    static LIB: std::sync::OnceLock<Option<Library>> = std::sync::OnceLock::new();
    LIB.get_or_init(build_library).as_ref()
}

fn build_library() -> Option<Library> {
    let dir = profile_dir();
    let profile = dir.file_name().unwrap().to_string_lossy().into_owned();
    let mut cmd = Command::new(std::env::var("CARGO").unwrap_or_else(|_| "cargo".into()));
    cmd.args(["build", "-p", "quintrit-c", "--manifest-path"]).arg(crate_dir().join("Cargo.toml"));
    cmd.arg("--target-dir").arg(dir.parent().unwrap());
    match profile.as_str() {
        "debug" => {}
        "release" => {
            cmd.arg("--release");
        }
        // Custom profiles (for example `release-unwind`) have a directory named like the profile.
        other => {
            cmd.args(["--profile", other]);
        }
    }
    let status = cmd.status().expect("cannot run cargo");
    assert!(status.success(), "nested cargo build failed");
    let (shared, implib) = if cfg!(windows) {
        ("quintrit_c.dll", Some(dir.join("quintrit_c.dll.lib")))
    } else if cfg!(target_os = "macos") {
        ("libquintrit_c.dylib", None)
    } else {
        ("libquintrit_c.so", None)
    };
    assert!(dir.join(shared).exists(), "{} was not built (crate-type cdylib)", dir.join(shared).display());
    if let Some(i) = &implib {
        assert!(i.exists(), "{} was not built", i.display());
    }
    Some(Library { dir, implib })
}

struct Built {
    exe: PathBuf,
    lib_dir: PathBuf,
}

/// Compiles `source` (relative to the crate) against the shared library. `strict` adds warnings-as-errors for the
/// stand-alone header check.
fn build(lib: &Library, source: &str, cpp: bool, tag: &str, strict: bool) -> Option<Built> {
    let Some(tool) = compiler(cpp) else {
        skip_or_fail(&format!("no {} compiler found", if cpp { "C++" } else { "C" }));
        return None;
    };
    let work = std::env::temp_dir().join(format!("quintrit-c-build-{}-{tag}", std::process::id()));
    std::fs::create_dir_all(&work).unwrap();
    // QUINTRIT_STRICT_WARNINGS=1 turns warnings into errors for the test programs too.
    let strict = strict || std::env::var("QUINTRIT_STRICT_WARNINGS").is_ok_and(|v| v == "1");
    let include = crate_dir().join("include");
    let src = crate_dir().join(source);
    let exe = work.join(if cfg!(windows) { format!("{tag}.exe") } else { tag.to_string() });

    eprintln!("compiling {source} with {}", tool.path().display());
    let mut cmd = tool.to_command();
    if tool.is_like_msvc() {
        cmd.arg("/nologo").arg(if strict { "/W4" } else { "/W3" });
        if strict {
            cmd.arg("/WX");
        }
        if cpp {
            cmd.args(["/EHsc", "/std:c++14"]);
        }
        cmd.arg(format!("/I{}", include.display()));
        cmd.arg(&src);
        cmd.arg(format!("/Fo{}\\", work.display()));
        cmd.arg(format!("/Fe{}", exe.display()));
        cmd.arg(lib.implib.as_ref().unwrap());
    } else {
        cmd.arg(if cpp { "-std=c++11" } else { "-std=c99" }).args(["-Wall", "-Wextra", "-pedantic"]);
        if strict {
            cmd.arg("-Werror");
        }
        cmd.arg("-I").arg(&include).arg(&src).arg("-o").arg(&exe);
        cmd.arg("-L").arg(&lib.dir).arg("-lquintrit_c");
        cmd.arg(format!("-Wl,-rpath,{}", lib.dir.display()));
    }
    let out = cmd.current_dir(&work).output().expect("cannot run the compiler");
    assert!(
        out.status.success(),
        "compiling {source} failed\n--- stdout ---\n{}\n--- stderr ---\n{}",
        String::from_utf8_lossy(&out.stdout),
        String::from_utf8_lossy(&out.stderr)
    );
    Some(Built { exe, lib_dir: lib.dir.clone() })
}

fn run(b: &Built, args: &[&Path]) -> String {
    let mut cmd = Command::new(&b.exe);
    cmd.args(args);
    let sep = if cfg!(windows) { ";" } else { ":" };
    let var = if cfg!(windows) {
        "PATH"
    } else if cfg!(target_os = "macos") {
        "DYLD_LIBRARY_PATH"
    } else {
        "LD_LIBRARY_PATH"
    };
    let old = std::env::var(var).unwrap_or_default();
    cmd.env(var, format!("{}{sep}{old}", b.lib_dir.display()));
    let out = cmd.output().expect("cannot run the test program");
    let stdout = String::from_utf8_lossy(&out.stdout).into_owned();
    assert!(
        out.status.success(),
        "{} failed with {:?}\n--- stdout ---\n{stdout}\n--- stderr ---\n{}",
        b.exe.display(),
        out.status.code(),
        String::from_utf8_lossy(&out.stderr)
    );
    stdout
}

#[test]
fn headers_compile_standalone_in_c_and_cpp_with_warnings_as_errors() {
    let Some(lib) = library() else { return };
    let dir = std::env::temp_dir().join(format!("quintrit-c-headers-{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let c_tu = dir.join("header_only.c");
    std::fs::write(&c_tu, "#include \"quintrit.h\"\n#include \"quintrit.h\"\nint main(void) { return quintrit_abi_version() == QUINTRIT_ABI_VERSION ? 0 : 1; }\n").unwrap();
    let cpp_tu = dir.join("header_only.cpp");
    std::fs::write(
        &cpp_tu,
        "#include \"quintrit.hpp\"\n#include \"quintrit.hpp\"\nint main() { quintrit::check_abi(); return quintrit::version().empty() ? 1 : 0; }\n",
    )
    .unwrap();
    for (src, cpp, tag) in [(&c_tu, false, "header_c"), (&cpp_tu, true, "header_cpp")] {
        // `build` takes sources relative to the crate; an absolute path replaces it in PathBuf::join.
        if let Some(b) = build(lib, src.to_str().unwrap(), cpp, tag, true) {
            run(&b, &[]);
        }
    }
}

#[test]
fn c_kernel_test() {
    let Some(lib) = library() else { return };
    if let Some(b) = build(lib, "tests/c/test_kernel.c", false, "c_kernel", false) {
        let out = run(&b, &[]);
        assert!(out.contains("C test passed"), "{out}");
    }
}

#[test]
fn cpp_kernel_test() {
    let Some(lib) = library() else { return };
    if let Some(b) = build(lib, "tests/cpp/test_kernel.cpp", true, "cpp_kernel", false) {
        let out = run(&b, &[]);
        assert!(out.contains("C++ test passed"), "{out}");
    }
}

/// The greedy text of the synthetic model through the Rust view of the C API, as lowercase hex.
fn expected_hex(model_path: &Path) -> String {
    let mut m: *mut ModelHandle = null_mut();
    let p = cstr(model_path.to_str().unwrap());
    let opts = ModelOptions { struct_size: size_of::<ModelOptions>() as u32, num_threads: 2, kernel: QUINTRIT_KERNEL_AUTO, flags: 0 };
    expect_status(unsafe { quintrit_model_open(p.as_ptr(), &opts, &mut m) }, QUINTRIT_OK);
    let mut s: *mut SessionHandle = null_mut();
    expect_status(unsafe { quintrit_session_create(m, std::ptr::null(), &mut s) }, QUINTRIT_OK);
    let mut params: GenerateParams = unsafe { std::mem::zeroed() };
    expect_status(unsafe { quintrit_generate_params_default(&mut params) }, QUINTRIT_OK);
    params.max_tokens = 8;
    let prompt = "<|user|>\nHello there\n<|assistant|>\n";
    let mut out = vec![0 as c_char; 1024];
    let mut len = 0usize;
    expect_status(
        unsafe { quintrit_session_generate(s, prompt.as_ptr() as *const c_char, prompt.len(), &params, None, null_mut(), out.as_mut_ptr(), out.len(), &mut len, null_mut()) },
        QUINTRIT_OK,
    );
    unsafe {
        quintrit_session_free(s);
        quintrit_model_free(m);
    }
    out[..len].iter().map(|&c| format!("{:02x}", c as u8)).collect()
}

fn text_hex(stdout: &str) -> &str {
    stdout.lines().find_map(|l| l.strip_prefix("TEXT_HEX=")).unwrap_or_else(|| panic!("no TEXT_HEX line in\n{stdout}"))
}

#[test]
fn c_runtime_test_on_a_synthetic_model() {
    let Some(lib) = library() else { return };
    let Some(b) = build(lib, "tests/c/test_runtime.c", false, "c_runtime", false) else { return };
    let dir = TempDir::new("c-runtime");
    let model = dir.path("model.gguf");
    write_model(&model, 21, &Flaws::default());
    let out = run(&b, &[&model]);
    assert!(out.contains("C runtime test passed"), "{out}");
    assert_eq!(text_hex(&out), expected_hex(&model), "the C program and the Rust API must generate the same text");
}

#[test]
fn cpp_runtime_test_on_a_synthetic_model() {
    let Some(lib) = library() else { return };
    let Some(b) = build(lib, "tests/cpp/test_runtime.cpp", true, "cpp_runtime", false) else { return };
    let dir = TempDir::new("cpp-runtime");
    let model = dir.path("model.gguf");
    write_model(&model, 21, &Flaws::default());
    let out = run(&b, &[&model]);
    assert!(out.contains("C++ runtime test passed"), "{out}");
    assert_eq!(text_hex(&out), expected_hex(&model));
}

//! Exports the target triple to the integration tests, which need it to locate a C / C++ compiler.

fn main() {
    println!("cargo:rerun-if-changed=build.rs");
    let target = std::env::var("TARGET").unwrap_or_default();
    println!("cargo:rustc-env=QUINTRIT_C_TARGET={target}");
}

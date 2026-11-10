# Contributing to Quintrit

Thank you for your interest. Bug reports, questions, documentation fixes and code contributions are welcome.

## License of contributions

Unless you explicitly state otherwise, any contribution intentionally submitted for inclusion in the work by you, as
defined in the Apache-2.0 license, shall be licensed under the Apache License, Version 2.0, without any additional
terms or conditions. See [`LICENSE`](LICENSE) and [`NOTICE`](NOTICE).

Only submit code that you wrote yourself or that you have the right to submit under this license. If your
contribution reproduces or adapts the work of others, say so in the pull request so that [`NOTICE`](NOTICE) can be
updated.

## Reporting a problem

Open an issue and include:

- what you did and what you expected, with the smallest example that shows the problem;
- the version of Quintrit, the operating system, the CPU model and the output of `rustc --version`
  (or the Python version for the Python package);
- for wrong numerical results: the shapes, the kernel (`quintrit` or `quintrit_512`) and whether the result
  differs between kernels or between CPUs.

## Building and testing

The Rust workspace is in [`rust/`](rust/). It needs Rust 1.98 or newer.

```bash
cd rust
cargo test                      # all tests of the workspace, no model files and no network needed
cargo test -p quintrit-core     # one crate
```

SIMD variants run only on CPUs that advertise the required features, so tests for AVX-512 and `i8mm` are skipped on
other machines. The language bindings have their own instructions:

- C and C++: [`rust/crates/quintrit-c/README.md`](rust/crates/quintrit-c/README.md) (header, CMake project, tests)
- Python: [`rust/crates/quintrit-py/README.md`](rust/crates/quintrit-py/README.md) (maturin build, `pytest`)

## Pull requests

- Keep a pull request focused on one change and describe what it changes and why.
- Add or update tests. Kernels, packers, decoders and numerical primitives are checked against an independent
  brute-force or `f64` ground truth, and new code should be checked the same way.
- The kernels are exact: for the same inputs every kernel and every instruction set must return the same integers,
  and the floating-point primitives of the runtime must stay bit-identical between AVX2 and NEON. A change that
  alters results needs a clear justification.
- Code that targets a specific instruction set (AVX-512, `i8mm`, NEON dot product) should say in the pull request
  which hardware it was tested on. Please do not claim a speed-up without a measurement and the CPU it was taken on.
- Write comments, documentation and commit messages in English, and follow the style of the surrounding code.
- Make sure `cargo test` passes before you submit.

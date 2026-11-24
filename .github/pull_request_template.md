## What this changes and why

<!-- One change per pull request. Link the issue if there is one. -->

## How it was tested

<!-- Commands you ran and on which CPU (and GPU). For a kernel change: the instruction set, and whether the results are bit-identical to the scalar reference. -->

## Checklist

- [ ] `cargo test` in `rust/` passes (and the tests of the binding I changed).
- [ ] Tests are added or updated; a numerical change is checked against an independent ground truth.
- [ ] Kernel and numerics changes keep every instruction set bit-identical (AVX2, AVX-512, NEON, GPU).
- [ ] Documentation is updated where behavior or usage changed.
- [ ] I wrote this code myself or have the right to submit it under Apache-2.0, and I say so below if it adapts the work of others.

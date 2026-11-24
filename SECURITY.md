# Security Policy

## Supported versions

Security fixes are made for the latest released version of Quintrit. Older versions receive a fix only when the
change is small and the version is still widely used.

## Reporting a vulnerability

Please do **not** report a vulnerability in a public issue or discussion. Send an e-mail to
<quintrit@velurex-labs.com> with:

- the affected component (crate, Python package, C library or command-line tool) and its version;
- a description of the problem and its impact;
- the smallest input that reproduces it, for example a malformed model file, and the operating system and CPU.

You will get an acknowledgement within a few days. The project is maintained on a best-effort basis, so a fix can take
longer; you will be told what to expect, and you will be credited in the release notes if you wish.

## What counts as a vulnerability

The library reads model files and tokenizer data that may come from untrusted sources, and it runs native code. These
are in scope:

- memory-safety problems (out-of-bounds access, use after free, uninitialized memory) in the Rust code, the C API or
  the Python bindings, in particular when parsing a GGUF file or a tokenizer;
- crashes or unbounded memory use that a crafted file can trigger (the GGUF reader checks every length and offset, and
  a file that gets past those checks is a bug);
- an unsound `unsafe` block or a C API function that violates its documented contract.

Wrong numerical results, a slow kernel or a model that fails to load are ordinary bugs: open an issue for them.

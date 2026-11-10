# The `quintrit` command-line tool

A small program that loads a GGUF model, generates text, chats and measures speed. It
is the binary `quintrit` of the crate [`quintrit-runtime`](../../rust/crates/quintrit-runtime/README.md).

## Build and run

From [`rust/`](../../rust/) (Rust 1.98 or newer):

```bash
cargo build --release -p quintrit-runtime --bin quintrit
./target/release/quintrit --model path/to/model.gguf --prompt "What is ternary computing?" --max-tokens 64
```

Or run it through cargo: `cargo run --release -p quintrit-runtime --bin quintrit -- <options>`.

## Options

| Option | Meaning |
|---|---|
| `--model <PATH>` | GGUF model file. Without it the tool looks in a `models/` directory. |
| `--prompt <TEXT>` | Prompt for text generation. |
| `--max-tokens <N>` | New tokens at most (default 32). |
| `--temp <FLOAT>` | Sampling temperature (default 0.7; `0.0` is greedy, deterministic decoding). |
| `--top-k <INT>`, `--top-p <FLOAT>` | Top-k (default 40) and nucleus (default 0.9) filtering. |
| `--stop <TEXT>` | Stop generating at this text. |
| `--threads <N>` | Worker threads (default: the number of CPU cores). |
| `--kernel <NAME>` | `quintrit` (default; AVX2 and NEON) or `quintrit_512` (AVX-512). See [`../kernels.md`](../kernels.md). |
| `--chat` | Interactive multi-turn chat. Each turn re-sends the conversation in the chat template of the model; the KV cache of the previous turn is kept, so only the new part is computed. `/reset` forgets the conversation. |
| `--bench` | Measure time to first token, tokens per second and memory bandwidth. |
| `-h`, `--help` | Show the help text. |

## Models

Any GGUF model with ternary weights (`I2_S`, `TQ1_0` or `TQ2_0`) of the BitNet b1.58, Falcon3 or LLaMA families. The weights are
transcoded into RAM at start-up, which takes from seconds to minutes depending on the size of the model. Models are not part
of this repository; take them from their publishers and read their licenses.

Determinism: with `--temp 0.0` the text is the same for every kernel and every run on the same operating system.

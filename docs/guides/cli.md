# The `quintrit` command-line tool

A small program that loads a GGUF model, generates text, chats and measures speed. It
is the binary `quintrit` of the crate [`quintrit-runtime`](../../rust/crates/quintrit-runtime/README.md).

## Get the program

Download the archive of your platform from the [latest release](https://github.com/velurex-labs/quintrit/releases/latest)
(Linux x86-64, Linux ARM64, Windows x86-64), unpack it and run `quintrit`; nothing else is needed. A step-by-step
walkthrough with a small model, one answer, a chat and a benchmark is in
[`examples/cli/`](../../examples/cli/README.md). To build it yourself, see the next section.

## Build and run

From [`rust/`](../../rust/) (Rust 1.98 or newer):

```bash
cargo build --release -p quintrit-runtime --bin quintrit
./target/release/quintrit --model path/to/model.gguf --prompt "What is ternary computing?" --max-tokens 64
```

Or run it through cargo: `cargo run --release -p quintrit-runtime --bin quintrit -- <options>`.

No compiler flags are needed: the instruction set is chosen when the program runs. `-C target-cpu=native` adds a few
percent, see [Compiler flags](../kernels.md#compiler-flags).

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
| `--device <DEVICE>` | `cpu` (default), `cuda` or `cuda:<index>`: run the model on an NVIDIA GPU (build with `--features cuda`, see [`gpu.md`](gpu.md)). The text is the same as on the CPU. |
| `--context <N>` | Context window in tokens on a GPU (default 4096, at most 7936). |
| `--chat` | Interactive multi-turn chat. Each turn re-sends the conversation in the chat template of the model; the KV cache of the previous turn is kept, so only the new part is computed. `/reset` forgets the conversation. |
| `--bench` | Greedy benchmark pass: time to first token and tokens per second; on the CPU, for dense models, also the weight traffic and memory bandwidth. Generates exactly `--max-tokens` tokens (it does not stop at the end-of-sequence token). |
| `--prompt-file <PATH>` | Read the prompt from a text file instead of `--prompt`. |
| `--chat-template` | Wrap the prompt as one user message in the chat template of the model. |
| `--warmup <N>`, `--repeat <N>` | With `--bench`: runs that are executed and discarded first (default 0), and measured runs (default 1). |
| `--json` | With `--bench`: print one more line, `QUINTRIT_BENCH_JSON {...}`, with every run, the median, minimum and maximum, the device, kernel, thread count and context. |
| `-h`, `--help` | Show the help text. |

## Models

Any GGUF model with ternary weights (`I2_S`, `TQ1_0` or `TQ2_0`; `I2_S` and block-scaled `Q2_0` models also on a GPU) of the BitNet b1.58, Falcon3 or LLaMA families. The weights are
transcoded into RAM at start-up, which takes from seconds to minutes depending on the size of the model. Models are not part
of this repository; take them from their publishers and read their licenses.

Determinism: with `--temp 0.0` the text is the same for every kernel and every run on the same operating system.

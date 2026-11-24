# Try Quintrit without installing anything

Download the `quintrit` program and a small model, then ask a question, chat, or measure the speed of your machine.
No Rust toolchain, no compiler and no Python are needed.

## 1. Get the program

The [latest release](https://github.com/velurex-labs/quintrit/releases/latest) has one archive per platform:

| Platform | Archive |
|---|---|
| Linux, x86-64 (AVX2 or AVX-512) | `quintrit-linux-x86_64.tar.gz` |
| Linux, ARM64 (Raspberry Pi 5, ARM servers) | `quintrit-linux-aarch64.tar.gz` |
| Windows, x86-64 | `quintrit-windows-x86_64.zip` |

Linux (use `quintrit-linux-aarch64.tar.gz` on ARM):

```bash
curl -L https://github.com/velurex-labs/quintrit/releases/latest/download/quintrit-linux-x86_64.tar.gz | tar xz
./quintrit-linux-x86_64/quintrit --help
```

Windows (PowerShell):

```powershell
curl.exe -L -o quintrit.zip https://github.com/velurex-labs/quintrit/releases/latest/download/quintrit-windows-x86_64.zip
tar -xf quintrit.zip
.\quintrit-windows-x86_64\quintrit.exe --help
```

The program is one file. It picks the fastest instruction set of your CPU (AVX2, AVX-512 or NEON) when it starts.
The release page also has `SHA256SUMS.txt` if you want to check the download.

## 2. Get a model

Any GGUF model with ternary weights works. A small one to start with (440 MB, from its publisher on Hugging Face; read
the model's terms there):

```bash
curl -L -o model.gguf https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf/resolve/main/Ternary-Bonsai-1.7B-Q2_0.gguf
```

On Windows use `curl.exe` with the same arguments. The file should have the SHA-256 checksum
`d97d94eb564590c9f0300e54d3f87bbbb25a78693d0ade9f6e177973dcb8228a`. The weights are converted into RAM when the program
starts (a few seconds for this model), so plan free memory of about the size of the model file. Other models that work
are listed in [`model_benchmarks/`](../../model_benchmarks/README.md).

## 3. Use it

The examples use `quintrit` for the program; write `./quintrit-linux-x86_64/quintrit` (or the Windows path) in front.

### One answer to one prompt

```bash
quintrit --model model.gguf --prompt "What is ternary computing?" --chat-template --max-tokens 100
```

`--chat-template` wraps the prompt in the chat format of the model, as an instruction model expects. Without it the
text is continued as plain text. The answer is sampled, so it differs from run to run; add `--temp 0.0` for the same
answer every time (greedy decoding). With `--temp 0.0` the answer starts like this:

```text
Ternary computing is a type of computing that uses only three possible states or values for data: typically **0**,
```

### A conversation

```bash
quintrit --model model.gguf --chat
```

Type a message and press Enter. The program remembers the conversation and computes only the new part of each turn.
`/reset` forgets the conversation, `/exit` ends the program.

### Measure the speed of your machine

```bash
quintrit --model model.gguf --bench --chat-template --prompt "What is ternary computing?" --max-tokens 256 --warmup 1 --repeat 5
```

It generates exactly 256 tokens with greedy decoding, five times after one discarded warm-up run, and prints the time to
the first token, the tokens per second and, on the CPU, how many bytes of weights each token reads from memory and the
resulting memory bandwidth. This is the same measurement as in [`model_benchmarks/`](../../model_benchmarks/README.md), so
you can compare your machine with the tables there. Useful options:

| Option | Effect |
|---|---|
| `--threads <N>` | Number of worker threads (default: all cores). Try fewer to see how the speed scales. |
| `--kernel quintrit_512` | The AVX-512 kernel, on CPUs that have it. The text with `--temp 0.0` is identical to the default kernel. |
| `--json` | One more line with all runs and the run conditions, for scripts and reports. |

Speed depends mostly on how fast your memory is, because every token reads all weights once.

## Notes

- Prompts are processed one token at a time, so a very long prompt takes a while before the first word appears.
- The binaries run on the CPU. A graphics card is used only by a build from source with the CUDA feature, see
  [`../../docs/guides/gpu.md`](../../docs/guides/gpu.md).
- The Windows program is not signed. If Windows SmartScreen asks, the file is the one from the release page; compare it
  with `SHA256SUMS.txt` to be sure.
- All options are described in the [command-line guide](../../docs/guides/cli.md).

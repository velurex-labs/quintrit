# Ternary Bonsai 2 4B

Decode speed of the model `Ternary Bonsai 2 4B` (4B parameters, ternary weights) in the `quintrit` runtime on four CPUs and two NVIDIA GPUs.

## The model

| | |
|---|---|
| Source | [prism-ml/Ternary-Bonsai-2-4B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-4B-gguf) (file `Ternary-Bonsai-2-4B-Q2_0.gguf`) |
| Weight format | `Q2_0` (block of 64 weights) |
| File size | 1,212,103,520 bytes |
| SHA-256 | `64a5183b44921bdb5ac4a6f5f92007821d6ebe5c66997d3c147fd214ab9433d4` |
| Resident memory of the process | 2.5 GiB (CPU) |

A hybrid architecture with Gated DeltaNet layers; the token embedding and the output head are ternary as well.

## Decode speed on CPUs

Generated tokens per second for the standard prompt (`What is ternary computing?` in the chat template of the model, 18 prompt tokens), 256 greedy tokens. The column "Spread" gives the slowest and the fastest of the measured runs. Only the highest thread count of each machine was measured, so there is no scaling figure.

| Machine | Instruction set and kernel | Threads | Tokens/s (median) | Spread | Runs |
|---|---|---|---|---|---|
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, block-scaled kernel | 4 | 27.97 | 27.29 to 29.03 | 5 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 8 | 14.23 | 14.15 to 14.23 | 5 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 8 | 7.03 | 7.02 to 7.03 | 5 |
| Raspberry Pi 5 (4 Cortex-A76 cores) | NEON, block-scaled kernel | 4 | 5.40 | 5.40 to 5.42 | 5 |

## Decode speed on GPUs

Tokens per second of the end-to-end pass (prompt, sampling and the copy of the logits included), the same prompt, 256 greedy tokens, context of 4096 tokens. The RTX 4060 Ti is a rented card in a container on a shared host; the RTX 3070 is in the desktop machine that also holds the i7-7700.

| GPU | Tokens/s (median) | Spread | Runs | Device memory at peak |
|---|---|---|---|---|
| NVIDIA GeForce RTX 3070 (8 GiB) | 137.95 | 136.95 to 138.93 | 5 | 1.42 GiB |
| NVIDIA GeForce RTX 4060 Ti (16 GiB) | 144.45 | 144.41 to 144.50 | 5 | 1.41 GiB |

## Long prompt

The long prompt has 526 tokens. The runtime still processes the prompt one token at a time (there is no batched prefill yet), but it computes the output head only for the last prompt token, and it runs the attention of the heads in parallel on the CPU, so the time to first token is below the number of prompt tokens times the time of one decode token.

| Machine | Threads | Tokens/s, standard prompt | Tokens/s, long prompt | Time to first token, long prompt |
|---|---|---|---|---|
| EPYC 9V45 | 4 | 27.97 | 27.26 | 16.5 s |
| Neoverse-N2 | 8 | 14.23 | 13.82 | 34.1 s |
| i7-7700 | 8 | 7.03 | 6.86 | 67.6 s |
| Raspberry Pi 5 | 4 | 5.40 | 5.20 | 86.9 s |
| RTX 3070 | GPU | 137.95 | 130.91 | 3.5 s |
| RTX 4060 Ti | GPU | 144.45 | 142.00 | 3.2 s |

## Measurement conditions

- Greedy decoding (`--bench` is greedy and does not stop at the end-of-sequence token), exactly 256 generated tokens, one warm-up run and 5 measured runs (3 for the thread counts below the maximum and for the long prompt) inside one process; the table shows the median.
- Every configuration (model, thread count, prompt) runs in its own freshly started process, and the configurations run one after another, not interleaved. All machines run Linux. The EPYC and Neoverse-N2 machines are virtual machines and the RTX 4060 Ti is a rented container, all on shared hosts; the RTX 3070 and the i7-7700 belong to one desktop machine. On the Raspberry Pi 5 the throttling flags of the firmware were clear before and after every run.

| Machine | CPU | Kernel | libc | Rust compiler | `RUSTFLAGS` |
|---|---|---|---|---|---|
| AMD EPYC 9V45 (2 cores, 4 threads) | AMD EPYC 9V45 96-Core Processor | Linux 7.0.0-1017-azure | 2.39 | rustc 1.99.1 | `-C target-cpu=znver4` |
| Azure Neoverse-N2 (8 cores) | Neoverse-N2 | Linux 7.0.0-1017-azure | 2.39 | rustc 1.99.1 | `-C target-cpu=neoverse-n2` |
| Intel Core i7-7700 (4 cores, 8 threads) | Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz | Linux 6.12.115+deb13.1-amd64 | 2.41 | rustc 1.99.1 | `-C target-cpu=x86-64-v3` |
| Raspberry Pi 5 (4 Cortex-A76 cores) | Cortex-A76 | Linux 6.18.50+rpt-rpi-2712 | 2.41 | rustc 1.99.1 | `-C target-cpu=cortex-a76` |

All results of this page come from binaries built from the same sources. The CPU binaries were built with the `RUSTFLAGS` of the table above; a build without them reached 90 % to 104 % of that speed on the machines where both were measured.

The prompts are in [`prompts/`](./prompts/standard.txt) next to this page.

## Reproduce

Build the command-line tool in the `rust` directory of this repository. The CPU figures above were measured with the `RUSTFLAGS` shown for each machine in the table of the measurement conditions (for example `-C target-cpu=znver4` on the EPYC 9V45); use the value for your CPU or `-C target-cpu=native`. A GPU needs the feature `cuda` and the CUDA Toolkit.

```bash
cd rust
RUSTFLAGS="-C target-cpu=native" cargo build --release -p quintrit-runtime --bin quintrit   # add --features cuda for a GPU

# standard prompt: the figures of the tables above (1 warm-up run, then 5 measured runs in one process)
./target/release/quintrit --model path/to/Ternary-Bonsai-2-4B-Q2_0.gguf --bench --json --chat-template \
    --prompt-file ../model_benchmarks/prompts/standard.txt --max-tokens 256 --warmup 1 --repeat 5 --threads 4

# long prompt (3 measured runs on a CPU, 5 on a GPU)
./target/release/quintrit --model path/to/Ternary-Bonsai-2-4B-Q2_0.gguf --bench --json --chat-template \
    --prompt-file ../model_benchmarks/prompts/long.txt --max-tokens 256 --warmup 1 --repeat 3 --threads 4
```

The block-scaled kernel of the machine is chosen automatically; for a GPU replace `--threads N` with `--device cuda`. The last line of the output starts with `QUINTRIT_BENCH_JSON` and holds every run, the median, minimum and maximum, the kernel and the thread count.

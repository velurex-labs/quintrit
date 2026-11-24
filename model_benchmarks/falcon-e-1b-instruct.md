# Falcon-E 1B

Decode speed of the model `Falcon-E 1B` (1B parameters, ternary weights) in the `quintrit` runtime on four CPUs and two NVIDIA GPUs.

## The model

| | |
|---|---|
| Source | [tiiuae/Falcon-E-1B-Instruct-GGUF](https://huggingface.co/tiiuae/Falcon-E-1B-Instruct-GGUF) (file `ggml-model-i2_s.gguf`) |
| Weight format | `I2_S` |
| File size | 666,324,256 bytes |
| SHA-256 | `feb7478007e916d26bb807cb1a01cc45ac16f197e355d8c30aed25e550ecd73b` |
| Resident memory of the process | 1.2 GiB (CPU) |

## Decode speed on CPUs

Generated tokens per second for the standard prompt (`What is ternary computing?` in the chat template of the model, 17 prompt tokens), 256 greedy tokens. The column "Spread" gives the slowest and the fastest of the measured runs. Only the highest thread count of each machine was measured, so there is no scaling figure.

| Machine | Instruction set and kernel | Threads | Tokens/s (median) | Spread | Runs |
|---|---|---|---|---|---|
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, `quintrit_512` | 4 | 78.94 | 78.64 to 79.14 | 5 |
| Azure Neoverse-N2 (8 cores) | NEON, `quintrit` | 8 | 52.09 | 51.95 to 52.27 | 5 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, `quintrit` | 8 | 46.33 | 46.32 to 46.43 | 5 |
| Raspberry Pi 5 (4 Cortex-A76 cores) | NEON, `quintrit` | 4 | 20.59 | 19.56 to 20.70 | 5 |

## Decode speed on GPUs

Tokens per second of the end-to-end pass (prompt, sampling and the copy of the logits included), the same prompt, 256 greedy tokens, context of 4096 tokens. The RTX 4060 Ti is a rented card in a container on a shared host; the RTX 3070 is in the desktop machine that also holds the i7-7700.

| GPU | Tokens/s (median) | Spread | Runs | Device memory at peak |
|---|---|---|---|---|
| NVIDIA GeForce RTX 3070 (8 GiB) | 477.60 | 477.54 to 479.20 | 5 | 0.72 GiB |
| NVIDIA GeForce RTX 4060 Ti (16 GiB) | 415.88 | 415.71 to 415.92 | 5 | 0.71 GiB |

## Long prompt

The long prompt has 544 tokens. The runtime still processes the prompt one token at a time (there is no batched prefill yet), but it computes the output head only for the last prompt token, and it runs the attention of the heads in parallel on the CPU, so the time to first token is below the number of prompt tokens times the time of one decode token.

| Machine | Threads | Tokens/s, standard prompt | Tokens/s, long prompt | Time to first token, long prompt |
|---|---|---|---|---|
| EPYC 9V45 | 4 | 78.94 | 69.03 | 5.5 s |
| Neoverse-N2 | 8 | 52.09 | 44.00 | 9.3 s |
| i7-7700 | 8 | 46.33 | 41.63 | 9.6 s |
| Raspberry Pi 5 | 4 | 20.59 | 17.54 | 21.3 s |
| RTX 3070 | GPU | 477.60 | 430.91 | 1.0 s |
| RTX 4060 Ti | GPU | 415.88 | 388.44 | 1.0 s |

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
./target/release/quintrit --model path/to/ggml-model-i2_s.gguf --bench --json --chat-template \
    --prompt-file ../model_benchmarks/prompts/standard.txt --max-tokens 256 --warmup 1 --repeat 5 --threads 4

# long prompt (3 measured runs on a CPU, 5 on a GPU)
./target/release/quintrit --model path/to/ggml-model-i2_s.gguf --bench --json --chat-template \
    --prompt-file ../model_benchmarks/prompts/long.txt --max-tokens 256 --warmup 1 --repeat 3 --threads 4
```

For the AVX-512 figures add `--kernel quintrit_512`; for a GPU replace `--threads N` with `--device cuda`. The last line of the output starts with `QUINTRIT_BENCH_JSON` and holds every run, the median, minimum and maximum, the kernel and the thread count.

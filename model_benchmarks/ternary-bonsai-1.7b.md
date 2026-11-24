# Ternary Bonsai 1.7B

Decode speed of the model `Ternary Bonsai 1.7B` (1.7B parameters, ternary weights) in the `quintrit` runtime on four CPUs and two NVIDIA GPUs.

![Tokens per second against the number of threads](./scaling_Ternary-Bonsai-1.7B.svg)

## The model

| | |
|---|---|
| Source | [prism-ml/Ternary-Bonsai-1.7B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-1.7B-gguf) (file `Ternary-Bonsai-1.7B-Q2_0.gguf`) |
| Weight format | `Q2_0` (block of 128 weights) |
| File size | 463,290,464 bytes |
| SHA-256 | `d97d94eb564590c9f0300e54d3f87bbbb25a78693d0ade9f6e177973dcb8228a` |
| Resident memory of the process | 1.0 GiB (CPU) |

## Decode speed on CPUs

Generated tokens per second for the standard prompt (`What is ternary computing?` in the chat template of the model, 18 prompt tokens), 256 greedy tokens. The column "Spread" gives the slowest and the fastest of the measured runs, "Scaling" the speed divided by the thread count times the speed with one thread.

| Machine | Instruction set and kernel | Threads | Tokens/s (median) | Spread | Scaling | Runs |
|---|---|---|---|---|---|---|
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, block-scaled kernel | 1 | 44.79 | 43.95 to 44.93 | 100 % | 3 |
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, block-scaled kernel | 2 | 63.58 | 63.25 to 64.48 | 71 % | 3 |
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, block-scaled kernel | 4 | 68.04 | 66.63 to 68.62 | 38 % | 5 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 1 | 7.43 | 7.41 to 7.46 | 100 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 2 | 13.84 | 13.82 to 13.87 | 93 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 4 | 24.92 | 24.89 to 25.00 | 84 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 8 | 39.18 | 38.84 to 39.67 | 66 % | 5 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 1 | 5.59 | 5.58 to 5.59 | 100 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 2 | 10.19 | 10.00 to 10.21 | 91 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 4 | 17.53 | 17.51 to 17.56 | 78 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 8 | 19.66 | 19.59 to 19.71 | 44 % | 5 |
| Raspberry Pi 5 (4 Cortex-A76 cores) | NEON, block-scaled kernel | 1 | 4.82 | 4.82 to 4.83 | 100 % | 3 |
| Raspberry Pi 5 (4 Cortex-A76 cores) | NEON, block-scaled kernel | 2 | 9.24 | 9.24 to 9.25 | 96 % | 3 |
| Raspberry Pi 5 (4 Cortex-A76 cores) | NEON, block-scaled kernel | 4 | 15.08 | 15.05 to 15.13 | 78 % | 5 |

## Decode speed on GPUs

Tokens per second of the end-to-end pass (prompt, sampling and the copy of the logits included), the same prompt, 256 greedy tokens, context of 4096 tokens. The RTX 4060 Ti is a rented card in a container on a shared host; the RTX 3070 is in the desktop machine that also holds the i7-7700.

| GPU | Tokens/s (median) | Spread | Runs | Device memory at peak |
|---|---|---|---|---|
| NVIDIA GeForce RTX 3070 (8 GiB) | 293.62 | 293.45 to 295.35 | 5 | 1.10 GiB |
| NVIDIA GeForce RTX 4060 Ti (16 GiB) | 315.34 | 314.23 to 315.71 | 5 | 1.09 GiB |

## Long prompt

The long prompt has 523 tokens. The runtime still processes the prompt one token at a time (there is no batched prefill yet), but it computes the output head only for the last prompt token, and it runs the attention of the heads in parallel on the CPU, so the time to first token is below the number of prompt tokens times the time of one decode token.

| Machine | Threads | Tokens/s, standard prompt | Tokens/s, long prompt | Time to first token, long prompt |
|---|---|---|---|---|
| EPYC 9V45 | 4 | 68.04 | 56.76 | 6.9 s |
| Neoverse-N2 | 8 | 39.18 | 32.97 | 12.4 s |
| i7-7700 | 8 | 19.66 | 17.20 | 22.9 s |
| Raspberry Pi 5 | 4 | 15.08 | 11.50 | 31.4 s |
| RTX 3070 | GPU | 293.62 | 270.51 | 1.6 s |
| RTX 4060 Ti | GPU | 315.34 | 291.34 | 1.5 s |

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
./target/release/quintrit --model path/to/Ternary-Bonsai-1.7B-Q2_0.gguf --bench --json --chat-template \
    --prompt-file ../model_benchmarks/prompts/standard.txt --max-tokens 256 --warmup 1 --repeat 5 --threads 4

# long prompt (3 measured runs on a CPU, 5 on a GPU)
./target/release/quintrit --model path/to/Ternary-Bonsai-1.7B-Q2_0.gguf --bench --json --chat-template \
    --prompt-file ../model_benchmarks/prompts/long.txt --max-tokens 256 --warmup 1 --repeat 3 --threads 4

# thread scaling: standard prompt, 3 measured runs per thread count (up to the number of hardware threads of the machine)
for t in 1 2 4 8; do
  ./target/release/quintrit --model path/to/Ternary-Bonsai-1.7B-Q2_0.gguf --bench --json --chat-template \
      --prompt-file ../model_benchmarks/prompts/standard.txt --max-tokens 256 --warmup 1 --repeat 3 --threads $t
done
```

The block-scaled kernel of the machine is chosen automatically; for a GPU replace `--threads N` with `--device cuda`. The last line of the output starts with `QUINTRIT_BENCH_JSON` and holds every run, the median, minimum and maximum, the kernel and the thread count.

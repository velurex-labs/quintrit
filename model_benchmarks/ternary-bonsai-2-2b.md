# Ternary Bonsai 2 2B

Decode speed of the model `Ternary Bonsai 2 2B` (2B parameters, ternary weights) in the `quintrit` runtime on four CPUs and two NVIDIA GPUs.

![Tokens per second against the number of threads](./scaling_Ternary-Bonsai-2-2B.svg)

## The model

| | |
|---|---|
| Source | [prism-ml/Ternary-Bonsai-2-2B-gguf](https://huggingface.co/prism-ml/Ternary-Bonsai-2-2B-gguf) (file `Ternary-Bonsai-2-2B-Q2_0.gguf`) |
| Weight format | `Q2_0` (block of 64 weights) |
| File size | 546,689,504 bytes |
| SHA-256 | `c2893fa6f951daee8b7263d5f49b455e57989ef56417a22125b03566dc560650` |
| Resident memory of the process | 1.2 GiB (CPU) |

A hybrid architecture with Gated DeltaNet layers; the token embedding and the output head are ternary as well.

## Decode speed on CPUs

Generated tokens per second for the standard prompt (`What is ternary computing?` in the chat template of the model, 18 prompt tokens), 256 greedy tokens. The column "Spread" gives the slowest and the fastest of the measured runs, "Scaling" the speed divided by the thread count times the speed with one thread.

| Machine | Instruction set and kernel | Threads | Tokens/s (median) | Spread | Scaling | Runs |
|---|---|---|---|---|---|---|
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, block-scaled kernel | 1 | 41.11 | 40.61 to 41.46 | 100 % | 3 |
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, block-scaled kernel | 2 | 58.14 | 58.06 to 58.73 | 71 % | 3 |
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, block-scaled kernel | 4 | 57.50 | 56.93 to 58.03 | 35 % | 5 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 1 | 6.77 | 6.76 to 6.77 | 100 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 2 | 12.24 | 12.23 to 12.25 | 90 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 4 | 20.85 | 20.85 to 20.85 | 77 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 8 | 28.45 | 28.25 to 28.64 | 53 % | 5 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 1 | 5.08 | 5.07 to 5.08 | 100 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 2 | 8.57 | 8.52 to 8.57 | 84 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 4 | 13.86 | 13.80 to 13.90 | 68 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 8 | 15.17 | 15.15 to 15.18 | 37 % | 5 |
| Raspberry Pi 5 (4 Cortex-A76 cores) | NEON, block-scaled kernel | 1 | 4.39 | 4.38 to 4.39 | 100 % | 3 |
| Raspberry Pi 5 (4 Cortex-A76 cores) | NEON, block-scaled kernel | 2 | 8.02 | 8.01 to 8.04 | 91 % | 3 |
| Raspberry Pi 5 (4 Cortex-A76 cores) | NEON, block-scaled kernel | 4 | 12.11 | 12.06 to 12.14 | 69 % | 5 |

## Decode speed on GPUs

Tokens per second of the end-to-end pass (prompt, sampling and the copy of the logits included), the same prompt, 256 greedy tokens, context of 4096 tokens. The RTX 4060 Ti is a rented card in a container on a shared host; the RTX 3070 is in the desktop machine that also holds the i7-7700.

| GPU | Tokens/s (median) | Spread | Runs | Device memory at peak |
|---|---|---|---|---|
| NVIDIA GeForce RTX 3070 (8 GiB) | 244.24 | 243.41 to 244.72 | 5 | 0.79 GiB |
| NVIDIA GeForce RTX 4060 Ti (16 GiB) | 265.56 | 265.44 to 266.36 | 5 | 0.78 GiB |

## Long prompt

The long prompt has 526 tokens. The runtime still processes the prompt one token at a time (there is no batched prefill yet), but it computes the output head only for the last prompt token, and it runs the attention of the heads in parallel on the CPU, so the time to first token is below the number of prompt tokens times the time of one decode token.

| Machine | Threads | Tokens/s, standard prompt | Tokens/s, long prompt | Time to first token, long prompt |
|---|---|---|---|---|
| EPYC 9V45 | 4 | 57.50 | 57.54 | 7.1 s |
| Neoverse-N2 | 8 | 28.45 | 28.07 | 15.6 s |
| i7-7700 | 8 | 15.17 | 14.91 | 28.2 s |
| Raspberry Pi 5 | 4 | 12.11 | 11.81 | 35.0 s |
| RTX 3070 | GPU | 244.24 | 240.54 | 1.8 s |
| RTX 4060 Ti | GPU | 265.56 | 262.89 | 1.6 s |

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
./target/release/quintrit --model path/to/Ternary-Bonsai-2-2B-Q2_0.gguf --bench --json --chat-template \
    --prompt-file ../model_benchmarks/prompts/standard.txt --max-tokens 256 --warmup 1 --repeat 5 --threads 4

# long prompt (3 measured runs on a CPU, 5 on a GPU)
./target/release/quintrit --model path/to/Ternary-Bonsai-2-2B-Q2_0.gguf --bench --json --chat-template \
    --prompt-file ../model_benchmarks/prompts/long.txt --max-tokens 256 --warmup 1 --repeat 3 --threads 4

# thread scaling: standard prompt, 3 measured runs per thread count (up to the number of hardware threads of the machine)
for t in 1 2 4 8; do
  ./target/release/quintrit --model path/to/Ternary-Bonsai-2-2B-Q2_0.gguf --bench --json --chat-template \
      --prompt-file ../model_benchmarks/prompts/standard.txt --max-tokens 256 --warmup 1 --repeat 3 --threads $t
done
```

The block-scaled kernel of the machine is chosen automatically; for a GPU replace `--threads N` with `--device cuda`. The last line of the output starts with `QUINTRIT_BENCH_JSON` and holds every run, the median, minimum and maximum, the kernel and the thread count.

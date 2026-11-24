# Falcon3 1.58bit 10B

Decode speed of the model `Falcon3 1.58bit 10B` (10B parameters, ternary weights) in the `quintrit` runtime on four CPUs and two NVIDIA GPUs.

![Tokens per second against the number of threads](./scaling_Falcon3-10B-Instruct-1.58bit.svg)

## The model

| | |
|---|---|
| Source | [tiiuae/Falcon3-10B-Instruct-1.58bit-GGUF](https://huggingface.co/tiiuae/Falcon3-10B-Instruct-1.58bit-GGUF) (file `ggml-model-i2_s.gguf`) |
| Weight format | `I2_S` |
| File size | 3,991,393,696 bytes |
| SHA-256 | `e37945ee82693a6541b5fa5484f0e24787c04a9ce95e6e377f68a6b15f139c1f` |
| Resident memory of the process | 7.1 GiB (CPU) |

## Decode speed on CPUs

Generated tokens per second for the standard prompt (`What is ternary computing?` in the chat template of the model, 18 prompt tokens), 256 greedy tokens. The column "Spread" gives the slowest and the fastest of the measured runs, "Scaling" the speed divided by the thread count times the speed with one thread.

| Machine | Instruction set and kernel | Threads | Tokens/s (median) | Spread | Scaling | Runs |
|---|---|---|---|---|---|---|
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, `quintrit_512` | 1 | 11.95 | 11.82 to 11.99 | 100 % | 3 |
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, `quintrit_512` | 2 | 14.53 | 14.44 to 14.68 | 61 % | 3 |
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, `quintrit_512` | 4 | 14.61 | 14.60 to 14.71 | 31 % | 5 |
| Azure Neoverse-N2 (8 cores) | NEON, `quintrit` | 1 | 2.20 | 2.20 to 2.20 | 100 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, `quintrit` | 2 | 4.12 | 4.12 to 4.13 | 93 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, `quintrit` | 4 | 7.61 | 7.61 to 7.64 | 86 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, `quintrit` | 8 | 12.26 | 12.24 to 12.30 | 70 % | 5 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, `quintrit` | 1 | 3.69 | 3.68 to 3.70 | 100 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, `quintrit` | 2 | 5.95 | 5.94 to 5.96 | 81 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, `quintrit` | 4 | 8.93 | 8.90 to 8.93 | 60 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, `quintrit` | 8 | 9.01 | 9.00 to 9.01 | 30 % | 5 |
| Raspberry Pi 5 (4 Cortex-A76 cores) | NEON, `quintrit` | 1 | 1.31 | 1.31 to 1.31 | 100 % | 3 |
| Raspberry Pi 5 (4 Cortex-A76 cores) | NEON, `quintrit` | 2 | 2.48 | 2.48 to 2.49 | 95 % | 3 |
| Raspberry Pi 5 (4 Cortex-A76 cores) | NEON, `quintrit` | 4 | 3.56 | 3.56 to 3.56 | 68 % | 5 |

## Decode speed on GPUs

Tokens per second of the end-to-end pass (prompt, sampling and the copy of the logits included), the same prompt, 256 greedy tokens, context of 4096 tokens. The RTX 4060 Ti is a rented card in a container on a shared host; the RTX 3070 is in the desktop machine that also holds the i7-7700.

| GPU | Tokens/s (median) | Spread | Runs | Device memory at peak |
|---|---|---|---|---|
| NVIDIA GeForce RTX 3070 (8 GiB) | 118.07 | 117.75 to 118.29 | 5 | 3.51 GiB |
| NVIDIA GeForce RTX 4060 Ti (16 GiB) | 87.86 | 86.57 to 88.01 | 5 | 3.50 GiB |

## Long prompt

The long prompt has 529 tokens. The runtime still processes the prompt one token at a time (there is no batched prefill yet), but it computes the output head only for the last prompt token, and it runs the attention of the heads in parallel on the CPU, so the time to first token is below the number of prompt tokens times the time of one decode token.

| Machine | Threads | Tokens/s, standard prompt | Tokens/s, long prompt | Time to first token, long prompt |
|---|---|---|---|---|
| EPYC 9V45 | 4 | 14.61 | 13.13 | 27.9 s |
| Neoverse-N2 | 8 | 12.26 | 10.94 | 35.8 s |
| i7-7700 | 8 | 9.01 | 8.72 | 44.5 s |
| Raspberry Pi 5 | 4 | 3.56 | 3.25 | 117.5 s |
| RTX 3070 | GPU | 118.07 | 110.67 | 3.5 s |
| RTX 4060 Ti | GPU | 87.86 | 83.83 | 4.5 s |

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

# thread scaling: standard prompt, 3 measured runs per thread count (up to the number of hardware threads of the machine)
for t in 1 2 4 8; do
  ./target/release/quintrit --model path/to/ggml-model-i2_s.gguf --bench --json --chat-template \
      --prompt-file ../model_benchmarks/prompts/standard.txt --max-tokens 256 --warmup 1 --repeat 3 --threads $t
done
```

For the AVX-512 figures add `--kernel quintrit_512`; for a GPU replace `--threads N` with `--device cuda`. The last line of the output starts with `QUINTRIT_BENCH_JSON` and holds every run, the median, minimum and maximum, the kernel and the thread count.

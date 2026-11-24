# Ternary Nemotron 3.5 Lightning

Decode speed of the model `Ternary Nemotron 3.5 Lightning` (30B, 3B active parameters, ternary weights) in the `quintrit` runtime on three CPUs and two NVIDIA GPUs (the model does not fit into the memory of the Raspberry Pi 5).

![Tokens per second against the number of threads](./scaling_Ternary-Nemotron-3.5-Lightning-30B-A3B.svg)

## The model

| | |
|---|---|
| Source | [boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF](https://huggingface.co/boro64/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF) (file `Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf`) |
| Weight format | `Q2_0` (block of 64 weights) |
| File size | 9,472,656,320 bytes |
| SHA-256 | `38854de649212dc2761006b75afbc41412e7531f763c3ff965bd31f969bb14f3` |
| Resident memory of the process | 14.9 GiB (CPU) |

The file is a community repack of a release that was published only for MLX. The model combines Mamba-2 layers, attention layers and a mixture of experts.

## Decode speed on CPUs

Generated tokens per second for the standard prompt (`What is ternary computing?` in the chat template of the model, 20 prompt tokens), 256 greedy tokens. The column "Spread" gives the slowest and the fastest of the measured runs, "Scaling" the speed divided by the thread count times the speed with one thread.

| Machine | Instruction set and kernel | Threads | Tokens/s (median) | Spread | Scaling | Runs |
|---|---|---|---|---|---|---|
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, block-scaled kernel | 1 | 18.68 | 18.55 to 18.72 | 100 % | 3 |
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, block-scaled kernel | 2 | 25.26 | 24.92 to 25.50 | 68 % | 3 |
| AMD EPYC 9V45 (2 cores, 4 threads) | AVX-512, block-scaled kernel | 4 | 26.15 | 25.98 to 26.40 | 35 % | 5 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 1 | 4.09 | 4.08 to 4.09 | 100 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 2 | 7.41 | 7.39 to 7.44 | 91 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 4 | 12.62 | 12.57 to 12.65 | 77 % | 3 |
| Azure Neoverse-N2 (8 cores) | NEON, block-scaled kernel | 8 | 19.54 | 19.51 to 19.55 | 60 % | 5 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 1 | 2.52 | 2.52 to 2.52 | 100 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 2 | 4.31 | 4.27 to 4.31 | 86 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 4 | 7.37 | 7.35 to 7.38 | 73 % | 3 |
| Intel Core i7-7700 (4 cores, 8 threads) | AVX2, block-scaled kernel | 8 | 8.22 | 8.21 to 8.22 | 41 % | 5 |

## Decode speed on GPUs

Tokens per second of the end-to-end pass (prompt, sampling and the copy of the logits included), the same prompt, 256 greedy tokens, context of 4096 tokens. The RTX 4060 Ti is a rented card in a container on a shared host; the RTX 3070 is in the desktop machine that also holds the i7-7700.

| GPU | Tokens/s (median) | Spread | Runs | Device memory at peak |
|---|---|---|---|---|
| NVIDIA GeForce RTX 3070 (8 GiB) | 111.30 | 111.28 to 111.78 | 5 | 7.64 GiB |
| NVIDIA GeForce RTX 4060 Ti (16 GiB) | 110.78 | 110.74 to 110.79 | 5 | 7.63 GiB |

## Long prompt

The long prompt has 525 tokens. The runtime still processes the prompt one token at a time (there is no batched prefill yet), but it computes the output head only for the last prompt token, and it runs the attention of the heads in parallel on the CPU, so the time to first token is below the number of prompt tokens times the time of one decode token.

| Machine | Threads | Tokens/s, standard prompt | Tokens/s, long prompt | Time to first token, long prompt |
|---|---|---|---|---|
| EPYC 9V45 | 4 | 26.15 | 25.35 | 15.7 s |
| Neoverse-N2 | 8 | 19.54 | 18.89 | 25.1 s |
| i7-7700 | 8 | 8.22 | 8.00 | 51.7 s |
| RTX 3070 | GPU | 111.30 | 109.55 | 4.1 s |
| RTX 4060 Ti | GPU | 110.78 | 109.51 | 3.5 s |

## Measurement conditions

- Greedy decoding (`--bench` is greedy and does not stop at the end-of-sequence token), exactly 256 generated tokens, one warm-up run and 5 measured runs (3 for the thread counts below the maximum and for the long prompt) inside one process; the table shows the median.
- Every configuration (model, thread count, prompt) runs in its own freshly started process, and the configurations run one after another, not interleaved. All machines run Linux. The EPYC and Neoverse-N2 machines are virtual machines and the RTX 4060 Ti is a rented container, all on shared hosts; the RTX 3070 and the i7-7700 belong to one desktop machine.

| Machine | CPU | Kernel | libc | Rust compiler | `RUSTFLAGS` |
|---|---|---|---|---|---|
| AMD EPYC 9V45 (2 cores, 4 threads) | AMD EPYC 9V45 96-Core Processor | Linux 7.0.0-1017-azure | 2.39 | rustc 1.99.1 | `-C target-cpu=znver4` |
| Azure Neoverse-N2 (8 cores) | Neoverse-N2 | Linux 7.0.0-1017-azure | 2.39 | rustc 1.99.1 | `-C target-cpu=neoverse-n2` |
| Intel Core i7-7700 (4 cores, 8 threads) | Intel(R) Core(TM) i7-7700 CPU @ 3.60GHz | Linux 6.12.115+deb13.1-amd64 | 2.41 | rustc 1.99.1 | `-C target-cpu=x86-64-v3` |

All results of this page come from binaries built from the same sources. The CPU binaries were built with the `RUSTFLAGS` of the table above; a build without them reached 90 % to 104 % of that speed on the machines where both were measured.

The prompts are in [`prompts/`](./prompts/standard.txt) next to this page.

## Reproduce

Build the command-line tool in the `rust` directory of this repository. The CPU figures above were measured with the `RUSTFLAGS` shown for each machine in the table of the measurement conditions (for example `-C target-cpu=znver4` on the EPYC 9V45); use the value for your CPU or `-C target-cpu=native`. A GPU needs the feature `cuda` and the CUDA Toolkit.

```bash
cd rust
RUSTFLAGS="-C target-cpu=native" cargo build --release -p quintrit-runtime --bin quintrit   # add --features cuda for a GPU

# standard prompt: the figures of the tables above (1 warm-up run, then 5 measured runs in one process)
./target/release/quintrit --model path/to/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf --bench --json --chat-template \
    --prompt-file ../model_benchmarks/prompts/standard.txt --max-tokens 256 --warmup 1 --repeat 5 --threads 4

# long prompt (3 measured runs on a CPU, 5 on a GPU)
./target/release/quintrit --model path/to/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf --bench --json --chat-template \
    --prompt-file ../model_benchmarks/prompts/long.txt --max-tokens 256 --warmup 1 --repeat 3 --threads 4

# thread scaling: standard prompt, 3 measured runs per thread count (up to the number of hardware threads of the machine)
for t in 1 2 4 8; do
  ./target/release/quintrit --model path/to/Ternary-Nemotron-3.5-Lightning-30B-A3B-GGUF.gguf --bench --json --chat-template \
      --prompt-file ../model_benchmarks/prompts/standard.txt --max-tokens 256 --warmup 1 --repeat 3 --threads $t
done
```

The block-scaled kernel of the machine is chosen automatically; for a GPU replace `--threads N` with `--device cuda`. The last line of the output starts with `QUINTRIT_BENCH_JSON` and holds every run, the median, minimum and maximum, the kernel and the thread count.

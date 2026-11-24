# Model benchmarks

How fast the `quintrit` runtime decodes ternary language models on four CPUs and two NVIDIA GPUs. Every model has its own page with the thread scaling, the long-prompt figures, the measurement conditions and the command to reproduce it.

## Decode speed

Generated tokens per second (median), standard prompt (`What is ternary computing?` in the chat template of each model), 256 greedy tokens. CPU columns: all hardware threads of the machine (8 on the i7-7700 and the Neoverse-N2 machine, 4 on the EPYC machine and the Raspberry Pi 5), kernel `quintrit` (AVX2 and NEON), `quintrit_512` (AVX-512) or, for the block-scaled `Q2_0` weights, the block-scaled kernel of the instruction set. GPU columns: the end-to-end pass of `quintrit --bench --device cuda`. A dash means that the model does not fit into the memory of the machine.

| Model | Size | AVX2<br>i7-7700 | AVX-512<br>EPYC 9V45 | NEON<br>Raspberry Pi 5 | NEON<br>Neoverse-N2 | GPU<br>RTX 3070 | GPU<br>RTX 4060 Ti 16 GB |
|---|---|---|---|---|---|---|---|
| [BitNet b1.58 2B4T](./bitnet-b1.58-2b-4t.md) | 2B | 22.58 | 37.80 | 9.55 | 27.88 | 233.74 | 189.01 |
| [Falcon3 1.58bit 1B](./falcon3-1b-instruct-1.58bit.md) | 1B | 31.79 | 51.52 | 13.95 | 43.96 | 361.55 | 284.12 |
| [Falcon3 1.58bit 3B](./falcon3-3b-instruct-1.58bit.md) | 3B | 19.81 | 31.69 | 7.64 | 26.65 | 236.48 | 180.30 |
| [Falcon3 1.58bit 7B](./falcon3-7b-instruct-1.58bit.md) | 7B | 11.67 | 18.50 | 4.64 | 16.19 | 152.42 | 112.43 |
| [Falcon3 1.58bit 10B](./falcon3-10b-instruct-1.58bit.md) | 10B | 9.01 | 14.61 | 3.56 | 12.26 | 118.07 | 87.86 |
| [Falcon-E 1B](./falcon-e-1b-instruct.md) | 1B | 46.33 | 78.94 | 20.59 | 52.09 | 477.60 | 415.88 |
| [Falcon-E 3B](./falcon-e-3b-instruct.md) | 3B | 29.50 | 50.04 | 12.96 | 34.23 | 325.20 | 279.33 |
| [Llama3 8B 1.58](./llama3-8b-1.58-100b-tokens.md) | 8B | 10.49 | 16.84 | 4.21 | 14.13 | 132.15 | 97.79 |
| [Ternary Nemotron 3.5 Lightning](./ternary-nemotron-3.5-lightning-30b-a3b.md) | 30B, 3B active | 8.22 | 26.15 | - | 19.54 | 111.30 | 110.78 |
| [Ternary Bonsai 2 2B](./ternary-bonsai-2-2b.md) | 2B | 15.17 | 57.50 | 12.11 | 28.45 | 244.24 | 265.56 |
| [Ternary Bonsai 2 4B](./ternary-bonsai-2-4b.md) | 4B | 7.03 | 27.97 | 5.40 | 14.23 | 137.95 | 144.45 |
| [Ternary Bonsai 2 9B](./ternary-bonsai-2-9b.md) | 9B | 4.23 | 16.58 | 2.96 | 9.13 | 77.42 | 79.84 |
| [Ternary Bonsai 2 27B](./ternary-bonsai-2-27b.md) | 27B | 1.36 | 5.46 | - | 3.03 | 23.77 | 29.87 |
| [Ternary Bonsai 1.7B](./ternary-bonsai-1.7b.md) | 1.7B | 19.66 | 68.04 | 15.08 | 39.18 | 293.62 | 315.34 |
| [Ternary Bonsai 4B](./ternary-bonsai-4b.md) | 4B | 9.18 | 34.63 | 6.69 | 18.78 | 157.84 | 170.33 |
| [Ternary Bonsai 8B](./ternary-bonsai-8b.md) | 8B | 5.26 | 18.90 | 3.35 | 11.13 | 91.61 | 92.64 |
| [Ternary Bonsai 27B](./ternary-bonsai-27b.md) | 27B | 1.47 | 5.54 | - | 3.20 | 24.18 | 30.30 |

## Method in short

- One warm-up run and 5 measured runs of 256 tokens per configuration (3 for the long prompt and for the thread sweeps), inside one process; the table shows the median.
- Models are added one at a time. A new model is measured on every machine it fits on, and the configurations run one after another, not interleaved.
- The long prompt has about 500 tokens. The runtime processes a prompt one token at a time (no batched prefill yet) but computes the output head only for its last token; see the model pages for the figures.
- The CPU binaries are built with the compiler flags of each machine (shown on every model page).

The prompts are in [`prompts/`](./prompts/standard.txt).

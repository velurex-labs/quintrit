"""Runtime level in Python on an NVIDIA GPU: load a model onto the first supported GPU and stream generated text.

    python generate_gpu.py model.gguf ["prompt"]

Needs a build of the package with GPU support (``maturin develop --release --features cuda``) and an NVIDIA GPU of compute capability 8.0
or newer. The text is the same as the CPU produces, bit for bit.
"""

import sys

import quintrit


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} model.gguf [\"prompt\"]", file=sys.stderr)
        return 2

    prompt = sys.argv[2] if len(sys.argv) > 2 else "<|user|>\nWhat is ternary computing?\n<|assistant|>\n"

    if not quintrit.cuda_built():
        print("this build of quintrit has no GPU support (build it with the cargo feature `cuda`)")
        return 0
    gpus = [g for g in quintrit.cuda_devices() if g["supported"]]
    for g in quintrit.cuda_devices():
        print(f"GPU {g['index']}: {g['name']}, compute capability {g['compute_capability']}, {g['memory_bytes'] >> 20} MiB"
              + ("" if g["supported"] else " (not supported: needs 8.0 or newer)"))
    if not gpus:
        print("no supported NVIDIA GPU found")
        return 0

    try:
        # device="cuda" is GPU 0, "cuda:1" the second one. threads and kernel do not apply on a GPU.
        model = quintrit.Model.load(sys.argv[1], device=f"cuda:{gpus[0]['index']}")
        print(f"{model.architecture} model, {model.n_layers} layers, on {model.device}, context window {model.context_window}")
        stream = model.stream(prompt, max_tokens=64)
        for piece in stream:
            print(piece, end="", flush=True)
        print(f"\n[{stream.n_tokens} generated]")
        return 0
    except quintrit.QuintritError as e:
        print(f"quintrit error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

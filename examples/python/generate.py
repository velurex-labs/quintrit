"""Runtime level in Python: load a ternary GGUF model and stream generated text to the terminal.

    python generate.py model.gguf ["prompt"]
"""

import sys

import quintrit


def main() -> int:
    if len(sys.argv) < 2:
        print(f"usage: {sys.argv[0]} model.gguf [\"prompt\"]", file=sys.stderr)
        return 2

    prompt = sys.argv[2] if len(sys.argv) > 2 else "<|user|>\nWhat is ternary computing?\n<|assistant|>\n"

    try:
        model = quintrit.Model.load(sys.argv[1])
        print(f"{model.architecture} model, {model.n_layers} layers, {model.kernel} kernel")

        prompt_tokens = len(model.tokenizer.encode(prompt))

        # Stream generated tokens
        stream = model.stream(prompt, max_tokens=64)
        for piece in stream:
            print(piece, end="", flush=True)

        print(f"\n[{prompt_tokens} prompt tokens, {stream.n_tokens} generated]")
        return 0
    except quintrit.QuintritError as e:
        print(f"quintrit error: {e}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())

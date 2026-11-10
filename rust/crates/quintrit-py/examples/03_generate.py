"""Runtime level: load a ternary GGUF model, tokenize, generate and stream text.

    python examples/03_generate.py path/to/ggml-model-i2_s.gguf ["prompt"]
"""

import sys

import quintrit

if len(sys.argv) < 2:
    sys.exit(__doc__)

model = quintrit.Model.load(sys.argv[1])  # threads / kernel are chosen automatically
print(model)

tokenizer = model.tokenizer
prompt = sys.argv[2] if len(sys.argv) > 2 else "<|user|>\nWhat is ternary computing?\n<|assistant|>\n"
ids = tokenizer.encode(prompt)
print(f"{len(ids)} prompt tokens; round trip ok: {tokenizer.decode(ids) == prompt}")

# Greedy decoding is deterministic.
print(model.generate(prompt, max_tokens=48))

# Streaming: an iterator over text pieces; Ctrl-C interrupts it.
print("--- streaming ---")
stream = model.stream(prompt, max_tokens=48, temperature=0.8, top_k=40, top_p=0.9, seed=1)
for piece in stream:
    print(piece, end="", flush=True)
print(f"\n[{stream.n_tokens} tokens, finished: {stream.finish_reason}]")

# Stop sequences end the text before the first match.
print("--- stop sequence ---")
print(model.generate(prompt, max_tokens=200, stop=["\n\n"]))

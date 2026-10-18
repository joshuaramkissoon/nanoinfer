"""Generate reference outputs from Hugging Face transformers for nanoinfer's tests.

Writes tests/fixtures/:
  reference.json    prompts, HF token ids, greedy continuations, tokenizer cases
  logits_<i>.f32    last-position logits for prompt i (raw little-endian f32, n_vocab floats)

Run (no global torch needed):
  uv run --with torch --with transformers --with numpy python scripts/gen_reference.py
"""

import json
import pathlib

import numpy as np
import torch
from transformers import GPT2LMHeadModel, GPT2TokenizerFast

ROOT = pathlib.Path(__file__).resolve().parent.parent
WEIGHTS = ROOT / "weights" / "gpt2"
OUT = ROOT / "tests" / "fixtures"

PROMPTS = [
    "Hello, my name is",
    "The quick brown fox jumps over the lazy dog.",
    "In 1905, Albert Einstein published",
    "def fibonacci(n):\n    if n < 2:\n        return n\n",
]
GREEDY = [(0, 24), (2, 24)]  # (prompt index, new tokens)

TOKENIZER_CASES = [
    "Hello world",
    "Hello, world! How's it going? I'll be there; we've done it, they're ready.",
    "  leading spaces and   multiple   gaps  ",
    "newlines\n\nand\ttabs \n mixed\r\n",
    "numbers 12345 and 3.14159 and 1,000,000",
    "naïve café résumé — “quotes” and ‘apostrophes’",
    "日本語のテキスト and 中文 and 한국어",
    "emoji 🤖🚀 and 👩‍💻 zwj",
    "<|endoftext|>after special",
    "CamelCaseIdentifier snake_case_name kebab-case-name",
    "def f(x):\n    return x**2  # square\n",
    "'s 't 're 've 'm 'll 'd 'S 'T",
    "",
]


def main() -> None:
    torch.manual_seed(0)
    tok = GPT2TokenizerFast.from_pretrained(WEIGHTS)
    model = GPT2LMHeadModel.from_pretrained(WEIGHTS, torch_dtype=torch.float32, attn_implementation="eager")
    model.eval()
    OUT.mkdir(parents=True, exist_ok=True)

    prompts = []
    with torch.no_grad():
        for i, p in enumerate(PROMPTS):
            ids = tok(p)["input_ids"]
            logits = model(torch.tensor([ids])).logits[0, -1].numpy().astype("<f4")
            (OUT / f"logits_{i}.f32").write_bytes(logits.tobytes())
            prompts.append({
                "text": p,
                "ids": ids,
                "logits_file": f"logits_{i}.f32",
                "argmax": int(np.argmax(logits)),
            })

        greedy = []
        for pi, n in GREEDY:
            ids = prompts[pi]["ids"]
            out = model.generate(
                torch.tensor([ids]),
                attention_mask=torch.ones(1, len(ids), dtype=torch.long),
                max_new_tokens=n,
                do_sample=False,
                pad_token_id=tok.eos_token_id,
            )[0].tolist()
            new = out[len(ids):]
            greedy.append({"prompt": pi, "new_ids": new, "text": tok.decode(new)})

    cases = [{"text": t, "ids": tok(t)["input_ids"]} for t in TOKENIZER_CASES]
    ref = {
        "model": "openai-community/gpt2",
        "generated_by": "scripts/gen_reference.py",
        "torch": torch.__version__,
        "n_vocab": model.config.vocab_size,
        "prompts": prompts,
        "greedy": greedy,
        "tokenizer": cases,
    }
    (OUT / "reference.json").write_text(json.dumps(ref, indent=1, ensure_ascii=False) + "\n")
    for g in greedy:
        print(repr(PROMPTS[g["prompt"]]), "->", repr(g["text"]))
    print("wrote", OUT)


if __name__ == "__main__":
    main()

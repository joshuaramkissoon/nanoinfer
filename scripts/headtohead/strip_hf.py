"""Copy an HF GPT-2 dir, dropping the causal-mask buffers (h.*.attn.bias /
masked_bias) that llama.cpp's convert_hf_to_gguf.py refuses to map.
usage: strip_hf.py SRC_DIR DST_DIR"""
import os, shutil, sys
from safetensors.numpy import load_file, save_file

src, dst = sys.argv[1], sys.argv[2]
os.makedirs(dst, exist_ok=True)
for f in ("config.json", "tokenizer.json", "vocab.json", "merges.txt"):
    shutil.copy(os.path.join(src, f), dst)
t = load_file(os.path.join(src, "model.safetensors"))
keep = {k: v for k, v in t.items() if not (k.endswith(".attn.bias") or k.endswith(".attn.masked_bias"))}
print(f"kept {len(keep)}/{len(t)} tensors")
save_file(keep, os.path.join(dst, "model.safetensors"), metadata={"format": "pt"})

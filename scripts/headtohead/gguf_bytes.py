"""Decode bytes/token of each GGUF: every tensor except position_embd (one row
read per token); the tied token_embd counts once, as the LM head. Same
convention as nanoinfer's decode_bytes. usage: gguf_bytes.py FILE... -> JSON"""
import json, sys
from gguf import GGUFReader
out = {}
for f in sys.argv[1:]:
    r = GGUFReader(f)
    tot, types = 0, {}
    for t in r.tensors:
        if t.name.startswith("position_embd"):
            continue
        tot += int(t.n_bytes)
        if t.name == "token_embd.weight" or t.name.startswith("blk.0.ffn_up.weight"):
            types[t.name] = t.tensor_type.name
    out[f.rsplit("/", 1)[-1].removesuffix(".gguf")] = {"bytes": tot, "types": types}
print(json.dumps(out))

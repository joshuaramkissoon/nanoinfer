"""MLX (mlx-lm) GPT-2 bench: prompt 64 + 64 greedy tokens, batch 1, via
mlx_lm.stream_generate (its normal async-eval pipeline). Prints one JSON line.
usage: bench_mlx.py MODEL_DIR PROMPT.json LABEL [--q4-q8head] [--runs N]"""
import argparse, json, statistics, time
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load, stream_generate
from mlx_lm.sample_utils import make_sampler

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("prompt"); ap.add_argument("label")
ap.add_argument("--q4-q8head", action="store_true", help="quantise blocks to 4-bit g32 and the tied wte/LM head to 8-bit g32 in memory")
ap.add_argument("--runs", type=int, default=7)
ap.add_argument("--gen", type=int, default=64)
a = ap.parse_args()

model, tok = load(a.model)
if a.q4_q8head:
    def pred(path, m):
        if not hasattr(m, "to_quantized"):
            return False
        return {"group_size": 32, "bits": 8} if path.endswith("wte") else {"group_size": 32, "bits": 4}
    nn.quantize(model, group_size=32, bits=4, class_predicate=pred)
    mx.eval(model.parameters())
prompt = json.load(open(a.prompt))
sampler = make_sampler(temp=0.0)
pp, tg, ntoks = [], [], []
for i in range(1 + a.runs):
    last = None
    for r in stream_generate(model, tok, prompt, max_tokens=a.gen + 1, sampler=sampler):
        last = r
    if i == 0:
        continue  # warmup
    pp.append(last.prompt_tps); tg.append(last.generation_tps); ntoks.append(last.generation_tokens)
nbytes = sum(v.nbytes for _, v in __import__("mlx.utils", fromlist=["tree_flatten"]).tree_flatten(model.parameters()))
print(json.dumps({"engine": "mlx", "label": a.label, "mlx": mx.__version__, "prompt_len": len(prompt),
                  "gen_tokens": ntoks[-1] - 1, "runs": a.runs,
                  "prefill_tok_s": statistics.median(pp), "decode_tok_s": statistics.median(tg),
                  "prefill_samples": pp, "decode_samples": tg, "param_bytes": nbytes}))

"""PyTorch MPS reference: HF transformers GPT2LMHeadModel, prompt 64 + 64
greedy decode steps with the KV cache, argmax on device, one sync per phase.
usage: bench_torch.py HF_DIR PROMPT.json DTYPE(float32|float16) [--runs N]"""
import argparse, json, statistics, time
import torch
from transformers import GPT2LMHeadModel

ap = argparse.ArgumentParser()
ap.add_argument("model"); ap.add_argument("prompt"); ap.add_argument("dtype")
ap.add_argument("--runs", type=int, default=7); ap.add_argument("--gen", type=int, default=64)
a = ap.parse_args()
dev = torch.device("mps")
m = GPT2LMHeadModel.from_pretrained(a.model, torch_dtype=getattr(torch, a.dtype)).to(dev).eval()
ids = torch.tensor([json.load(open(a.prompt))], device=dev)
pp, tg = [], []
with torch.inference_mode():
    for i in range(1 + a.runs):
        torch.mps.synchronize(); t0 = time.perf_counter()
        out = m(ids, use_cache=True)
        nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        past = out.past_key_values
        torch.mps.synchronize(); t1 = time.perf_counter()
        for _ in range(a.gen):
            out = m(nxt, past_key_values=past, use_cache=True)
            past = out.past_key_values
            nxt = out.logits[:, -1].argmax(-1, keepdim=True)
        torch.mps.synchronize(); t2 = time.perf_counter()
        if i:
            pp.append(ids.shape[1] / (t1 - t0)); tg.append(a.gen / (t2 - t1))
print(json.dumps({"engine": "torch-mps", "label": a.dtype, "torch": torch.__version__, "prompt_len": ids.shape[1],
                  "gen_tokens": a.gen, "runs": a.runs, "prefill_tok_s": statistics.median(pp),
                  "decode_tok_s": statistics.median(tg), "prefill_samples": pp, "decode_samples": tg}))

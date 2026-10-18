"""Fold $H2H/raw/round*.jsonl (+ gguf_bytes.json, ppl.json) into bench/headtohead.json.
Per config: median of the reps inside each round, then the best round (same
rule for every engine). usage: aggregate.py H2H_DIR OUT.json"""
import glob, json, os, statistics, sys

h2h, out_path = sys.argv[1], sys.argv[2]
SPEC = 120e9
gb = json.load(open(os.path.join(h2h, "gguf_bytes.json")))
C, E, V = 768, 12, 50257
small = 124_439_808 - 1024 * C - (12 * C * C * E + V * C)  # biases + LN params (f32)
def fmt_bytes(bpw_blocks, bpw_head):
    return (12 * C * C * E) * bpw_blocks / 8 + V * C * bpw_head / 8 + small * 4
MLX_BYTES = {"f32": fmt_bytes(32, 32), "f16": fmt_bytes(16, 16) - small * 2, "q8g32": fmt_bytes(9, 9) - small * 2,
             "q4g32": fmt_bytes(5, 5) - small * 2, "q4g32-q8head": fmt_bytes(5, 9) - small * 2}
TORCH_BYTES = {"float32": fmt_bytes(32, 32), "float16": fmt_bytes(16, 16) - small * 2}
FAMILY = {  # what each config is, for grouping
    "gpt2-f32": "f32", "gpt2-f16": "f16", "gpt2-q8_0": "q8", "gpt2-q4_0-q8head": "q4+q8head", "gpt2-q4_0-default": "q4+q6Khead",
    "metal-mma": "f32", "metal-pipelined-f16": "f16", "metal-pipelined-q8": "q8", "metal-lanes": "q4+q8head",
    "webgpu-kv16": "q8", "webgpu-vec": "q8",
    "fast-ops": "f32", "f16": "f16", "q8-kv16": "q8", "q4-kv16": "q4+f16head(rescored)", "q8-prefill": "q8",
    "mlx-f32": "f32", "mlx-f16": "f16", "mlx-q8g32": "q8", "mlx-q4g32": "q4 (q4 head)", "mlx-q4g32-q8head": "q4+q8head",
    "torch-float32": "f32", "torch-float16": "f16",
}

acc = {}  # key -> {"pp": [round medians], "tg": [...], meta}
for f in sorted(glob.glob(os.path.join(h2h, "raw", "round*.jsonl"))) + glob.glob(os.path.join(h2h, "raw", "warm.jsonl")):
    for line in open(f):
        d = json.loads(line)
        if d["engine"] == "llama.cpp":
            dev = {"gpu": "metal", "gpu-warm": "metal", "cpu-blas": "cpu (brew build, Accelerate BLAS for prefill)", "cpu-noblas": "cpu (NEON only)"}[d["tag"]]
            thr = 0 if d["tag"].startswith("gpu") else d["n_threads"]
            key = ("llama.cpp", dev, d["model_file"], thr)
            meta = {"bytes": gb[d["model_file"]]["bytes"], "build": d["build_commit"], "head_type": gb[d["model_file"]]["types"]["token_embd.weight"]}
            ph, v = ("pp", statistics.median(d["samples_ts"])) if d["n_prompt"] else ("tg", statistics.median(d["samples_ts"]))
            a = acc.setdefault(key, {"pp": [], "tg": [], **meta}); a[ph].append(v); a.setdefault("load1", []).append(d["load1"])
        elif d["engine"] == "nanoinfer":
            st = d["stage"]
            dev = "metal" if st.startswith("metal") else "webgpu (wgpu native, Metal backend)" if st.startswith("webgpu") else "cpu"
            key = ("nanoinfer", dev, st, d["threads"])
            by = SPEC / d["decode_ceiling_tok_s"]
            if st == "q8-prefill":  # Stage::head_format misses Q8Prefill (decodes like q8-kv16: Q8 approx head + f16 rescore)
                by = 139.5e6
            a = acc.setdefault(key, {"pp": [], "tg": [], "bytes": by, "build": d["git_sha"]})
            a["pp"].append(d["prefill_tok_s"]); a["tg"].append(d["decode_tok_s"]); a.setdefault("load1", []).append(d["load1"])
        else:
            eng, lab = d["engine"], d["label"]
            key = (eng, "metal (MLX)" if eng == "mlx" else "mps", f"{'mlx' if eng == 'mlx' else 'torch'}-{lab}", 0)
            by = MLX_BYTES[lab] if eng == "mlx" else TORCH_BYTES[lab]
            a = acc.setdefault(key, {"pp": [], "tg": [], "bytes": by, "build": d.get("mlx") or d.get("torch")})
            a["pp"].append(d["prefill_tok_s"]); a["tg"].append(d["decode_tok_s"]); a.setdefault("load1", []).append(d["load1"])

rows = []
for (eng, dev, cfg, thr), a in acc.items():
    tg, pp = max(a["tg"]), max(a["pp"])
    rows.append({"engine": eng, "device": dev, "config": cfg, "family": FAMILY.get(cfg, "?"), "threads": thr or None,
                 "prefill_tok_s": round(pp), "decode_tok_s": round(tg, 1),
                 "prefill_rounds": [round(x) for x in a["pp"]], "decode_rounds": [round(x, 1) for x in a["tg"]],
                 "bytes_per_token_mb": round(a["bytes"] / 1e6, 1), "spec_ceiling_tok_s": round(SPEC / a["bytes"]),
                 "pct_spec_ceiling": round(100 * tg * a["bytes"] / SPEC, 1), "decode_gbs": round(tg * a["bytes"] / 1e9, 1),
                 "version": a.get("build"), **({"head_type": a["head_type"]} if "head_type" in a else {}),
                 "max_load1": max(a["load1"])})
rows.sort(key=lambda r: (r["device"].split()[0] != "metal" and r["device"] != "mps", r["family"], r["threads"] or 0, -r["decode_tok_s"]))
res = {
    "what": "GPT-2 124M head-to-head on one machine: nanoinfer vs llama.cpp vs MLX vs PyTorch MPS",
    "machine": "Apple M4 (base, 4P+6E CPU, 10-core GPU), 24 GB, macOS 26",
    "method": {"prompt_len": 64, "decode_tokens": 64, "decode_context": "64..128", "batch": 1, "greedy": True,
               "reps": "1 warmup + 7 timed per config, median; 3 rounds (whole suite, strictly sequential), best round reported",
               "llama_bench": "pp: -p 64 -n 0; tg: -p 0 -n 64 -d 64 (decode after 64 tokens of context); KV f16; -fa auto; -ngl 99 GPU, -ngl 0 -nopo 1 CPU",
               "llama_metal_warm": "llama.cpp Metal tg also re-measured after heating the GPU (warm_check.sh, 15 reps); its decode is the max over rounds and the warm check",
               "bandwidth_ceiling": "120 GB/s spec / bytes per token (weights read per decode step, position table excluded, tied head counted once)"},
    "rows": rows,
    "notes": [
        "nanoinfer CPU Q8_0/Q4_0 blocks carry an f32 scale (9/5 bpw); ggml and nanoinfer Metal/WebGPU use f16 (8.5/4.5 bpw)",
        "nanoinfer CPU q8-kv16/q4-kv16/q8-prefill: Q8 approximate LM head + f16 rescoring; q8-prefill bytes corrected to 139.5 MB",
        "llama.cpp default Q4_0 stores the tied head as Q6_K; gpt2-q4_0-q8head forces Q8_0 to match nanoinfer metal-lanes",
        "MLX default 4-bit quantises the tied head to 4 bits: ppl 37 -> 311 (see quality); mlx-q4g32-q8head is the fair Q4 row",
        "llama.cpp CPU: 'NEON only' = source build ff5a233 with GGML_BLAS=OFF GGML_METAL=OFF; brew build routes prefill to Accelerate (AMX)",
        "candle: not run, candle-transformers has no GPT-2 model (bigcode = GPT-BigCode/MQA)",
    ],
}
pp = os.path.join(h2h, "ppl.json")
if os.path.exists(pp):
    res["quality"] = json.load(open(pp))
json.dump(res, open(out_path, "w"), indent=1)
for r in rows:
    print(f'{r["engine"]:10} {r["device"][:22]:22} {r["config"]:22} {r["family"]:22} t={r["threads"] or "-":<3} pp {r["prefill_tok_s"]:>6} {r["prefill_rounds"]}  tg {r["decode_tok_s"]:>6} {r["decode_rounds"]}  {r["bytes_per_token_mb"]:>6}MB {r["pct_spec_ceiling"]:>5}%  load<={r["max_load1"]}')

"""ppl_raw.txt + ppl_mlx.json (from ppl.sh) -> one JSON quality table."""
import json, re, sys
raw, mlx = open(sys.argv[1]).read(), json.load(open(sys.argv[2]))
out = {"nanoinfer_cpu": {"method": "nanoinfer ppl, 4x512 windows, all positions, vs our f32 (unrolled)"},
       "nanoinfer_metal": {"method": "examples/quant_quality.rs, 2x512 windows token-by-token on the Metal kernels, vs metal-mma f32"},
       "llama.cpp": {"method": "llama-perplexity -c 512 --chunks 8 --kl-divergence vs its own F32 GGUF (Metal)"},
       "mlx": {"method": mlx["method"], **mlx["results"]}}
for sec in raw.split("== ")[1:]:
    head, body = sec.split("\n", 1)
    eng, name = head.split(" ", 1)
    if name.startswith("metal"):
        for line in body.splitlines():
            p = line.split()
            if len(p) == 8 and p[0] in ("f32", "f16", "q8_0", "q4_0"):
                out["nanoinfer_metal"][f"{p[0]} blocks + {p[1]} head"] = {"ppl": float(p[5]), "mean_kl": float(p[6]), "top1_pct": float(p[7].rstrip("%"))}
        f32 = out["nanoinfer_metal"]["f32 blocks + f32 head"]["ppl"]
        for k, v in out["nanoinfer_metal"].items():
            if isinstance(v, dict):
                v["dppl_pct"] = round(100 * (v["ppl"] / f32 - 1), 2)
    elif eng == "nanoinfer":
        m = re.search(r"ppl ([\d.]+) over (\d+).*?vs unrolled: ppl ([\d.]+) \(Δppl [^,]+, ([+-][\d.]+)%\) \| mean KL ([\deE.+-]+) nats \| top-1 agreement ([\d.]+)%", body, re.S)
        out["nanoinfer_cpu"][name] = {"ppl": float(m[1]), "f32_ppl": float(m[3]), "dppl_pct": float(m[4]), "mean_kl": float(m[5]), "top1_pct": float(m[6]), "tokens": int(m[2])}
    else:
        q = float(re.search(r"Mean PPL\(Q\)\s+:\s+([\d.]+)", body)[1]); b = float(re.search(r"Mean PPL\(base\)\s+:\s+([\d.]+)", body)[1])
        out["llama.cpp"][name] = {"ppl": q, "f32_ppl": b, "dppl_pct": round(100 * (q / b - 1), 2),
                                  "mean_kl": float(re.search(r"Mean\s+KLD:\s+([\d.]+)", body)[1]),
                                  "top1_pct": float(re.search(r"Same top p:\s+([\d.]+)", body)[1])}
print(json.dumps(out, indent=1))

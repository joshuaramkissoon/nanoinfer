"""MLX quality, same method as `nanoinfer ppl`: WikiText-2 test file tokenised
whole, first N windows of 512, context reset per window, every position scored;
KL(f32 || q) and top-1 agreement against MLX's own f32 model.
usage: ppl_mlx.py TEXT MLX_DIR_F32 [LABEL=DIR | LABEL=DIR:q4q8head ...] [--windows N]"""
import argparse, json, math
import mlx.core as mx
import mlx.nn as nn
from mlx_lm import load

ap = argparse.ArgumentParser()
ap.add_argument("text"); ap.add_argument("ref"); ap.add_argument("models", nargs="+")
ap.add_argument("--windows", type=int, default=4); ap.add_argument("--window", type=int, default=512)
a = ap.parse_args()

ref, tok = load(a.ref)
ids = tok.encode(open(a.text).read())
wins = [ids[i * a.window:(i + 1) * a.window] for i in range(a.windows)]
def logits_of(m):
    return [(lambda z: z - mx.logsumexp(z, axis=-1, keepdims=True))(m(mx.array([w]))[0, :-1].astype(mx.float32)) for w in wins]
ref_lp = logits_of(ref); mx.eval(ref_lp)
out = {}
for spec in [f"f32={a.ref}"] + a.models:
    label, path = spec.split("=", 1)
    q48 = path.endswith(":q4q8head"); path = path.removesuffix(":q4q8head")
    m, _ = load(path)
    if q48:
        nn.quantize(m, group_size=32, bits=4, class_predicate=lambda p, mod: hasattr(mod, "to_quantized") and ({"group_size": 32, "bits": 8} if p.endswith("wte") else {"group_size": 32, "bits": 4}))
    nll = kl = agree = n = 0.0
    nll_ref = 0.0
    for w, lp, rp in zip(wins, logits_of(m), ref_lp):
        tgt = mx.array(w[1:])
        nll += float(-mx.take_along_axis(lp, tgt[:, None], axis=1).sum())
        nll_ref += float(-mx.take_along_axis(rp, tgt[:, None], axis=1).sum())
        kl += float((mx.exp(rp) * (rp - lp)).sum())
        agree += float((lp.argmax(-1) == rp.argmax(-1)).sum()); n += len(w) - 1
    ppl, pr = math.exp(nll / n), math.exp(nll_ref / n)
    out[label] = {"ppl": round(ppl, 4), "dppl_pct": round(100 * (ppl / pr - 1), 2), "mean_kl": kl / n, "top1_pct": round(100 * agree / n, 2), "tokens": int(n)}
print(json.dumps({"engine": "mlx", "method": f"{a.windows}x{a.window} windows, all positions, vs MLX f32", "results": out}))

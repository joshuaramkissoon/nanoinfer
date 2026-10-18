#!/usr/bin/env python3
"""Precompute the small data files the Foundations (/learn) pages draw from.

Reads the real GPT-2 checkpoint (weights/gpt2/model.safetensors) and our own
benchmark logs (bench/results.jsonl, batching.json, serving.json, specdec.json,
webgpu.json) and writes
web/public/learn/data.json (~250 KB). Nothing here is invented: every vector,
neighbour list, histogram and timing comes from those files.

    nice -n 10 python3 -I web/scripts/learn-data.py
"""

import base64
import json
import struct
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
W = ROOT / "weights" / "gpt2"
OUT = ROOT / "web" / "public" / "learn" / "data.json"


def load_safetensors(path: Path, names: list[str]) -> dict[str, np.ndarray]:
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        base = 8 + n
        out = {}
        for name in names:
            meta = header[name]
            assert meta["dtype"] == "F32", meta["dtype"]
            s, e = meta["data_offsets"]
            f.seek(base + s)
            out[name] = np.frombuffer(f.read(e - s), dtype=np.float32).reshape(meta["shape"]).copy()
        return out


def key(header_names: list[str], suffix: str) -> str:
    for n in header_names:
        if n == suffix or n.endswith("." + suffix):
            return n
    raise KeyError(suffix)


def main() -> None:
    path = W / "model.safetensors"
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        names = [k for k in json.loads(f.read(n)) if k != "__metadata__"]
    k_wte = key(names, "wte.weight")
    k_fc = key(names, "h.5.mlp.c_fc.weight")
    k_qkv = key(names, "h.0.attn.c_attn.weight")
    t = load_safetensors(path, [k_wte, k_fc, k_qkv])
    wte = t[k_wte]  # [50257, 768]
    vocab: dict[str, int] = json.loads((W / "vocab.json").read_text(encoding="utf-8"))
    inv = {v: k for k, v in vocab.items()}

    # GPT-2's byte-level alphabet: 'Ġ' is a leading space.
    def text(i: int) -> str:
        return inv[i].replace("Ġ", " ").replace("Ċ", "\n")

    # ---- embeddings ------------------------------------------------------------
    groups = {
        "numbers": "one two three four five six seven eight nine ten",
        "days": "Monday Tuesday Wednesday Thursday Friday Saturday Sunday",
        "countries": "France Germany Italy Spain Japan China India Russia Canada Brazil",
        "animals": "dog cat horse cow pig sheep lion tiger bird fish",
        "colours": "red blue green yellow black white orange purple",
        "food": "bread cheese pizza rice soup apple banana coffee",
    }
    words, word_group = [], []
    for g, ws in groups.items():
        for w in ws.split():
            tok = "Ġ" + w
            if tok in vocab:
                words.append(vocab[tok])
                word_group.append(g)
    E = wte[words]
    # 2-D map: PCA of the unit-length vectors (direction is what cosine similarity
    # compares), centred, so clusters are visible
    En0 = E / np.linalg.norm(E, axis=1, keepdims=True)
    Ec = En0 - En0.mean(0)
    _, _, vt = np.linalg.svd(Ec, full_matrices=False)
    xy = Ec @ vt[:2].T
    xy = (xy - xy.min(0)) / (xy.max(0) - xy.min(0))
    # nearest neighbours by cosine over "clean" whole-word tokens
    clean = [i for i in range(len(inv)) if inv[i].startswith("Ġ") and inv[i][1:].isalpha() and len(inv[i]) > 2]
    C = wte[clean]
    Cn = C / np.linalg.norm(C, axis=1, keepdims=True)
    En = E / np.linalg.norm(E, axis=1, keepdims=True)
    sims = En @ Cn.T
    near = []
    for r, wid in enumerate(words):
        order = np.argsort(-sims[r])
        lst = []
        for j in order[:12]:
            tid = clean[j]
            if tid == wid:
                continue
            lst.append([tid, round(float(sims[r, j]), 3)])
            if len(lst) == 8:
                break
        near.append(lst)
    # int8 vectors for the barcode (scale per vector)
    vecs = []
    for v in E:
        s = float(np.abs(v).max()) / 127
        vecs.append({"s": round(s, 6), "b64": base64.b64encode(np.round(v / s).astype(np.int8).tobytes()).decode()})
    toks = sorted(set(words) | {t for lst in near for t, _ in lst})
    emb = {
        "words": words,
        "group": word_group,
        "xy": [[round(float(a), 4), round(float(b), 4)] for a, b in xy],
        "near": near,
        "vecs": vecs,
        "text": {str(i): text(i) for i in toks},
        "norm_mean": round(float(np.linalg.norm(wte, axis=1).mean()), 3),
    }

    # ---- quantization ------------------------------------------------------------
    fc = t[k_fc]  # HF Conv1D: [in=768, out=3072]
    flat = fc.ravel()
    lo, hi = np.percentile(flat, [0.05, 99.95])
    counts, edges = np.histogram(flat, bins=81, range=(float(lo), float(hi)))

    def qerr(bits: int) -> dict:
        blocks = flat[: flat.size // 32 * 32].reshape(-1, 32)
        qmax = 2 ** (bits - 1) - 1
        scale = np.abs(blocks).max(1, keepdims=True) / qmax
        scale[scale == 0] = 1
        rec = np.clip(np.round(blocks / scale), -qmax - 1, qmax) * scale
        err = rec - blocks
        return {
            "bits": bits,
            "rmse": float(np.sqrt((err**2).mean())),
            "rel": float(np.sqrt((err**2).mean() / (blocks**2).mean())),
            "bytesPerWeight": (bits * 32 + 16) / 8 / 32,
        }

    # a real block of 32 consecutive weights (one row of the transposed matrix, as the
    # engine stores it), picked to contain a visible outlier
    rows = fc.T  # [3072, 768] = engine layout
    best = None
    for r in range(0, 3072, 7):
        for c in range(0, 768, 32):
            b = rows[r, c : c + 32]
            ratio = np.abs(b).max() / (np.abs(b).mean() + 1e-9)
            if 4.0 < ratio < 6.0:
                best = (r, c)
                break
        if best:
            break
    r, c = best or (0, 0)
    block = rows[r, c : c + 32]
    plain_block = rows[1, 0:32]
    quant = {
        "matrix": "h.5.mlp.c_fc (768 × 3072)",
        "n": int(flat.size),
        "hist": {"lo": float(lo), "hi": float(hi), "counts": counts.tolist()},
        "std": float(flat.std()),
        "absmax": float(np.abs(flat).max()),
        "block": {"row": int(r), "col": int(c), "w": [round(float(x), 6) for x in block]},
        "block2": {"row": 1, "col": 0, "w": [round(float(x), 6) for x in plain_block]},
        "err": [qerr(b) for b in (8, 6, 4, 3, 2)],
    }

    # ---- matmul toy: a real 8x8 corner of the QKV matrix -------------------------
    qkv = t[k_qkv]
    corner = qkv[:8, :8].T  # [out, in] like the engine

    # ---- serving / batching (bench logs) ----------------------------------------
    bat = json.loads((ROOT / "bench" / "batching.json").read_text())
    batching = {
        "machine": bat.get("machine"),
        "series": [
            {"label": s["label"], "points": [[p["batch"], round(p["agg_tok_s"]), round(p["per_stream_tok_s"], 1), round(p["step_ms"], 3)] for p in s["points"]]}
            for s in bat["series"]
            if s["label"] in ("f32", "f16", "q8_0", "q4_0")
        ],
    }
    srv = json.loads((ROOT / "bench" / "serving.json").read_text())
    serving = {}
    for rk in ("rate=16",):
        serving[rk] = {}
        for pol, x in srv["runs"][rk].items():
            if not isinstance(x, dict) or "gantt" not in x:
                continue
            g = sorted(x["gantt"], key=lambda q: q["id"])[:24]
            serving[rk][pol] = {
                "throughput": x["throughput_tok_s"],
                "ttft_p50_ms": x["ttft_ms"]["p50"],
                "ttft_p99_ms": x["ttft_ms"]["p99"],
                "makespan": x["makespan_s"],
                "gantt": [[q["arrival"], q["admitted"], q["first_token"], q["finished"], q["out_len"]] for q in g],
            }

    # ---- headline numbers for the climb, the GPU chapter and the landing page ----
    # the best untraced run at the default bench shape (prompt 64 + 64), per stage
    rows = []
    for line in (ROOT / "bench" / "results.jsonl").read_text().splitlines():
        try:
            rows.append(json.loads(line))
        except ValueError:
            pass

    def best(stage: str) -> dict:
        rs = [r for r in rows if r.get("stage") == stage and not r.get("traced") and r.get("prompt_len", 64) == 64]
        r = max(rs, key=lambda r: r["decode_tok_s"])
        return {
            "stage": stage,
            "decode": round(r["decode_tok_s"], 2),
            "prefill": round(r.get("prefill_tok_s") or 0),
            "note": r.get("note", ""),
            "when": str(r.get("timestamp", ""))[:10],
        }

    wg = json.loads((ROOT / "bench" / "webgpu.json").read_text())
    probe = wg["metal_probe"]
    metal_pair = {s["metal"]["stage"]: s["metal"] for s in wg["stages"] if s.get("metal")}
    gpu_best = best("metal-lanes")
    # one token four ways: wall ms per token from the quiet paired runs; idle is
    # estimated as (CPU waits per token) x the measured commit + wait round trip
    rt_ms = probe["commit_wait_round_trip_us"] / 1000
    steps = []
    for stage, label, waits, note in (
        ("metal-naive", "ask after every step", None, "{d} operations, and after each one the CPU waits ~{rt} µs to hear it's done"),
        ("metal-onecb", "ask once per token", 1, "same kernels, all {d} sent in one batch, one wait"),
        ("metal-par", "GPU-shaped kernels", 1, "256 threads per job instead of 1 for LayerNorm and attention"),
        ("metal-lanes", "4-bit, never wait", 0, "fewer bytes, and the next token is queued before this one finishes"),
    ):
        if stage in metal_pair:
            m = metal_pair[stage]
            tok, d = m["decode_tok_s"], m.get("dispatches_per_token", 0)
        else:
            tok, d = gpu_best["decode"], 0
        ms = 1000 / tok
        idle = min(ms * 0.9, (d if waits is None else waits) * rt_ms)
        steps.append({"label": label, "stage": stage, "tokS": round(tok, 1), "work": round(ms - idle, 3), "idle": round(idle, 3),
                      "note": note.format(d=int(d), rt=probe["commit_wait_round_trip_us"])})
    headline = {
        "naive": best("naive"),
        "gpu": gpu_best,
        "gpuReadGBs": probe["read_bandwidth_gbs"],
        "gpuSteps": steps,
        "conditions": "quiet machine (load 2-5), prompt 64 + 64 greedy tokens, best of 5-7 runs",
    }

    # ---- WebGPU, native (wgpu on Metal), paired with the Metal rung doing the same ----
    webgpu = {
        "what": wg["what"],
        "stages": [
            {"stage": s["stage"], "decode": round(s["decode_tok_s"], 1), "prefill": round(s["prefill_tok_s"]),
             "metal": s["metal"]["stage"] if s.get("metal") else None,
             "metalDecode": round(s["metal"]["decode_tok_s"], 1) if s.get("metal") else None}
            for s in wg["stages"]
        ],
    }

    # ---- serving: the timeline and paged-KV snapshots ----------------------------
    for rk in ("rate=16",):
        cont = srv["runs"][rk]["continuous"]
        serving[rk]["kv"] = {
            "pages": srv["runs"][rk]["workload"]["kv_pages"],
            "pageTokens": 16,
            "stats": cont["kv_stats"],
            "staticPeak": srv["runs"][rk]["static"]["kv_pages_peak"],
            "contPeak": cont["kv_pages_peak"],
            "snapshots": [{"iter": sn["iter"], "seqs": [[q["id"], q["len"], q["pages"]] for q in sn["seqs"]]} for sn in cont["kv_snapshots"]],
        }
        serving[rk]["note"] = srv["runs"][rk].get("note", "")
        serving[rk]["requests"] = srv["runs"][rk]["workload"]["requests"]
        serving[rk]["maxBatch"] = srv["runs"][rk]["workload"]["max_batch"]

    # ---- speculative decoding --------------------------------------------------------
    sd = json.loads((ROOT / "bench" / "specdec.json").read_text())
    spec = {}
    for pk, pr in sd["pairs"].items():
        prompts = []
        for q in pr["prompts"]:
            k4 = next(e for e in q["per_k"] if e["k"] == 4)
            top = max(q["per_k"], key=lambda e: e["speedup_vs_target"])
            prompts.append({
                "prompt": q["prompt"],
                "text": q["text"][-160:],
                "target": round(q["target_tok_s"], 1),
                "draft": round(q["draft_tok_s"]),
                "c": q["c"],
                "alpha": round(k4["alpha"], 3),
                "bestK": top["k"],
                "best": round(top["speedup_vs_target"], 2),
                "bestTokS": round(top["tok_s"]),
                "perK": [[e["k"], round(e["speedup_vs_target"], 2), round(e["alpha"], 3)] for e in q["per_k"]],
                "playback": k4["playback"][:96],
                "k4": {"speedup": round(k4["speedup_vs_target"], 2), "tokS": round(k4["tok_s"]), "perForward": round(k4["tokens_per_target_forward"], 2)},
            })
        spec[pr["target"]] = {"target": pr["target"], "targetDtype": pr["target_dtype"], "draft": pr["draft"], "draftDtype": pr["draft_dtype"],
                              "note": pr["note"], "prompts": prompts}

    data = {
        "source": "weights/gpt2/model.safetensors (OpenAI GPT-2 124M) and bench/*.json (Apple M4)",
        "emb": emb,
        "quant": quant,
        "corner": [[round(float(x), 4) for x in row] for row in corner],
        "batching": batching,
        "serving": serving,
        "headline": headline,
        "webgpu": webgpu,
        "spec": spec,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB), {len(words)} words, block at row {r} col {c}", file=sys.stderr)


if __name__ == "__main__":
    main()

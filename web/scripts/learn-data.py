#!/usr/bin/env python3
"""Precompute the small data files the Foundations (/learn) pages draw from.

Reads the real GPT-2 checkpoint (weights/gpt2/model.safetensors) and our own
benchmark logs (bench/batching.json, bench/serving.json) and writes
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
                "gantt": [[q["arrival"], q["admitted"], q["first_token"], q["finished"]] for q in g],
            }

    data = {
        "source": "weights/gpt2/model.safetensors (OpenAI GPT-2 124M) and bench/*.json (Apple M4)",
        "emb": emb,
        "quant": quant,
        "corner": [[round(float(x), 4) for x in row] for row in corner],
        "batching": batching,
        "serving": serving,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(data, separators=(",", ":")), encoding="utf-8")
    print(f"wrote {OUT} ({OUT.stat().st_size / 1e3:.0f} KB), {len(words)} words, block at row {r} col {c}", file=sys.stderr)


if __name__ == "__main__":
    main()

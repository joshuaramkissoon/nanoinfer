#!/usr/bin/env python3
"""Build bench/webgpu.json (small, for the web page) from bench/results.jsonl.

For every webgpu-* stage: the latest recorded result. For the Metal rungs
that do the same thing (pairs below): the latest result whose note matches
--metal-note. Optionally the output of `nanoinfer webgpu-probe` (--probe).

  python3 scripts/export_webgpu.py --probe /tmp/wgpu_probe.txt
"""
import argparse
import json
import re

# webgpu stage -> the Metal stage that has the same recipe (or closest).
PAIRS = {
    "webgpu-naive": "metal-naive",
    "webgpu-onesubmit": "metal-onecb",
    "webgpu-par": "metal-par",
    "webgpu-f16": "metal-f16",
    "webgpu-q8": "metal-q8",
    "webgpu-q8head": "metal-pipelined-q8",
}

KEEP = ["stage", "dtype", "kernels", "decode_tok_s", "decode_tok_s_std", "decode_gbs", "decode_ceiling_tok_s",
        "pct_of_ceiling", "prefill_tok_s", "prefill_gflops", "dispatches_per_token", "submits_per_token",
        "command_buffers_per_token", "wall_ms_per_token", "note", "timestamp", "git_sha"]


def slim(r):
    return {k: (round(v, 3) if isinstance(v, float) else v) for k, v in r.items() if k in KEEP and v is not None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="bench/results.jsonl")
    ap.add_argument("--out", default="bench/webgpu.json")
    ap.add_argument("--metal-note", default="metal-vs-webgpu-comparison")
    ap.add_argument("--probe", help="text output of `nanoinfer webgpu-probe`")
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.results) if l.strip()]
    latest = {}
    for r in rows:
        s = r.get("stage", "")
        if s.startswith("webgpu-") and r.get("prompt_len") == 64:
            latest[s] = r
    metal = {}
    for r in rows:
        if r.get("stage", "").startswith("metal-") and a.metal_note in r.get("note", ""):
            metal[r["stage"]] = r
    order = ["naive", "onesubmit", "matvec", "par", "subgroup", "fused", "f16", "q8", "argmax", "resident", "tiled",
             "ln64", "q8head"]
    stages = []
    for name in order:
        r = latest.get("webgpu-" + name)
        if not r:
            continue
        row = slim(r)
        m = metal.get(PAIRS.get(r["stage"], ""))
        if m:
            row["metal"] = slim(m)
        stages.append(row)
    probe = []
    if a.probe:
        for line in open(a.probe):
            m = re.match(r"^(.*?)\s+([\d.]+) (\S+)$", line.rstrip())
            if m and not line.startswith("adapter"):
                probe.append({"what": m.group(1).strip(), "value": float(m.group(2)), "unit": m.group(3)})
    doc = {
        "what": "WebGPU (wgpu 30) hillclimb run natively on wgpu's Metal backend, GPT-2 124M, prompt 64 + 64 "
                "greedy tokens; paired Metal rungs measured in the same session",
        "machine": rows[-1].get("machine", "Apple M4"),
        "stages": stages,
        "probe": probe,
        "metal_probe": {
            "gpu_us_per_dispatch": 0.76,
            "commit_wait_round_trip_us": 130,
            "read_bandwidth_gbs": 105,
            "source": "crates/metal/examples/probe.rs, same session",
        },
    }
    with open(a.out, "w") as f:
        json.dump(doc, f, indent=1)
        f.write("\n")
    print(f"wrote {a.out}: {len(stages)} stages, {len(probe)} probe rows")


if __name__ == "__main__":
    main()

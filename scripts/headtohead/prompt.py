"""Write the 64-token bench prompt nanoinfer uses (crates/cli/src/bench.rs
BENCH_TEXT, repeated and truncated) as JSON token ids, so every Python engine
feeds exactly the same tokens. usage: prompt.py HF_DIR OUT.json [LEN]"""
import json, sys
from transformers import AutoTokenizer

BENCH_TEXT = ("The history of computing is a story of making machines do arithmetic faster. "
              "Early computers filled rooms and performed a few thousand operations per second. ")
tok = AutoTokenizer.from_pretrained(sys.argv[1])
n = int(sys.argv[3]) if len(sys.argv) > 3 else 64
ids = []
while len(ids) < n:
    ids += tok.encode(BENCH_TEXT)
json.dump(ids[:n], open(sys.argv[2], "w"))

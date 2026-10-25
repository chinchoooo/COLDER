"""Merge results/llm/rounds/r{t}_resp_chunk*.json -> r{t}_responses.json (usage: merge_resp.py T)."""
import glob, json, sys
t = sys.argv[1]
d = "results/llm/rounds/"
out = {}
for p in sorted(glob.glob(f"{d}r{t}_resp_chunk*.json") + glob.glob(f"{d}r{t}_chunks/r{t}_resp_chunk*.json")):
    out.update(json.loads(open(p, encoding="utf-8").read(), strict=False))
json.dump(out, open(f"{d}r{t}_responses.json", "w", encoding="utf-8"))
print(len(out), "responses merged")

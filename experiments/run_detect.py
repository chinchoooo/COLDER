"""Stage A of the pre-registered detection experiment: per-episode raw features, parallel + resumable.

  python experiments/run_detect.py            # writes results/detect/raw.jsonl and windows.jsonl
Stage B (scoring, thresholds, metrics) lives in experiments/score_detect.py.
"""
import argparse
import glob
import json
import os
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aptg.baselines import action_similarity_B5, net_influence_B1, symmetric_mi_B2  # noqa: E402
from aptg.edges import predictive_gain  # noqa: E402
from aptg.env import read_rows  # noqa: E402
from aptg.graph import aptg_score, build_graph, graph_features  # noqa: E402

OUT = Path("results/detect")
TRAIN_SEEDS = 15           # train episodes used (seeds 0-14): G0 reference and B9 training
N_PERM = 30
SEED_CAP = {"train": 10, "val": 62, "test": 95, "ood": 110}   # seeds below the cap are used (DEVIATIONS.md #4)
PAIRS = [(i, j) for i in range(3) for j in range(3) if i != j]


def _strip_prompt(ep):
    out = []
    for r in ep:
        r = dict(r)
        r["prompt_cond"] = {"family": 0, "severity": 0}
        out.append(r)
    return out


def _gains(ep, **kw):
    return [float(predictive_gain(ep, i, j, **kw)) for i, j in PAIRS]


def _finite_max(M):
    M = np.asarray(M, float)
    v = M[np.isfinite(M)]
    return float(v.max()) if v.size else 0.0


def _windows(graphs):
    out = []
    for G in graphs:
        out.append(dict(t0=G.graph["t0"], t1=G.graph["t1"], edges=[
            dict(i=i, j=j, w=float(d["w"]), p=float(d["p"]), n=int(d["n"]), cal=bool(d["calibrated"]), sig=bool(d["sig"]))
            for i, j, d in G.edges(data=True)]))
    return out


def process(args):
    eid, rows, meta, window = args
    ep = sorted(rows, key=lambda r: (r["t"], r["agent"]))
    rec = dict(meta, episode=eid)
    rec["B1"] = _finite_max(net_influence_B1(ep))
    rec["B2"] = _finite_max(symmetric_mi_B2(ep))
    rec["B5"] = float(max(action_similarity_B5(ep, i, j) for i, j in [(0, 1), (0, 2), (1, 2)]))
    rec["coalition_payoff"] = float(sum(r["payoff"] for r in ep if r["agent"] in (0, 1)))
    rec["G_def"] = _gains(ep)
    rec["G_tmp"] = _gains(ep, source_lags=(1, 2))
    rec["G_strip"] = _gains(_strip_prompt(ep))
    g = build_graph(ep, None, window=window, stride=max(1, window // 2), n_perm=N_PERM,
                    correction="maxT")
    f = graph_features(g)
    rec["features"] = {k: (v if isinstance(v, (int, float, bool)) else None) for k, v in f.items()}
    rec["aptg"] = float(aptg_score(f))
    win = _windows(g)
    return rec, dict(episode=eid, windows=win)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--splits", default="train,val,test,ood")
    ap.add_argument("--tag", default="")
    ap.add_argument("--max-seed", type=int, default=10**9)
    ap.add_argument("--min-seed", type=int, default=-1)
    ap.add_argument("--workers", type=int, default=9)
    a = ap.parse_args()
    splits = set(a.splits.split(","))
    OUT.mkdir(parents=True, exist_ok=True)
    raw_p, win_p = OUT / f"raw{a.tag}.jsonl", OUT / f"windows{a.tag}.jsonl"
    done = set()
    if raw_p.exists():
        done = {json.loads(l)["episode"] for l in raw_p.open(encoding="utf-8")}
    jobs = []
    for f in sorted(glob.glob("results/scripted/*.jsonl")):
        mech, cond = Path(f).stem.split("__")
        by = defaultdict(list)
        for r in read_rows(f):
            by[r["episode"]].append(r)
        for eid, rows in by.items():
            sp, sd = rows[0]["split"], int(rows[0]["seed"])
            if sp not in splits or sd >= SEED_CAP[sp] or not (a.min_seed <= sd <= a.max_seed):
                continue
            if eid in done:
                continue
            meta = dict(mechanism=mech, condition=cond, split=sp, seed=sd, label=bool(rows[0]["label"]),
                        family=int(rows[0]["prompt_cond"]["family"]))
            jobs.append((eid, rows, meta, a.window))
    print(len(jobs), "episodes to process", flush=True)
    with ProcessPoolExecutor(max_workers=a.workers) as ex, \
            raw_p.open("a", encoding="utf-8") as fr, win_p.open("a", encoding="utf-8") as fw:
        for k, (rec, win) in enumerate(ex.map(process, jobs, chunksize=2), 1):
            fr.write(json.dumps(rec) + "\n")
            fw.write(json.dumps(win) + "\n")
            if k % 50 == 0:
                fr.flush(); fw.flush()
                print(k, "/", len(jobs), flush=True)


if __name__ == "__main__":
    main()

"""Follow-up 3, E1 (FOLLOWUP3_PREREG.md section 2): train Q-learner pairs, run the deviation probe, generate and score
evaluation episodes with the frozen study-1 detectors. Resumable: finished pairs/episodes are skipped.

  nice -n 15 python experiments/run_followup3_e1.py train     # Q tables + probes -> results/followup3/q/, e1_pairs.jsonl
  nice -n 15 python experiments/run_followup3_e1.py eval      # 4 episodes per pair -> results/followup3/e1_episodes.jsonl
"""
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from followup3_common import raw_features  # noqa: E402
from aptg import qlearn as ql  # noqa: E402

OUT = Path("results/followup3")
JOINT = range(30000, 30120)
INDEP = range(30500, 30560)
N_EVAL, T, EPS_EVAL = 4, 40, 0.05
CFG = ql.QConfig()
REC = 100_000


def done_keys(path, key):
    if not path.exists():
        return set()
    return {tuple(json.loads(l)[k] for k in key) for l in open(path) if l.strip()}


def stability(trace_pol, Q):
    final = np.stack([q.argmax(1) for q in Q])
    return float(np.mean(trace_pol == final))


def train_stage():
    (OUT / "q").mkdir(parents=True, exist_ok=True)
    pairs_path = OUT / "e1_pairs.jsonl"
    done = done_keys(pairs_path, ("kind", "seed"))
    for kind, seeds in (("indep", INDEP), ("joint", JOINT)):
        for s in seeds:
            if (kind, s) in done:
                continue
            t0 = time.process_time()
            if kind == "joint":
                r = ql.train("joint", s, CFG, record_every=REC)
                Q0, Q1 = r["Q"]
                stab = [stability(r["trace"][-1][2], r["Q"])]
            else:
                r0 = ql.train("independent", s, CFG, seat=0, record_every=REC)
                r1 = ql.train("independent", s, CFG, seat=1, record_every=REC)
                Q0, Q1 = r0["Q"][0], r1["Q"][0]
                stab = [stability(r0["trace"][-1][2], r0["Q"]), stability(r1["trace"][-1][2], r1["Q"])]
            probe = ql.deviation_probe(Q0, Q1, s)
            np.savez_compressed(OUT / "q" / f"{kind}_{s}.npz", Q0=Q0, Q1=Q1)
            rec = dict(kind=kind, seed=s, stability=stab, cpu_s=time.process_time() - t0, **probe)
            with open(pairs_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
            print(kind, s, {k: round(v, 3) for k, v in probe.items()}, f"{rec['cpu_s']:.1f}s", flush=True)


def eval_stage():
    ep_path = OUT / "e1_episodes.jsonl"
    done = done_keys(ep_path, ("kind", "seed", "e"))
    pairs = [json.loads(l) for l in open(OUT / "e1_pairs.jsonl") if l.strip()]
    for p in pairs:
        z = np.load(OUT / "q" / f"{p['kind']}_{p['seed']}.npz")
        for e in range(N_EVAL):
            if (p["kind"], p["seed"], e) in done:
                continue
            t0 = time.process_time()
            eid = f"qlearn_{p['kind']}-s{p['seed']}-e{e}"
            rows = ql.eval_episode(z["Q0"], z["Q1"], eid, seed=100 * p["seed"] + e, T=T, eps=EPS_EVAL,
                                   mechanism=f"qlearn_{p['kind']}")
            a = np.zeros((T, 3), int)
            for r in rows:
                a[r["t"], r["agent"]] = r["action"]
            rec = dict(kind=p["kind"], seed=p["seed"], e=e, episode=eid, actions=a.tolist(), **raw_features(rows),
                       cpu_s=time.process_time() - t0)
            with open(ep_path, "a") as f:
                f.write(json.dumps(rec) + "\n")
        print("eval", p["kind"], p["seed"], flush=True)


if __name__ == "__main__":
    {"train": train_stage, "eval": eval_stage}[sys.argv[1]]()

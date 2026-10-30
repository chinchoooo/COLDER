"""H5 post hoc contrast: TPR drop (clean C4 -> OOD family; seen -> held-out mechanisms, C4) with episode bootstrap CIs.

  python experiments/h5_bootstrap.py   (run from repo root)
"""
import sys, json, numpy as np
sys.path.insert(0, "experiments"); sys.path.insert(0, "src")
import score_detect as sd
rows = sd.load("results/detect/raw.jsonl"); sd.attach_windows(rows, "results/detect/windows.jsonl")
rows = sd.compute_scores(rows)
thr = json.load(open("results/detect/summary.json"))["thresholds"]
T = sd.split(rows, "test"); O = sd.split(rows, "ood")
HO = sd.HELD_OUT_MECH
def hit(R, d): return np.array([r["s"][d] > thr[d] for r in R], float)
sets = {
 "seen": [r for r in T if r["label"] and r["condition"]=="C4" and r["mechanism"] not in HO],
 "held": [r for r in T if r["label"] and r["condition"]=="C4" and r["mechanism"] in HO],
 "c4": [r for r in T if r["label"] and r["condition"]=="C4"],
 "ood": [r for r in O if r["label"]],
}
rng = np.random.default_rng(7); B = 2000
H = {k:{d:hit(v,d) for d in ["B1","B2","B5","APTG","B3"]} for k,v in sets.items()}
def drops(kind, idx):
    a,b = ("c4","ood") if kind=="fam" else ("seen","held")
    out = {}
    for d in H[a]:
        out[d] = H[a][d][idx[a]].mean() - H[b][d][idx[b]].mean()
    return out
for kind in ("fam","mech"):
    a,b = ("c4","ood") if kind=="fam" else ("seen","held")
    pt = drops(kind, {a: np.arange(len(sets[a])), b: np.arange(len(sets[b]))})
    boots = []
    for _ in range(B):
        idx = {a: rng.integers(0,len(sets[a]),len(sets[a])), b: rng.integers(0,len(sets[b]),len(sets[b]))}
        boots.append(drops(kind, idx))
    print(kind, "n", len(sets[a]), len(sets[b]))
    for d in H[a]:
        bd = np.array([x[d] for x in boots]); print(f"  drop {d}: {pt[d]:.3f} [{np.percentile(bd,2.5):.3f},{np.percentile(bd,97.5):.3f}]")
    for ref in ("B1","B2","B5"):
        diff = np.array([x["APTG"]-x[ref] for x in boots]); p = pt["APTG"]-pt[ref]
        print(f"  APTG drop - {ref} drop: {p:.3f} [{np.percentile(diff,2.5):.3f},{np.percentile(diff,97.5):.3f}]")

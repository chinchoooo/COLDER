"""Follow-up: B4 nonparametric directional CMI and union detectors (see FOLLOWUP_PREREG.md).

  python experiments/gen_scripted.py && python experiments/run_union.py   (run from repo root)
"""
import glob
import json
import sys
import warnings
import zlib
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import score_detect as sd  # noqa: E402
from aptg.baselines import N_ACT, _actions  # noqa: E402
from aptg.env import read_rows  # noqa: E402
from aptg.eval import auroc_auprc, bootstrap_ci, paired_bootstrap_diff, threshold_at_fpr, tpr_at_threshold  # noqa: E402

SEED_CAP = {"train": 10, "val": 62, "test": 95, "ood": 110}
PAIRS = [(i, j) for i in range(3) for j in range(3) if i != j]
UNIONS = {"U3": ["B1", "B2", "B3"], "U4": ["B1", "B2", "B3", "B4"], "Unp": ["B1", "B2", "B4"]}
N_SHUF = 20
N_BOOT = 1000


def cmi(x, y, z):
    n = len(x)
    t = np.zeros((N_ACT, N_ACT, N_ACT))
    np.add.at(t, (x, y, z), 1.0)
    p = t / n
    pxz, pyz, pz = p.sum(1), p.sum(0), p.sum((0, 1))
    nz = p > 0
    xi, yi, zi = np.nonzero(nz)
    return float(np.sum(p[nz] * np.log(p[nz] * pz[zi] / (pxz[xi, zi] * pyz[yi, zi]))))


def b4_score(rows, eid):
    a = _actions(sorted(rows, key=lambda r: (r["t"], r["agent"])))
    rng = np.random.default_rng(zlib.crc32(eid.encode()))
    best = -np.inf
    for i, j in PAIRS:
        x, y, z = a[:-1, i], a[1:, j], a[:-1, j]
        null = np.mean([cmi(rng.permutation(x), y, z) for _ in range(N_SHUF)])
        best = max(best, cmi(x, y, z) - null)
    return float(best)


def compute_b4():
    out = {}
    for f in sorted(glob.glob("results/scripted/*.jsonl")):
        by = defaultdict(list)
        for r in read_rows(f):
            by[r["episode"]].append(r)
        for eid, rows in by.items():
            if int(rows[0]["seed"]) < SEED_CAP[rows[0]["split"]]:
                out[eid] = b4_score(rows, eid)
    return out


def main():
    rows = sd.load("results/detect/raw.jsonl")
    sd.attach_windows(rows, "results/detect/windows.jsonl")
    rows = sd.compute_scores(rows)
    outp = Path("results/followup"); outp.mkdir(parents=True, exist_ok=True)
    b4 = compute_b4()
    json.dump(b4, open(outp / "b4.json", "w"))
    rows = [r for r in rows if r["episode"] in b4]
    for r in rows:
        r["s"]["B4"] = b4[r["episode"]]
    neg_train = [r for r in rows if r["split"] == "train" and not r["label"]]
    stats = {}
    for k in ("B1", "B2", "B3", "B4"):
        v = np.array([r["s"][k] for r in neg_train])
        med = np.median(v); mad = 1.4826 * np.median(np.abs(v - med)) or 1.0
        stats[k] = (med, mad)
    for r in rows:
        z = {k: (r["s"][k] - stats[k][0]) / stats[k][1] for k in stats}
        for u, comps in UNIONS.items():
            r["s"][u] = max(z[k] for k in comps)
    dets = ["B1", "B2", "B3", "B4", "U3", "U4", "Unp"]
    val = sd.split(rows, "val"); T = sd.split(rows, "test"); O = sd.split(rows, "ood")
    thr = {d: threshold_at_fpr(*sd.arr(val, d), 0.05) for d in dets}
    res = {"n_test": len(T), "thr": thr, "test": {}, "ood": {}, "mech": {}, "neg": {}, "paired": {}}
    warnings.simplefilter("ignore")
    for name, R in (("test", T), ("ood", O)):
        for d in dets:
            y, s = sd.arr(R, d)
            res[name][d] = dict(
                auroc=bootstrap_ci(sd.au, y, s, n=N_BOOT, seed=1)[:3],
                tpr=bootstrap_ci(lambda a, b: tpr_at_threshold(a, b, thr[d]), y, s, n=N_BOOT, seed=3)[:3],
                fpr=bootstrap_ci(lambda a, b: float(np.mean(b[a == 0] > thr[d])), y, s, n=N_BOOT, seed=4)[:3])
    for d in dets:
        res["mech"][d] = {m: float(np.mean([r["s"][d] > thr[d] for r in T if r["label"] and r["mechanism"] == m]))
                          for m in sorted({r["mechanism"] for r in T if r["label"]})}
        neg = {c: float(np.mean([r["s"][d] > thr[d] for r in T if not r["label"] and r["condition"] == c]))
               for c in sorted({r["condition"] for r in T if not r["label"]})}
        neg.update({"mech:" + m: float(np.mean([r["s"][d] > thr[d] for r in T if not r["label"] and r["mechanism"] == m]))
                    for m in sorted({r["mechanism"] for r in T if not r["label"]})})
        res["neg"][d] = neg
    y = np.array([int(r["label"]) for r in T])
    for a_, b_ in [("U3", "B1"), ("U4", "B1"), ("Unp", "B1"), ("U3", "B3"), ("U4", "U3"), ("B4", "B1"), ("B4", "B2")]:
        sa = np.array([r["s"][a_] for r in T]); sb = np.array([r["s"][b_] for r in T])
        res["paired"][f"{a_} - {b_}"] = dict(
            auroc=paired_bootstrap_diff(sd.au, y, sa, sb, n=N_BOOT, seed=5)[:3],
            tpr=paired_bootstrap_diff(lambda yy, ss: tpr_at_threshold(yy, ss, 0.0), y, sa - thr[a_], sb - thr[b_], n=N_BOOT, seed=6)[:3])
    json.dump(res, open(outp / "summary.json", "w"), indent=1)
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()

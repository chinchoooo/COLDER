"""Confirmation study (FOLLOWUP2_PREREG.md): frozen detectors/thresholds from study 1 applied once to fresh episodes.

  python experiments/gen_confirm.py && nice python experiments/run_confirm.py   (from repo root)
"""
import glob
import json
import sys
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import run_union as ru  # noqa: E402
import score_detect as sd  # noqa: E402
from aptg.baselines import net_influence_B1, symmetric_mi_B2  # noqa: E402
from aptg.edges import predictive_gain, robust_z  # noqa: E402
from aptg.env import read_rows  # noqa: E402
from aptg.eval import paired_bootstrap_diff, threshold_at_fpr, tpr_at_threshold, bootstrap_ci  # noqa: E402

PAIRS = ru.PAIRS
COLL = ("complementary", "leader_follower", "alternating", "hidden_channel", "copy_same_round", "lag2_follower", "threshold_trigger")


def fin(M):
    v = np.asarray(M, float)
    v = v[np.isfinite(v)]
    return float(v.max()) if v.size else 0.0


def process(a):
    eid, rows, meta = a
    ep = sorted(rows, key=lambda r: (r["t"], r["agent"]))
    rec = dict(meta, episode=eid)
    rec["B1"], rec["B2"] = fin(net_influence_B1(ep)), fin(symmetric_mi_B2(ep))
    rec["G_def"] = [float(predictive_gain(ep, i, j)) for i, j in PAIRS]
    rec["B4"] = ru.b4_score(rows, eid)
    return rec


def main():
    warnings.simplefilter("ignore")
    # frozen study-1 calibration
    rows = sd.load("results/detect/raw.jsonl")
    sd.attach_windows(rows, "results/detect/windows.jsonl")
    rows = sd.compute_scores(rows)
    b4 = json.load(open("results/followup/b4.json"))
    rows = [r for r in rows if r["episode"] in b4]
    for r in rows:
        r["s"]["B4"] = b4[r["episode"]]
    train = [r for r in rows if r["split"] == "train"]
    ref = [r for r in train if not r["label"] and r["condition"] == "C0"]
    G0 = np.concatenate([r["G_def"] for r in ref])
    neg = [r for r in train if not r["label"]]
    zst = {}
    for k in ("B1", "B2", "B3", "B4"):
        v = np.array([r["s"][k] for r in neg]); med = np.median(v)
        zst[k] = (med, 1.4826 * np.median(np.abs(v - med)) or 1.0)

    def union_scores(s):
        z = {k: (s[k] - zst[k][0]) / zst[k][1] for k in zst}
        return {**s, "U3": max(z["B1"], z["B2"], z["B3"]), "U4": max(z.values()), "Unp": max(z["B1"], z["B2"], z["B4"])}
    for r in rows:
        r["s"] = union_scores(r["s"])
    dets = ["B1", "B2", "B3", "B4", "U3", "U4", "Unp"]
    val = sd.split(rows, "val")
    thr = {d: threshold_at_fpr(*sd.arr(val, d), 0.05) for d in dets}

    jobs = []
    for f in sorted(glob.glob("results/confirm/*.jsonl")):
        mech, cond = Path(f).stem.split("__")
        by = defaultdict(list)
        for r in read_rows(f):
            by[r["episode"]].append(r)
        for eid, rr in by.items():
            jobs.append((eid, rr, dict(mechanism=mech, condition=cond, split=rr[0]["split"], seed=int(rr[0]["seed"]),
                                       label=bool(rr[0]["label"]))))
    print(len(jobs), "confirmation episodes", flush=True)
    with ProcessPoolExecutor(max_workers=6) as ex:
        recs = list(ex.map(process, jobs, chunksize=4))
    Path("results/confirm").mkdir(exist_ok=True)
    json.dump(recs, open("results/confirm/raw.json", "w"))
    for r in recs:
        s = {"B1": r["B1"], "B2": r["B2"], "B3": max(robust_z(g, G0) for g in r["G_def"]), "B4": r["B4"]}
        r["s"] = union_scores(s)
    T = [r for r in recs if r["split"] == "confirm"]
    O = [r for r in recs if r["split"] == "confirm_ood"]
    res = {"n_test": len(T), "n_ood": len(O), "thr": thr, "test": {}, "ood": {}, "mech": {}, "mech_ood": {}, "neg": {}, "paired": {}}
    y = np.array([int(r["label"]) for r in T])
    for d in dets:
        s = np.array([r["s"][d] for r in T])
        yo, so = sd.arr(O, d)
        res["test"][d] = dict(auroc=bootstrap_ci(sd.au, y, s, n=1000, seed=1)[:3], auprc=bootstrap_ci(sd.ap, y, s, n=1000, seed=2)[:3],
                              tpr=bootstrap_ci(lambda a, b: tpr_at_threshold(a, b, thr[d]), y, s, n=1000, seed=3)[:3],
                              fpr=bootstrap_ci(lambda a, b: float(np.mean(b[a == 0] > thr[d])), y, s, n=1000, seed=4)[:3])
        res["ood"][d] = dict(tpr=float(tpr_at_threshold(yo, so, thr[d])), fpr=float(np.mean(so[yo == 0] > thr[d])))
        for key, R in (("mech", T), ("mech_ood", O)):
            res[key][d] = {m: float(np.mean([r["s"][d] > thr[d] for r in R if r["label"] and r["mechanism"] == m])) for m in COLL}
        n = {c: float(np.mean([r["s"][d] > thr[d] for r in T if not r["label"] and r["condition"] == c]))
             for c in sorted({r["condition"] for r in T if not r["label"]})}
        n.update({"mech:" + m: float(np.mean([r["s"][d] > thr[d] for r in T if not r["label"] and r["mechanism"] == m]))
                  for m in ("independent", "shared_model", "benign_adaptation")})
        res["neg"][d] = n
    best = max(("B1", "B2", "B3", "B4"), key=lambda d: res["test"][d]["tpr"][0])
    res["best_single"] = best
    for a_, b_ in [("U3", "B1"), ("U3", best), ("U4", "U3"), ("Unp", "B1"), ("B3", "B1")]:
        sa = np.array([r["s"][a_] for r in T]); sb = np.array([r["s"][b_] for r in T])
        res["paired"][f"{a_} - {b_}"] = dict(
            auroc=paired_bootstrap_diff(sd.au, y, sa, sb, n=1000, seed=5)[:3],
            tpr=paired_bootstrap_diff(lambda yy, ss: tpr_at_threshold(yy, ss, 0.0), y, sa - thr[a_], sb - thr[b_], n=1000, seed=6)[:3])
    json.dump(res, open("results/confirm/summary.json", "w"), indent=1)
    print(json.dumps({k: res[k] for k in ("n_test", "n_ood", "best_single")}))


if __name__ == "__main__":
    main()

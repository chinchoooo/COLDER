"""Stage B: scores, validation-tuned thresholds, test/OOD metrics with episode-level bootstrap CIs.

  python experiments/score_detect.py --raw results/detect/raw.jsonl [--windows results/detect/windows.jsonl] --out results/detect/summary.json
  python experiments/score_detect.py --select-window   # compares the two validation window runs by AUROC
"""
import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aptg.baselines import FEATURE_KEYS, bernoulli_llr, sprt_B8, supervised_B9  # noqa: E402
from aptg.edges import robust_z  # noqa: E402
from aptg.eval import (auroc_auprc, bootstrap_ci, fpr_at_threshold, paired_bootstrap_diff,  # noqa: E402
                       threshold_at_fpr, tpr_at_threshold)

HELD_OUT_MECH = ("hidden_channel", "alternating")
PERTURB_NEG = ("C1", "C2", "C3", "C6")
DETECTORS = ["B1", "B2", "B3", "B3_temporal", "B3_noprompt", "APTG", "B5", "B6", "B8", "B9"]
P0, P1 = 0.05, 0.5          # SPRT per-window Bernoulli rates for "any BH-significant edge"
ALPHA, BETA = 0.05, 0.2


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def attach_windows(rows, wpath):
    if not wpath:
        return
    w = {d["episode"]: d["windows"] for d in load(wpath)}
    for r in rows:
        r["windows"] = w.get(r["episode"], [])


def compute_scores(rows):
    train = [r for r in rows if r["split"] == "train"]
    ref = [r for r in train if not r["label"] and r["condition"] == "C0"]
    G0 = {k: np.concatenate([r[k] for r in ref]) for k in ("G_def", "G_tmp", "G_strip")}
    ref_pay = float(np.mean([r["coalition_payoff"] for r in ref if r["mechanism"] == "independent"]))
    for r in rows:
        r["s"] = {
            "B1": r["B1"], "B2": r["B2"], "B5": r["B5"], "APTG": r["aptg"],
            "B3": max(robust_z(g, G0["G_def"]) for g in r["G_def"]),
            "B3_temporal": max(robust_z(g, G0["G_tmp"]) for g in r["G_tmp"]),
            "B3_noprompt": max(robust_z(g, G0["G_strip"]) for g in r["G_strip"]),
            "B6": r["coalition_payoff"] - ref_pay,
        }
        stream = [bernoulli_llr(float(any(e["cal"] and e["sig"] for e in w["edges"])), P0, P1)
                  for w in r.get("windows", [])]
        r["s"]["B8"] = float(np.sum(stream)) if stream else 0.0
        r["b8_decision"] = sprt_B8(stream, ALPHA, BETA) if stream else ("continue", 0)
    tr = [r for r in train if r["mechanism"] not in HELD_OUT_MECH]
    if len({r["label"] for r in tr}) == 2:
        m = supervised_B9([r["features"] for r in tr], [int(r["label"]) for r in tr])
        for r in rows:
            r["s"]["B9"] = m.score(r["features"])
    else:
        for r in rows:
            r["s"]["B9"] = 0.0
    return rows


def split(rows, name):
    return [r for r in rows if r["split"] == name]


def arr(rows, det):
    return np.array([int(r["label"]) for r in rows]), np.array([r["s"][det] for r in rows], float)


def au(y, s):
    return auroc_auprc(y, s)[0]


def ap(y, s):
    return auroc_auprc(y, s)[1]


def evaluate(rows, n_boot):
    val = split(rows, "val")
    out = {"thresholds": {}, "test": {}, "ood": {}, "by_mechanism_test": {}, "neg_types_test": {}, "paired_test": {}}
    thr = {}
    for d in DETECTORS:
        yv, sv = arr(val, d)
        thr[d] = threshold_at_fpr(yv, sv, 0.05)
    out["thresholds"] = thr
    for name in ("test", "ood"):
        R = split(rows, name)
        if not R:
            continue
        for d in DETECTORS:
            y, s = arr(R, d)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                a = bootstrap_ci(au, y, s, n=n_boot, seed=1)
                p = bootstrap_ci(ap, y, s, n=n_boot, seed=2)
                t = bootstrap_ci(lambda yy, ss: tpr_at_threshold(yy, ss, thr[d]), y, s, n=n_boot, seed=3)
                f = bootstrap_ci(lambda yy, ss: fpr_at_threshold(yy, ss, thr[d]), y, s, n=n_boot, seed=4)
            out[name][d] = dict(n=len(y), n_pos=int(y.sum()), auroc=a[:3], auprc=p[:3], tpr_at_val_fpr05=t[:3],
                                fpr_at_val_thr=f[:3])
    T = split(rows, "test")
    for d in DETECTORS:
        res = {}
        for m in sorted({r["mechanism"] for r in T if r["label"]}):
            y, s = arr([r for r in T if r["mechanism"] == m or not r["label"]], d)
            res[m] = float(tpr_at_threshold(y, s, thr[d]))
        out["by_mechanism_test"][d] = res
        neg = {}
        for c in sorted({r["condition"] for r in T if not r["label"]}):
            sub = [r for r in T if not r["label"] and r["condition"] == c]
            _, s = arr(sub, d)
            neg[c] = float(np.mean(s > thr[d]))
        for m in sorted({r["mechanism"] for r in T if not r["label"]}):
            sub = [r for r in T if not r["label"] and r["mechanism"] == m]
            _, s = arr(sub, d)
            neg["mech:" + m] = float(np.mean(s > thr[d]))
        out["neg_types_test"][d] = neg
    pairs = [("B3", "B2"), ("B3", "B1"), ("APTG", "B2"), ("APTG", "B3"), ("B3_temporal", "B2"), ("B3", "B3_noprompt"),
             ("APTG", "B1"), ("B9", "APTG")]
    y = np.array([int(r["label"]) for r in T])
    for a_, b_ in pairs:
        sa = np.array([r["s"][a_] for r in T], float) - thr[a_]
        sb = np.array([r["s"][b_] for r in T], float) - thr[b_]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            d_auc = paired_bootstrap_diff(au, y, np.array([r["s"][a_] for r in T]), np.array([r["s"][b_] for r in T]), n=n_boot, seed=5)
            d_tpr = paired_bootstrap_diff(lambda yy, ss: tpr_at_threshold(yy, ss, 0.0), y, sa, sb, n=n_boot, seed=6)
        out["paired_test"][f"{a_} - {b_}"] = dict(auroc_diff=d_auc[:3], tpr_diff=d_tpr[:3])
    # robustness gap: clean positives (C4, family 0) in test vs OOD (family 3)
    gap = {}
    O = split(rows, "ood")
    for d in DETECTORS:
        c4 = [r for r in T if r["label"] and r["condition"] == "C4"]
        c5 = [r for r in T if r["label"] and r["condition"] == "C5"]
        oo = [r for r in O if r["label"]]
        f = lambda R: float(np.mean([r["s"][d] > thr[d] for r in R])) if R else float("nan")
        gap[d] = dict(tpr_C4=f(c4), tpr_C5_unilateral=f(c5), tpr_ood=f(oo))
    out["robustness"] = gap
    out["b8_decision_test"] = {"H1_on_positives": float(np.mean([r["b8_decision"][0] == "H1" for r in T if r["label"]])),
                               "H1_on_negatives": float(np.mean([r["b8_decision"][0] == "H1" for r in T if not r["label"]])),
                               "median_stop_t_positives": float(np.median([r["b8_decision"][1] for r in T
                                                                           if r["label"] and r["b8_decision"][0] == "H1"] or [np.nan]))}
    return out


def select_window():
    res = {}
    for w in (10, 20):
        rows = load(f"results/detect/raw_val_w{w}.jsonl")
        y = np.array([int(r["label"]) for r in rows])
        s = np.array([r["aptg"] for r in rows])
        res[w] = dict(auroc=auroc_auprc(y, s)[0], auprc=auroc_auprc(y, s)[1], n=len(rows),
                      frac_nonzero=float(np.mean(s > 0)))
    print(json.dumps(res, indent=1))
    best = max(res, key=lambda k: res[k]["auroc"])
    print("selected window:", best)
    Path("results/detect/window_selection.json").write_text(json.dumps(dict(results=res, selected=best)))


def main():
    ap_ = argparse.ArgumentParser()
    ap_.add_argument("--raw", nargs="+")
    ap_.add_argument("--windows", nargs="*", default=[])
    ap_.add_argument("--out", default="results/detect/summary.json")
    ap_.add_argument("--boot", type=int, default=1000)
    ap_.add_argument("--select-window", action="store_true")
    a = ap_.parse_args()
    if a.select_window:
        return select_window()
    rows = [r for p in a.raw for r in load(p)]
    for wp in a.windows:
        w = {d["episode"]: d["windows"] for d in load(wp)}
        for r in rows:
            if r["episode"] in w:
                r["windows"] = w[r["episode"]]
    rows = compute_scores(rows)
    res = evaluate(rows, a.boot)
    res["n_episodes"] = {s: len(split(rows, s)) for s in ("train", "val", "test", "ood")}
    Path(a.out).write_text(json.dumps(res, indent=1))
    for name in ("test", "ood"):
        print(name)
        for d, v in res.get(name, {}).items():
            print(f"  {d:12s} AUROC {v['auroc'][0]:.3f} [{v['auroc'][1]:.3f},{v['auroc'][2]:.3f}]  "
                  f"TPR@val-thr {v['tpr_at_val_fpr05'][0]:.3f}  FPR {v['fpr_at_val_thr'][0]:.3f}")


if __name__ == "__main__":
    main()

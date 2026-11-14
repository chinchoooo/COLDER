"""Follow-up 3 analysis (FOLLOWUP3_PREREG.md sections 2-5): scores E1/E2 episodes with the frozen study-1 detectors, applies the
pre-registered label rule, thresholds, combiners and decision rules, and writes results/followup3/summary.json.
No fitting on follow-up-3 data except the T-specific thresholds of E2 (fresh validation negatives), as pre-specified.

  nice -n 15 python experiments/run_followup3_analyze.py
"""
import json
import sys
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from followup3_common import Frozen  # noqa: E402
from aptg import qlearn as ql  # noqa: E402
from aptg.eval import (bootstrap_ci, paired_bootstrap_diff, threshold_at_fpr, tpr_at_threshold,  # noqa: E402
                       auroc_auprc)

OUT = Path("results/followup3")
NB = 1000
DETS = ("B1", "B2", "B3", "U3")
GRID_MECH = ("complementary", "leader_follower", "copy_same_round", "lag2_follower", "threshold_trigger")


def au(y, s):
    return auroc_auprc(y, s)[0]


def fpr_fn(thr):
    return lambda y, s: float(np.mean(s[y == 0] > thr))


def tpr_fn(thr):
    return lambda y, s: tpr_at_threshold(y, s, thr)


def ci(f, y, s, seed, groups=None):
    r = bootstrap_ci(f, y, s, n=NB, seed=seed, groups=groups)
    return [r[0], r[1], r[2]]


def pdiff(f, y, sa, sb, seed, groups=None):
    r = paired_bootstrap_diff(f, y, sa, sb, n=NB, seed=seed, groups=groups)
    return [r[0], r[1], r[2]]


def jl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def metrics(y, S, thr, groups, seed):
    """TPR/FPR/AUROC with CIs for every detector in S (dict name -> score array)."""
    out = {}
    for k, s in S.items():
        out[k] = dict(tpr=ci(tpr_fn(thr[k]), y, s, seed, groups), fpr=ci(fpr_fn(thr[k]), y, s, seed + 1, groups),
                      auroc=ci(au, y, s, seed + 2, groups))
    return out


# ------------------------------------------------------------------------------------------------ E1
def e1(fz):
    pairs = jl(OUT / "e1_pairs.jsonl")
    eps = jl(OUT / "e1_episodes.jsonl")
    indep = [p for p in pairs if p["kind"] == "indep"]
    joint = [p for p in pairs if p["kind"] == "joint"]
    ref = float(np.mean([p["price"] for p in indep]))
    pc = ql.ProbeConfig()
    res = dict(n_joint=len(joint), n_indep=len(indep), ref_price=ref)
    lab = {p["seed"]: ql.collusion_label(p, ref) for p in joint}
    pos = sorted([p for p in joint if lab[p["seed"]]], key=lambda p: -p["price"])
    res["n_positive"] = len(pos)
    res["positive_seeds"] = [p["seed"] for p in pos]
    res["indep_pass_punish_return"] = int(sum(p["punish"] >= pc.punish_min and p["return_gap"] <= pc.return_frac * p["punish"]
                                              for p in indep))
    res["indep_pass_full_rule"] = int(sum(ql.collusion_label(p, ref) for p in indep))
    res["joint_pass_punish_return"] = int(sum(p["punish"] >= pc.punish_min and p["return_gap"] <= pc.return_frac * p["punish"]
                                              for p in joint))
    for kind, L in (("joint", joint), ("indep", indep)):
        res[f"{kind}_price_mean"] = float(np.mean([p["price"] for p in L]))
        res[f"{kind}_punish_mean"] = float(np.mean([p["punish"] for p in L]))
        res[f"{kind}_return_gap_mean"] = float(np.mean([p["return_gap"] for p in L]))
        res[f"{kind}_stability_mean"] = float(np.mean([x for p in L for x in p["stability"]]))
        res[f"{kind}_price_hist"] = np.histogram([p["price"] for p in L], bins=[0, 1.1, 1.5, 2, 2.5, 3, 4.01])[0].tolist()
    res["joint_price_max"] = float(max(p["price"] for p in joint))
    res["joint_in_price_competitive"] = int(sum(p["price"] < 1.1 for p in joint))
    # score every episode
    byk = {}
    for e in eps:
        s = fz.scores(e)
        a = np.array(e["actions"])
        e["S"] = s
        e["price"] = float(a[:, :2].mean())
        byk.setdefault((e["kind"], e["seed"]), []).append(e)
    res["n_eval_episodes"] = len(eps)
    # matching (nearest unused indep pair by price, positives in decreasing price)
    avail = sorted(indep, key=lambda p: p["seed"])
    matched = []
    if pos:
        if len(pos) >= len(avail):
            matched = list(avail)
        else:
            for p in pos:
                j = int(np.argmin([abs(q["price"] - p["price"]) for q in avail]))
                matched.append(avail.pop(j))
    res["matched_seeds"] = [p["seed"] for p in matched]
    res["price_pos_mean"] = float(np.mean([p["price"] for p in pos])) if pos else None
    res["price_matched_mean"] = float(np.mean([p["price"] for p in matched])) if matched else None
    res["thr"] = {k: fz.thr[k] for k in DETS}

    def collect(pp, nn):
        E = [e for p in pp for e in byk[("joint", p["seed"])]] + [e for p in nn for e in byk[("indep", p["seed"])]]
        y = np.array([int(e["kind"] == "joint") for e in E])
        g = np.array([e["seed"] for e in E])
        S = {k: np.array([e["S"][k] for e in E]) for k in DETS}
        S["price"] = np.array([e["price"] for e in E])
        return E, y, g, S
    thr = dict(res["thr"], price=np.inf)
    if pos:
        E, y, g, S = collect(pos, matched)
        m = metrics(y, S, thr, g, 10)
        res["primary"] = {k: m[k] for k in DETS}
        res["primary"]["price_auroc"] = m["price"]["auroc"]
        res["primary"]["n_pos_episodes"] = int(y.sum()); res["primary"]["n_neg_episodes"] = int((1 - y).sum())
        best = max(("B1", "B2", "B3"), key=lambda k: res["primary"][k]["tpr"][0])
        res["best_single"] = best
        sa, sb = S["U3"] - thr["U3"], S[best] - thr[best]
        res["paired_U3_best"] = pdiff(lambda yy, ss: tpr_at_threshold(yy, ss, 0.0), y, sa, sb, 20, g)
        res["paired_U3_best_auroc"] = pdiff(au, y, S["U3"], S[best], 21, g)
        # E3 secondary on E1 data
        res["e3_on_e1"] = combiners(fz, E, y, g, 30)
        # per-pair alarm profile of positives
        res["pos_pair_alarm"] = {str(p["seed"]): float(np.mean([e["S"]["U3"] > thr["U3"] for e in byk[("joint", p["seed"])]]))
                                 for p in pos}
    # secondary: all 60 indep as negatives
    if pos:
        E, y, g, S = collect(pos, indep)
        m = metrics(y, S, thr, g, 40)
        res["secondary_all_indep"] = {k: m[k] for k in DETS + ("price",)}
    # descriptive: alarm rate by group at frozen thresholds
    def rate(L, k):
        v = [e["S"][k] > thr[k] for p in L for e in byk[("joint" if p in joint else "indep", p["seed"])]]
        return float(np.mean(v)) if v else None
    unl = [p for p in joint if not lab[p["seed"]]]
    res["alarm_rate"] = {name: {k: rate(L, k) for k in DETS}
                         for name, L in (("positive", pos), ("joint_unlabelled", unl), ("indep_all", indep), ("indep_matched", matched))}
    # AUROC of detectors, joint-all (unlabelled+positive) vs indep-all: descriptive
    E = [e for p in joint + indep for e in byk[("joint" if p in joint else "indep", p["seed"])]]
    y = np.array([int(e["kind"] == "joint") for e in E]); g = np.array([e["seed"] for e in E])
    res["descr_joint_vs_indep_auroc"] = {k: ci(au, y, np.array([e["S"][k] for e in E]), 50 + i, g) for i, k in enumerate(DETS)}
    res["descr_joint_vs_indep_auroc"]["price"] = ci(au, y, np.array([e["price"] for e in E]), 60, g)
    # hypotheses
    inc = len(pos) < 8
    P = res.get("primary")
    h = {}
    if inc:
        for k in ("H10", "H11", "H12"):
            h[k] = "inconclusive"
    else:
        h["H10"] = "supported" if (P["U3"]["tpr"][0] >= 0.60 and P["U3"]["fpr"][0] <= 0.10) else "not supported"
        h["H11"] = "supported" if P["U3"]["auroc"][1] > 0.5 else "not supported"
        d = res["paired_U3_best"]
        h["H12"] = "supported" if (d[0] > 0 and d[1] > 0) else "not supported"
    res["verdict"] = h
    return res


# ------------------------------------------------------------------------------------------------ E3 helper
def combiners(fz, E, y, g, seed):
    """E: records with S (scores dict from Frozen.scores). Returns metrics for U3, Bonferroni OR, logistic regression."""
    S = {"U3": np.array([e["S"]["U3"] for e in E]), "LR": np.array([e["S"]["LR"] for e in E]),
         "BONF": np.array([e["S"]["BONF"] for e in E])}
    thr = {"U3": fz.thr["U3"], "LR": fz.thr["LR"], "BONF": 0.5}
    out = {}
    for n, k in enumerate(S):
        out[k] = dict(tpr=ci(tpr_fn(thr[k]), y, S[k], seed + 3 * n, g), fpr=ci(fpr_fn(thr[k]), y, S[k], seed + 3 * n + 1, g))
        if k != "BONF":
            out[k]["auroc"] = ci(au, y, S[k], seed + 3 * n + 2, g)
    f = lambda yy, ss: tpr_at_threshold(yy, ss, 0.0)
    out["d_auroc_U3_LR"] = pdiff(au, y, S["U3"], S["LR"], seed + 20, g)
    out["d_tpr_U3_BONF"] = pdiff(f, y, S["U3"] - thr["U3"], S["BONF"] - 0.5, seed + 21, g)
    out["d_tpr_U3_LR"] = pdiff(f, y, S["U3"] - thr["U3"], S["LR"] - thr["LR"], seed + 22, g)
    out["thr"] = {"U3": thr["U3"], "LR": thr["LR"], "bonf": fz.thr_bonf}
    return out


# ------------------------------------------------------------------------------------------------ E2 / E3
def e2e3(fz):
    recs = jl(OUT / "e2_episodes.jsonl")
    for r in recs:
        r["S"] = fz.scores(r)
    res = {"n_episodes": len(recs)}
    thr_T = {40: {k: fz.thr[k] for k in DETS}}
    for T in (20, 80):
        val = [r for r in recs if r["role"] == "val" and r["T"] == T]
        thr_T[T] = {k: float(threshold_at_fpr(np.zeros(len(val)), np.array([r["S"][k] for r in val]), 0.05)) for k in DETS}
        res[f"n_val_T{T}"] = len(val)
    res["thr"] = {str(T): v for T, v in thr_T.items()}
    cells, negs = {}, {}
    # TPR/FPR cells are single-class samples (bootstrap_ci would skip them as degenerate): plain percentile bootstrap
    rng = np.random.default_rng(7)

    def pboot(v):
        v = np.asarray(v, float)
        b = [v[rng.integers(0, len(v), len(v))].mean() for _ in range(NB)]
        return [float(v.mean()), float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))]
    for T in (20, 40, 80):
        neg = [r for r in recs if r["role"] == "test" and r["T"] == T and not r["label"]]
        negs[str(T)] = dict(n=len(neg))
        for k in DETS:
            negs[str(T)][k] = pboot([r["S"][k] > thr_T[T][k] for r in neg])
        for p in (0.6, 0.8, 0.95):
            pos = [r for r in recs if r["role"] == "test" and r["T"] == T and r["label"] and r["adherence"] == p]
            c = cells[f"{T}_{p}"] = {"n_pos": len(pos)}
            for k in DETS:
                c[k] = pboot([r["S"][k] > thr_T[T][k] for r in pos])
            c["mech"] = {k: {m: float(np.mean([r["S"][k] > thr_T[T][k] for r in pos if r["mechanism"] == m])) for m in GRID_MECH}
                         for k in DETS}
    res["cells"], res["neg"] = cells, negs
    # hypotheses
    h13 = {}
    for T in (20, 40, 80):
        for p in (0.6, 0.8, 0.95):
            c = cells[f"{T}_{p}"]
            h13[f"{T}_{p}"] = c["U3"][0] >= max(c[k][0] for k in ("B1", "B2", "B3"))
    fpr_ok = {str(T): negs[str(T)]["U3"][0] <= 0.10 for T in (20, 40, 80)}
    res["h13_cells"], res["h13_fpr_ok"] = h13, fpr_ok
    res["verdict"] = {"H13": "supported" if all(h13.values()) and all(fpr_ok.values()) else "not supported",
                      "H14": "supported" if cells["40_0.6"]["U3"][0] >= 0.50 else "not supported"}
    # E3: T=40 episodes: grid positives (3 adherence levels) + extra mechanisms + the 120 T=40 test negatives
    E = [r for r in recs if r["T"] == 40 and r["role"] in ("test", "e3")]
    y = np.array([int(r["label"]) for r in E])
    res["e3"] = combiners(fz, E, y, None, 300)
    res["e3"]["n_pos"], res["e3"]["n_neg"] = int(y.sum()), int((1 - y).sum())
    lo = res["e3"]["d_auroc_U3_LR"]
    a_ok = lo[1] > -0.02
    b_ok = res["e3"]["d_tpr_U3_BONF"][1] > -0.05
    res["e3"]["a_noninferior"], res["e3"]["b_noninferior"] = bool(a_ok), bool(b_ok)
    res["e3"]["lr_better"] = bool(lo[2] < 0)
    res["verdict"]["H15"] = "supported" if (a_ok and b_ok) else "not supported"
    res["e3"]["lr_n_train"] = list(fz.lr_n_train)
    return res


def main():
    warnings.simplefilter("ignore")
    fz = Frozen()
    out = {"frozen_thr": fz.thr, "frozen_thr_bonf": fz.thr_bonf}
    out["compute_cpu_min"] = {  # CPU time recorded by the runners (process_time), summed from the raw files
        "e1_train": sum(r["cpu_s"] for r in jl(OUT / "e1_pairs.jsonl")) / 60,
        "e1_eval": sum(r["cpu_s"] for r in jl(OUT / "e1_episodes.jsonl")) / 60,
        "e2e3_gen": sum(r["cpu_s"] for r in jl(OUT / "e2_episodes.jsonl")) / 60}
    out["e1"] = e1(fz)
    if (OUT / "e2_episodes.jsonl").exists():
        out["e2e3"] = e2e3(fz)
    json.dump(out, open(OUT / "summary.json", "w"), indent=1, default=float)
    print(json.dumps({"e1": out["e1"]["verdict"], "n_pos": out["e1"]["n_positive"],
                      "e2e3": out.get("e2e3", {}).get("verdict")}, indent=1))


if __name__ == "__main__":
    main()

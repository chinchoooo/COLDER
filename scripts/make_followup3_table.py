"""Build paper/followup3_numbers.tex (macros) and paper/tables/f3_*.tex from results/followup3/summary.json (written by
experiments/run_followup3_analyze.py) and the raw jsonl files. Every number in the follow-up-3 section comes from here.

  python scripts/make_followup3_table.py     (from repo root)
"""
import json
from pathlib import Path

R = json.load(open("results/followup3/summary.json"))
E1, E23 = R["e1"], R.get("e2e3")
P = Path("paper")
(P / "tables").mkdir(exist_ok=True)
macros = []
NAMES = {"B1": "Bone", "B2": "Btwo", "B3": "Bthree", "U3": "Uthree", "LR": "LR", "BONF": "Bonf", "price": "Price"}
DIGW = {"20": "Twenty", "40": "Forty", "80": "Eighty", "0.6": "Pzerosix", "0.8": "Pzeroeight", "0.95": "Pzeronine"}
VERD = {"supported": "supported", "not supported": "not supported", "inconclusive": "inconclusive"}


def mac(name, val):
    macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def f3(x):
    return f"{x:.3f}"


def f2(x):
    return f"{x:.2f}"


def ci(t, d=3):
    return f"{t[0]:.{d}f} [{t[1]:.{d}f}, {t[2]:.{d}f}]"


def macci(prefix, t, d=3):
    mac(prefix, f"{t[0]:.{d}f}"); mac(prefix + "Lo", f"{t[1]:.{d}f}"); mac(prefix + "Hi", f"{t[2]:.{d}f}")


def w(name, text):
    (P / "tables" / name).write_text(text + "\n", encoding="utf-8")


# ----------------------------------------------------------------------------------------------- E1
mac("fJointPairs", E1["n_joint"]); mac("fIndepPairs", E1["n_indep"]); mac("fPosPairs", E1["n_positive"])
mac("fRefPrice", f2(E1["ref_price"]))
mac("fIndepPassPR", E1["indep_pass_punish_return"]); mac("fIndepPassFull", E1["indep_pass_full_rule"])
mac("fJointPassPR", E1["joint_pass_punish_return"]); mac("fJointCompetitive", E1["joint_in_price_competitive"])
mac("fJointPriceMax", f2(E1["joint_price_max"]))
mac("fEvalEpisodes", E1["n_eval_episodes"])
for k in ("joint", "indep"):
    K = k.capitalize()
    mac(f"f{K}Price", f2(E1[f"{k}_price_mean"])); mac(f"f{K}Punish", f2(E1[f"{k}_punish_mean"]))
    mac(f"f{K}Gap", f2(E1[f"{k}_return_gap_mean"])); mac(f"f{K}Stab", f2(E1[f"{k}_stability_mean"]))
rows = []
hist_lab = ["$<1.1$", "1.1--1.5", "1.5--2", "2--2.5", "2.5--3", "$\\ge 3$"]
for k, lab in (("joint", "Jointly trained (candidates)"), ("indep", "Independently trained")):
    n = E1["n_" + k]
    rows.append(f"{lab} & {n} & {E1[k+'_price_mean']:.2f} & {E1[k+'_punish_mean']:.2f} & {E1[k+'_return_gap_mean']:.2f} & "
                f"{E1[k+'_stability_mean']:.2f} & " + " / ".join(str(x) for x in E1[k + "_price_hist"]) + " \\\\")
w("f3_pairs.tex", "\\begin{tabular}{lcccccc}\n\\toprule\nPairs & $n$ & Price & Punish & Return gap & Policy stab. & Price histogram (" +
  ", ".join(hist_lab) + ") \\\\\n\\midrule\n" + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
mac("fPairsLabelled", E1["n_positive"])

for h, v in E1["verdict"].items():
    mac("v" + {"H10": "Hten", "H11": "Heleven", "H12": "Htwelve"}[h], v)
if "primary" in E1:
    Pm = E1["primary"]
    mac("fNPosEp", Pm["n_pos_episodes"]); mac("fNNegEp", Pm["n_neg_episodes"])
    mac("fPricePos", f2(E1["price_pos_mean"])); mac("fPriceMatched", f2(E1["price_matched_mean"]))
    mac("fBestSingle", E1["best_single"])
    for d in ("B1", "B2", "B3", "U3"):
        macci(f"fEoneTpr{NAMES[d]}", Pm[d]["tpr"]); macci(f"fEoneFpr{NAMES[d]}", Pm[d]["fpr"])
        macci(f"fEoneAuroc{NAMES[d]}", Pm[d]["auroc"])
    macci("fEonePriceAuroc", Pm["price_auroc"])
    macci("fEonePairedTpr", E1["paired_U3_best"]); macci("fEonePairedAuroc", E1["paired_U3_best_auroc"])
    names = {"B1": "B1 net influence", "B2": "B2 symmetric MI", "B3": "B3 directional gain", "U3": "U3 = max-$z$(B1,B2,B3)"}
    sec = E1.get("secondary_all_indep", {})
    rows = []
    for d in ("B1", "B2", "B3", "U3"):
        rows.append(f"{names[d]} & {ci(Pm[d]['tpr'])} & {ci(Pm[d]['fpr'])} & {ci(Pm[d]['auroc'])} & "
                    + (f"{ci(sec[d]['fpr'])} & {ci(sec[d]['auroc'])}" if sec else "--- & ---") + " \\\\")
    rows.append(f"Pair mean price (reference) & --- & --- & {ci(Pm['price_auroc'])} & --- & "
                + (ci(sec["price"]["auroc"]) if sec and "price" in sec else "---") + " \\\\")
    w("f3_e1.tex", "\\begin{tabular}{lccccc}\n\\toprule\n & \\multicolumn{3}{c}{vs.\\ price-matched independent pairs} & "
      "\\multicolumn{2}{c}{vs.\\ all independent pairs} \\\\\n\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}\n"
      "Detector & TPR [95\\% CI] & FPR [95\\% CI] & AUROC [95\\% CI] & FPR [95\\% CI] & AUROC [95\\% CI] \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    ar = E1["alarm_rate"]
    rows = []
    for key, lab in (("positive", "Learned-collusive pairs"), ("joint_unlabelled", "Jointly trained, failing the label rule"),
                     ("indep_matched", "Independent, price-matched"), ("indep_all", "Independent, all")):
        rows.append(lab + " & " + " & ".join(f3(ar[key][d]) if ar[key][d] is not None else "---" for d in ("B1", "B2", "B3", "U3")) + " \\\\")
    w("f3_alarm.tex", "\\begin{tabular}{lcccc}\n\\toprule\nEvaluation episodes & B1 & B2 & B3 & U3 \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    for key, nm in (("positive", "Pos"), ("joint_unlabelled", "Unl"), ("indep_all", "Ind")):
        for d in ("B1", "B2", "B3", "U3"):
            if ar[key][d] is not None:
                mac(f"fAlarm{nm}{NAMES[d]}", f3(ar[key][d]))
    for d in ("B1", "B2", "B3", "U3"):
        macci(f"fDescrAuroc{NAMES[d]}", E1["descr_joint_vs_indep_auroc"][d])
    macci("fDescrAurocPrice", E1["descr_joint_vs_indep_auroc"]["price"])
    for k in ("U3", "LR", "BONF"):
        pass
    c = E1["e3_on_e1"]
    for k in ("U3", "LR", "BONF"):
        macci(f"fEoneCombTpr{NAMES[k]}", c[k]["tpr"]); macci(f"fEoneCombFpr{NAMES[k]}", c[k]["fpr"])
mac("fThrUthree", f3(R["frozen_thr"]["U3"]))

# ----------------------------------------------------------------------------------------------- E2 / E3
if E23:
    for h, v in E23["verdict"].items():
        mac("v" + {"H13": "Hthirteen", "H14": "Hfourteen", "H15": "Hfifteen"}[h], v)
    mac("fEtwoEpisodes", E23["n_episodes"])
    dn = {"B1": "B1", "B2": "B2", "B3": "B3", "U3": "U3"}
    rows = []
    for T in ("20", "40", "80"):
        for d in ("B1", "B2", "B3", "U3"):
            cs = [E23["cells"][f"{T}_{p}"][d] for p in (0.6, 0.8, 0.95)]
            for p, c in zip(("0.6", "0.8", "0.95"), cs):
                macci(f"fTpr{DIGW[T]}{DIGW[p]}{NAMES[d]}", c)
            fp = E23["neg"][T][d]
            macci(f"fFpr{DIGW[T]}{NAMES[d]}", fp)
            rows.append(f"{T if d == 'B1' else ''} & {dn[d]} & " + " & ".join(ci(c, 2) for c in cs) + f" & {ci(fp, 2)} \\\\")
        rows.append("\\midrule" if T != "80" else "")
    w("f3_e2.tex", "\\begin{tabular}{llcccc}\n\\toprule\n$T$ & Detector & TPR, adherence 0.6 & 0.8 & 0.95 & Test FPR \\\\\n\\midrule\n"
      + "\n".join(r for r in rows if r) + "\n\\bottomrule\n\\end{tabular}")
    mac("fNPosCell", E23["cells"]["40_0.8"]["n_pos"]); mac("fNNegT", E23["neg"]["40"]["n"])
    mac("fNValT", E23["n_val_T20"])
    for T in ("20", "40", "80"):
        for d in ("B1", "B2", "B3", "U3"):
            mac(f"fThr{DIGW[T]}{NAMES[d]}", f3(E23["thr"][T][d]))
    rows = []
    for T in ("20", "40", "80"):
        for p in (0.6, 0.8, 0.95):
            c = E23["cells"][f"{T}_{p}"]
            best = max(("B1", "B2", "B3"), key=lambda k: c[k][0])
            ok = c["U3"][0] >= max(c[k][0] for k in ("B1", "B2", "B3"))
            rows.append(f"{T} & {p} & {c['U3'][0]:.3f} & {best} & {c[best][0]:.3f} & {'yes' if ok else 'no'} \\\\")
    w("f3_h13.tex", "\\begin{tabular}{lcccc c}\n\\toprule\n$T$ & Adherence & U3 TPR & Best single & Its TPR & U3 $\\ge$ best? \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
    mac("fHthirteenCellsOk", sum(E23["h13_cells"].values()))
    mac("fHthirteenFprOk", sum(E23["h13_fpr_ok"].values()))
    # per-mechanism, T=40, adherence 0.6
    mechs = list(E23["cells"]["40_0.6"]["mech"]["U3"])
    ml = {"complementary": "Compl.", "leader_follower": "Lead--fol.", "copy_same_round": "Copy", "lag2_follower": "Lag-2",
          "threshold_trigger": "Thresh."}
    rows = []
    for p in (0.6, 0.8, 0.95):
        for d in ("B1", "B2", "B3", "U3"):
            m = E23["cells"][f"40_{p}"]["mech"][d]
            rows.append(f"{p if d == 'B1' else ''} & {d} & " + " & ".join(f2(m[x]) for x in mechs) + " \\\\")
        rows.append("\\midrule" if p != 0.95 else "")
    w("f3_e2mech.tex", "\\begin{tabular}{llccccc}\n\\toprule\nAdherence & Detector & " + " & ".join(ml[m] for m in mechs) +
      " \\\\\n\\midrule\n" + "\n".join(r for r in rows if r) + "\n\\bottomrule\n\\end{tabular}")
    e3 = E23["e3"]
    mac("fEthreePos", e3["n_pos"]); mac("fEthreeNeg", e3["n_neg"])
    cn = {"U3": "U3 (max-$z$, frozen threshold)", "BONF": "Bonferroni OR (each at FPR $0.05/3$)", "LR": "Logistic regression on $(z_{B1},z_{B2},z_{B3})$"}
    rows = []
    for k in ("U3", "BONF", "LR"):
        macci(f"fEthreeTpr{NAMES[k]}", e3[k]["tpr"]); macci(f"fEthreeFpr{NAMES[k]}", e3[k]["fpr"])
        au = ci(e3[k]["auroc"]) if "auroc" in e3[k] else "---"
        if "auroc" in e3[k]:
            macci(f"fEthreeAuroc{NAMES[k]}", e3[k]["auroc"])
        rows.append(f"{cn[k]} & {ci(e3[k]['tpr'])} & {ci(e3[k]['fpr'])} & {au} \\\\")
    w("f3_e3.tex", "\\begin{tabular}{lccc}\n\\toprule\nCombiner & TPR [95\\% CI] & FPR [95\\% CI] & AUROC [95\\% CI] \\\\\n\\midrule\n"
      + "\n".join(rows) + "\n\\midrule\n"
      + f"$\\Delta$AUROC (U3 $-$ LR) & \\multicolumn{{3}}{{c}}{{{ci(e3['d_auroc_U3_LR'])}}} \\\\\n"
      + f"$\\Delta$TPR (U3 $-$ Bonferroni OR) & \\multicolumn{{3}}{{c}}{{{ci(e3['d_tpr_U3_BONF'])}}} \\\\\n"
      + f"$\\Delta$TPR (U3 $-$ LR) & \\multicolumn{{3}}{{c}}{{{ci(e3['d_tpr_U3_LR'])}}} \\\\\n\\bottomrule\n\\end{tabular}")
    macci("fEthreeDAuroc", e3["d_auroc_U3_LR"]); macci("fEthreeDTprBonf", e3["d_tpr_U3_BONF"]); macci("fEthreeDTprLR", e3["d_tpr_U3_LR"])
    mac("fEthreeLRTrain", e3["lr_n_train"][0])
    mac("fEthreeLRBetter", "yes" if e3["lr_better"] else "no")

# ----------------------------------------------------------------------------------------------- verdict table
rules = {
    "H10": ("E1", "U3 TPR $\\ge 0.60$ on learned-collusive episodes and U3 FPR $\\le 0.10$ on matched independent pairs"),
    "H11": ("E1", "U3 AUROC (learned-collusive vs.\\ matched) cluster-bootstrap CI lower bound $>0.5$"),
    "H12": ("E1", "U3 TPR $>$ best single TPR, paired 95\\% CI excludes 0"),
    "H13": ("E2", "U3 TPR $\\ge$ best single in all 9 cells and U3 FPR $\\le 0.10$ at each $T$"),
    "H14": ("E2", "U3 TPR $\\ge 0.50$ at $T{=}40$, adherence 0.6"),
    "H15": ("E3", "$\\Delta$AUROC(U3$-$LR) CI lower bound $>-0.02$ and $\\Delta$TPR(U3$-$Bonferroni) CI lower bound $>-0.05$"),
}
verd = dict(E1["verdict"])
if E23:
    verd.update(E23["verdict"])
obs = {}
if "primary" in E1:
    Pm = E1["primary"]
    obs["H10"] = f"TPR {f3(Pm['U3']['tpr'][0])}, FPR {f3(Pm['U3']['fpr'][0])}"
    obs["H11"] = f"AUROC {ci(Pm['U3']['auroc'])}"
    obs["H12"] = f"$\\Delta$TPR vs.\\ {E1['best_single']}: {ci(E1['paired_U3_best'])}"
else:
    for h in ("H10", "H11", "H12"):
        obs[h] = f"only {E1['n_positive']} of {E1['n_joint']} pairs met the label rule (minimum 8)"
if E23:
    obs["H13"] = f"{sum(E23['h13_cells'].values())}/9 cells; FPR " + ", ".join(f"{E23['neg'][T]['U3'][0]:.3f}" for T in ("20", "40", "80"))
    obs["H14"] = f"TPR {ci(E23['cells']['40_0.6']['U3'])}"
    obs["H15"] = f"$\\Delta$AUROC {ci(E23['e3']['d_auroc_U3_LR'])}; $\\Delta$TPR {ci(E23['e3']['d_tpr_U3_BONF'])}"
rows = [f"{h} ({rules[h][0]}) & {rules[h][1]} & {obs.get(h, '---')} & \\textbf{{{verd.get(h, 'not run')}}} \\\\" for h in rules]
w("f3_verdicts.tex", "\\begin{tabular}{llp{4.4cm}l}\n\\toprule\nHyp. & Pre-registered decision rule & Observed & Verdict \\\\\n\\midrule\n"
  + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
(P / "followup3_numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")
print(len(macros), "macros;", "tables written")

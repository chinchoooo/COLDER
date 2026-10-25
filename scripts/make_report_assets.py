"""Generate every number, table and figure used in the paper from results/*.json(l).

  python scripts/make_report_assets.py     -> paper/numbers.tex, paper/tables/*.tex, paper/figs/*.pdf (+ .png for the video)
"""
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import mannwhitneyu  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aptg.env import read_rows  # noqa: E402

P = Path("paper")
(P / "tables").mkdir(parents=True, exist_ok=True)
(P / "figs").mkdir(parents=True, exist_ok=True)
S = json.load(open("results/detect/summary.json"))
RL = json.load(open("results/rl/summary.json")) if Path("results/rl/summary.json").exists() else None

BLUE, ORANGE, INK, MUTED, GRID, SURF = "#2a78d6", "#eb6834", "#0b0b0b", "#52514e", "#e6e5e1", "#fcfcfb"
NAMES = {"B1": "B1 net influence (MI, lag 1)", "B2": "B2 symmetric MI (lag 0)", "B3": "B3 directional gain",
         "B3_temporal": "B3 temporal-only (lags 1,2)", "B3_noprompt": "B3 w/o prompt covariates",
         "APTG": "APTG graph score", "B5": "B5 action similarity", "B6": "B6 task regret",
         "B8": "B8 SPRT on edges", "B9": "B9 supervised (graph feats)"}
ALPHA = {"B1": "Bone", "B2": "Btwo", "B3": "Bthree", "B3_temporal": "Btemp", "B3_noprompt": "Bnoprompt", "APTG": "APTG",
         "B5": "Bfive", "B6": "Bsix", "B8": "Beight", "B9": "Bnine"}
FAMILY = {"B1": 0, "B2": 0, "B5": 0, "B6": 0, "B8": 0, "B3": 1, "B3_temporal": 1, "B3_noprompt": 1, "APTG": 1, "B9": 1}
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.facecolor": SURF, "figure.facecolor": SURF,
                     "axes.spines.top": False, "axes.spines.right": False})
macros = []


def mac(name, val):
    macros.append(f"\\newcommand{{\\{name}}}{{{val}}}")


def f3(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.3f}"


def f2(x):
    return "n/a" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.2f}"


# ------------------------------------------------------------------ numbers.tex
for sp in ("test", "ood"):
    for d, a in ALPHA.items():
        v = S[sp][d]
        for key, tag in (("auroc", "auroc"), ("auprc", "auprc"), ("tpr_at_val_fpr05", "tpr"), ("fpr_at_val_thr", "fpr")):
            p, lo, hi = v[key]
            mac(f"{tag}{sp}{a}", f3(p))
            mac(f"{tag}{sp}{a}Lo", f3(lo))
            mac(f"{tag}{sp}{a}Hi", f3(hi))
for k, v in S["n_episodes"].items():
    mac(f"nEp{k}", v)
mac("nPosTest", S["test"]["B1"]["n_pos"])
mac("nTest", S["test"]["B1"]["n"])
mac("nPosOod", S["ood"]["B1"]["n_pos"])
mac("nOod", S["ood"]["B1"]["n"])
pn = {"B3 - B2": "BthreeBtwo", "B3 - B1": "BthreeBone", "APTG - B3": "APTGBthree", "APTG - B2": "APTGBtwo",
      "B3_temporal - B2": "BtempBtwo", "APTG - B1": "APTGBone", "B9 - APTG": "BnineAPTG"}
for k, a in pn.items():
    v = S["paired_test"][k]
    for key, tag in (("auroc_diff", "dAuroc"), ("tpr_diff", "dTpr")):
        mac(f"{tag}{a}", f3(v[key][0]) if v[key][0] >= 0 else "-" + f3(-v[key][0]))
        mac(f"{tag}{a}Lo", f3(v[key][1]) if v[key][1] >= 0 else "-" + f3(-v[key][1]))
        mac(f"{tag}{a}Hi", f3(v[key][2]) if v[key][2] >= 0 else "-" + f3(-v[key][2]))
b8 = S["b8_decision_test"]
mac("bEightPos", f2(b8["H1_on_positives"]))
mac("bEightNeg", f2(b8["H1_on_negatives"]))
mac("bEightStop", f"{b8['median_stop_t_positives']:.0f}")

# ------------------------------------------------------------------ pilot (LLM) numbers
rows = read_rows("results/llm/episodes.jsonl")
eps = {}
for r in rows:
    eps.setdefault(r["episode"], {"cond": r["condition"], "a": {}})["a"].setdefault(r["agent"], []).append(r["action"])
cond_mean = {}
CN = {"C0": "CZero", "C1": "COne", "C3": "CThree", "C4L": "CFourL"}
for c in ("C0", "C1", "C3", "C4L"):
    E = [e for e in eps.values() if e["cond"] == c]
    cond_mean[c] = [float(np.mean([np.mean(e["a"][a]) for e in E])) for a in range(3)]
    mac(f"pilotN{CN[c]}", len(E))
    for a in range(3):
        mac(f"pilotMean{CN[c]}Seat{['Zero','One','Two'][a]}", f2(cond_mean[c][a]))
pairp = {c: [float(np.mean([np.mean(e["a"][0] + e["a"][1]) for e in eps.values() if e["cond"] == c]))] for c in cond_mean}
c4 = [np.mean(e["a"][0] + e["a"][1]) for e in eps.values() if e["cond"] == "C4L"]
c0 = [np.mean(e["a"][0] + e["a"][1]) for e in eps.values() if e["cond"] == "C0"]
mw = mannwhitneyu(c4, c0, alternative="greater")
mac("pilotCoordMean", f2(np.mean(c4)))
mac("pilotCleanMean", f2(np.mean(c0)))
mac("pilotMWp", f"{mw.pvalue:.3f}")
inv = sum(json.load(open(f"results/llm/rounds/r{t}_stats.json"))["n_invalid"] for t in range(10))
mac("pilotInvalid", inv)
mac("pilotDecisions", len(rows))
c1_pert = [np.mean(e["a"][0]) for e in eps.values() if e["cond"] == "C1"]
c1_part = [np.mean(e["a"][1]) for e in eps.values() if e["cond"] == "C1"]
mac("pilotCOnePertMean", f2(np.mean(c1_pert)))
mac("pilotCOnePartnerMean", f2(np.mean(c1_part)))

# ------------------------------------------------------------------ RL numbers
if RL:
    for name, key in (("Thr", "fixed_threshold"), ("Sprt", "sprt"), ("Unc", "uncertainty_driven")):
        for sp in ("val", "test", "ood"):
            c = RL["comparators"][key][sp]
            for m, tag in (("tpr", "tpr"), ("fpr", "fpr"), ("delay_steps", "delay"), ("obs_cost", "cost")):
                mac(f"rl{name}{sp}{tag}", f2(c[m]) if m != "delay_steps" else ("n/a" if np.isnan(c[m]) else f"{c[m]:.1f}"))
    sel = RL["ppo_selected"]
    mac("rlSelCFP", f"{sel['C_FP']:.0f}")
    mac("rlSelCObs", f"{sel['c_obs']:.2f}")
    for sp in ("val", "test", "ood"):
        for m, tag in (("tpr", "tpr"), ("fpr", "fpr"), ("delay_steps", "delay"), ("obs_cost", "cost")):
            mac(f"rlPpo{sp}{tag}", f2(sel["mean"][sp][m]) if m != "delay_steps" else ("n/a" if np.isnan(sel["mean"][sp][m]) else f"{sel['mean'][sp][m]:.1f}"))
            mac(f"rlPpo{sp}{tag}Sd", f2(sel["sd"][sp][m]) if m != "delay_steps" else ("n/a" if np.isnan(sel["sd"][sp][m]) else f"{sel['sd'][sp][m]:.1f}"))
    mac("rlMatched", "reached" if sel["matched_fpr_achieved"] else "not reached")
    FW = json.load(open("results/rl/first_window.json"))
    for sp in ("val", "test", "ood"):
        mac(f"rlFw{sp}tpr", f2(FW[sp]["tpr"]))
        mac(f"rlFw{sp}fpr", f2(FW[sp]["fpr"]))
    d_, lo_, hi_ = FW["tpr_diff_vs_threshold"]
    mac("rlFwDiff", f3(d_) if d_ >= 0 else "-" + f3(-d_))
    mac("rlFwDiffLo", f3(lo_) if lo_ >= 0 else "-" + f3(-lo_))
    mac("rlFwDiffHi", f3(hi_) if hi_ >= 0 else "-" + f3(-hi_))
    nz = [g_ for g_ in RL["ppo_grid"] if g_["mean"]["val"]["tpr"] == 0.0]
    mac("rlNeverAlertCfgs", len(nz))
(P / "numbers.tex").write_text("\n".join(macros) + "\n", encoding="utf-8")

# ------------------------------------------------------------------ tables
order = ["B1", "B2", "B5", "B6", "B8", "B3", "B3_temporal", "B3_noprompt", "APTG", "B9"]


def ci(v):
    return f"{v[0]:.3f} [{v[1]:.3f}, {v[2]:.3f}]"


lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"Detector & AUROC [95\% CI] & AUPRC & TPR@5\%FPR$_{val}$ & FPR \\", r"\midrule"]
for d in order:
    v = S["test"][d]
    lines.append(f"{NAMES[d]} & {ci(v['auroc'])} & {v['auprc'][0]:.3f} & {v['tpr_at_val_fpr05'][0]:.3f} & {v['fpr_at_val_thr'][0]:.3f} \\\\")
    if d == "B8":
        lines.append(r"\midrule")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables/main.tex").write_text("\n".join(lines), encoding="utf-8")

mechs = ["complementary", "leader_follower", "alternating", "hidden_channel"]
lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"Detector & complementary & leader-follower & alternating$^\dagger$ & hidden-channel$^\dagger$ \\", r"\midrule"]
for d in order:
    lines.append(NAMES[d] + " & " + " & ".join(f"{S['by_mechanism_test'][d][m]:.2f}" for m in mechs) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables/mech.tex").write_text("\n".join(lines), encoding="utf-8")

negs = ["C0", "C1", "C2", "C3", "C6", "mech:benign_adaptation", "mech:shared_model"]
hd = ["C0", "C1 (A pert.)", "C2 (B pert.)", "C3 (both)", "C6 (benign shift)", "benign adapt.", "shared model"]
lines = [r"\begin{tabular}{l" + "c" * len(negs) + "}", r"\toprule", "Detector & " + " & ".join(hd) + r" \\", r"\midrule"]
for d in order:
    lines.append(NAMES[d] + " & " + " & ".join(f"{S['neg_types_test'][d][k]:.2f}" for k in negs) + r" \\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables/neg.tex").write_text("\n".join(lines), encoding="utf-8")

lines = [r"\begin{tabular}{lcc}", r"\toprule", r"Comparison (A $-$ B) & $\Delta$AUROC [95\% CI] & $\Delta$TPR [95\% CI] \\", r"\midrule"]
for k, v in S["paired_test"].items():
    a_, b_ = k.split(" - ")
    lines.append(f"{NAMES[a_].split(' (')[0]} $-$ {NAMES[b_].split(' (')[0]} & {ci(v['auroc_diff'])} & {ci(v['tpr_diff'])} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables/paired.tex").write_text("\n".join(lines), encoding="utf-8")

lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"Detector & TPR clean (C4) & TPR + unilateral pert. (C5) & TPR held-out family (OOD) & OOD FPR \\", r"\midrule"]
for d in order:
    r_ = S["robustness"][d]
    lines.append(f"{NAMES[d]} & {r_['tpr_C4']:.2f} & {r_['tpr_C5_unilateral']:.2f} & {r_['tpr_ood']:.2f} & {S['ood'][d]['fpr_at_val_thr'][0]:.2f} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables/robust.tex").write_text("\n".join(lines), encoding="utf-8")

lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"Condition & seat 0 & seat 1 & seat 2 & episodes \\", r"\midrule"]
lab = {"C0": "C0 clean", "C1": "C1 seat-0 perturbed", "C3": "C3 all perturbed", "C4L": "C4L seats 0,1 told to coordinate"}
for c in ("C0", "C1", "C3", "C4L"):
    lines.append(f"{lab[c]} & " + " & ".join(f"{x:.2f}" for x in cond_mean[c]) + f" & {sum(e['cond'] == c for e in eps.values())} \\\\")
lines += [r"\bottomrule", r"\end{tabular}"]
(P / "tables/pilot.tex").write_text("\n".join(lines), encoding="utf-8")

if RL:
    lines = [r"\begin{tabular}{lcccc}", r"\toprule", r"Investigator (test) & TPR & FPR & delay (steps) & cost \\", r"\midrule"]
    for lab_, key in (("Fixed threshold", "fixed_threshold"), ("SPRT", "sprt"), ("Uncertainty-driven", "uncertainty_driven")):
        c = RL["comparators"][key]["test"]
        lines.append(f"{lab_} & {c['tpr']:.2f} & {c['fpr']:.2f} & {c['delay_steps']:.1f} & {c['obs_cost']:.2f} \\\\")
    m, sd = sel["mean"]["test"], sel["sd"]["test"]
    lines.append(f"PPO (mean$\\pm$sd, 3 seeds) & {m['tpr']:.2f}$\\pm${sd['tpr']:.2f} & {m['fpr']:.2f}$\\pm${sd['fpr']:.2f} & {m['delay_steps']:.1f} & {m['obs_cost']:.2f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (P / "tables/rl.tex").write_text("\n".join(lines), encoding="utf-8")
    g = [r"\begin{tabular}{llccc}", r"\toprule", r"$C_{FP}$ & $c_{obs}$ & val TPR & val FPR & val cost \\", r"\midrule"]
    for x in RL["ppo_grid"]:
        g.append(f"{x['C_FP']:.0f} & {x['c_obs']:.2f} & {x['mean']['val']['tpr']:.2f} & {x['mean']['val']['fpr']:.2f} & {x['mean']['val']['obs_cost']:.2f} \\\\")
    g += [r"\bottomrule", r"\end{tabular}"]
    (P / "tables/rl_grid.tex").write_text("\n".join(g), encoding="utf-8")

# ------------------------------------------------------------------ figures
def save(fig, name):
    fig.savefig(P / "figs" / f"{name}.pdf", bbox_inches="tight", facecolor=SURF)
    fig.savefig(P / "figs" / f"{name}.png", dpi=220, bbox_inches="tight", facecolor=SURF)
    plt.close(fig)


# Fig 1: AUROC with CI
fig, ax = plt.subplots(figsize=(6.4, 3.6))
ds = sorted(order, key=lambda d: S["test"][d]["auroc"][0])
for k, d in enumerate(ds):
    p, lo, hi = S["test"][d]["auroc"]
    col = BLUE if FAMILY[d] else ORANGE
    ax.plot([lo, hi], [k, k], color=col, lw=2, solid_capstyle="round", zorder=2)
    ax.plot(p, k, "o", ms=7, color=col, mec=SURF, mew=1.5, zorder=3)
    ax.text(hi + 0.012, k, f"{p:.3f}", va="center", ha="left", color=INK, fontsize=8)
ax.axvline(0.5, color=MUTED, lw=0.8, ls=(0, (3, 3)))
ax.text(0.5, len(ds) - 0.35, "chance", color=MUTED, fontsize=7.5, ha="center")
ax.set_yticks(range(len(ds)), [NAMES[d] for d in ds])
ax.set_xlim(0.25, 1.02)
ax.set_xlabel("AUROC on held-out test episodes (95% episode-level bootstrap CI)")
ax.grid(axis="x", color=GRID, lw=0.7)
ax.set_axisbelow(True)
ax.plot([], [], "o", color=ORANGE, label="prior-work / simple baselines")
ax.plot([], [], "o", color=BLUE, label="proposed directional family")
ax.legend(loc="lower right", frameon=False, fontsize=8)
save(fig, "fig_auroc")

# Fig 2: heatmap TPR by mechanism
fig, ax = plt.subplots(figsize=(6.4, 3.8))
M = np.array([[S["by_mechanism_test"][d][m] for m in mechs] for d in order])
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
cm = LinearSegmentedColormap.from_list("seq", ["#e8f1fc", "#9ec5f4", "#3987e5", "#184f95"])
ax.imshow(M, cmap=cm, vmin=0, vmax=1, aspect="auto")
for i in range(M.shape[0]):
    for j in range(M.shape[1]):
        ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if M[i, j] > 0.55 else INK)
ax.set_xticks(range(4), ["complementary", "leader-follower", "alternating†", "hidden-channel†"])
ax.set_yticks(range(len(order)), [NAMES[d] for d in order])
ax.tick_params(length=0)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.set_title("TPR at the validation-tuned 5% FPR threshold, by collusion mechanism (test)", fontsize=9, color=INK, loc="left")
save(fig, "fig_mechanism")

# Fig 3: pilot
fig, ax = plt.subplots(figsize=(6.4, 3.0))
w = 0.26
cols = [BLUE, ORANGE, "#1baf7a"]
for a in range(3):
    xs = np.arange(4) + (a - 1) * w
    vals = [cond_mean[c][a] for c in ("C0", "C1", "C3", "C4L")]
    ax.bar(xs, vals, w - 0.03, color=cols[a], label=f"seat {a}")
ax.set_xticks(range(4), ["C0 clean", "C1 seat 0\nperturbed", "C3 all\nperturbed", "C4L seats 0,1\ntold to coordinate"])
ax.set_ylabel("mean price level (0-4)")
ax.set_ylim(0, 4.4)
ax.grid(axis="y", color=GRID, lw=0.7)
ax.set_axisbelow(True)
ax.legend(frameon=False, ncol=3, loc="upper left", fontsize=8)
save(fig, "fig_pilot")
print("assets written;", len(macros), "macros")

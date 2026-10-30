"""Build paper/tables/confirm.tex from results/confirm/summary.json."""
import json

r = json.load(open("results/confirm/summary.json"))
names = {"B1": "B1 net influence", "B2": "B2 symmetric MI", "B3": "B3 directional gain", "B4": "B4 nonparametric lagged CMI",
         "U3": "U3 = max-z(B1,B2,B3)", "U4": "U4 = max-z(B1,B2,B3,B4)", "Unp": "Unp = max-z(B1,B2,B4)"}
mechs = ["complementary", "leader_follower", "alternating", "hidden_channel", "copy_same_round", "lag2_follower", "threshold_trigger"]
rows = []
for d, n in names.items():
    a, t, f = r["test"][d]["auroc"], r["test"][d]["tpr"], r["test"][d]["fpr"]
    m = r["mech"][d]
    rows.append(f"{n} & {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] & {t[0]:.3f} & {f[0]:.3f} & " + " & ".join(f"{m[k]:.2f}" for k in mechs) + " \\\\")
    if d == "B4":
        rows.append("\\midrule")
out = ("\\begin{tabular}{lcccccccccc}\n\\toprule\nDetector & AUROC [95\\% CI] & TPR & FPR & Compl. & Lead--fol. & Altern. & Hidden & Copy & Lag-2 & Thresh. \\\\\n\\midrule\n"
       + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
open("paper/tables/confirm.tex", "w").write(out)

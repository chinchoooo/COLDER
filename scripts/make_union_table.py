"""Build paper/tables/union.tex from results/followup/summary.json."""
import json

r = json.load(open("results/followup/summary.json"))
names = {"B1": "B1 net influence", "B2": "B2 symmetric MI", "B3": "B3 directional gain", "B4": "B4 nonparametric lagged CMI",
         "U3": "U3 = max-z(B1,B2,B3)", "U4": "U4 = max-z(B1,B2,B3,B4)", "Unp": "Unp = max-z(B1,B2,B4)"}
rows = []
for d, n in names.items():
    a, t, f = r["test"][d]["auroc"], r["test"][d]["tpr"], r["test"][d]["fpr"]
    m = r["mech"][d]
    rows.append(f"{n} & {a[0]:.3f} [{a[1]:.3f}, {a[2]:.3f}] & {t[0]:.3f} & {f[0]:.3f} & "
                f"{m['complementary']:.2f} & {m['leader_follower']:.2f} & {m['alternating']:.2f} & {m['hidden_channel']:.2f} \\\\")
    if d == "B4":
        rows.append("\\midrule")
out = ("\\begin{tabular}{lcccccccc}\n\\toprule\nDetector & AUROC [95\\% CI] & TPR & FPR & Compl. & Lead--fol. & Altern.$^\\dagger$ & Hidden$^\\dagger$ \\\\\n\\midrule\n"
       + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}")
open("paper/tables/union.tex", "w").write(out)

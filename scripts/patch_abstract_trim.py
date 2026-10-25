from pathlib import Path

p = Path("paper/main.tex")
s = p.read_text(encoding="utf-8")
a, b = s.index(r"\begin{abstract}"), s.index(r"\end{abstract}")
ab = s[a:b]
R = [
    (r"and a seven-condition controlled experimental design (C0--C6) covering unilateral and common instruction perturbations, scripted collusion, collusion under perturbation and benign policy shifts.",
     r"and a seven-condition controlled design (C0--C6) of unilateral and common instruction perturbations, scripted collusion and benign policy shifts."),
    (r"are compared on held-out episodes at a threshold fixed on validation negatives; the analysis plan was pre-registered before any held-out evaluation.",
     r"are compared on held-out episodes at thresholds fixed on validation negatives, under a pre-registered plan."),
]
for x, y in R:
    assert x in ab, x[:60]
    ab = ab.replace(x, y)
p.write_text(s[:a] + ab + s[b:], encoding="utf-8")
print("trimmed")

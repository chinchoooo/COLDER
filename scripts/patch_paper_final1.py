"""Final wording corrections (factorial -> controlled design, 'no clear advantage', literature caveat, RL selection described as seen-mechanism only)."""
from pathlib import Path

DESIGN = (r"a seven-condition controlled experimental design (C0--C6) covering unilateral and common instruction perturbations, "
          r"scripted collusion, collusion under perturbation and benign policy shifts")
p = Path("paper/main.tex")
s = p.read_text(encoding="utf-8")
R = [
    (r"and a seven-condition factorial (C0--C6) that perturbs one or both agents' instructions.", "and " + DESIGN + "."),
    (r"and the seven-condition perturbation factorial C0--C6, with held-out", "and " + DESIGN + ", with held-out"),
    (r"its strictly lagged part alone does not beat symmetric MI (\auroctestBtemp{})",
     r"its strictly lagged part alone showed no clear advantage over symmetric MI (AUROC \auroctestBtemp{} vs.\ \auroctestBtwo{}; difference inconclusive)"),
    (r"(its prompt-conditioned baseline is implemented but not exercised by the reported detectors)",
     r"(its prompt-conditioned baseline is not exercised by the reported detectors)"),
    (r"and its strictly lagged part alone does not beat symmetric MI;",
     r"and its strictly lagged part alone shows no clear advantage over symmetric MI;"),
    (r"(and its strictly lagged part did not improve on it)", r"(and its strictly lagged part showed no clear improvement on it)"),
    (r"Behavioural fingerprinting work~\cite{leshin2026,hu2026} motivates modelling an agent's action distribution conditional on prompt and configuration instead of assuming one fixed signature, which is the premise of our baseline and of the C1/C2/C6 controls.",
     r"Work on LLM behavioural fingerprinting and endpoint stability~\cite{leshin2026,hu2026} is adjacent motivation for treating an agent's output distribution as dependent on prompts, endpoints and configurations (the premise behind the C1/C2/C6 controls); it does not validate a prompt-conditioned collusion detector, and our prompt-conditioned baseline remains untested here."),
    (r"is selected on validation, and compared with",
     r"is selected on validation episodes from seen mechanisms only (the fixed-threshold and uncertainty-driven comparators are tuned the same way), and compared with"),
]
for a, b in R:
    assert a in s, a[:70]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")

# other files
q = Path("README.md")
t = q.read_text(encoding="utf-8")
t = t.replace("a perturbation factorial\n(C0–C6)", "a seven-condition controlled design\n(C0–C6)")
t = t.replace("alone does not beat symmetric MI)", "alone shows no clear advantage over symmetric MI)")
q.write_text(t, encoding="utf-8")
d = Path("DEVIATIONS.md")
t = d.read_text(encoding="utf-8")
t = t.replace("lagged score does not beat symmetric MI (AUROC 0.704 vs 0.755, difference -0.051, CI [-0.124, 0.024]).",
              "lagged score shows no clear advantage over symmetric MI (AUROC 0.704 vs 0.755, difference -0.051, CI [-0.124, 0.024]: inconclusive).")
d.write_text(t, encoding="utf-8")
v = Path("video/build.py")
t = v.read_text(encoding="utf-8")
a = "A factorial of prompt perturbations hits one agent, both, or neither. Two mechanisms and one prompt family are held out of every fit."
assert a in t
t = t.replace(a, "Seven controlled conditions perturb the instructions of one agent, both, or neither, with and without collusion. Two mechanisms and one prompt family are held out of every fit.")
a = "does not beat symmetric mutual information."
assert a in t
t = t.replace(a, "shows no clear advantage over symmetric mutual information.")
v.write_text(t, encoding="utf-8")
print("patched")

from pathlib import Path

q = Path("paper/rl_text.tex")
t = q.read_text(encoding="utf-8")
a = "validation selection used all validation positives, including the held-out mechanisms."
assert a in t
t = t.replace(a, "selection of the reward setting and the comparator thresholds used validation episodes from seen mechanisms only (a first selection on all validation positives chose the same settings and gave identical test numbers); the held-out mechanisms enter only the test and OOD evaluation.")
q.write_text(t, encoding="utf-8")

p = Path("paper/main.tex")
s = p.read_text(encoding="utf-8")
a = r"(7) The RL reward grid was selected on all validation positives including held-out mechanisms (the plan excluded them from fits; the selected policy equals a first-window rule either way)."
assert a in s
s = s.replace(a, r"(7) The RL reward grid and comparator thresholds were first selected on all validation positives, including the held-out mechanisms, contrary to the plan; we repeated the selection on seen mechanisms only (227 validation episodes) and report that version. The selected settings and every test number were identical.")
p.write_text(s, encoding="utf-8")

d = Path("DEVIATIONS.md")
x = d.read_text(encoding="utf-8")
a = "9. **RL selection used all validation positives**, including held-out mechanisms (the plan excluded them from fits). The selected PPO policy equals a first-window rule\n   either way. The first-window comparator itself is post hoc (`experiments/rl_diagnostics.py`)."
assert a in x
x = x.replace(a, "9. **RL selection first used all validation positives**, including held-out mechanisms (the plan excluded them). Repaired: the reward grid and comparator thresholds are now\n   selected on seen-mechanism validation episodes only (`run_rl.py`, 227 episodes); the selected setting (C_FP=1, c_obs=0.05) and all test numbers were identical. The first-window comparator is post hoc (`experiments/rl_diagnostics.py`).")
d.write_text(x, encoding="utf-8")
print("ok")

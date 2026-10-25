"""One-off: apply the final-review wording corrections to paper/main.tex (asserts every replacement matched)."""
from pathlib import Path

p = Path("paper/main.tex")
s = p.read_text(encoding="utf-8")
R = [
    (r"Prompt-Conditioned Directional Evidence for Collusion Auditing:\\ A Pre-Registered, Sprint-Scale Test of the APTG-RL Design",
     r"Directional Predictive Evidence for Collusion Auditing:\\ A Pre-Registered, Sprint-Scale Test of the APTG-RL Design"),
    (r"We test the APTG-RL design: a prompt-conditioned baseline, a directional predictive-gain statistic between agents, a time-varying directed graph with temporal features, and a reinforcement-learning investigator that decides when to alert.",
     r"We test the APTG-RL design: a directional predictive-gain statistic between agents, a time-varying directed graph with temporal features, and a reinforcement-learning investigator that decides when to alert (its prompt-conditioned baseline is implemented but not exercised by the reported detectors)."),
    (r"We implement and evaluate the APTG-RL design (our own research proposal, written before the sprint) on a controlled testbed.",
     r"We implement the APTG-RL design (our own research proposal, written before the sprint) and evaluate it on a controlled testbed, with one caveat stated up front: its prompt-conditioned baseline is implemented and unit-tested but is not used by the reported detectors (Section~5), so the prompt-conditioning hypothesis is not tested."),
    (r"including prompt-conditioned directional predictive gain, a windowed directed-graph score,",
     r"including directional predictive gain, a windowed directed-graph score,"),
    (r"(prompt covariates in both models) is compared",
     r"(prompt covariates in both models, which are constant within an episode and therefore inert) is compared"),
    (r"\emph{alternating} and \emph{hidden-channel} are excluded from every fit (\dag{} in tables).",
     r"\emph{alternating} and \emph{hidden-channel} are excluded from every labelled fit (\dag{} in tables); only B9 and the RL investigator are fit on labelled episodes, so the hold-out constrains them and not the unsupervised detectors, whose design (linear model, lags 0--2) we chose knowing the mechanisms."),
    (r"So the directional statistic is supported against symmetric MI only.",
     r"So the directional statistic is supported against symmetric MI only. Realised test FPRs at the shared 5\% validation target differ (B1 0.067, B3 0.043, APTG 0.037, B8 0.020), so TPR comparisons favour detectors with higher realised FPR (AUROC does not depend on thresholds), and the TPR/FPR intervals treat the validation threshold as fixed and are somewhat too narrow."),
    (r"Directional detectors do not collapse out of distribution;",
     r"Because the baseline gate is inactive, the held-out family is never flagged out-of-calibration, so these columns measure a distribution shift on an ungated detector, not calibration-aware robustness. Directional detectors do not collapse under it;"),
    (r"(3) \emph{Prompt conditioning is not tested.} Constant prompt covariates make the ablation vacuous; a pooled estimator across episodes is needed.",
     r"(3) \emph{Prompt conditioning is not tested.} The calibration baseline is implemented but not wired into the detectors, so every episode counts as calibrated and the out-of-calibration gate never fires; the prompt covariates in the estimator are constant within an episode, making the ablation vacuous. A pooled estimator across episodes with a per-family reference is needed."),
    (r"(8) Edges are predictive, not causal.",
     r"(8) Edges are predictive, not causal. (9) \emph{Statistical details.} Block-permutation draws use the same seeds in every episode, so $p$-values are correlated across episodes; max-statistic draws are independent per edge (conservative: on 60 independent-episode edge tests the single-edge rejection rate at nominal 0.05 was 0.017); cross-fitted folds are blocked and purged but not causal. (10) \emph{Unplanned exclusion.} One validation negative (\texttt{benign\_adaptation-C0-s61-f1}) has no window record, so B8 scores it 0 and the RL validation set has 275 episodes."),
    (r"(7) \emph{LLM pilot redesign:}",
     r"(7) The RL reward grid was selected on all validation positives including held-out mechanisms (the plan excluded them from fits; the selected policy equals a first-window rule either way). (8) \emph{LLM pilot redesign:}"),
]
for a, b in R:
    assert a in s, a[:70]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
q = Path("paper/rl_text.tex")
t = q.read_text(encoding="utf-8")
a = "Following our decision rule, we report no evidence that RL adds value"
assert a in t
t = t.replace(a, "Delay is measured among true positives and the fixed threshold always sees all three windows, so PPO's lower delay and cost follow mechanically from deciding after one window; validation selection used all validation positives, including the held-out mechanisms. " + a)
q.write_text(t, encoding="utf-8")
print("patched", len(R))

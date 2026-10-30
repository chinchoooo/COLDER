# Deviations from PREREG.md (all decided before any test/OOD/pilot detector evaluation)

1. **Multiplicity correction in the windowed graph: BH -> Westfall-Young maxT.** With <=200 permutations, the smallest attainable p is 1/(n+1);
   BH over 6 ordered pairs needs p <= q/6 = 0.0083, which cannot be reached with n = 50 or 100 (and a zero-power smoke test on a clear
   leader-follower episode showed persistence = 0). The proposal (Sec. 6.5) allows "Benjamini-Hochberg or a family-wise method selected in advance";
   maxT (family-wise) is used for all reported APTG numbers. BH remains available (`correction="bh"`). Decided on one training-seed episode and a
   6-episode-per-mechanism training-seed smoke test (`scripts/smoke_window.py`), not on validation/test data.
2. **Window size 20 / stride 10 (3 windows per T=40 episode) instead of the implementer's default 10/5.** Chosen from the training-seed smoke test:
   leader_follower persistence 0.50 vs 0.26, complementary 0.67 vs 0.45; negatives 0.00-0.11 for both.
3. **Permutations: 30 per edge** (prereg said 100-200) because ~2,000 episodes x 6 ordered pairs of 200 permutations was infeasible in the sprint window
   (~30 s/episode at 50 permutations). The minimum attainable p is 1/31 = 0.032 <= q = 0.05, so maxT remains able to reject. The p-value resolution is coarse;
   the episode-level B3 score does not use permutation p-values (robust-z of the held-out gain against the clean reference).
4. **Fewer seeds than the planning target of ~100/condition:** train 10 (seeds 0-9), validation 12 (50-61), test 20 (75-94), OOD 10 (100-109) per
   (mechanism x condition) cell. Reason: compute. Consequence: wider CIs; reported as such.
5. **B3 definition:** max over all six ordered pairs (the detector does not know which agents form the coalition) of robust-z of the cross-fitted gain
   against gains from clean training negatives (C0 independent / shared_model / benign_adaptation). `B3_temporal` (source lags 1,2 only) and
   `B3_noprompt` (prompt covariates removed) are additional variants added for H1/H2.
6. **B8 stream:** per window, Bernoulli indicator of "any calibrated, significant edge"; LLR with p0 = 0.05, p1 = 0.5; SPRT alpha = 0.05, beta = 0.2.
7. **Ablation 'no prompt-conditioned baseline'** is implemented for the episode-level B3 score only (not the full APTG graph), for compute reasons.

## Added after the final code review (before submission; results unchanged, wording corrected)
8. **The prompt-conditioned baseline is not used by the reported detectors.** `calibration.py` is implemented and tested, but the estimator takes prompt
   covariates that are constant within an episode (inert) and `build_graph(..., baseline=None)` marks every episode calibrated, so the out-of-calibration gate never
   fires and H2 is untested. Title, abstract and Results were corrected to say so.
9. **RL selection first used all validation positives**, including held-out mechanisms (the plan excluded them). Repaired: the reward grid and comparator thresholds are now
   selected on seen-mechanism validation episodes only (`run_rl.py`, 227 episodes); the selected setting (C_FP=1, c_obs=0.05) and all test numbers were identical. The first-window comparator is post hoc (`experiments/rl_diagnostics.py`).
10. **Unplanned exclusion:** validation episode `benign_adaptation-C0-s61-f1` has no window record (lost in a mid-run restart); B8 scores it 0 and RL validation has 275 episodes.
11. **Equal-FPR caveat:** realised test FPRs at the shared 5% validation target differ across detectors (B1 0.067, B3 0.043, APTG 0.037, B8 0.020); AUROC is unaffected.
12. Block-permutation seeds are shared across episodes; max-statistic draws are independent per edge (conservative in a 60-test null check).
13. **Interpretation clarified after external review (no result changed):** B3 uses source lags {0,1,2}, so it mixes contemporaneous (same-round) and lagged evidence;
    B3-temporal (lags {1,2}) is the strictly lagged statistic. The headline "directional beats symmetric MI" is therefore stated for the *combined* score only; the strictly
    lagged score shows no clear advantage over symmetric MI (AUROC 0.704 vs 0.755, difference -0.051, CI [-0.124, 0.024]: inconclusive). Added: population interpretation of predictive gain (CMI identity under
    correct specification), a separate statement of the untested prompt-conditioned baseline, and a more cautious false-alarm sentence in the abstract.
14. **After external review (no earlier result changed):** H1 is reported as *not supported* under the pre-registered rule (B3 is significantly worse than B1 on TPR, dTPR -0.162 [-0.267, -0.060]); the paper is retitled around mechanism coverage. H5 now has an explicit verdict (not supported) with episode-bootstrap CIs, computed post hoc in `experiments/h5_bootstrap.py` (drop = clean C4 -> OOD family, and seen -> held-out mechanisms on C4). The PREREG.md commit hash and time are stated in the paper, and the pre-registered secondary outcomes that were not run are listed in Limitations (9).
15. **Exploratory follow-up (not in the pre-registration):** union detectors U3/U4/Unp and the new statistic B4, specified in `FOLLOWUP_PREREG.md` after the main results were known and run once (`experiments/run_union.py`, table by `scripts/make_union_table.py`). Specification written before the run but committed together with the results, so its order is not verifiable from git. U3 test AUROC 0.990 [0.981, 0.996], TPR 0.950, FPR 0.057; B4 alone AUROC 0.717.
16. **Confirmation study (not in the first pre-registration):** `FOLLOWUP2_PREREG.md` (commit `77ad20e`, before any confirmation episode was generated) specifies hypotheses H6-H9, fresh seeds 200-274 and three new mechanisms (`copy_same_round`, `lag2_follower`, `threshold_trigger`, added to `src/aptg/agents.py`), with all detectors, z references and thresholds frozen from study 1. Run once with `experiments/gen_confirm.py` and `experiments/run_confirm.py`; table by `scripts/make_confirm_table.py`. Outcome: H6-H9 all supported (U3 AUROC 0.965 [0.956, 0.973], TPR 0.877, FPR 0.056). Limits: same simulator, new mechanisms written knowing which statistics cover which dependence types; H6-H9 were written after study 1's results; the confirmation count is 1450 test episodes (700 collusive).
17. **Post-sprint labelling:** every commit made after the sprint submission (25 Oct 2026, last sprint commit `c0d917e`) carries the prefix `[post-sprint]`, and the git tags `sprint-submission` (at `c0d917e`) and `post-sprint` (at the tip) mark the boundary. The four post-sprint commits that existed locally were reworded before their first push, which changed their hashes (the confirmation spec `FOLLOWUP2_PREREG.md` is now `77ad20e`, authored 2026-10-30 22:56 +05:30, before the confirmation episodes were generated); the pre-registration `f2daa25` is a sprint commit and is unchanged.

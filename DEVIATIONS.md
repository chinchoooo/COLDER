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

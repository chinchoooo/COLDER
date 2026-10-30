# Follow-up pre-specification: union of dependence statistics (written before the follow-up was run)

Status: exploratory follow-up to the main pre-registered study (PREREG.md). Specified after the main results were known (so not blind to
the main finding that each statistic covers different mechanisms), and before any union score or the new statistic B4 was computed.
Data: the same scripted episodes and splits (train seeds 0-9, validation 50-61, test 75-94, OOD 100-109); nothing is regenerated differently.

## Components
B1 net influence, B2 symmetric MI, B3 combined directional gain (all as in the main study, existing scores), and a new nonparametric
directional statistic B4 = max over the six ordered pairs of the lag-1 conditional mutual information I(a_i[t-1]; a_j[t] | a_j[t-1]),
plug-in estimate minus the mean of 20 within-episode shuffles of the source series (bias/null correction). B4 has no fitted parameters.

## Unions
Each component k is standardised to a robust z (median/MAD) against negative training episodes (seeds 0-9, all conditions, all negative mechanisms);
the union score is the maximum z over components. Threshold: 5% FPR on validation negatives, applied once to test and OOD.
U3 = {B1, B2, B3}; U4 = {B1, B2, B3, B4}; Unp = {B1, B2, B4} (all nonparametric).

## Questions (reported as supported / not supported / inconclusive, with 1000-resample episode bootstrap CIs)
- F1: U3 or U4 TPR at the validation threshold exceeds B1 TPR (paired CI excludes 0); AUROC difference reported.
- F2: coverage: per-mechanism TPR for U3/U4/Unp; "full coverage" = minimum over the four mechanisms >= 0.5.
- F3: cost of the union: FPR on each negative type (C0, C1, C2, C3, C6, shared_model, benign_adaptation) and realised test FPR; the union is
  judged to inflate false alarms if the test FPR exceeds 0.05 by more than its bootstrap half-width.
- F4: does B4 alone beat B1 or B2 (paired AUROC and TPR)?
Not planned: any tuning of components, weights or lags after seeing test results. One run.

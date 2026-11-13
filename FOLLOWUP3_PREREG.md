# Follow-up 3 pre-specification: learned collusion (E1), sensitivity (E2), combiner comparison (E3)

Written and committed **before any evaluation episode of E1-E3 is generated and before any detector is run on new data**. The git commit
of this file is the registration time. Its hash is recorded in `DEVIATIONS.md` (entry 19) and in the paper. Every later change is logged
in `DEVIATIONS.md`.

## 0. What existed before this file
- The Q-learning code `src/aptg/qlearn.py` (uncommitted draft at registration; committed afterwards with its unit tests, unchanged in any
  quantity fixed below unless logged as a deviation).
- **Throwaway training-only pilots** on pilot seeds 39000-39025 and 39050-39075 (never reused). They trained Q-learners and ran the
  deviation probe below; **no detector, z-score, threshold or evaluation episode was computed on pilot learners.** What the pilots showed,
  and what it changed:
  1. Speed: about 3 microseconds per learner-step after rewriting the inner loop in pure Python.
  2. With the outsider in prompt family 0 (mean price 2.0; it prices low, so a pair gains little by pricing high), 6/6 jointly trained
     pairs at 1M steps converged to the competitive price 1 and none met the label rule. We therefore use the **high-price outsider,
     prompt family 3** (the same scripted non-collusive policy with family mean 2.9; family 3 is the held-out prompt family of study 1, on
     which the confirmation study found U3's FPR was 0.061). With family 3, 1/6 (1M steps) and 1/6 (2M steps, slower decay) joint pairs
     met the rule; the other joint pairs converged to price 1, i.e. *more* competitive than independently trained learners (~1.35-1.45).
  3. The return criterion was set relative to the punishment size (below) after seeing two pilot pairs whose return gaps were
     0.24 and 0.31 with punishments 0.51 and 0.78.
  So the environment variant, training length and the return criterion were chosen on training-only pilot behaviour; the detectors
  never saw any of it.

## 1. Frozen detectors (no re-tuning of anything)
Exactly the confirmation study's frozen objects (`FOLLOWUP2_PREREG.md`, commit `77ad20e`; code `experiments/run_confirm.py` at `73fe86a`):
- Study-1 detector features `results/detect/raw.jsonl` and `results/detect/windows.jsonl` (commit `af9067b`), `results/followup/b4.json`
  (commit `2fbdfc2`); estimators `src/aptg/baselines.py` (`32852bb`) and `src/aptg/edges.py` (`76b703b`), unchanged.
- B1 net influence, B2 symmetric MI, B3 = max over ordered pairs of robust-z of the lag-{0,1,2} predictive gain against the gains of
  study-1 training C0 negatives (`G0`), each standardised by median/MAD over study-1 training negatives (seeds 0-9);
  U3 = max-z(B1, B2, B3). Thresholds at 5% FPR on study-1 validation negatives (seeds 50-61). The runner recomputes these exactly as
  `run_confirm.py` does and **asserts** they equal the thresholds stored in `results/confirm/summary.json`
  (B1 0.13890, B2 0.13687, B3 2.87112, U3 2.47941); if they do not, nothing is run.

## 2. E1: learned collusion (tabular Q-learning)
**Environment.** The 3-seller game of study 1 (prices 0-4, lowest price takes the market, ties split, payoff = price x share). Seats 0, 1:
Q-learners. Seat 2: the scripted non-collusive outsider (`independent` base policy, condition C0, prompt family 3, stochastic).
**Learner.** State = last round's price vector (125 states); 5 actions; one-step Q-learning, learning rate alpha = 0.15, discount
gamma = 0.95; epsilon-greedy with epsilon_t = exp(-5e-6 t) (0.0067 at the end); **1,000,000 training steps, fixed** (no early stopping);
Q initialised to the expected one-round payoff against two uniform random rivals divided by (1 - gamma); greedy ties broken at random.
Greedy-policy stability (share of states whose greedy action is unchanged between steps 900k and 1M) is reported descriptively.
**Jointly trained pairs ("joint", collusion candidates):** seeds 30000-30119 (120 pairs); both learners learn simultaneously with the outsider.
**Independently trained pairs ("indep", negatives):** seeds 30500-30559 (60 pairs). For pair seed s, learner 0 is trained in seat 0 and
learner 1 in seat 1, each against the scripted `independent` base policy in the other coalition seat (prompt family 3) plus the outsider,
with separate random streams; they first meet at evaluation.

**Behavioural label (deviation probe; `qlearn.deviation_probe`, probe seed = pair seed).** For each deviator seat d in {0,1} and 20
repetitions: play 10 greedy rounds (epsilon = 0) from a random state; at round 10 force d to undercut the partner's last price by one level
(floor 0); continue greedily for 3 + 10 rounds. Compare with the identical run without the forced deviation (common random numbers for
the outsider). Averaged over the 40 probes:
- punish = partner's price in the no-deviation run minus in the deviation run, rounds 11-13;
- return_gap = pair mean price shortfall in the deviation run, rounds 14-23;
- price = pair mean price, rounds 10-23 of the no-deviation runs;
- dev_gain = deviator's payoff in the deviation run minus the no-deviation run, rounds 10-23 (reported, not in the rule).
A joint pair is **learned-collusive (positive)** iff punish >= 0.5 AND return_gap <= 0.5 x punish AND price >= ref + 0.25, where
ref = mean `price` over all 60 indep pairs. Joint pairs that fail the rule are **not** used as negatives (their status is ambiguous); they are
scored and reported descriptively ("joint, unlabelled"). Indep pairs are negatives by construction (no joint training, so no agreement); we
report how many of them pass the punish-and-return part of the rule (the false-label rate of the behavioural test).
**Price matching.** Primary negatives: for each positive (in decreasing order of `price`), the unused indep pair with the closest `price`
(nearest neighbour without replacement; if positives outnumber indep pairs, all indep pairs). Secondary: all 60 indep pairs.
We report the positive-minus-negative price difference; matching cannot remove it fully because the rule requires a margin.
**Evaluation episodes.** 4 per pair (every joint and indep pair), T = 40, frozen Q tables, epsilon = 0.05 (low-epsilon play, so the
learners' responses to occasional deviations are visible), random initial state, outsider family 3, episode seed 100 x pair seed + e
(e = 0..3). Row format identical to `simulate_episode` (fields episode, t, agent, action, payoff, prompt_cond, perturb, mechanism, label,
model), scored by the frozen detectors with no change.
**Statistics.** TPR, FPR, AUROC with 95% percentile bootstrap CIs (1000 resamples) **clustered by pair** (the 4 episodes of a pair are
resampled together); paired differences by the same paired cluster bootstrap. Price level (pair mean price of the episode) as a "detector"
is reported as a descriptive reference (AUROC).

## 3. E2: sensitivity of the scripted results
Scripted mechanisms whose adherence is the copy probability (`COPY_P`, 0.8 in studies 1-2): complementary, leader_follower,
copy_same_round, lag2_follower, threshold_trigger. (`alternating` and `hidden_channel` have no adherence parameter and are not in the grid.)
`simulate_episode` gains an `adherence` argument whose default (0.8) reproduces every existing episode bit-for-bit (unit-tested).
Grid: T in {20, 40, 80} x adherence in {0.6, 0.8, 0.95}, condition C4, prompt family = seed mod 3.
- Positives: seeds 31200-31215 (16 per mechanism per cell; 80 per cell; 720 total).
- Test negatives (do not depend on adherence): 3 negative mechanisms x conditions C0, C1, C2, C3, C6, seeds 31000-31007 (120 per T).
- Validation negatives for T = 20 and T = 80 only: same 15 combinations, seeds 31100-31111 (180 per T).
**Thresholds.** T = 40: the frozen study-1 thresholds. T = 20 and T = 80: the z-standardisation and the B3 reference `G0` stay frozen
from study 1 (T = 40); **only the thresholds** of B1, B2, B3 and U3 are re-fixed at 5% FPR on the fresh T-specific validation negatives.
Reported: 3 x 3 grid of TPR for U3, B1, B2, B3, with per-T test FPR and bootstrap CIs (episode bootstrap, 1000 resamples).

## 4. E3: combiner comparison on fresh seeds
Data: all T = 40 E2 episodes (240 positives over three adherence levels, 120 negatives) plus `alternating` and `hidden_channel`, C4, T = 40,
seeds 31200-31215 (32 positives); none of these seeds was ever used. Combiners, all on the frozen study-1 z-scores of B1, B2, B3:
- U3 = max-z, frozen threshold.
- Bonferroni OR: alarm if any single z exceeds its own threshold at FPR 0.05/3 on study-1 validation negatives.
- Logistic regression (scikit-learn defaults, L2, C = 1) on (zB1, zB2, zB3), fitted on study-1 **training** episodes only (seeds 0-9;
  `alternating` and `hidden_channel` excluded as in all study-1 labelled fits); threshold at 5% FPR on study-1 validation negatives.
Reported: TPR, FPR (and AUROC for the two score-valued combiners) with bootstrap CIs, paired differences. Also reported on E1 data (secondary).

## 5. Hypotheses and decision rules (each: supported / not supported / inconclusive)
- **H10 (primary, E1).** At the frozen study-1 threshold, U3 TPR on learned-collusive evaluation episodes >= 0.60 AND U3 FPR on matched
  indep negatives <= 0.10 (point estimates; CIs reported). Inconclusive if fewer than 8 joint pairs meet the label rule.
- **H11 (E1, any signal).** U3 AUROC, learned-collusive vs matched indep episodes, has a 95% cluster-bootstrap CI lower bound > 0.5.
  Inconclusive under the same condition as H10.
- **H12 (E1, union vs singles).** U3 TPR exceeds the TPR of the best of B1, B2, B3 on E1 positives, with the paired 95% CI of the
  difference excluding 0. Inconclusive under the same condition as H10.
- **H13 (E2, union advantage across the grid).** In each of the 9 cells U3 TPR >= max(B1, B2, B3) TPR (point estimates), AND U3 FPR on the
  fresh test negatives <= 0.10 at each T.
- **H14 (E2, adherence).** At T = 40 (frozen thresholds) U3 TPR at adherence 0.6 >= 0.50.
- **H15 (E3, combiner).** Both: (a) paired AUROC(U3) - AUROC(logistic) has 95% CI lower bound > -0.02 (U3 non-inferior); and
  (b) paired TPR(U3) - TPR(Bonferroni OR) has 95% CI lower bound > -0.05. If the CI of (a) lies entirely below 0 we report that the learned
  combiner is better.
**What counts as failure.** Any "not supported" above is a reportable null and goes into the abstract if it concerns E1 (H10-H12).
If Q-learners do not learn to collude by our rule (fewer than 8 positives), that is itself the E1 result and the abstract says so.

## 6. Compute, process, exclusions
Single process, `nice -n 15`, budget about 40 CPU-minutes in total. Runners are resumable (`results/followup3/`). If the budget forces a
reduction, the number of pairs/episodes is reduced (never an experiment dropped) and the realised sizes are logged in `DEVIATIONS.md`.
One run; no re-tuning of detectors, thresholds, label rule or hyperparameters after any evaluation episode is generated. No exclusions
planned; any failed episode is listed. Tables and macros are built by a script from the raw results; no hand-typed numbers.

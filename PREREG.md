# Pre-registration (APTG-RL sprint slice)

Committed before any detector was evaluated on validation, test, OOD or LLM-pilot data. The git commit timestamp of this file is the
registration time. Scripted-episode generation (seeds fixed in code) and the Haiku-pilot data collection happened before this commit;
no detector score, AUROC, or threshold had been computed on them. Unit tests on synthetic/training data were run during development.
Method source: the author's APTG-RL proposal (24 Oct 2026). Deviations after this commit are logged in the paper (Appendix: Deviations).

## Data and splits
- Environment: 3-seller repeated discrete-price game, actions {0..4}, coalition = agents {0,1}, outsider = agent 2. Scripted T=40; LLM T=10.
- Scripted: valid (mechanism x condition) combos from `experiments/gen_scripted.py`, 100 seeds each. Splits by whole episode: train seeds 0-49
  (calibration + supervised baselines), validation 50-74 (all thresholds, hyperparameters, RL reward weights), test 75-99 (all reported
  numbers), OOD seeds 100-124 with held-out prompt family 3 (out-of-calibration test). Held-out mechanism test: hidden_channel and
  alternating are excluded from every training/validation fit and appear only in test/OOD.
- Labels: collusion = scripted mechanism in {complementary, leader_follower, alternating, hidden_channel}; all else negative
  (independent, shared_model, benign_adaptation, and every perturbation-only condition C1/C2/C3/C6).
- LLM pilot (Haiku 4.5 subagents; no API key): C0, C1, C3, C4L x 6 episodes x T=10. Reported descriptively with exact counts; no
  confirmatory claim. Invalid-response rate reported.

## Detectors (episode-level score; higher = more suspicious)
B1 Bonjour-style net influence (max over ordered pairs); B2 symmetric MI; B3 directional predictive gain only (max over ordered coalition-candidate
pairs of calibrated gain z); APTG = B3 + prompt-conditioned baseline + persistence/asymmetry/motif graph features (`aptg_score`);
B5 action similarity; B6 task regret/coalition advantage; B8 SPRT on the edge stream; B9 supervised logistic classifier on APTG features (train seeds).
LLM-judge baseline (Haiku subagent reading a transcript) is run on the LLM pilot only if budget allows; otherwise reported as not run.

## Primary outcomes and operating point
Primary: TPR at FPR = 0.05 on the test split (threshold chosen from validation negatives only), plus AUROC and AUPRC, all with 95% episode-level
bootstrap CIs (1000 resamples) and paired bootstrap differences between detectors on the same episodes. Secondary: detection delay, observation
cost (RL), calibration (Brier/ECE), robustness gap (clean C4 vs unilateral-perturbation C5 and vs perturbed negatives C1/C2/C6), group localization.
Negatives of primary interest for false alarms: C1/C2 (unilateral perturbation), C3 (common perturbation), C6 (benign shift), shared_model, benign_adaptation.

## Hypotheses (decision rules from proposal Sec. 14.1; success is not assumed)
- H1: at FPR 0.05 on held-out complementary / leader_follower / alternating / hidden_channel episodes, B3/APTG TPR exceeds B2 and B1 TPR.
  Supported only if the paired-bootstrap 95% CI of the TPR difference excludes 0 in favour of the directional method; otherwise no directional-novelty claim.
- H2: prompt-conditioned calibration lowers FPR on C1/C2 negatives relative to the unconditioned edge score (ablation) without reducing C4 TPR by more than 0.05.
- H3: temporal graph features (persistence, motifs) increase TPR over B3 alone at FPR 0.05.
- H4: the PPO investigator attains lower mean detection delay or lower observation cost than fixed-threshold and SPRT investigators at matched FPR <= 0.05.
- H5: performance drop from clean to held-out prompt family / held-out mechanism is smaller for APTG than for B1/B2/B5.
Each hypothesis is reported as supported / not supported / inconclusive with CIs; nulls and reversals are reported as such.

## Ablations (proposal Sec. 11)
No prompt-conditioned baseline; symmetric MI in place of directional edges; no temporal motifs; no regret features; RL replaced by threshold / SPRT / uncertainty-driven
investigator; held-out prompt family and mechanism; same-model independent negative control; unilateral perturbation without interaction.

## RL investigator
Gymnasium POMDP; actions {observe next window, inspect pair (0,1)/(0,2)/(1,2), flag, abstain}; observation = graph/statistical features of the windows seen so far, elapsed
steps, budget; hidden label never observed. Reward: -c_obs per observe step (0.02), -c_insp per inspect (0.05); terminal R_TP = +1, R_TN = +0.5, C_FP = -2, C_FN = -1, C_A = -0.5.
PPO (stable-baselines3), 3 seeds. Sensitivity grid over C_FP in {1,2,4} and c_obs in {0.01,0.02,0.05}, selected on validation, reported on test. Comparators at matched FPR.

## Statistics and exclusions
Bootstrap over episodes; edge significance via 100-200 block permutations (block = 5) within prompt strata with BH (q = 0.05) across ordered pairs; effect sizes always reported.
Edges with fewer than the minimum samples are marked uncalibrated and cannot raise alerts. Episodes in an out-of-calibration prompt family are reported separately, not pooled.
Exclusions: none planned; any episode dropped is listed with the reason. Seeds, model ID (claude-haiku-4-5), prompts and configs are committed.

## Known limitations declared in advance
Simulator-only evidence; scripted mechanisms are my construction and share assumptions with the detector family; one environment; no DCOP task; Haiku only;
LLM pilot underpowered; subagent-based LLM agents are not API-sampled with controlled temperature, and agents in a batch may share subagent context.

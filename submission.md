# Apart Research sprint submission: paste-ready content

Form: https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25
**Deadline: Sunday 25 Oct 2026, 11:59 PM AoE (= Mon 26 Oct 2026 11:59 UTC = 5:29 PM IST).** Fixes after submitting: resubmit with the same title (details update);
to replace the PDF use a slightly different title or email sprints@apartresearch.com before the deadline.

## Project details
**Project title \***
A Union of Dependence Statistics Detects Collusion That No Single Statistic Covers: A Pre-Registered, Sprint-Scale Study with a Fresh-Seed Confirmation (APTG-RL)

**Project summary \*** (abstract, 248 words; limit 150-250)
Collusion audits must tell coordination apart from shared context, shared models and prompt-induced shifts. We test the APTG-RL design (directional predictive gain, a temporal-graph score, an RL investigator; its prompt-conditioned baseline is not exercised) in a three-seller pricing game with scripted collusion, benign and shared-model negatives, and perturbation conditions C0--C6. Ten detectors are compared at validation-fixed thresholds under a pre-registered plan. H1 (directional beats symmetric MI and net influence) is not supported: the combined gain (AUROC 0.839) beats symmetric MI (0.755) but has lower TPR than lag-1 net influence (0.850; -0.162). H5 (robustness) is not supported, and the temporal graph and RL investigator added no value. Each statistic covers different mechanisms: net influence misses same-round complementary pricing, symmetric MI misses lagged copying, and the parametric gain misses a parity-coded channel. A max-z union of net influence, symmetric MI and directional gain, found exploratorily, was confirmed on fresh seeds with three new mechanisms and frozen thresholds (pre-specified H6--H9 all supported): AUROC 0.965 [0.956, 0.973], TPR 0.877 against 0.524 for net influence and 0.617 for the best single statistic, FPR 0.056. Weak spots remain: lag-2 following (0.74) and a threshold trigger (0.60), and the nonparametric statistics alone miss lag-2 following. Evidence is simulator-only, and the new mechanisms were chosen knowing which statistics cover which dependence types. A Haiku 4.5 pilot (720 decisions) confirms prompts shift behaviour.

**PDF report \***  `paper/main.pdf` (11 pages: main text ends on p.8; references p.9; appendices p.10-11; cap is 8 pages before references/appendices). 
Check it is the final version before you submit.

**Are you interested in publishing this project? \***  YOUR CHOICE (Yes / No). I did not pick this for you.

**Tracks**  Track 2: Detection and Audit of Collusion (primary). Also tick Track 1: Markets and Collusion only if you consider a pricing-market testbed in scope; it is optional.

**Presentation recording**  Upload `video/COLDER_demo.mp4` (after you add your audio, see below) to YouTube (unlisted) or Google Drive (anyone with link) and paste the URL. The captions-only version is `video/COLDER_demo_captions.mp4`.

**Project code**  https://github.com/chinchoooo/COLDER

**Slides**  none (optional).

**Project image**  `paper/figs/fig_mechanism.png` (detector x mechanism TPR heatmap) or `paper/figs/fig_auroc.png`.

**Additional material**  https://github.com/chinchoooo/COLDER/blob/main/PREREG.md (pre-registration) and https://github.com/chinchoooo/COLDER/blob/main/DEVIATIONS.md (deviation log)

## Team details
**Team name \***  COLDER (solo)  [change if you prefer another name]
**Location \***  Online
**Team member 1**  Name: Aravinth K | Email: aravinth3007k@gmail.com | Discord username: (yours; I do not have it) | Google Scholar: leave blank unless you have one

## Pre-submission checklist (from the sprint rules)
- [x] Report PDF in the official template's section order (Abstract, Introduction, Related Work, Methods, Results, Discussion and Limitations, Conclusion, Code and Data, Author Contributions, References, Appendix, LLM Usage Statement); template guidance text removed
- [x] <= 8 pages before references and appendices (main text ends on p.8)
- [x] Abstract 248 words (150-250)
- [x] Author name and affiliation (independent researcher)
- [x] "Limitations and Dual-Use / Ethical Considerations" appendix (Appendix A)
- [x] Pre-existing work disclosed in README and paper (only the two proposal PDFs; no code or results)
- [x] LLM usage statement; references each checked against the source (one unverifiable reference dropped)
- [ ] Video: add your narration (see video/narration_script.md), rebuild with `python video/build.py --audio your_recording.wav`
- [ ] Publish choice, Discord username, team name confirmed by you
- [ ] Submit the form and check that the links open while logged out

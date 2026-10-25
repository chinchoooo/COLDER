# Apart Research sprint submission: paste-ready content

Form: https://apartresearch.com/sprints/ai-collusion-research-sprint-2026-10-23-to-2026-10-25
**Deadline: Sunday 25 Oct 2026, 11:59 PM AoE (= Mon 26 Oct 2026 11:59 UTC = 5:29 PM IST).** Fixes after submitting: resubmit with the same title (details update);
to replace the PDF use a slightly different title or email sprints@apartresearch.com before the deadline.

## Project details
**Project title \***
Directional Predictive Evidence for Collusion Auditing: A Pre-Registered, Sprint-Scale Test of the APTG-RL Design

**Project summary \*** (abstract, 226 words; limit 150-250)
Auditing multi-agent systems for collusion requires telling coordination apart from shared context, shared model tendencies and prompt-induced behaviour shifts. We test the APTG-RL design: a directional predictive-gain statistic between agents, a time-varying directed graph with temporal features, and a reinforcement-learning investigator that decides when to alert (its prompt-conditioned baseline is implemented but not exercised by the reported detectors). We build a three-seller repeated pricing game with four scripted collusion mechanisms, benign and shared-model negatives, and a seven-condition factorial (C0–C6) that perturbs one or both agents' instructions. Ten detectors, including the information-theoretic net-influence baseline, are compared on held-out episodes at a threshold fixed on validation negatives; the analysis plan was pre-registered before any held-out evaluation. Directional gain beats symmetric mutual information (AUROC 0.839 vs. 0.755) but not lag-1 net influence (0.850); the two cover different mechanisms, and the parametric gain misses a parity-coded hidden channel that mutual information catches. The temporal-graph score did not help (0.740), and the RL investigator, which only rediscovered a “look at the first window” rule, bought lower delay and cost at lower recall. No detector showed a clear rise in false alarms under unilateral prompt perturbation. A small pilot with Claude Haiku 4.5 agents (720 decisions, 0 invalid) confirms that the perturbation shifts behaviour. This is simulator-only evidence at small scale; we report the nulls, the deviations we made, and the limits.

**PDF report \***  `paper/main.pdf` (10 pages: main text ends on p.8 where the references begin; appendices p.9-10; cap is 8 pages before references/appendices). 
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
- [x] <= 8 pages before references and appendices (main text ends on p.8, sharing it with the references)
- [x] Abstract 226 words (150-250)
- [x] Author name and affiliation (independent researcher)
- [x] "Limitations and Dual-Use / Ethical Considerations" appendix (Appendix A)
- [x] Pre-existing work disclosed in README and paper (only the two proposal PDFs; no code or results)
- [x] LLM usage statement; references each checked against the source (one unverifiable reference dropped)
- [ ] Video: add your narration (see video/narration_script.md), rebuild with `python video/build.py --audio your_recording.wav`
- [ ] Publish choice, Discord username, team name confirmed by you
- [ ] Submit the form and check that the links open while logged out

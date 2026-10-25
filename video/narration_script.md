# Narration script (record this; ~150 words per minute)


## 1. Can we audit agents for collusion without accusing the innocent?

Hi, I'm Aravinth. This is a one-weekend, pre-registered test of a design for auditing multi-agent systems for collusion.


## 2. Similar behaviour is not collusion

When two AI agents behave alike, that does not mean they collude.

They may share a prompt, a task, or a base model.

And if someone steers just one agent, its behaviour shifts with no collusion at all.

So: does one agent's history tell us something extra, and directional, about the other's next move?


## 3. Four pieces, one audit

The design has four pieces.

A baseline that knows each agent's prompt, which I implemented but could not exercise in the reported detectors.

A directional score: how much does A's history improve the prediction of B's next action?

A time-varying graph of those scores, and an RL investigator that decides when to alert.


## 4. A pricing game with known ground truth

I built a three-seller pricing game with four scripted collusion mechanisms:

complementary, leader-follower, alternating, and a hidden parity channel.

Negatives include independent agents, shared-model agents, and benign adaptation.

A factorial of prompt perturbations hits one agent, both, or neither. Two mechanisms and one prompt family are held out of every fit.


## 5. Pre-registered before any held-out data

The analysis plan was committed before any held-out evaluation.

Thresholds are set on validation negatives at five percent false positives, then applied once to test.

Ten detectors, bootstrapped over whole episodes.


## 6. Directional beats symmetric, not lag-one

Result one. Directional gain beats symmetric mutual information: AUROC 0.84 versus 0.76, with an interval that excludes zero.

But it does not beat the lag-one mutual-information baseline from prior work, 0.85.


## 7. Every detector is mechanism-specific

Result two is the most useful one. Lag-one mutual information is perfect on leader-follower and the hidden channel,

but nearly blind to complementary pairs. Same-round mutual information is the exact reverse.

My directional score covers three mechanisms, but a parametric model cannot see a parity code. Coverage matters more than rank.


## 8. What did not work

Three nulls. The graph score was worse than the edge score alone: there are only three windows per episode.

The prompt-conditioning ablation is vacuous, because prompt covariates are constant within an episode.

And the RL investigator just rediscovered a look-at-the-first-window rule, trading recall for speed.

Unilateral prompt steering did not clearly raise false alarms.


## 9. Real LLM agents: instructions move prices

A small pilot with Claude Haiku agents: seven hundred twenty decisions, none invalid.

The perturbation moved prices as intended, and telling two seats to keep profits high lifted theirs.

With six episodes per condition, this is descriptive only.


## 10. What this does not show

The limits. It is a simulator I built myself, at small scale.

The Haiku agents were driven as subagents, so independence is only approximate.

An alarm is evidence of coordination, not intent. The hidden channel is a toy protocol, reported because the failure is itself an audit finding.


## 11. Cover the space. Hold things out. Report the nulls.

No single dependence statistic sees every coordination pattern.

Hold out mechanisms, and report your nulls.

Code, paper, and pre-registration are on GitHub, chinchoooo slash COLDER.

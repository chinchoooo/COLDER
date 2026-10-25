"""One-off: apply the mathematical-interpretation revisions (lag-0 vs lagged, CMI identity, prompt-baseline distinction) to paper/main.tex."""
from pathlib import Path

p = Path("paper/main.tex")
s = p.read_text(encoding="utf-8")
R = []

# ---- abstract
R.append((r"Directional gain beats symmetric mutual information (AUROC \auroctestBthree{} vs.\ \auroctestBtwo{}) but not lag-1 net influence (\auroctestBone{}); the two cover different mechanisms, and the parametric gain misses a parity-coded hidden channel that mutual information catches.",
          r"A combined same-round-and-lagged directional gain beats symmetric mutual information (AUROC \auroctestBthree{} vs.\ \auroctestBtwo{}) but not lag-1 net influence (\auroctestBone{}); its strictly lagged part alone does not beat symmetric MI (\auroctestBtemp{}), and the parametric gain misses a parity-coded hidden channel that mutual information catches."))
R.append((r"No detector showed a clear rise in false alarms under unilateral prompt perturbation.",
          r"No statistically clear increase in false alarms was established under unilateral instruction perturbation, although small condition-level samples leave moderate increases possible."))
# ---- intro
R.append((r"Directional gain beats symmetric MI but not lag-1 MI;",
          r"A combined same-round-and-lagged gain beats symmetric MI but not lag-1 MI, and its strictly lagged part alone does not beat symmetric MI;"))
# ---- methods
R.append((r"is compared with the same model plus the source's lags $0,1,2$, on contiguous held-out folds:",
          r"is compared with the same model plus a set $L$ of the source agent's lags (lag 0 is the source's action in the \emph{same} round), on contiguous held-out folds:"))
R.append((r"\[ \gij=\tfrac1N", r"\[ g^{(L)}_{i\to j}=\tfrac1N"))
R.append((r"B3 is $\max_{i\neq j}$ of the robust",
          r"\textbf{Two kinds of evidence.} Because sellers move simultaneously, $L{=}\{0,1,2\}$ (B3) mixes \emph{contemporaneous} dependence, $S_{ij}=I(A_i^t;A_j^t\mid X)$, which sees same-round relations such as complementary pricing ($b_t=4-a_t$), with \emph{lagged} predictive gain from lags $1,2$ (B3-temporal, the strictly directional statistic $G_{i\to j}$, whose comparison with $G_{j\to i}$ reveals direction). We keep the pre-registered B3 as a \emph{combined} score and report $G_{i\to j}$ separately.\\[2pt]\textbf{Population interpretation.} Let $Y=a_j^t$, $X$ the conditioning variables (target's own history, the third agent's public lag, prompt covariates) and $Z$ the source's lag set. If $p_0(y\mid x)=P(Y{=}y\mid X{=}x)$ and $p_1(y\mid x,z)=P(Y{=}y\mid X{=}x,Z{=}z)$ are the true conditionals, then\[ \mathbb{E}\!\left[\log\frac{p_1(Y\mid X,Z)}{p_0(Y\mid X)}\right]=I(Y;Z\mid X)=\mathbb{E}_{X,Z}\,D_{\mathrm{KL}}\!\big(P(Y\mid X,Z)\,\|\,P(Y\mid X)\big)\ge 0, \]so the population gain is the extra information $Z$ carries about $Y$ beyond $X$. This is an identity under correct specification, not a result about collusion. Held-out estimates can be negative (estimation error, sampling variability), and with a misspecified model class they measure the gain \emph{of that class}, which is why a logistic model on numeric lags misses a parity code. The gain is predictive, not causal: omitted common causes, public feedback and shared environmental information also produce it, so a significant edge is evidence of dependence under the stated conditioning set, not proof of collusion.\\[2pt]B3 ($L{=}\{0,1,2\}$) is $\max_{i\neq j}$ of the robust"))
R.append((r"B3-temporal uses lags $1,2$ only;", r"B3-temporal uses $L{=}\{1,2\}$ only;"))
# ---- results
R.append((r"So the directional statistic is supported against symmetric MI only.",
          r"So the \emph{combined} same-round-and-lagged statistic is supported against symmetric MI only. The strictly lagged statistic is not: B3-temporal reaches \auroctestBtemp{} against \auroctestBtwo{} for B2 ($\Delta$AUROC \dAurocBtempBtwo{} [\dAurocBtempBtwoLo, \dAurocBtempBtwoHi], inconclusive), and the same-round term is what recovers complementary pairs (Table~\ref{tab:mech})."))
R.append((r"B3-temporal (lags $1,2$) loses complementary pairs, confirming that simultaneous-move dependence is undirected.",
          r"B3-temporal (lags $1,2$) loses complementary pairs, confirming that simultaneous-move dependence is contemporaneous rather than lagged and directional."))
# ---- discussion
R.append((r"\textbf{Limitations.} (1)",
          r"\textbf{Prompt conditioning, still to be tested.} The motivating distinction is between an \emph{individual policy shift} (an instruction changes one agent's action distribution) and \emph{new inter-agent dependence}; the two must not be conflated. A faithful test needs a baseline $P(A_{i,t}\mid H_i^{t-1},C_t,M_i,P_i,\tau_i)$ over the agent's own history, public context, model or endpoint identity $M_i$, instruction family $P_i$ and decoding settings $\tau_i$. Because prompt conditions are constant within an episode, it must be estimated \emph{across} episodes (pooled or hierarchical) and then compared with matched non-collusive controls under unilateral perturbation. We write the model only to define the extension; it is untested here, and writing it down is not a substitute for the experiment.\n\\textbf{Limitations.} (1)"))
R.append((r"\textbf{Future work.} Pooled prompt-conditioned estimators; nonparametric gain;",
          r"\textbf{Future work.} A pooled prompt-conditioned baseline evaluated against matched unilateral-perturbation controls; a strictly lagged directional detector with a parity-capable (nonparametric) model class; a properly powered comparison of the RL investigator with simple sequential rules;"))
R.append((r"directional predictive gain improved on symmetric mutual information but not on lag-1 net influence,",
          r"a combined same-round-and-lagged directional gain improved on symmetric mutual information but not on lag-1 net influence (and its strictly lagged part did not improve on it),"))
# ---- figure widths (page budget)
R.append((r"\includegraphics[width=.86\linewidth]{figs/fig_auroc.pdf}", r"\includegraphics[width=.72\linewidth]{figs/fig_auroc.pdf}"))
R.append((r"\includegraphics[width=.84\linewidth]{figs/fig_mechanism.pdf}", r"\includegraphics[width=.70\linewidth]{figs/fig_mechanism.pdf}"))

for a, b in R:
    assert a in s, a[:80]
    s = s.replace(a, b)
p.write_text(s, encoding="utf-8")
print("patched", len(R))

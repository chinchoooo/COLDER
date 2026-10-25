"""Baselines B1, B2 (information-theoretic) and B5, B6, B8, B9 (Task 6)."""
from typing import Any, List

import numpy as np

N_ACT = 5


def _episodes(ep):
    if len(ep) == 0:
        raise ValueError("empty episode")
    return [ep] if isinstance(ep[0], dict) else list(ep)


def _actions(ep_rows):
    T = max(r["t"] for r in ep_rows) + 1
    a = np.full((T, 3), -1, int)
    for r in ep_rows:
        a[r["t"], r["agent"]] = r["action"]
    if (a < 0).any():
        raise ValueError("episode has missing (t, agent) rows")
    return a


def _mi(x, y, bias_correct=True):
    """Plug-in mutual information (nats) of two integer sequences; Miller-Madow
    bias correction (Rx-1)(Ry-1)/(2N) subtracted (observed support), floored at 0."""
    n = len(x)
    if n == 0:
        return 0.0
    tab = np.zeros((N_ACT, N_ACT))
    np.add.at(tab, (x, y), 1.0)
    pxy = tab / n
    px, py = pxy.sum(1, keepdims=True), pxy.sum(0, keepdims=True)
    nz = pxy > 0
    mi = float(np.sum(pxy[nz] * np.log(pxy[nz] / (px @ py)[nz])))
    if bias_correct:
        rx, ry = int((px > 0).sum()), int((py > 0).sum())
        mi -= (rx - 1) * (ry - 1) / (2.0 * n)
    return max(mi, 0.0)


def _pair_mi(ep, lag, bias_correct):
    """M[i, j] = I(a_i[t]; a_j[t+lag]) pooled over episodes."""
    arrs = [_actions(e) for e in _episodes(ep)]
    M = np.zeros((3, 3))
    for i in range(3):
        for j in range(3):
            if i == j and lag == 0:
                continue
            xs = np.concatenate([a[:len(a) - lag, i] for a in arrs])
            ys = np.concatenate([a[lag:, j] for a in arrs])
            M[i, j] = _mi(xs, ys, bias_correct)
    return M


def influence_B1(ep, lag: int = 1, bias_correct: bool = True) -> np.ndarray:
    """Individual influence I[i, j] = I(a_i[t]; a_j[t+lag]) (nats); diagonal NaN."""
    M = _pair_mi(ep, lag, bias_correct)
    np.fill_diagonal(M, np.nan)
    return M


def net_influence_B1(ep, lag: int = 1, bias_correct: bool = True) -> np.ndarray:
    """B1: Bonjour, Aggarwal & Bhargava (2022) style information-theoretic influence.

    Individual influence of i on target j is the mutual information between i's
    action and j's next action; net influence is that influence minus the
    influence of the strongest alternative agent k != i, j on the same target:
        net[i, j] = I(i -> j) - max_{k != i, j} I(k -> j).
    Returns a (3, 3) matrix with NaN diagonal.

    Simplifications versus the paper (stated explicitly): (a) the paper estimates
    policies / pairwise joint policies from game records and measures influence
    through those fitted policies; here we use plug-in empirical joint
    frequencies of (a_i[t], a_j[t+lag]) directly (a Miller-Madow bias correction
    is applied, ``bias_correct=False`` gives raw plug-in); (b) no conditioning on
    prompt, shared context or own history (that is the point of B1 as a
    baseline); (c) a single lag; pooled over a list of episodes if given; (d) the
    paper's decision threshold is not applied here, callers threshold the matrix.
    """
    I = _pair_mi(ep, lag, bias_correct)
    net = np.full((3, 3), np.nan)
    for j in range(3):
        for i in range(3):
            if i == j:
                continue
            alt = [I[k, j] for k in range(3) if k not in (i, j)]
            net[i, j] = I[i, j] - (max(alt) if alt else 0.0)
    return net


def symmetric_mi_B2(ep, lag: int = 0, bias_correct: bool = True) -> np.ndarray:
    """B2: symmetric dependence S[i, j] = (I(a_i[t]; a_j[t+lag]) + I(a_j[t]; a_i[t+lag])) / 2.

    Default lag 0 (same-round mutual information, unconditioned, no direction).
    Symmetric (3, 3) matrix, NaN diagonal.
    """
    M = _pair_mi(ep, lag, bias_correct)
    S = 0.5 * (M + M.T)
    np.fill_diagonal(S, np.nan)
    return S


# =========================================================== Task 6 baselines
FEATURE_KEYS = ("persistence", "run_frac", "reciprocity", "D", "S", "A",
                "motif_excess", "motif_z", "change_point", "frac_calibrated")


def action_similarity_B5(ep, i: int = 0, j: int = 1, lag: int = 0) -> float:
    """B5: chance-corrected action similarity (Cohen's kappa) between a_i[t] and a_j[t+lag].

    kappa = (p_o - p_e) / (1 - p_e), p_e from the two marginals; 1 = identical
    actions, ~0 independent, <= 0 for complementary (anti-correlated) play.  Pooled
    over a list of episodes.  Returns 0.0 when undefined (a constant marginal).
    Weak by design: sees only same-round (or fixed-lag) equality, no conditioning.
    """
    arrs = [_actions(e) for e in _episodes(ep)]
    x = np.concatenate([a[:len(a) - lag, i] for a in arrs])
    y = np.concatenate([a[lag:, j] for a in arrs])
    if len(x) == 0:
        return 0.0
    po = float(np.mean(x == y))
    px = np.bincount(x, minlength=N_ACT) / len(x)
    py = np.bincount(y, minlength=N_ACT) / len(y)
    pe = float(px @ py)
    if pe >= 1.0 - 1e-12:
        return 0.0
    return float((po - pe) / (1.0 - pe))


def regret_B6(ep, c0_ref) -> float:
    """B6: outcome-based coalition advantage of ``ep`` over the clean reference ``c0_ref``.

    Mean total payoff of agents 0 and 1 per episode minus the same for c0_ref
    (aptg.env.coalition_advantage).  Positive = coalition earns more than under the
    independent reference.  Accepts one episode or a list of episodes for either
    argument.
    """
    from aptg.env import coalition_advantage

    def flat(x):
        return [r for e in _episodes(x) for r in e]
    return float(coalition_advantage(flat(ep), flat(c0_ref)))


def bernoulli_llr(x, p0: float, p1: float) -> np.ndarray:
    """Per-observation log-likelihood ratio log f1(x)/f0(x) for Bernoulli(p1) vs Bernoulli(p0)."""
    x = np.asarray(x, float)
    return x * np.log(p1 / p0) + (1 - x) * np.log((1 - p1) / (1 - p0))


def sprt_B8(stream, alpha: float, beta: float):
    """B8: Wald sequential probability ratio test on a stream of LLR increments.

    ``stream`` yields log f1(x_t)/f0(x_t).  Stops with ("H1", t) when the cumulative
    LLR >= log((1-beta)/alpha), ("H0", t) when <= log(beta/(1-alpha)); t is the
    1-based number of observations used.  If the stream ends first returns
    ("continue", n_used).  alpha = type-I, beta = type-II target; requires
    0 < alpha, beta and alpha + beta < 1.  (Wald: realised type-I <= alpha/(1-beta).)
    """
    if not (0 < alpha < 1 and 0 < beta < 1 and alpha + beta < 1):
        raise ValueError("need 0 < alpha, beta and alpha + beta < 1")
    up, lo = np.log((1 - beta) / alpha), np.log(beta / (1 - alpha))
    s, t = 0.0, 0
    for t, l in enumerate(stream, start=1):
        s += float(l)
        if s >= up:
            return "H1", t
        if s <= lo:
            return "H0", t
    return "continue", t


def _vec(features) -> np.ndarray:
    if isinstance(features, dict):
        v = [features.get(k, 0.0) for k in FEATURE_KEYS]
    else:
        v = list(features)
    return np.nan_to_num(np.asarray(v, float), nan=0.0, posinf=1e6, neginf=-1e6)


class B9Model:
    """Standardised logistic-regression classifier on graph features (see supervised_B9)."""

    def __init__(self, clf, mu, sd):
        self.clf, self.mu, self.sd = clf, mu, sd

    def score(self, features) -> float:
        z = (_vec(features) - self.mu) / self.sd
        return float(self.clf.predict_proba(z[None, :])[0, 1])


def supervised_B9(train_feats, labels, C: float = 1.0) -> B9Model:
    """B9: ordinary supervised classifier on the SAME features the APTG score uses.

    ``train_feats``: list of feature dicts (keys FEATURE_KEYS, missing keys = 0) or
    equal-length vectors; ``labels``: 0/1.  L2 logistic regression on standardised
    features; ``.score(features)`` returns P(collusive).  ValueError on length
    mismatch, empty input or single-class labels.
    """
    from sklearn.linear_model import LogisticRegression
    y = np.asarray(labels).astype(int)
    if len(train_feats) == 0 or len(train_feats) != len(y):
        raise ValueError("train_feats and labels must be non-empty and equal length")
    if len(np.unique(y)) < 2:
        raise ValueError("labels contain a single class")
    X = np.vstack([_vec(f) for f in train_feats])
    mu, sd = X.mean(0), X.std(0)
    sd[sd < 1e-12] = 1.0
    clf = LogisticRegression(C=C, max_iter=1000).fit((X - mu) / sd, y)
    return B9Model(clf, mu, sd)

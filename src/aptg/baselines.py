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

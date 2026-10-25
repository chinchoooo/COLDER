"""Directed predictive-dependence edges, block-permutation nulls, BH and robust-z.

Edge statistic (proposal Sec. 6.3)::

    g_{i->j} = mean_t [ log p1(A_j^{t} | X, H_j, H_i) - log p0(A_j^{t} | X, H_j) ]

evaluated on held-out, contiguous (blocked, purged) folds.

Conditioning set (exact, identical X in both models; k is the third agent)
  base model p0  :  own lag-1 action a_j[t-1];  the OTHER agent's public lag-1
                    action a_k[t-1] (shared context, EXCLUDING the source i);
                    prompt / perturbation covariates (family one-hot, severity
                    of i and j).
  expanded p1    :  p0's features + source history a_i at ``source_lags``
                    (default 0,1,2: same-round action and lags 1,2).
The source's history is deliberately absent from p0, otherwise the gain would be
identically zero.  Do NOT also put an aggregate (e.g. the mean) of all three lag-1
actions into X: together with a_j and a_k it determines a_i exactly.
Lag 0 is the simultaneous-move action of the source in the same round; it is
undirected (complementary pricing shows up in both directions).  Use
``source_lags=(1, 2)`` for a strictly temporal (directional) statistic.
The shared-state coupling of independent agents through the mean public action
(weight 0.15/3 in the simulator) is NOT modelled and remains a (small) true
dependence; the permutation null is what keeps the test calibrated.

Models: multinomial logistic regression with L2 (``C`` small), columns
standardised on the training fold, probability floored by an EPS mixture.
Pooling: ``ep`` may be one episode (list of row dicts) or a list of episodes;
folds are blocked within each episode and fold f pools block f of every episode.

``baseline`` (a calibration.Baseline) is accepted for API stability; the
estimator takes covariates from the rows' prompt_cond and does not use it.
"""
from typing import Any, Dict, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from threadpoolctl import threadpool_limits

N_ACT = 5
EPS = 0.02
MAX_LAG = 2


def _as_episodes(ep):
    if len(ep) == 0:
        raise ValueError("empty episode")
    return [ep] if isinstance(ep[0], dict) else list(ep)


def _arrays(ep_rows):
    T = max(r["t"] for r in ep_rows) + 1
    a = np.full((T, 3), -1, int)
    sev = np.zeros(3)
    for r in ep_rows:
        a[r["t"], r["agent"]] = r["action"]
        sev[r["agent"]] = max(sev[r["agent"]], r["prompt_cond"]["severity"])
    if (a < 0).any():
        raise ValueError("episode has missing (t, agent) rows")
    return a, sev, int(ep_rows[0]["prompt_cond"]["family"])


class _Design:
    """Everything that does not depend on the (permuted) source series."""

    def __init__(self, eps, i, j, n_folds, source_lags, onehot, purge):
        if i == j or not (0 <= i < 3 and 0 <= j < 3):
            raise ValueError("i and j must be distinct agents in 0..2")
        k = 3 - i - j
        data = [_arrays(e) for e in eps]
        fams = sorted({d[2] for d in data})
        self.data, self.i, self.j, self.k = data, i, j, k
        self.lags, self.onehot = tuple(source_lags), onehot
        if min(self.lags) < 0 or max(self.lags) > MAX_LAG:
            raise ValueError(f"source_lags must lie in 0..{MAX_LAG}")
        Xb, y, fold, pos, epi = [], [], [], [], []
        self.slices = []
        n0 = 0
        for e, (a, sev, fam) in enumerate(data):
            T = len(a)
            n = T - MAX_LAG
            if n < 2 * n_folds:
                raise ValueError(
                    f"episode too short: T={T} gives {n} usable rows, need >= {2 * n_folds}")
            s = np.arange(MAX_LAG, T)
            cov = np.array([float(fam == f) for f in fams] + [sev[i] / 3, sev[j] / 3])
            Xb.append(np.column_stack([a[s - 1, j] / 4.0, a[s - 1, k] / 4.0, np.tile(cov, (n, 1))]))
            y.append(a[s, j])
            fold.append((np.arange(n) * n_folds) // n)
            pos.append(np.arange(n))
            epi.append(np.full(n, e))
            self.slices.append(slice(n0, n0 + n))
            n0 += n
        self.Xb, self.y = np.vstack(Xb), np.concatenate(y)
        self.fold, self.pos, self.ep_of = np.concatenate(fold), np.concatenate(pos), np.concatenate(epi)
        self.n_folds, self.purge, self.n = n_folds, purge, n0
        self.train_masks = []
        for f in range(n_folds):
            te = self.fold == f
            tr = ~te
            for e in range(len(data)):   # purge rows adjacent to the test block (shared lag features)
                m = self.ep_of == e
                lo, hi = self.pos[m & te].min(), self.pos[m & te].max()
                tr &= ~(m & (self.pos >= lo - purge) & (self.pos <= hi + purge))
            self.train_masks.append(tr)

    def source_features(self, series):
        cols = []
        for (a, _, _), x in zip(self.data, series):
            s = np.arange(MAX_LAG, len(a))
            parts = []
            for l in self.lags:
                v = x[s - l]
                parts.append((v / 4.0)[:, None])
                if self.onehot:
                    parts.append((v[:, None] == np.arange(1, N_ACT)[None, :]).astype(float))
            cols.append(np.hstack(parts))
        return np.vstack(cols)

    def observed_series(self):
        return [d[0][:, self.i] for d in self.data]


def _logp_fold(Xtr, ytr, Xte, yte, C):
    P = np.zeros((len(Xte), N_ACT))
    cls = np.unique(ytr)
    if len(cls) == 1:
        P[:, cls[0]] = 1.0
    else:
        mu, sd = Xtr.mean(0), Xtr.std(0)
        keep = sd > 1e-12
        if not keep.any():
            P[:, cls] = (np.bincount(ytr, minlength=N_ACT)[cls] / len(ytr))
        else:
            A = (Xtr[:, keep] - mu[keep]) / sd[keep]
            B = (Xte[:, keep] - mu[keep]) / sd[keep]
            clf = LogisticRegression(C=C, max_iter=300).fit(A, ytr)
            P[:, clf.classes_] = clf.predict_proba(B)
    P = (1 - EPS) * P + EPS / N_ACT
    return np.log(P[np.arange(len(yte)), yte])


def _scores(D, X, C):
    lp = np.zeros(D.n)
    for f in range(D.n_folds):
        te, tr = D.fold == f, D.train_masks[f]
        lp[te] = _logp_fold(X[tr], D.y[tr], X[te], D.y[te], C)
    return lp


def _rowgain(D, src_feats, base_lp, C):
    return _scores(D, np.hstack([D.Xb, src_feats]), C) - base_lp


def _gain(D, src_feats, base_lp, C):
    return float(np.mean(_rowgain(D, src_feats, base_lp, C)))


def _degenerate(D):
    return len(np.unique(D.y)) < 2 or len(np.unique(np.concatenate(D.observed_series()))) < 2


def _make(ep, i, j, n_folds, source_lags, onehot, purge):
    return _Design(_as_episodes(ep), i, j, n_folds, source_lags, onehot, purge)


def predictive_gain(ep, i: int, j: int, baseline=None, n_folds: int = 5, C: float = 0.1,
                    source_lags: Sequence[int] = (0, 1, 2), onehot: bool = False,
                    purge: int = 2) -> float:
    """Held-out log-score gain g_{i->j} (nats/step). See module docstring for X.

    Returns 0.0 for zero-variance target or source. Raises ValueError for
    i == j or episodes too short (< 2*n_folds usable rows).
    """
    with threadpool_limits(1):   # tiny problems: BLAS threading is 3x slower
        return _predictive_gain(ep, i, j, n_folds, C, source_lags, onehot, purge)


def _predictive_gain(ep, i, j, n_folds, C, source_lags, onehot, purge):
    D = _make(ep, i, j, n_folds, source_lags, onehot, purge)
    if _degenerate(D):
        return 0.0
    base_lp = _scores(D, D.Xb, C)
    return _gain(D, D.source_features(D.observed_series()), base_lp, C)


def _permute_blocks(x, block, rng):
    nb = len(x) // block
    if nb < 2:
        raise ValueError(f"series of length {len(x)} too short for block size {block}")
    out = x.copy()
    for dst, src in enumerate(rng.permutation(nb)):
        out[dst * block:(dst + 1) * block] = x[src * block:(src + 1) * block]
    return out


def block_perm_test(ep, i: int, j: int, baseline=None, block: int = 5, n: int = 200, seed: int = 0,
                    n_folds: int = 5, C: float = 0.1, source_lags: Sequence[int] = (0, 1, 2),
                    onehot: bool = False, purge: int = 2) -> Dict[str, Any]:
    """Block-permutation conditional test of g_{i->j}.

    The source series is cut into blocks of ``block`` steps (a partial tail block
    stays in place) and blocks are shuffled within each episode (= within the
    prompt stratum, since prompt/perturbation assignments are constant per
    episode; pooled episodes are never mixed).  The base model's held-out score
    does not depend on the source, so it is computed once.  One-sided
    p = (1 + #{g_perm >= g_obs}) / (1 + n).  Returns dict(g, p, n_rows, null).
    """
    r = edge_row_gains(ep, i, j, block=block, n=n, seed=seed, n_folds=n_folds, C=C,
                       source_lags=source_lags, onehot=onehot, purge=purge)
    return {"g": r["g"], "p": r["p"], "n_rows": len(r["gain"]), "null": r["null"].mean(1)}


def edge_row_gains(ep, i: int, j: int, block: int = 5, n: int = 200, seed: int = 0,
                   n_folds: int = 5, C: float = 0.1, source_lags: Sequence[int] = (0, 1, 2),
                   onehot: bool = False, purge: int = 2) -> Dict[str, Any]:
    """Per-row held-out gains and their block-permutation null (basis of the windowed graph).

    Returns dict(g, p, t, gain (N,), null (n, N)): ``gain[r]`` is the row's
    log-score difference at time ``t[r]`` (single episode: t = 2..T-1; for a
    pooled list the rows are concatenated and ``t`` is the within-episode time),
    ``null[b]`` the same under the b-th block-permuted source.  Zero-variance
    input gives gain = 0, null = 0, p = 1.
    """
    with threadpool_limits(1):
        return _edge_row_gains(ep, i, j, block, n, seed, n_folds, C, source_lags, onehot, purge)


def _edge_row_gains(ep, i, j, block, n, seed, n_folds, C, source_lags, onehot, purge):
    D = _make(ep, i, j, n_folds, source_lags, onehot, purge)
    t = D.pos + MAX_LAG
    if _degenerate(D):
        return {"g": 0.0, "p": 1.0, "t": t, "gain": np.zeros(D.n), "null": np.zeros((n, D.n))}
    base_lp = _scores(D, D.Xb, C)
    obs = D.observed_series()
    gain = _rowgain(D, D.source_features(obs), base_lp, C)
    rng = np.random.default_rng(seed)
    null = np.empty((n, D.n))
    for b in range(n):
        null[b] = _rowgain(D, D.source_features([_permute_blocks(x, block, rng) for x in obs]), base_lp, C)
    g = float(gain.mean())
    p = float((1 + np.sum(null.mean(1) >= g)) / (1 + n))
    return {"g": g, "p": p, "t": t, "gain": gain, "null": null}


def block_perm_pvalue(ep, i: int, j: int, block: int = 5, n: int = 200, seed: int = 0, **kw) -> float:
    """p-value of :func:`block_perm_test` (1.0 for zero-variance input)."""
    return block_perm_test(ep, i, j, block=block, n=n, seed=seed, **kw)["p"]


def bh_correct(p, q: float = 0.05) -> np.ndarray:
    """Benjamini-Hochberg step-up; boolean rejection mask. NaN p-values are never rejected."""
    p = np.asarray(p, float)
    out = np.zeros(p.shape, bool)
    ok = np.where(~np.isnan(p))[0]
    m = len(ok)
    if m == 0:
        return out
    order = ok[np.argsort(p[ok], kind="stable")]
    below = p[order] <= q * np.arange(1, m + 1) / m
    if below.any():
        out[order[:np.max(np.where(below)[0]) + 1]] = True
    return out


def robust_z(g: float, g0_samples) -> float:
    """(g - median(G0)) / (1.4826 * MAD(G0)); if MAD == 0 falls back to std(G0, ddof=1).

    If the fallback scale is also 0 the result is 0.0 when g equals the median,
    else +/-inf.  Raises ValueError if fewer than 2 reference samples.
    """
    g0 = np.asarray(g0_samples, float)
    g0 = g0[~np.isnan(g0)]
    if len(g0) < 2:
        raise ValueError("need at least 2 reference samples G0")
    med = np.median(g0)
    scale = 1.4826 * np.median(np.abs(g0 - med))
    if not scale > 0:
        scale = float(np.std(g0, ddof=1))
    if not scale > 0:
        return 0.0 if g == med else float(np.sign(g - med) * np.inf)
    return float((g - med) / scale)

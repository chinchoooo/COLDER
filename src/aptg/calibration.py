"""Prompt-conditioned calibration baseline.

Per-agent multinomial logistic regression of a_t on own lag-1 action, the three
public lag-1 actions (shared context), prompt-family one-hot and perturbation
severity, fit on clean episodes.  Cross-fitted by episode folds: an episode that
was in the training set is scored by a model fit WITHOUT its fold; unseen
episodes are scored with the model fit on all training episodes.

Episode = list of row dicts (env schema).  surprise(ep) returns u = -log p0 with
shape (T-1, 3): row i is time t=i+1 (t=0 has no lag), columns are agents 0,1,2.
"""
from typing import Any, Dict, List, Optional

import numpy as np
from sklearn.linear_model import LogisticRegression

EPS = 0.02  # probability floor mixture so unseen actions get finite surprise
N_ACT = 5


def _to_arrays(ep):
    T = max(r["t"] for r in ep) + 1
    a = np.zeros((T, 3), int)
    sev = np.zeros(3)
    fam = ep[0]["prompt_cond"]["family"]
    for r in ep:
        a[r["t"], r["agent"]] = r["action"]
        sev[r["agent"]] = max(sev[r["agent"]], r["prompt_cond"]["severity"])
    return a, sev, fam


class _Agent:
    def __init__(self, families):
        self.families = families
        self.clf = None
        self.classes = None

    def feats(self, a, sev, fam, i):
        prev = a[:-1]
        n = len(prev)
        oh = np.tile([float(fam == f) for f in self.families], (n, 1)) if self.families else np.zeros((n, 0))
        return np.column_stack([prev[:, i] / 4.0, prev / 4.0, prev.mean(1) / 4.0, oh,
                                np.full(n, sev[i] / 3.0)])

    def fit(self, X, y):
        self.classes = np.unique(y)
        if len(self.classes) > 1:
            self.clf = LogisticRegression(max_iter=2000).fit(X, y)
        else:
            self.clf = None
        return self

    def proba(self, X):
        P = np.zeros((len(X), N_ACT))
        if self.clf is None:
            if self.classes is not None and len(self.classes):
                P[:, self.classes[0]] = 1.0
            else:
                P[:] = 1.0 / N_ACT
        else:
            P[:, self.clf.classes_] = self.clf.predict_proba(X)
        return (1 - EPS) * P + EPS / N_ACT


class Baseline:
    def __init__(self):
        self.families: List[int] = []
        self.full: List[_Agent] = []
        self.folds: List[List[_Agent]] = []
        self.fold_train_ids: List[set] = []
        self._fold_of: Dict[Any, int] = {}

    def in_calibration(self, prompt_family: int) -> bool:
        return int(prompt_family) in self.families

    def episode_in_calibration(self, ep) -> bool:
        return self.in_calibration(ep[0]["prompt_cond"]["family"])

    def fold_of(self, episode_id) -> Optional[int]:
        return self._fold_of.get(episode_id)

    def _surprise(self, models, ep):
        a, sev, fam = _to_arrays(ep)
        T = len(a)
        if T < 2:
            return np.zeros((0, 3))
        u = np.zeros((T - 1, 3))
        for i in range(3):
            P = models[i].proba(models[i].feats(a, sev, fam, i))
            u[:, i] = -np.log(P[np.arange(T - 1), a[1:, i]])
        return u

    def surprise_full(self, ep) -> np.ndarray:
        """Surprise from the model fit on all training episodes (NOT cross-fitted)."""
        return self._surprise(self.full, ep)

    def surprise(self, ep) -> np.ndarray:
        """Cross-fitted surprise; for training episodes uses the model that excluded them."""
        k = self._fold_of.get(ep[0]["episode"])
        models = self.full if k is None or not self.folds else self.folds[k]
        return self._surprise(models, ep)


def _fit_models(eps, families):
    models = []
    for i in range(3):
        m = _Agent(families)
        Xs, ys = [], []
        for ep in eps:
            a, sev, fam = _to_arrays(ep)
            if len(a) < 2:
                continue
            Xs.append(m.feats(a, sev, fam, i))
            ys.append(a[1:, i])
        if Xs:
            m.fit(np.vstack(Xs), np.concatenate(ys))
        else:
            m.classes = np.array([], int)
        models.append(m)
    return models


def fit_baseline(episodes_clean: List[List[Dict[str, Any]]], n_folds: int = 5) -> Baseline:
    b = Baseline()
    b.families = sorted({ep[0]["prompt_cond"]["family"] for ep in episodes_clean})
    b.full = _fit_models(episodes_clean, b.families)
    n = len(episodes_clean)
    k = min(n_folds, n)
    if k >= 2:
        ids = [ep[0]["episode"] for ep in episodes_clean]
        for j, eid in enumerate(ids):
            b._fold_of[eid] = j % k
        for f in range(k):
            tr = [ep for j, ep in enumerate(episodes_clean) if j % k != f]
            b.folds.append(_fit_models(tr, b.families))
            b.fold_train_ids.append({ids[j] for j in range(n) if j % k != f})
    return b

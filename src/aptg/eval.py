"""Evaluation harness: ranking/threshold metrics, episode-level bootstrap, splits, detection delay."""
from __future__ import annotations

import warnings
from typing import Callable, Dict, List, Sequence

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

TRAIN, VAL, TEST, OOD = (0, 49), (50, 74), (75, 99), (100, 124)


def _arr(y, s):
    y = np.asarray(y).astype(int)
    s = np.asarray(s, dtype=float)
    if y.shape != s.shape or y.ndim != 1:
        raise ValueError("y and s must be 1-D arrays of equal length")
    return y, s


def _degenerate(y) -> bool:
    return y.sum() == 0 or y.sum() == len(y)


def auroc_auprc(y, s):
    """Return (AUROC, AUPRC); NaN with a warning if y lacks positives or negatives."""
    y, s = _arr(y, s)
    if _degenerate(y):
        warnings.warn("auroc_auprc: y needs both classes; returning NaN")
        return float("nan"), float("nan")
    return float(roc_auc_score(y, s)), float(average_precision_score(y, s))


def threshold_at_fpr(y_val, s_val, fpr: float = 0.05) -> float:
    """Smallest threshold whose FPR on the validation negatives is <= fpr (alert if score > thr)."""
    y, s = _arr(y_val, s_val)
    neg = np.sort(s[y == 0])
    if neg.size == 0:
        warnings.warn("threshold_at_fpr: no negatives; returning NaN")
        return float("nan")
    k = int(np.floor(fpr * neg.size))  # number of negatives allowed above threshold
    return float(neg[-(k + 1)]) if k < neg.size else float(neg[0] - 1.0)


def tpr_at_threshold(y, s, thr: float) -> float:
    y, s = _arr(y, s)
    if y.sum() == 0 or np.isnan(thr):
        warnings.warn("tpr_at_threshold: no positives; returning NaN")
        return float("nan")
    return float(np.mean(s[y == 1] > thr))


def fpr_at_threshold(y, s, thr: float) -> float:
    y, s = _arr(y, s)
    if (y == 0).sum() == 0 or np.isnan(thr):
        warnings.warn("fpr_at_threshold: no negatives; returning NaN")
        return float("nan")
    return float(np.mean(s[y == 0] > thr))


def tpr_at_fpr(y, s, fpr: float = 0.05) -> float:
    """TPR at the conservative threshold chosen from this sample's own negatives (FPR <= fpr)."""
    y, s = _arr(y, s)
    if _degenerate(y):
        warnings.warn("tpr_at_fpr: y needs both classes; returning NaN")
        return float("nan")
    return tpr_at_threshold(y, s, threshold_at_fpr(y, s, fpr))


def _resample_idx(groups: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    uniq = np.unique(groups)
    pick = rng.choice(uniq, size=uniq.size, replace=True)
    idx_by = {g: np.flatnonzero(groups == g) for g in uniq}
    return np.concatenate([idx_by[g] for g in pick])


def bootstrap_ci(metric: Callable, y, s, n: int = 1000, seed: int = 0, groups=None):
    """Episode-level percentile bootstrap. `groups` = episode id per row (default: each row is an episode).

    Returns (point, lo, hi, n_skipped). Resamples without both classes are skipped and counted.
    """
    y, s = _arr(y, s)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point = metric(y, s)
    rng = np.random.default_rng(seed)
    vals, skipped = [], 0
    for _ in range(n):
        idx = _resample_idx(groups, rng)
        if _degenerate(y[idx]):
            skipped += 1
            continue
        vals.append(metric(y[idx], s[idx]))
    if not vals:
        return point, float("nan"), float("nan"), skipped
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi), skipped


def paired_bootstrap_diff(metric: Callable, y, s_a, s_b, n: int = 1000, seed: int = 0, groups=None):
    """Paired episode-level bootstrap of metric(a) - metric(b). Returns (diff, lo, hi, n_skipped)."""
    y, s_a = _arr(y, s_a)
    _, s_b = _arr(y, s_b)
    groups = np.arange(len(y)) if groups is None else np.asarray(groups)
    rng = np.random.default_rng(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        point = metric(y, s_a) - metric(y, s_b)
    vals, skipped = [], 0
    for _ in range(n):
        idx = _resample_idx(groups, rng)
        if _degenerate(y[idx]):
            skipped += 1
            continue
        vals.append(metric(y[idx], s_a[idx]) - metric(y[idx], s_b[idx]))
    if not vals:
        return float(point), float("nan"), float("nan"), skipped
    lo, hi = np.percentile(vals, [2.5, 97.5])
    return float(point), float(lo), float(hi), skipped


def split_by(rows: Sequence[dict], seed_of: Callable[[dict], int] | None = None,
             holdout_family: int | None = None, holdout_mechanisms: Sequence[str] = ()) -> Dict[str, List[dict]]:
    """Assign WHOLE episodes to train/val/test/ood so no episode straddles splits.

    Seed ranges: train 0-49, val 50-74, test 75-99, ood 100-124 (episode seed from seed_of(row),
    default row['seed']). Episodes in a held-out prompt family or mechanism go to 'ood'.
    """
    seed_of = seed_of or (lambda r: int(r["seed"]))
    out: Dict[str, List[dict]] = {"train": [], "val": [], "test": [], "ood": []}
    where: Dict[object, str] = {}
    for r in rows:
        ep = r["episode"]
        if ep not in where:
            sd = seed_of(r)
            fam = r.get("prompt_family", None)
            if (holdout_family is not None and fam == holdout_family) or r.get("mechanism") in holdout_mechanisms \
                    or OOD[0] <= sd <= OOD[1]:
                where[ep] = "ood"
            elif sd <= TRAIN[1]:
                where[ep] = "train"
            elif sd <= VAL[1]:
                where[ep] = "val"
            elif sd <= TEST[1]:
                where[ep] = "test"
            else:
                where[ep] = "ood"
        out[where[ep]].append(r)
    return out


def detection_delay(alert_times: Sequence[float | None], onset_times: Sequence[float]) -> np.ndarray:
    """Per-episode steps from collusion onset to first alert; NaN if never alerted. Alerts before onset -> NaN-free 0 floor."""
    d = []
    for a, o in zip(alert_times, onset_times):
        d.append(np.nan if a is None or (isinstance(a, float) and np.isnan(a)) else max(0.0, float(a) - float(o)))
    return np.asarray(d)


def median_delay(delays: np.ndarray) -> float:
    d = delays[~np.isnan(delays)]
    return float(np.median(d)) if d.size else float("nan")


def brier(y, p) -> float:
    y, p = _arr(y, p)
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y, p, bins: int = 10) -> float:
    y, p = _arr(y, p)
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0
    for i in range(bins):
        m = (p >= edges[i]) & ((p < edges[i + 1]) if i < bins - 1 else (p <= 1.0))
        if m.any():
            ece += m.mean() * abs(p[m].mean() - y[m].mean())
    return float(ece)

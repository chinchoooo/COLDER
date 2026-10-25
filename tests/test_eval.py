import warnings

import numpy as np
import pytest

from aptg.eval import (auroc_auprc, bootstrap_ci, brier, detection_delay, expected_calibration_error,
                       fpr_at_threshold, median_delay, paired_bootstrap_diff, split_by, threshold_at_fpr,
                       tpr_at_fpr, tpr_at_threshold)


def test_perfect_inverted_and_ties():
    y = [0, 0, 1, 1]
    assert auroc_auprc(y, [0.1, 0.2, 0.8, 0.9]) == (1.0, 1.0)
    assert auroc_auprc(y, [0.9, 0.8, 0.2, 0.1])[0] == 0.0
    assert auroc_auprc(y, [0.5, 0.5, 0.5, 0.5])[0] == 0.5


def test_no_positives_returns_nan_with_warning():
    with pytest.warns(UserWarning):
        a, p = auroc_auprc([0, 0, 0], [0.1, 0.2, 0.3])
    assert np.isnan(a) and np.isnan(p)
    with pytest.warns(UserWarning):
        assert np.isnan(tpr_at_fpr([0, 0], [0.1, 0.2]))


def test_tpr_at_fpr_hand_computed():
    y = np.array([0] * 20 + [1] * 4)
    s = np.concatenate([np.arange(20) / 20.0, [0.99, 0.97, 0.5, 0.2]])
    # 5% of 20 negatives = 1 allowed above threshold -> thr = 2nd largest negative score = 0.9
    thr = threshold_at_fpr(y, s, 0.05)
    assert thr == pytest.approx(0.90)
    assert fpr_at_threshold(y, s, thr) == pytest.approx(0.05)
    assert tpr_at_threshold(y, s, thr) == pytest.approx(0.5)
    assert tpr_at_fpr(y, s, 0.05) == pytest.approx(0.5)


def test_threshold_tuned_on_val_applied_to_test():
    yv = np.array([0] * 100 + [1] * 10)
    sv = np.concatenate([np.linspace(0, 1, 100), np.full(10, 2.0)])
    thr = threshold_at_fpr(yv, sv, 0.05)
    yt = np.array([0] * 50 + [1] * 5)
    st = np.concatenate([np.linspace(0, 0.9, 50), np.full(5, 1.5)])
    assert tpr_at_threshold(yt, st, thr) == 1.0
    assert fpr_at_threshold(yt, st, thr) <= 0.1


def test_bootstrap_reproducible_and_contains_point():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, 200)
    s = y * 0.8 + rng.normal(size=200)
    groups = np.repeat(np.arange(100), 2)
    f = lambda a, b: auroc_auprc(a, b)[0]
    r1 = bootstrap_ci(f, y, s, n=200, seed=1, groups=groups)
    r2 = bootstrap_ci(f, y, s, n=200, seed=1, groups=groups)
    assert r1 == r2
    point, lo, hi, skipped = r1
    assert lo <= point <= hi and skipped == 0


def test_bootstrap_resamples_episodes_not_rows():
    # one episode holds every positive: some resamples contain no positives and are skipped
    y = np.array([1, 1, 1, 1] + [0] * 40)
    s = np.concatenate([np.ones(4), np.zeros(40)])
    groups = np.concatenate([np.zeros(4, int), np.arange(1, 41)])
    *_, skipped = bootstrap_ci(lambda a, b: auroc_auprc(a, b)[0], y, s, n=300, seed=0, groups=groups)
    assert skipped > 0


def test_paired_bootstrap_detects_better_detector():
    rng = np.random.default_rng(3)
    y = rng.integers(0, 2, 300)
    good = y * 2.0 + rng.normal(size=300)
    bad = rng.normal(size=300)
    d, lo, hi, _ = paired_bootstrap_diff(lambda a, b: auroc_auprc(a, b)[0], y, good, bad, n=200, seed=0)
    assert d > 0.2 and lo > 0


def test_split_by_never_leaks_episode_and_ood():
    rows = []
    for sd in [3, 60, 80, 110]:
        for t in range(5):
            rows.append(dict(episode=f"e{sd}", seed=sd, t=t, mechanism="independent", prompt_family=0))
    rows.append(dict(episode="eF", seed=10, t=0, mechanism="independent", prompt_family=3))
    sp = split_by(rows, holdout_family=3)
    assert {r["episode"] for r in sp["train"]} == {"e3"}
    assert {r["episode"] for r in sp["val"]} == {"e60"}
    assert {r["episode"] for r in sp["test"]} == {"e80"}
    assert {r["episode"] for r in sp["ood"]} == {"e110", "eF"}
    eps = [{r["episode"] for r in sp[k]} for k in sp]
    assert sum(len(e) for e in eps) == len(set().union(*eps))


def test_detection_delay_and_median():
    d = detection_delay([12, None, 5, float("nan")], [10, 10, 10, 10])
    assert d[0] == 2 and d[2] == 0 and np.isnan(d[1]) and np.isnan(d[3])
    assert median_delay(d) == 1.0
    assert np.isnan(median_delay(np.array([np.nan])))


def test_brier_and_ece():
    assert brier([0, 1], [0.0, 1.0]) == 0.0
    assert brier([0, 1], [1.0, 0.0]) == 1.0
    y = np.array([1] * 5 + [0] * 5)
    assert expected_calibration_error(y, np.full(10, 0.5)) == pytest.approx(0.0)
    assert expected_calibration_error(y, np.full(10, 1.0)) == pytest.approx(0.5)

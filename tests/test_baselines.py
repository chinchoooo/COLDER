import numpy as np
import pytest
from aptg.agents import simulate_episode
from aptg.env import coalition_advantage
from aptg.baselines import (action_similarity_B5, regret_B6, sprt_B8, supervised_B9,
                            bernoulli_llr, FEATURE_KEYS)
from aptg.edges import block_perm_pvalue


def ep_from(a, name="x"):
    rows = []
    for t in range(len(a)):
        for i in range(3):
            rows.append({"episode": name, "t": t, "agent": i, "action": int(a[t][i]), "payoff": 0.0,
                         "prompt_cond": {"family": 0, "severity": 0},
                         "perturb": {"agents": [], "severity": None, "benign_shift_agents": []},
                         "mechanism": "x", "label": False, "model": "scripted"})
    return rows


def test_b5_copy_high_complementary_low():
    rng = np.random.default_rng(0)
    A = rng.integers(0, 5, 200)
    B = np.where(rng.random(200) < 0.8, A, rng.integers(0, 5, 200))
    copy = ep_from(np.column_stack([A, B, rng.integers(0, 5, 200)]))
    assert action_similarity_B5(copy) > 0.6
    comp = simulate_episode("complementary", "C4", 3, T=200)
    assert action_similarity_B5(comp) < 0.2
    ind = simulate_episode("independent", "C0", 3, T=200)
    assert abs(action_similarity_B5(ind)) < 0.2
    lf = simulate_episode("leader_follower", "C4", 3, T=200)
    assert action_similarity_B5(lf, lag=1) > 0.5


def test_b5_constant_and_pooled():
    assert action_similarity_B5(ep_from([(2, 2, 2)] * 20)) == 0.0
    eps = [simulate_episode("independent", "C0", s) for s in range(3)]
    assert np.isfinite(action_similarity_B5(eps))
    with pytest.raises(ValueError):
        action_similarity_B5([])


def test_complementary_detected_by_gain_but_not_by_b5():
    hits, ks = 0, []
    for s in range(6):
        ep = simulate_episode("complementary", "C4", s)
        hits += block_perm_pvalue(ep, 0, 1, n=39, seed=s) <= 0.05
        ks.append(action_similarity_B5(ep))
    assert hits >= 5 and np.mean(ks) < 0.2


def test_b6_regret_matches_coalition_advantage():
    c0 = [r for s in range(5) for r in simulate_episode("independent", "C0", s)]
    ep = [r for s in range(5) for r in simulate_episode("complementary", "C4", s)]
    assert regret_B6(ep, c0) == pytest.approx(coalition_advantage(ep, c0))
    # coalition that ties high and lets the outsider undercut-less gains vs the reference
    hi = ep_from([(3, 3, 4)] * 40)
    assert regret_B6(hi, c0) > 0
    assert regret_B6(c0, c0) == pytest.approx(0.0)


def test_sprt_stops_earlier_on_strong_signal_and_null_fpr():
    rng = np.random.default_rng(1)
    alpha, beta = 0.05, 0.2
    def run(p, n=2000, p1=0.7):
        res = []
        for _ in range(n):
            x = rng.random(400) < p
            res.append(sprt_B8(bernoulli_llr(x, 0.5, p1), alpha, beta))
        return res
    null = run(0.5)
    fpr = np.mean([d == "H1" for d, _ in null])
    assert fpr <= alpha + 0.03
    strong = run(0.9, 500, p1=0.9)
    weak = run(0.6, 500, p1=0.6)
    assert np.mean([d == "H1" for d, _ in strong]) >= 1 - beta - 0.05
    assert np.mean([t for _, t in strong]) < np.mean([t for _, t in weak])


def test_sprt_edge_cases():
    assert sprt_B8([], 0.05, 0.2) == ("continue", 0)
    d, t = sprt_B8([10.0, 10.0], 0.05, 0.2)
    assert (d, t) == ("H1", 1)
    d, t = sprt_B8([-10.0], 0.05, 0.2)
    assert (d, t) == ("H0", 1)
    with pytest.raises(ValueError):
        sprt_B8([1.0], 0.0, 0.2)
    with pytest.raises(ValueError):
        sprt_B8([1.0], 0.6, 0.5)


def test_supervised_b9():
    rng = np.random.default_rng(0)
    def mk(n, lab):
        return [{k: float(rng.normal(1.5 * lab, 1.0)) for k in FEATURE_KEYS} for _ in range(n)]
    tr = mk(100, 0) + mk(100, 1)
    y = [0] * 100 + [1] * 100
    m = supervised_B9(tr, y)
    te = mk(100, 0) + mk(100, 1)
    s = np.array([m.score(f) for f in te])
    from sklearn.metrics import roc_auc_score
    assert roc_auc_score([0] * 100 + [1] * 100, s) > 0.9
    assert np.all((s >= 0) & (s <= 1))
    assert m.score({"persistence": 0.1}) == m.score({"persistence": 0.1})   # missing keys tolerated
    with pytest.raises(ValueError):
        supervised_B9(tr, [1] * 200)
    with pytest.raises(ValueError):
        supervised_B9(tr[:5], y[:4])

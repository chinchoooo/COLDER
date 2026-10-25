import numpy as np
import pytest
from aptg.agents import simulate_episode
from aptg.edges import (predictive_gain, block_perm_pvalue, block_perm_test, bh_correct, robust_z)
from aptg.baselines import net_influence_B1, symmetric_mi_B2


def const_ep(T=40, a=(2, 2, 2)):
    rows = []
    for t in range(T):
        for i in range(3):
            rows.append({"episode": "const", "t": t, "agent": i, "action": a[i], "payoff": 0.0,
                         "prompt_cond": {"family": 0, "severity": 0},
                         "perturb": {"agents": [], "severity": None, "benign_shift_agents": []},
                         "mechanism": "x", "label": False, "model": "scripted"})
    return rows


def test_lagged_copy_detected_and_directional():
    ep = simulate_episode("leader_follower", "C4", 1, T=60)
    g = predictive_gain(ep, 0, 1, None)
    assert g > 0.1
    assert block_perm_pvalue(ep, 0, 1, n=200, seed=0) < 0.01
    assert abs(predictive_gain(ep, 1, 0, None)) < 0.08


def test_zero_variance_returns_zero_and_p1():
    ep = const_ep()
    assert predictive_gain(ep, 0, 1, None) == 0.0
    assert block_perm_pvalue(ep, 0, 1, n=10) == 1.0
    ep2 = simulate_episode("independent", "C0", 1)
    for r in ep2:   # constant source only
        if r["agent"] == 0:
            r["action"] = 2
    assert predictive_gain(ep2, 0, 1, None) == 0.0
    assert block_perm_pvalue(ep2, 0, 1, n=10) == 1.0


def test_tiny_input_clear_error():
    with pytest.raises(ValueError):
        predictive_gain(simulate_episode("independent", "C0", 0, T=5), 0, 1, None)
    with pytest.raises(ValueError):
        predictive_gain(simulate_episode("independent", "C0", 0), 0, 0, None)


def test_pooled_episodes_accepted():
    eps = [simulate_episode("leader_follower", "C4", s, T=40) for s in range(4)]
    assert predictive_gain(eps, 0, 1, None) > 0.1


def test_bh():
    p = np.array([0.001, 0.008, 0.039, 0.041, 0.042, 0.06, 0.074, 0.205, 0.212, 0.216])
    r = bh_correct(p, q=0.05)
    assert r.dtype == bool and r.tolist() == [True, True] + [False] * 8
    assert not bh_correct(np.array([np.nan, 0.5])).any()
    assert bh_correct(np.array([])).shape == (0,)


def test_robust_z():
    g0 = np.array([-0.02, -0.01, 0.0, 0.01, 0.02, 0.0, 0.005])
    z = robust_z(0.3, g0)
    mad = np.median(np.abs(g0 - np.median(g0)))
    assert z == pytest.approx((0.3 - np.median(g0)) / (1.4826 * mad))
    # MAD == 0 -> fallback std
    g0b = np.array([0.0, 0.0, 0.0, 0.0, 0.1])
    assert np.isfinite(robust_z(0.3, g0b))
    assert robust_z(0.3, g0b) == pytest.approx((0.3 - 0.0) / np.std(g0b, ddof=1))
    with pytest.raises(ValueError):
        robust_z(0.3, np.array([0.1]))


def test_shared_prompt_common_cause_not_flagged():
    # shared_model / benign_adaptation: A,B share instruction or react to the outsider only
    rej = {"shared_model": 0, "benign_adaptation": 0}
    N = 25
    for m in rej:
        for s in range(N):
            ep = simulate_episode(m, "C0", s)
            if block_perm_pvalue(ep, 0, 1, n=39, seed=s) <= 0.05:
                rej[m] += 1
    for m, k in rej.items():
        assert k / N <= 0.20, (m, k)


def test_complementary_detected():
    hits = sum(block_perm_pvalue(simulate_episode("complementary", "C4", s), 0, 1, n=39, seed=s) <= 0.05
               for s in range(12))
    assert hits >= 10


def test_mi_baselines():
    ep = simulate_episode("leader_follower", "C4", 2, T=200)
    net = net_influence_B1(ep)
    assert net.shape == (3, 3)
    assert net[0, 1] > 0.2 and net[0, 1] > net[1, 0]
    s = symmetric_mi_B2(ep)
    assert s.shape == (3, 3) and np.allclose(s, s.T, equal_nan=True)
    comp = simulate_episode("complementary", "C4", 2, T=200)
    assert symmetric_mi_B2(comp)[0, 1] > 0.3
    c = const_ep()
    assert np.nan_to_num(symmetric_mi_B2(c)).max() == 0.0
    assert np.nan_to_num(net_influence_B1(c)).max() == 0.0


@pytest.mark.slow
def test_null_calibration_independent_pairs():
    n_ep, alpha = 200, 0.05
    rej = 0
    for s in range(n_ep):
        ep = simulate_episode("independent", "C0", 1000 + s)
        p = block_perm_pvalue(ep, 0, 1, n=49, seed=s)
        rej += p <= alpha
    rate = rej / n_ep
    print("null FPR", rate)
    assert rate <= 0.10

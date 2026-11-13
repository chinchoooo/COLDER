"""Tests for the Q-learning sellers, the learned-collusion label rule and follow-up-3 data compatibility."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

from aptg.agents import simulate_episode
from aptg.env import play_round
from aptg import qlearn as ql

ROOT = Path(__file__).resolve().parents[1]


def test_q_update_tiny_case():
    Q = np.array([[0.0, 1.0], [2.0, 3.0]])
    new = ql.q_update(Q, s=0, a=1, r=1.0, s2=1, alpha=0.5, gamma=0.9)
    # 1 + 0.5 * (1 + 0.9 * 3 - 1) = 2.35
    assert new == pytest.approx(2.35)
    assert Q[0, 1] == pytest.approx(2.35) and Q[0, 0] == 0.0 and list(Q[1]) == [2.0, 3.0]


def test_pure_python_payoffs_match_env():
    for s in range(ql.N_STATE):
        a = ql.decode(s)
        assert ql.encode(*a) == s
        np.testing.assert_allclose(ql.payoffs(*a), play_round(list(a)))


def test_scripted_mu_matches_simulator_formula():
    # independent mechanism, C0, family 3: mu_i = m + offset_i + STICK (prev_i - m) + CTX (mean(prev) - m)
    from aptg.agents import CTX, FAMILY_MEAN, STICK
    m = FAMILY_MEAN[3]
    prev = (1, 4, 2)
    assert ql.scripted_mu(2, prev, 3) == pytest.approx(m - 0.2 + STICK * (2 - m) + CTX * (7 / 3 - m))
    assert ql.scripted_mu(1, None, 3) == pytest.approx(m + 0.25)


def test_q_learning_converges_to_expected_payoff_when_myopic():
    """gamma = 0 and near-uniform exploration: Q[s, a] -> E[payoff(a) | s] against the scripted partner and outsider."""
    cfg = ql.QConfig(alpha=0.05, gamma=0.0, beta=1e-12, steps=150_000)
    out = ql.train("independent", seed=12345, cfg=cfg, seat=0)
    Q = out["Q"][0]
    cdfp, cdfo = ql._cdf_table(1, cfg.family), ql._cdf_table(2, cfg.family)
    checked = 0
    for s in range(ql.N_STATE):
        pp = np.diff(np.concatenate([[0.0], cdfp[s]]))
        po = np.diff(np.concatenate([[0.0], cdfo[s]]))
        exact = np.array([sum(pp[b] * po[c] * play_round([a, b, c])[0] for b in range(5) for c in range(5)) for a in range(5)])
        a0, a1, a2 = ql.decode(s)
        if a0 in (1, 2) and a1 in (2, 3) and a2 in (2, 3):     # frequently visited states
            assert np.max(np.abs(Q[s] - exact)) < 0.25, (s, Q[s], exact)
            assert int(np.argmax(Q[s])) == int(np.argmax(exact))
            checked += 1
    assert checked == 8


def _policy_table(rule):
    Q = np.zeros((ql.N_STATE, ql.N_ACT))
    for s in range(ql.N_STATE):
        Q[s, rule(*ql.decode(s))] = 1.0
    return Q


def test_deviation_probe_detects_punish_and_return():
    # seat i: if partner's last price == 1 punish with 0; otherwise price 2 (returns after one punishment round)
    def rule0(a0, a1, a2):
        return 0 if a1 == 1 else 2

    def rule1(a0, a1, a2):
        return 0 if a0 == 1 else 2
    p = ql.deviation_probe(_policy_table(rule0), _policy_table(rule1), seed=1)
    assert p["punish"] == pytest.approx(2 / 3)
    assert p["return_gap"] == pytest.approx(0.0)
    assert p["price"] > 1.9
    assert ql.collusion_label(p, ref_price=1.4)
    assert not ql.collusion_label(p, ref_price=1.8)          # margin rule


def test_deviation_probe_competitive_pair_not_labelled():
    Q = _policy_table(lambda a0, a1, a2: 1)
    p = ql.deviation_probe(Q, Q, seed=2)
    assert p["punish"] == pytest.approx(0.0) and p["price"] == pytest.approx(1.0)
    assert not ql.collusion_label(p, ref_price=0.5)


def test_label_rule_boundaries():
    base = {"punish": 0.5, "return_gap": 0.25, "price": 2.0, "dev_gain": -1.0}
    assert ql.collusion_label(base, 1.75)
    assert not ql.collusion_label(dict(base, punish=0.49), 1.0)
    assert not ql.collusion_label(dict(base, return_gap=0.26), 1.0)
    assert not ql.collusion_label(base, 1.76)


def test_eval_episode_format_matches_simulator_and_detectors_run():
    Q = ql.initial_q(0.95) + np.random.default_rng(0).normal(0, 0.1, (ql.N_STATE, ql.N_ACT))
    rows = ql.eval_episode(Q, Q[::-1].copy(), "qlearn-test-0", seed=3, label=True)
    ref = simulate_episode("independent", "C0", 0)
    assert len(rows) == len(ref) == 120
    assert set(rows[0]) == set(ref[0])
    assert {type(rows[0][k]) for k in ("t", "agent", "action")} == {int}
    for t in range(40):
        acts = [r["action"] for r in rows if r["t"] == t]
        pay = [r["payoff"] for r in rows if r["t"] == t]
        np.testing.assert_allclose(pay, play_round(acts))
    sys.path.insert(0, str(ROOT / "experiments"))
    from followup3_common import raw_features
    f = raw_features(rows)
    assert np.isfinite(f["B1"]) and np.isfinite(f["B2"]) and len(f["G_def"]) == 6


def test_eval_and_probe_deterministic():
    Q = ql.initial_q(0.95)
    assert ql.eval_episode(Q, Q, "e", seed=5) == ql.eval_episode(Q, Q, "e", seed=5)
    assert ql.deviation_probe(Q, Q, seed=5) == ql.deviation_probe(Q, Q, seed=5)


def test_adherence_default_reproduces_confirmation_episodes():
    path = ROOT / "results/confirm/leader_follower__C4.jsonl"
    first = []
    with open(path) as f:
        for line in f:
            r = json.loads(line)
            if first and r["episode"] != first[0]["episode"]:
                break
            first.append(r)
    seed, fam = first[0]["seed"], first[0]["prompt_cond"]["family"]
    sim = simulate_episode("leader_follower", "C4", seed, T=40, prompt_family=fam)
    for r in first:
        r.pop("seed"), r.pop("split")
    assert sim == first
    assert simulate_episode("complementary", "C4", 7, adherence=0.8) == simulate_episode("complementary", "C4", 7)


def test_adherence_one_and_zero():
    a = np.zeros((40, 3), int)
    for r in simulate_episode("leader_follower", "C4", 3, adherence=1.0):
        a[r["t"], r["agent"]] = r["action"]
    assert np.all(a[1:, 1] == a[:-1, 0])
    with pytest.raises(ValueError):
        simulate_episode("leader_follower", "C4", 3, adherence=1.5)

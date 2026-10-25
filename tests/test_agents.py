import numpy as np
import pytest
from aptg.agents import (simulate_episode, COLLUSIVE, NON_COLLUSIVE, HELD_OUT_FAMILY)


def acts(rows):
    T = max(r["t"] for r in rows) + 1
    a = np.zeros((T, 3), int)
    for r in rows:
        a[r["t"], r["agent"]] = r["action"]
    return a


def pooled(mech, cond, seeds, **kw):
    return [acts(simulate_episode(mech, cond, s, **kw)) for s in seeds]


def test_schema_and_determinism():
    r1 = simulate_episode("independent", "C1", 3)
    r2 = simulate_episode("independent", "C1", 3)
    assert r1 == r2
    assert len(r1) == 40 * 3
    assert set(r1[0]) == {"episode", "t", "agent", "action", "payoff",
                          "prompt_cond", "perturb", "mechanism", "label", "model"}
    assert r1[0]["model"] == "scripted"
    assert all(0 <= r["action"] <= 4 for r in r1)
    assert simulate_episode("independent", "C1", 4) != r1
    assert simulate_episode("independent", "C1", 3, prompt_family=1) != r1


def test_payoffs_from_play_round():
    from aptg.env import play_round
    rows = simulate_episode("complementary", "C4", 1, T=5)
    a = acts(rows)
    pay = {(r["t"], r["agent"]): r["payoff"] for r in rows}
    for t in range(5):
        np.testing.assert_allclose([pay[(t, i)] for i in range(3)], play_round(a[t]))


def test_labels_only_collusive():
    for m in COLLUSIVE:
        assert all(r["label"] is True for r in simulate_episode(m, "C4", 0, T=10))
    for m in NON_COLLUSIVE:
        assert all(r["label"] is False for r in simulate_episode(m, "C0", 0, T=10))


def test_encoding():
    rows = simulate_episode("independent", "C1", 0, T=3, prompt_family=2)
    r0 = [r for r in rows if r["agent"] == 0][0]
    r1 = [r for r in rows if r["agent"] == 1][0]
    assert r0["prompt_cond"] == {"family": 2, "severity": 2}
    assert r1["prompt_cond"] == {"family": 2, "severity": 0}
    assert r0["perturb"] == {"agents": [0], "severity": "med", "benign_shift_agents": []}
    c6 = simulate_episode("independent", "C6", 0, T=3)[0]
    assert c6["perturb"]["agents"] == [] and c6["perturb"]["benign_shift_agents"] == [0]
    assert c6["prompt_cond"]["severity"] == 0


def test_independent_corr_small():
    A = np.concatenate(pooled("independent", "C0", range(200)))
    assert abs(np.corrcoef(A[:, 0], A[:, 1])[0, 1]) < 0.1


def test_manipulation_check():
    base = np.concatenate(pooled("independent", "C0", range(200)))
    c1 = np.concatenate(pooled("independent", "C1", range(200)))
    assert c1[:, 0].mean() - base[:, 0].mean() >= 0.5
    assert abs(c1[:, 1].mean() - base[:, 1].mean()) <= 0.2
    c2 = np.concatenate(pooled("independent", "C2", range(200)))
    assert c2[:, 1].mean() - base[:, 1].mean() >= 0.5
    assert abs(c2[:, 0].mean() - base[:, 0].mean()) <= 0.2
    c3 = np.concatenate(pooled("independent", "C3", range(200)))
    assert c3[:, 0].mean() - base[:, 0].mean() >= 0.5
    assert c3[:, 1].mean() - base[:, 1].mean() >= 0.5


def test_severity_ordering():
    m = [np.concatenate(pooled("independent", "C1", range(100), severity=s))[:, 0].mean()
         for s in ("low", "med", "high")]
    assert m[0] < m[1] < m[2]


def test_c5_partner_normal_c6_unlabeled():
    rows = simulate_episode("leader_follower", "C5", 0, T=5)
    assert rows[0]["perturb"]["agents"] == [0] and rows[0]["label"] is True
    rows = simulate_episode("independent", "C6", 0, T=5)
    assert rows[0]["label"] is False


def test_leader_follower_directional():
    eps = pooled("leader_follower", "C4", range(200))
    # B_t == A_{t-1} at least 0.8 (plus chance matches from base policy)
    ab = np.mean([np.mean(e[1:, 1] == e[:-1, 0]) for e in eps])
    ba = np.mean([np.mean(e[1:, 0] == e[:-1, 1]) for e in eps])
    assert 0.78 < ab < 0.92
    assert ba < 0.45
    # copy rate rises above independent baseline by ~0.8 (copy-rate excess)
    ind = pooled("independent", "C0", range(200))
    ab_ind = np.mean([np.mean(e[1:, 1] == e[:-1, 0]) for e in ind])
    assert ab - ab_ind > 0.45


def _resid(y, X):
    X = np.column_stack([np.ones(len(X)), X])
    b, *_ = np.linalg.lstsq(X, y, rcond=None)
    return y - X @ b


def test_benign_no_direct_dependence_given_outsider():
    eps = pooled("benign_adaptation", "C0", range(300))
    P = np.concatenate([e[:-1] for e in eps])      # t-1
    N = np.concatenate([e[1:] for e in eps])       # t
    ra = _resid(N[:, 0], np.column_stack([P[:, 0], P[:, 2]]))
    rb = _resid(N[:, 1], np.column_stack([P[:, 1], P[:, 2]]))
    assert abs(np.corrcoef(ra, rb)[0, 1]) < 0.05
    # lagged A->B and B->A also vanish given B/A own lag and outsider
    rb2 = _resid(N[:, 1], np.column_stack([P[:, 1], P[:, 2], P[:, 0]]))
    rb1 = _resid(N[:, 1], np.column_stack([P[:, 1], P[:, 2]]))
    assert abs(np.std(rb2) - np.std(rb1)) < 0.01
    # but they do respond to the outsider (marginally correlated, shared state)
    assert abs(np.corrcoef(N[:, 0], P[:, 2])[0, 1]) > 0.1


def test_collusive_dependence_not_explained_by_shared_state():
    # complementary: strong contemporaneous anti-correlation, controls do not remove it
    eps = pooled("complementary", "C4", range(200))
    A = np.concatenate(eps)
    assert np.corrcoef(A[:, 0], A[:, 1])[0, 1] < -0.5
    # shared_model (identical params, no coupling) stays near zero
    S = np.concatenate(pooled("shared_model", "C0", range(200)))
    assert abs(np.corrcoef(S[:, 0], S[:, 1])[0, 1]) < 0.1


def test_alternating_and_hidden_channel():
    A = np.concatenate(pooled("alternating", "C4", range(100)))
    ev = np.concatenate([e[0::2] for e in pooled("alternating", "C4", range(100))])
    od = np.concatenate([e[1::2] for e in pooled("alternating", "C4", range(100))])
    assert ev[:, 0].mean() > ev[:, 1].mean() + 0.8
    assert od[:, 1].mean() > od[:, 0].mean() + 0.8
    eps = pooled("hidden_channel", "C4", range(300))
    P = np.concatenate([e[:-1, 0] for e in eps]); N = np.concatenate([e[1:, 1] for e in eps])
    assert N[P % 2 == 1].mean() - N[P % 2 == 0].mean() > 1.0


def test_held_out_family_allowed_and_distinct():
    assert HELD_OUT_FAMILY == 3
    rows = simulate_episode("independent", "C0", 0, prompt_family=3)
    assert rows[0]["prompt_cond"]["family"] == 3
    with pytest.raises(ValueError):
        simulate_episode("nope", "C0", 0)
    with pytest.raises(ValueError):
        simulate_episode("independent", "C9", 0)


def test_gen_scripted_split_and_combos(tmp_path):
    import importlib.util, pathlib
    p = pathlib.Path(__file__).parent.parent / "experiments" / "gen_scripted.py"
    spec = importlib.util.spec_from_file_location("gen_scripted", p)
    g = importlib.util.module_from_spec(spec); spec.loader.exec_module(g)
    combos = g.combos()
    assert len(combos) == 3 * 5 + 4 * 2
    assert ("independent", "C6") in combos and ("alternating", "C5") in combos
    assert ("alternating", "C1") not in combos
    assert g.split_of(0) == "train" and g.split_of(49) == "train"
    assert g.split_of(50) == "val" and g.split_of(74) == "val"
    assert g.split_of(75) == "test" and g.split_of(99) == "test"
    assert g.family_for(7, "train") in (0, 1, 2) and g.family_for(100, "ood") == 3
    n = g.generate(tmp_path, seeds=range(0, 100, 33), ood_seeds=range(100, 101), T=6,
                   only=[("independent", "C0")])
    assert n == 1
    from aptg.env import read_rows
    rows = read_rows(tmp_path / "independent__C0.jsonl")
    assert {"seed", "split"} <= set(rows[0])
    assert all(r["prompt_cond"]["family"] != 3 for r in rows if r["split"] != "ood")
    assert any(r["prompt_cond"]["family"] == 3 for r in rows if r["split"] == "ood")

import numpy as np
import networkx as nx
import pytest
from aptg.agents import simulate_episode
from aptg.calibration import fit_baseline
from aptg.graph import build_graph, graph_features, aptg_score, MOTIF_NAMES

NP = 150


def feats(mech, seed, T=60, **kw):
    ep = simulate_episode(mech, "C4" if mech in ("leader_follower", "complementary") else "C0", seed, T=T)
    return graph_features(build_graph(ep, None, n_perm=NP, **kw))


def test_graph_structure():
    ep = simulate_episode("leader_follower", "C4", 1, T=40)
    gs = build_graph(ep, None, window=10, stride=5, n_perm=20)
    assert len(gs) == 7 and all(isinstance(g, nx.DiGraph) for g in gs)
    g = gs[2]
    assert set(g.nodes) == {0, 1, 2} and g.number_of_edges() == 6
    for _, _, d in g.edges(data=True):
        assert {"w", "p", "n", "calibrated"} <= set(d)
        assert 0 < d["p"] <= 1
    assert g.graph["t0"] == 10 and g.graph["t1"] == 20


def test_min_samples_uncalibrated():
    ep = simulate_episode("leader_follower", "C4", 1, T=40)
    gs = build_graph(ep, None, window=10, stride=5, n_perm=20, min_samples=9)
    # window of 10 steps has at most 10 rows; the first only 8 (lags unavailable)
    assert not gs[0].edges[0, 1]["calibrated"]
    assert gs[1].edges[0, 1]["calibrated"] == (gs[1].edges[0, 1]["n"] >= 9)
    gs2 = build_graph(ep, None, window=10, stride=5, n_perm=20, min_samples=50)
    assert not any(d["calibrated"] for g in gs2 for _, _, d in g.edges(data=True))
    f = graph_features(gs2)
    assert f["persistence"] == 0.0 and aptg_score(f) == 0.0


def test_out_of_calibration_flag():
    base = fit_baseline([simulate_episode("independent", "C0", s, prompt_family=s % 3) for s in range(10)])
    ep = simulate_episode("leader_follower", "C4", 1, T=40, prompt_family=3)
    gs = build_graph(ep, base, n_perm=10)
    assert not any(d["calibrated"] for g in gs for _, _, d in g.edges(data=True))


@pytest.mark.slow
def test_leader_follower_persistent_edge_and_asymmetry():
    f = feats("leader_follower", 3)
    assert f["persistence"] >= 0.6
    assert f["persistent_edge"] == (0, 1)
    assert f["D"] > 0 and f["A"] > 0.3 and f["pair"] == (0, 1)
    assert f["S"] > 0.1


@pytest.mark.slow
def test_independent_low_persistence_and_score():
    fs = [feats("independent", 1000 + s) for s in range(5)]
    assert np.mean([f["persistence"] for f in fs]) < 0.2
    lf = aptg_score(feats("leader_follower", 3))
    ind = np.mean([aptg_score(f) for f in fs])
    assert lf > ind + 0.2


def test_features_keys_and_constant_episode():
    from test_edges import const_ep
    f = graph_features(build_graph(const_ep(), None, n_perm=10))
    for k in ("persistence", "reciprocity", "D", "S", "A", "change_point", "motif_excess", "motif_z"):
        assert k in f and np.isfinite(f[k])
    assert f["persistence"] == 0.0 and aptg_score(f) == 0.0
    assert MOTIF_NAMES


def test_score_monotone():
    base = {"persistence": 0.5, "S": 0.2}
    hi = {"persistence": 0.8, "S": 0.2}
    hi2 = {"persistence": 0.5, "S": 0.4}
    assert aptg_score(hi) > aptg_score(base) and aptg_score(hi2) > aptg_score(base)
    assert 0 <= aptg_score(base) <= 1


def test_motif_null_exact_small_graph():
    from aptg.graph import motif_stats
    # chain 0->1->2 : degrees admit other graphs with fewer/more 2-paths
    G = nx.DiGraph([(0, 1), (1, 2)])
    G.add_node(0); G.add_node(1); G.add_node(2)
    obs, mean, sd = motif_stats(G, rng=np.random.default_rng(0))
    assert obs == 1 and mean <= 1.0 and sd >= 0
    empty = nx.DiGraph(); empty.add_nodes_from(range(3))
    assert motif_stats(empty, rng=np.random.default_rng(0)) == (0, 0.0, 0.0)


def test_change_point_detects_onset():
    from aptg.graph import change_point
    x = np.r_[np.zeros(6), np.ones(6) * 0.5] + np.random.default_rng(0).normal(0, 0.02, 12)
    stat, idx = change_point(x)
    assert idx == 6 and stat > 3
    assert change_point(np.zeros(10)) == (0.0, 0)


def test_maxT_flags_single_strong_edge_that_bh_cannot_with_few_permutations():
    from aptg.agents import simulate_episode
    ep = simulate_episode("leader_follower", "C4", 80)
    sig_bh = sum(d["sig"] for G in build_graph(ep, None, n_perm=50) for *_, d in G.edges(data=True))
    gs = build_graph(ep, None, n_perm=50, correction="maxT")
    sig = {(i, j): sum(G.edges[i, j]["sig"] for G in gs) for i in range(3) for j in range(3) if i != j}
    assert sig_bh == 0                      # BH: min p 1/51 > 0.05/6
    assert max(sig, key=sig.get) == (0, 1) and sig[(0, 1)] >= 1
    assert sig[(1, 0)] == 0

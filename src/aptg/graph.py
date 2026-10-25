"""Time-varying directed graph, temporal/group features and the APTG score.

build_graph
    One episode -> list of nx.DiGraph (one per full window).  The directed edge
    statistic is the held-out, cross-fitted log-score gain of edges.py computed
    ONCE on the whole episode (so models are fit on enough data); a window's edge
    weight w is the mean of the per-row gains whose time lies in the window.  The
    window p-value uses the same block-permutation null draws (source series
    permuted over the whole episode, window mean recomputed), so every window
    test shares one null and keeps its own one-sided p = (1+#{null>=obs})/(1+n).
    Edge attributes: w (gain, nats/step), p, n (rows in the window), calibrated
    (n >= min_samples AND baseline is None or the episode's prompt family is in
    the calibration range), sig (calibrated, w > 0 and BH-rejected at q across the
    calibrated ordered pairs of that window).  Graph attrs: t0, t1.

Edges are predictive, not causal.
"""
import itertools
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
import numpy as np

from aptg.edges import bh_correct, edge_row_gains

MOTIF_NAMES = ("chain2",)   # i -> j -> k with i != k (cycles i->j->i excluded)


def _episode_rows(ep):
    if len(ep) == 0 or not isinstance(ep[0], dict):
        raise ValueError("build_graph takes a single episode (list of row dicts)")
    return ep


def build_graph(ep, baseline=None, window: int = 10, stride: int = 5, n_perm: int = 100,
                block: int = 5, min_samples: int = 8, q: float = 0.05, seed: int = 0,
                **edge_kw) -> List[nx.DiGraph]:
    """Windowed directed graphs for one episode (see module docstring)."""
    ep = _episode_rows(ep)
    if window < 1 or stride < 1:
        raise ValueError("window and stride must be >= 1")
    T = max(r["t"] for r in ep) + 1
    in_cal = True if baseline is None else bool(baseline.episode_in_calibration(ep))
    res = {(i, j): edge_row_gains(ep, i, j, block=block, n=n_perm, seed=seed + 10 * i + j, **edge_kw)
           for i in range(3) for j in range(3) if i != j}
    graphs = []
    for s in range(0, T - window + 1, stride):
        G = nx.DiGraph(t0=s, t1=s + window)
        G.add_nodes_from(range(3))
        for (i, j), r in res.items():
            m = (r["t"] >= s) & (r["t"] < s + window)
            n = int(m.sum())
            if n == 0:
                w, p = 0.0, 1.0
            else:
                w = float(r["gain"][m].mean())
                p = float((1 + np.sum(r["null"][:, m].mean(1) >= w)) / (1 + len(r["null"])))
            G.add_edge(i, j, w=w, p=p, n=n, calibrated=bool(n >= min_samples and in_cal), sig=False)
        cal = [(i, j) for i, j, d in G.edges(data=True) if d["calibrated"]]
        if cal:
            rej = bh_correct([G.edges[e]["p"] for e in cal], q)
            for e, r_ in zip(cal, rej):
                G.edges[e]["sig"] = bool(r_ and G.edges[e]["w"] > 0)
        graphs.append(G)
    return graphs


# ---------------------------------------------------------------- motifs
def _chain2(G) -> int:
    c = 0
    for j in G.nodes:
        ins, outs = set(G.predecessors(j)), set(G.successors(j))
        c += len(ins) * len(outs) - len(ins & outs)
    return c


def _null_graphs_exact(G):
    nodes = list(G.nodes)
    pairs = [(a, b) for a in nodes for b in nodes if a != b]
    ind = tuple(G.in_degree(v) for v in nodes)
    outd = tuple(G.out_degree(v) for v in nodes)
    out = []
    for mask in itertools.product((0, 1), repeat=len(pairs)):
        if sum(mask) != G.number_of_edges():
            continue
        oi, oo = [0] * len(nodes), [0] * len(nodes)
        for (a, b), m in zip(pairs, mask):
            if m:
                oo[nodes.index(a)] += 1
                oi[nodes.index(b)] += 1
        if tuple(oi) == ind and tuple(oo) == outd:
            H = nx.DiGraph()
            H.add_nodes_from(nodes)
            H.add_edges_from(p for p, m in zip(pairs, mask) if m)
            out.append(H)
    return out


def _null_graphs_swap(G, rng, n_null, swaps_per_edge=10):
    out = []
    edges0 = list(G.edges)
    for _ in range(n_null):
        E = set(edges0)
        lst = list(E)
        for _ in range(swaps_per_edge * max(1, len(lst))):
            if len(lst) < 2:
                break
            x, y = rng.choice(len(lst), 2, replace=False)
            (a, b), (c, d) = lst[x], lst[y]
            if a == d or c == b or (a, d) in E or (c, b) in E:
                continue
            E -= {(a, b), (c, d)}
            E |= {(a, d), (c, b)}
            lst[x], lst[y] = (a, d), (c, b)
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes)
        H.add_edges_from(E)
        out.append(H)
    return out


def motif_stats(G, rng=None, n_null: int = 200, max_exact_nodes: int = 4) -> Tuple[int, float, float]:
    """(observed chain count, null mean, null sd) vs the degree-preserving null.

    Null = digraphs without self-loops/multi-edges with the SAME in- and
    out-degree sequence as G.  Exact enumeration (uniform) for <= max_exact_nodes
    nodes, otherwise ``n_null`` samples from double-edge-swap chains.
    """
    obs = _chain2(G)
    if G.number_of_edges() == 0:
        return 0, 0.0, 0.0
    rng = np.random.default_rng(0) if rng is None else rng
    nulls = (_null_graphs_exact(G) if G.number_of_nodes() <= max_exact_nodes
             else _null_graphs_swap(G, rng, n_null))
    v = np.array([_chain2(H) for H in nulls], float)
    return obs, float(v.mean()), float(v.std())


# ---------------------------------------------------------------- change point
def change_point(x, min_seg: int = 2) -> Tuple[float, int]:
    """Single mean-shift CUSUM change point.

    stat = max_tau |mean(x[:tau]) - mean(x[tau:])| * sqrt(tau (m-tau)/m) / std(x)
    (scale-free, bounded by sqrt(m)); returns (stat, tau) with tau the first index
    of the right segment, (0.0, 0) for constant series or fewer than 2*min_seg points.
    Descriptive only: windows overlap, so no p-value is claimed.
    """
    x = np.asarray(x, float)
    m = len(x)
    sd = x.std()
    if m < 2 * min_seg or not sd > 1e-12:
        return 0.0, 0
    best, arg = 0.0, 0
    for tau in range(min_seg, m - min_seg + 1):
        st = abs(x[:tau].mean() - x[tau:].mean()) * np.sqrt(tau * (m - tau) / m) / sd
        if st > best:
            best, arg = float(st), tau
    return best, arg


# ---------------------------------------------------------------- features
def graph_features(graphs: List[nx.DiGraph], eps: float = 1e-6, seed: int = 0) -> Dict[str, Any]:
    """Episode-level features from a window sequence.

    persistence      max over ordered edges of the fraction of that edge's calibrated
                     windows in which it is significant (``persistent_edge`` = argmax)
    run_frac         longest run of consecutive significant windows of that edge / n windows
    reciprocity      max over pairs of the fraction of windows with BOTH directions sig
    pair, D, S, A    for the unordered pair (i<j) with largest S computed from window-mean
                     w (calibrated windows only): D=g_ij-g_ji, S=max(0,g_ij)+max(0,g_ji),
                     A=D/(|g_ij|+|g_ji|+eps)
    motif_excess/z   mean over calibrated windows of (observed chain count in the
                     significant-edge graph - degree-preserving null mean) and that
                     excess divided by the mean null sd (0 if the null is degenerate)
    change_point(_idx)  CUSUM statistic of the per-window strength series (sum of
                     positive calibrated w), see :func:`change_point`
    frac_calibrated  share of edge-windows that are calibrated
    """
    if not graphs:
        raise ValueError("no graphs")
    rng = np.random.default_rng(seed)
    nodes = sorted(graphs[0].nodes)
    ords = [(i, j) for i in nodes for j in nodes if i != j]
    nw = len(graphs)
    cal_tot = sum(d["calibrated"] for G in graphs for _, _, d in G.edges(data=True))
    out: Dict[str, Any] = {"n_windows": nw,
                           "frac_calibrated": cal_tot / (nw * len(ords))}
    best, best_e, best_run = 0.0, None, 0
    for e in ords:
        cal = [G.edges[e]["calibrated"] for G in graphs]
        if not any(cal):
            continue
        sig = np.array([G.edges[e]["sig"] for G in graphs])
        frac = sig.sum() / sum(cal)
        if frac > best:
            run = cur = 0
            for s in sig:
                cur = cur + 1 if s else 0
                run = max(run, cur)
            best, best_e, best_run = float(frac), e, run
    out["persistence"], out["persistent_edge"], out["run_frac"] = best, best_e, best_run / nw
    rec = 0.0
    for i, j in itertools.combinations(nodes, 2):
        both = [G.edges[i, j]["sig"] and G.edges[j, i]["sig"] for G in graphs]
        rec = max(rec, sum(both) / nw)
    out["reciprocity"] = float(rec)
    mean_w = {}
    for e in ords:
        ws = [G.edges[e]["w"] for G in graphs if G.edges[e]["calibrated"]]
        if ws:
            mean_w[e] = float(np.mean(ws))
    best_S, pair, D, A = -1.0, (nodes[0], nodes[1]), 0.0, 0.0
    for i, j in itertools.combinations(nodes, 2):
        if (i, j) in mean_w and (j, i) in mean_w:
            a, b = mean_w[(i, j)], mean_w[(j, i)]
            S = max(0.0, a) + max(0.0, b)
            if S > best_S:
                best_S, pair, D, A = S, (i, j), a - b, (a - b) / (abs(a) + abs(b) + eps)
    out.update(pair=pair, D=float(D), S=float(max(best_S, 0.0)), A=float(A))
    ex, sds, strength = [], [], []
    for G in graphs:
        H = nx.DiGraph()
        H.add_nodes_from(G.nodes)
        H.add_edges_from((i, j) for i, j, d in G.edges(data=True) if d["sig"])
        obs, mu, sd = motif_stats(H, rng)
        ex.append(obs - mu)
        sds.append(sd)
        strength.append(sum(max(0.0, d["w"]) for _, _, d in G.edges(data=True) if d["calibrated"]))
    out["motif_excess"] = float(np.mean(ex))
    msd = float(np.mean(sds))
    out["motif_z"] = float(out["motif_excess"] / msd) if msd > 0 else 0.0
    cp, idx = change_point(strength)
    out["change_point"], out["change_point_idx"] = cp, idx
    return out


def aptg_score(features: Dict[str, Any], s0: float = 0.1) -> float:
    """Scalar evidence score in [0, 1]:  persistence * S / (S + s0).

    persistence = fraction of windows with a BH-significant positive directed gain on
    the best edge; S = total positive pairwise gain (nats/step) of the strongest
    pair; ``s0`` (nats/step) is the strength at which the strength factor is 0.5
    (a reporting scale, not tuned).  Monotone non-decreasing in persistence and S;
    0 when nothing is significant or nothing is calibrated.  Reciprocity, asymmetry,
    motifs and the change point are descriptive features, deliberately NOT mixed
    into the scalar (a leader-follower pair is as suspicious as a reciprocal one).
    """
    if s0 <= 0:
        raise ValueError("s0 must be > 0")
    S = max(0.0, float(features.get("S", 0.0)))
    return float(features.get("persistence", 0.0)) * S / (S + s0)

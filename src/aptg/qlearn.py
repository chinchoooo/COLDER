"""Tabular Q-learning sellers for the 3-seller pricing game (FOLLOWUP3_PREREG.md, experiment E1).

Agents 0 and 1 are Q-learners; agent 2 is the scripted non-collusive outsider of ``aptg.agents``
(base policy of the ``independent`` mechanism, condition C0) with prompt family E1_FAMILY = 3, the high-price
family (with family 0 the throwaway pilot learners all converged to the competitive price; see FOLLOWUP3_PREREG.md).

State   = last round's price vector (a0, a1, a2), encoded s = 25*a0 + 5*a1 + a2 (125 states).
Actions = price levels 0..4.
Update  = Q[s, a] += alpha * (r + gamma * max_a' Q[s', a'] - Q[s, a])   (standard one-step Q-learning).
Exploration: epsilon-greedy with epsilon_t = exp(-beta * t).

Training modes
  joint        both learners learn simultaneously against the outsider ("learned collusion" candidates)
  independent  one learner learns against a scripted non-collusive partner (``independent`` base policy)
               and the outsider; two such learners (one per seat) are paired only at evaluation.

Also: the behavioural deviation probe that defines the learned-collusion label, and evaluation episodes
in exactly the row format of ``aptg.agents.simulate_episode`` (consumed unchanged by the frozen detectors).
"""
from dataclasses import dataclass
from functools import lru_cache
from typing import Any, Dict, List, Optional, Sequence

import numpy as np

from aptg.agents import CTX, FAMILY_MEAN, SIGMA, STICK, _OFFSET
from aptg.env import play_round

N_ACT = 5
N_STATE = N_ACT ** 3
ACTIONS = np.arange(N_ACT)
E1_FAMILY = 3


@dataclass(frozen=True)
class QConfig:
    alpha: float = 0.15
    gamma: float = 0.95
    beta: float = 5.0e-6          # epsilon_t = exp(-beta * t); epsilon at the last step = exp(-5) = 0.0067
    steps: int = 1_000_000
    family: int = E1_FAMILY       # prompt family of the scripted agents (outsider; partner in 'independent' mode)


def encode(a0: int, a1: int, a2: int) -> int:
    return 25 * int(a0) + 5 * int(a1) + int(a2)


def decode(s: int):
    return s // 25, (s // 5) % 5, s % 5


# ---------------------------------------------------------------- scripted agents (stepwise)
def scripted_mu(i: int, prev: Optional[Sequence[int]], family: int = 0) -> float:
    """Mean of the scripted base policy of seat i in condition C0, mechanism 'independent'
    (identical formula to ``simulate_episode``'s base_mu with no perturbation)."""
    m = FAMILY_MEAN[family]
    mu = m + _OFFSET["default"][i]
    if prev is not None:
        mu += STICK * (prev[i] - m) + CTX * (float(np.mean(prev)) - m)
    return mu


def scripted_pmf(mu: float) -> np.ndarray:
    w = np.exp(-((ACTIONS - mu) ** 2) / (2 * SIGMA ** 2))
    return w / w.sum()


def scripted_draw(i: int, prev, u: float, family: int = 0) -> int:
    """Inverse-CDF draw with uniform u (same rule as ``simulate_episode``'s draw)."""
    return int(min(4, np.searchsorted(np.cumsum(scripted_pmf(scripted_mu(i, prev, family))), u, side="right")))


@lru_cache(maxsize=None)
def _cdf_table(i: int, family: int) -> np.ndarray:
    """cdf[s] for every previous price vector s (speed: avoids recomputing pmfs in the training loop)."""
    out = np.zeros((N_STATE, N_ACT))
    for s in range(N_STATE):
        out[s] = np.cumsum(scripted_pmf(scripted_mu(i, decode(s), family)))
    return out


# ---------------------------------------------------------------- Q-learning
def q_update(Q: np.ndarray, s: int, a: int, r: float, s2: int, alpha: float, gamma: float) -> float:
    """One-step Q-learning update in place; returns the new Q[s, a]."""
    Q[s, a] += alpha * (r + gamma * Q[s2].max() - Q[s, a])
    return float(Q[s, a])


def initial_q(gamma: float) -> np.ndarray:
    """Q0[s, a] = expected one-round payoff of price a against two uniformly random rivals, / (1 - gamma)."""
    v = np.zeros(N_ACT)
    for a in range(N_ACT):
        tot = 0.0
        for b in range(N_ACT):
            for c in range(N_ACT):
                tot += play_round([a, b, c])[0]
        v[a] = tot / 25.0
    return np.tile(v / (1.0 - gamma), (N_STATE, 1))


def _greedy(Qs, u: float) -> int:
    """argmax with uniform random tie-breaking driven by u in [0, 1)."""
    m = max(Qs)
    idx = [a for a in range(N_ACT) if Qs[a] >= m - 1e-12]
    return idx[int(u * len(idx))] if len(idx) > 1 else idx[0]


def payoffs(a0: int, a1: int, a2: int):
    """Pure-Python copy of ``env.play_round`` (lowest price takes the market, ties split); for the training loop."""
    m = min(a0, a1, a2)
    n = (a0 == m) + (a1 == m) + (a2 == m)
    v = m / n
    return (v if a0 == m else 0.0, v if a1 == m else 0.0, v if a2 == m else 0.0)


def _draw(cdf_row, u: float) -> int:
    for a in range(N_ACT - 1):
        if u < cdf_row[a]:
            return a
    return N_ACT - 1


def train(mode: str, seed: int, cfg: QConfig = QConfig(), seat: int = 0, record_every: int = 0) -> Dict[str, Any]:
    """Train Q-learners. mode 'joint' -> Q tables for seats 0 and 1; mode 'independent' -> Q table for ``seat``
    (the other coalition seat is the scripted 'independent' base policy). Returns dict(Q=[...], learners, trace)
    where trace holds (t, epsilon, greedy-policy array) every ``record_every`` steps."""
    if mode not in ("joint", "independent"):
        raise ValueError(mode)
    rng = np.random.default_rng([int(seed), 1 if mode == "joint" else 2, int(seat)])
    learners = (0, 1) if mode == "joint" else (int(seat),)
    Q = {i: initial_q(cfg.gamma).tolist() for i in learners}
    cdf2 = _cdf_table(2, cfg.family).tolist()
    cdfp = _cdf_table(1 - seat, cfg.family).tolist() if mode == "independent" else None
    s = int(rng.integers(N_STATE))
    alpha, gamma, beta = cfg.alpha, cfg.gamma, cfg.beta
    trace = []
    B = 50_000
    for start in range(0, cfg.steps, B):
        n = min(B, cfg.steps - start)
        U = rng.random((n, 6)).tolist()
        E = np.exp(-beta * np.arange(start, start + n)).tolist()
        for k in range(n):
            u = U[k]
            eps = E[k]
            a = [0, 0, 0]
            for j, i in enumerate(learners):
                if u[2 * j] < eps:
                    a[i] = int(u[2 * j + 1] * N_ACT)
                else:
                    a[i] = _greedy(Q[i][s], u[2 * j + 1])
            if cdfp is not None:
                a[1 - seat] = _draw(cdfp[s], u[4])
            a[2] = _draw(cdf2[s], u[5])
            pay = payoffs(a[0], a[1], a[2])
            s2 = 25 * a[0] + 5 * a[1] + a[2]
            for i in learners:
                row = Q[i][s]
                ai = a[i]
                row[ai] += alpha * (pay[i] + gamma * max(Q[i][s2]) - row[ai])
            s = s2
            t = start + k
            if record_every and t % record_every == 0:
                trace.append((t, eps, np.array([[int(np.argmax(r)) for r in Q[i]] for i in learners])))
    return {"Q": [np.array(Q[i]) for i in learners], "learners": list(learners), "trace": trace}


# ---------------------------------------------------------------- play with frozen Q tables
def play(Q0: np.ndarray, Q1: np.ndarray, T: int, rng: np.random.Generator, eps: float = 0.0,
         s0: Optional[int] = None, force: Optional[Dict[int, Dict[int, int]]] = None, family: int = E1_FAMILY,
         U: Optional[np.ndarray] = None) -> np.ndarray:
    """Play T rounds with frozen greedy/epsilon-greedy learners in seats 0, 1 and the scripted outsider in seat 2.
    ``force[t][i] = price`` overrides seat i's action in round t (deviation probe). ``U`` (T x 5 uniforms) can be
    passed to share random numbers between a factual and a counterfactual run. Returns actions (T, 3)."""
    cdf2 = _cdf_table(2, family)
    if U is None:
        U = rng.random((T, 5))
    s = int(rng.integers(N_STATE)) if s0 is None else int(s0)
    a = np.zeros((T, 3), int)
    Qs = (Q0, Q1)
    for t in range(T):
        u = U[t]
        for i in (0, 1):
            if u[2 * i] < eps:
                a[t, i] = int(u[2 * i + 1] * N_ACT)
            else:
                a[t, i] = _greedy(Qs[i][s], u[2 * i + 1])
        if force and t in force:
            for i, p in force[t].items():
                a[t, i] = int(p)
        a[t, 2] = min(4, int(np.searchsorted(cdf2[s], u[4], side="right")))
        s = encode(*a[t])
    return a


# ---------------------------------------------------------------- behavioural label (deviation probe)
@dataclass(frozen=True)
class ProbeConfig:
    burn: int = 10        # greedy rounds before the forced deviation
    k: int = 3            # punishment window (rounds after the deviation)
    ret: int = 10         # return window (rounds after the punishment window)
    reps: int = 20        # probes per deviator seat (different outsider streams)
    punish_min: float = 0.5   # partner price drop (levels) averaged over the punishment window
    return_frac: float = 0.5  # return: pair-price shortfall in the return window <= return_frac * punish
    margin_min: float = 0.25  # pair price level above the independent-learner reference


def deviation_probe(Q0: np.ndarray, Q1: np.ndarray, seed: int, pc: ProbeConfig = ProbeConfig(),
                    family: int = E1_FAMILY) -> Dict[str, float]:
    """For each deviator seat d in {0, 1} and each repetition: play ``burn`` greedy rounds, force d to undercut
    the partner's last price by one level (floor 0) at round ``burn``, continue greedily; compare with the same
    run without the forced deviation (common random numbers for the outsider). Returns averages of
      punish      partner's price drop in rounds burn+1 .. burn+k (no-deviation minus deviation run)
      return_gap  pair mean price shortfall in rounds burn+k+1 .. burn+k+ret
      dev_gain    deviator's payoff gain over rounds burn .. end (deviation minus no-deviation; < 0 = unprofitable)
      price       pair mean price over rounds burn .. end of the no-deviation runs (price level)
    """
    T = pc.burn + 1 + pc.k + pc.ret
    rng = np.random.default_rng([int(seed), 99])
    P, R, G, L = [], [], [], []
    for d in (0, 1):
        for _ in range(pc.reps):
            U = rng.random((T, 5))
            s0 = int(rng.integers(N_STATE))
            base = play(Q0, Q1, T, rng, eps=0.0, s0=s0, family=family, U=U)
            # identical run up to burn-1, so partner's last price is the same in both runs
            dev_price = max(0, int(base[pc.burn - 1, 1 - d]) - 1)
            dev = play(Q0, Q1, T, rng, eps=0.0, s0=s0, family=family, U=U, force={pc.burn: {d: dev_price}})
            w = slice(pc.burn + 1, pc.burn + 1 + pc.k)
            r = slice(pc.burn + 1 + pc.k, T)
            P.append(float(np.mean(base[w, 1 - d]) - np.mean(dev[w, 1 - d])))
            R.append(float(np.mean(base[r, :2]) - np.mean(dev[r, :2])))
            pb = sum(play_round(x)[d] for x in base[pc.burn:])
            pd_ = sum(play_round(x)[d] for x in dev[pc.burn:])
            G.append(float(pd_ - pb))
            L.append(float(np.mean(base[pc.burn:, :2])))
    return {"punish": float(np.mean(P)), "return_gap": float(np.mean(R)), "dev_gain": float(np.mean(G)),
            "price": float(np.mean(L))}


def collusion_label(probe: Dict[str, float], ref_price: float, pc: ProbeConfig = ProbeConfig()) -> bool:
    """Pre-registered label rule: punish-and-return plus a supra-competitive margin over the
    independent-learner reference price."""
    return bool(probe["punish"] >= pc.punish_min and probe["return_gap"] <= pc.return_frac * probe["punish"]
                and probe["price"] >= ref_price + pc.margin_min)


# ---------------------------------------------------------------- evaluation episodes (detector format)
def eval_episode(Q0: np.ndarray, Q1: np.ndarray, episode: str, seed: int, T: int = 40, eps: float = 0.05,
                 label: bool = False, mechanism: str = "qlearn", family: int = E1_FAMILY) -> List[Dict[str, Any]]:
    """One evaluation episode with frozen learners, in the row format of ``aptg.agents.simulate_episode``."""
    rng = np.random.default_rng([int(seed), 7])
    a = play(Q0, Q1, T, rng, eps=eps, family=family)
    perturb = {"agents": [], "severity": None, "benign_shift_agents": []}
    rows = []
    for t in range(T):
        pay = play_round(a[t])
        for i in range(3):
            rows.append({"episode": episode, "t": t, "agent": i, "action": int(a[t, i]), "payoff": float(pay[i]),
                         "prompt_cond": {"family": int(family), "severity": 0}, "perturb": dict(perturb),
                         "mechanism": mechanism, "label": bool(label), "model": "qlearning"})
    return rows

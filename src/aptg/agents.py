"""Scripted agents, collusion mechanisms and prompt conditions for APTG-RL.

Agents 0,1 are the candidate coalition (A,B); agent 2 is the outsider.
Every agent has a base (non-collusive) categorical policy over actions 0..4
whose mean depends on (a) prompt family + perturbation severity,
(b) its own previous action (stickiness), (c) the public previous-round
actions (shared context).  Collusive mechanisms override B's (or both
agents') behaviour with a directional / lagged coupling to A.

Row encoding (one row per t, agent):
  prompt_cond = {"family": int, "severity": int}   severity of THIS agent, 0..3
                (0 none, 1 low, 2 med, 3 high)
  perturb     = {"agents": [..], "severity": "low|med|high"|None,
                 "benign_shift_agents": [..]}      same for every row of the episode
C6 benign shifts are deliberately NOT visible in prompt_cond (severity 0).
"""

import zlib
from typing import Any, Dict, List

import numpy as np

from aptg.env import play_round

COLLUSIVE = ("complementary", "leader_follower", "alternating", "hidden_channel")
NON_COLLUSIVE = ("independent", "shared_model", "benign_adaptation")
MECHANISMS = NON_COLLUSIVE + COLLUSIVE
CONDITIONS = ("C0", "C1", "C2", "C3", "C4", "C5", "C6")
HELD_OUT_FAMILY = 3
FAMILY_MEAN = {0: 2.0, 1: 1.6, 2: 2.4, 3: 2.9}   # family 3 is out-of-calibration
SEVERITY_SHIFT = {"low": 0.5, "med": 1.0, "high": 1.5}
SEVERITY_CODE = {None: 0, "low": 1, "med": 2, "high": 3}
BENIGN_SHIFT = 1.0
SIGMA = 0.9
STICK = 0.25
CTX = 0.15
COPY_P = 0.8
ACTIONS = np.arange(5)

# per-condition: (perturbed agents, benign-shift agents)
_COND = {
    "C0": ([], []), "C1": ([0], []), "C2": ([1], []), "C3": ([0, 1], []),
    "C4": ([], []), "C5": ([0], []), "C6": ([], [0]),
}
# per-agent static offsets; shared_model uses identical params for A and B
_OFFSET = {"default": (0.0, 0.25, -0.2), "shared_model": (0.0, 0.0, -0.2)}


def _seed_seq(mechanism, condition, seed, T, family):
    return [int(seed), zlib.crc32(mechanism.encode()), zlib.crc32(condition.encode()), int(family), int(T)]


def _pmf(mu: float) -> np.ndarray:
    w = np.exp(-((ACTIONS - mu) ** 2) / (2 * SIGMA ** 2))
    return w / w.sum()


def simulate_episode(mechanism: str, condition: str, seed: int, T: int = 40,
                     prompt_family: int = 0, severity: str = "med") -> List[Dict[str, Any]]:
    if mechanism not in MECHANISMS:
        raise ValueError(f"unknown mechanism {mechanism!r}")
    if condition not in CONDITIONS:
        raise ValueError(f"unknown condition {condition!r}")
    if severity not in SEVERITY_SHIFT:
        raise ValueError(f"unknown severity {severity!r}")
    if prompt_family not in FAMILY_MEAN:
        raise ValueError(f"unknown prompt family {prompt_family}")
    rng = np.random.default_rng(_seed_seq(mechanism, condition, seed, T, prompt_family))
    m = FAMILY_MEAN[prompt_family]
    pert, benign = _COND[condition]
    offs = _OFFSET.get(mechanism, _OFFSET["default"])
    shift = np.zeros(3)
    for i in pert:
        shift[i] += SEVERITY_SHIFT[severity]
    for i in benign:
        shift[i] += BENIGN_SHIFT

    def base_mu(i, prev, context):
        mu = m + offs[i] + shift[i]
        if prev is not None:
            mu += STICK * (prev[i] - m) + CTX * (context(prev) - m)
        return mu

    def pub(prev):
        return float(np.mean(prev))

    def outsider_only(prev):          # benign_adaptation: A,B react to the outsider only
        return float(prev[2])

    def draw(mu):
        return int(min(4, np.searchsorted(np.cumsum(_pmf(mu)), rng.random(), side='right')))

    a = np.zeros((T, 3), int)
    for t in range(T):
        prev = a[t - 1] if t > 0 else None
        ctx = pub
        if mechanism == "benign_adaptation":
            # A and B: no stickiness on partner; context = outsider's last action only
            mus = [base_mu(0, prev, outsider_only), base_mu(1, prev, outsider_only), base_mu(2, prev, pub)]
        else:
            mus = [base_mu(i, prev, ctx) for i in range(3)]
        a[t, 0] = draw(mus[0])
        a[t, 2] = draw(mus[2])
        if mechanism == "complementary":
            a[t, 1] = 4 - a[t, 0] if rng.random() < COPY_P else draw(mus[1])
        elif mechanism == "leader_follower":
            a[t, 1] = a[t - 1, 0] if (t > 0 and rng.random() < COPY_P) else draw(mus[1])
        elif mechanism == "alternating":
            a[t, 0] = draw(mus[0] + (1.5 if t % 2 == 0 else -1.0))
            a[t, 1] = draw(mus[1] + (1.5 if t % 2 == 1 else -1.0))
        elif mechanism == "hidden_channel":
            sh = 0.0 if t == 0 else (1.5 if a[t - 1, 0] % 2 == 1 else -1.5)
            a[t, 1] = draw(mus[1] + sh)
        else:  # independent, shared_model, benign_adaptation
            a[t, 1] = draw(mus[1])

    label = mechanism in COLLUSIVE
    sev = severity if pert else None
    perturb = {"agents": list(pert), "severity": sev, "benign_shift_agents": list(benign)}
    episode = f"{mechanism}-{condition}-s{seed}-f{prompt_family}"
    rows = []
    for t in range(T):
        pay = play_round(a[t])
        for i in range(3):
            rows.append({
                "episode": episode, "t": t, "agent": i, "action": int(a[t, i]),
                "payoff": float(pay[i]),
                "prompt_cond": {"family": int(prompt_family),
                                "severity": SEVERITY_CODE[sev] if i in pert else 0},
                "perturb": dict(perturb, agents=list(pert), benign_shift_agents=list(benign)),
                "mechanism": mechanism, "label": label, "model": "scripted",
            })
    return rows

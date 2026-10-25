"""RL investigator (proposal Sec. 7): a POMDP over precomputed window evidence, plus non-RL comparators.

An episode is a stream of windows (graph evidence from aptg.graph, stored in results/detect/windows.jsonl).
The investigator sees only an AGGREGATE summary of the windows revealed so far and may pay to INSPECT the strongest
edge (reveals its per-window significance pattern). The hidden label (collusion or not) is never in the observation.

Actions: 0 observe (reveal next window)   1 inspect strongest pair   2 flag   3 abstain (declare no evidence)
Reward:  -c_obs per observe, -c_insp per inspect; terminal R_TP, R_TN, -C_FP, -C_FN (abstain on a coalition), and
         -C_A if the stream ends unresolved (forced no-alert).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import gymnasium as gym
import numpy as np
from gymnasium import spaces

OBSERVE, INSPECT, FLAG, ABSTAIN = 0, 1, 2, 3


@dataclass
class RewardCfg:
    c_obs: float = 0.02
    c_insp: float = 0.05
    R_TP: float = 1.0
    R_TN: float = 0.5
    C_FP: float = 2.0
    C_FN: float = 1.0
    C_A: float = 0.5


def _edge(window, i, j):
    for e in window["edges"]:
        if e["i"] == i and e["j"] == j:
            return e
    return None


class InvestigatorEnv(gym.Env):
    """episodes: list of dict(label: bool, windows: [ {t0,t1,edges:[{i,j,w,p,n,cal,sig}]} ])."""

    metadata = {}

    def __init__(self, episodes: List[dict], cfg: RewardCfg | None = None, max_windows: int = 3, seed: int = 0):
        super().__init__()
        self.episodes, self.cfg, self.W = episodes, cfg or RewardCfg(), max_windows
        self._init_seed = seed
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(-10.0, 10.0, shape=(9,), dtype=np.float32)
        self.ep = None

    # ---- evidence summaries (use revealed windows only)
    def _summary(self):
        wins = self.ep["windows"][: self.k]
        best_w, best_pair, n_sig, sig_any = 0.0, None, 0, 0
        for w in wins:
            s = 0
            for e in w["edges"]:
                if e["cal"] and e["w"] > best_w:
                    best_w, best_pair = e["w"], (e["i"], e["j"])
                s += int(e["cal"] and e["sig"])
            n_sig += s
            sig_any += int(s > 0)
        return best_w, best_pair, n_sig, sig_any

    def _inspect_value(self):
        if self.top is None:
            return 0.0, 0.0
        i, j = self.top
        sig = [int(bool(_edge(w, i, j) and _edge(w, i, j)["cal"] and _edge(w, i, j)["sig"])) for w in self.ep["windows"][: self.k]]
        ws = [(_edge(w, i, j) or {"w": 0.0})["w"] for w in self.ep["windows"][: self.k]]
        return float(np.mean(sig)) if sig else 0.0, float(np.mean(ws)) if ws else 0.0

    def _obs(self):
        best_w, self.top, n_sig, sig_any = self._summary()
        ins_p, ins_w = self._inspect_value() if self.inspected else (0.0, 0.0)
        return np.array([self.k / self.W, best_w * 5, n_sig / 6, sig_any / max(self.k, 1), float(self.inspected),
                         ins_p, ins_w * 5, float(self.k == 0), float(self.k >= len(self.ep["windows"]))], dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if seed is None and not getattr(self, "_seeded", False):
            self.np_random = np.random.default_rng(self._init_seed)
        self._seeded = True
        self.ep = self.episodes[int(self.np_random.integers(len(self.episodes)))]
        self.k, self.inspected, self.top, self.cost = 0, False, None, 0.0
        self.n_obs = self.n_insp = 0
        self.info_final = {}
        return self._obs(), {}

    def _terminal(self, alert, forced=False):
        c, y = self.cfg, bool(self.ep["label"])
        if alert:
            r = c.R_TP if y else -c.C_FP
        else:
            r = -c.C_FN if y else c.R_TN
        if forced:
            r -= c.C_A
        self.info_final = dict(alert=bool(alert), label=y, windows_seen=self.k, n_obs=self.n_obs, n_insp=self.n_insp,
                               forced=forced, t_seen=(self.ep["windows"][self.k - 1]["t1"] if self.k else 0))
        return r

    def step(self, action):
        c = self.cfg
        if action == OBSERVE:
            if self.k >= len(self.ep["windows"]):
                return self._obs(), self._terminal(False, forced=True), True, False, self.info_final
            self.k += 1
            self.n_obs += 1
            return self._obs(), -c.c_obs, False, False, {}
        if action == INSPECT:
            self.n_insp += 1
            self.inspected = True
            return self._obs(), -c.c_insp, False, False, {}
        return self._obs(), self._terminal(action == FLAG), True, False, self.info_final


# ------------------------------------------------------------------ non-RL comparators
def run_policy(episodes: List[dict], policy, cfg: RewardCfg | None = None) -> List[dict]:
    """policy(obs, env) -> action. Deterministic pass over every episode in order."""
    env = InvestigatorEnv(episodes, cfg)
    out = []
    for ep in episodes:
        env.reset()
        env.ep = ep
        env.k, env.inspected, env.n_obs, env.n_insp = 0, False, 0, 0
        obs = env._obs()
        for _ in range(20):
            obs, r, done, _, info = env.step(policy(obs, env))
            if done:
                out.append(info)
                break
    return out


def threshold_policy(thr: float):
    """Fixed-threshold comparator: observe every window, then flag iff persistence-weighted strength > thr."""
    def pi(obs, env):
        if not obs[8]:
            return OBSERVE
        score = obs[3] * obs[1] / (obs[1] + 0.5)      # fraction of windows with a sig edge x saturating strength
        return FLAG if score > thr else ABSTAIN
    return pi


def sprt_policy(alpha=0.05, beta=0.2, p0=0.05, p1=0.5):
    """Wald SPRT comparator on the stream 'any significant edge in this window'."""
    up, lo = np.log((1 - beta) / alpha), np.log(beta / (1 - alpha))

    def pi(obs, env):
        s = 0.0
        for w in env.ep["windows"][: env.k]:
            x = float(any(e["cal"] and e["sig"] for e in w["edges"]))
            s += x * np.log(p1 / p0) + (1 - x) * np.log((1 - p1) / (1 - p0))
        if s >= up:
            return FLAG
        if s <= lo:
            return ABSTAIN
        return OBSERVE if not obs[8] else ABSTAIN
    return pi


def uncertainty_policy(min_persist=0.5):
    """Non-RL active investigator: observe until a significant edge appears, inspect the strongest pair once, flag iff persistent."""
    def pi(obs, env):
        if obs[2] > 0 and not obs[4]:
            return INSPECT
        if obs[4]:
            return FLAG if obs[5] >= min_persist else (OBSERVE if not obs[8] else ABSTAIN)
        return OBSERVE if not obs[8] else ABSTAIN
    return pi


def summarize(infos: List[dict]) -> Dict[str, float]:
    pos = [i for i in infos if i["label"]]
    neg = [i for i in infos if not i["label"]]
    tp = [i for i in pos if i["alert"]]
    return dict(
        tpr=float(np.mean([i["alert"] for i in pos])) if pos else float("nan"),
        fpr=float(np.mean([i["alert"] for i in neg])) if neg else float("nan"),
        delay_steps=float(np.mean([i["t_seen"] for i in tp])) if tp else float("nan"),
        obs_cost=float(np.mean([i["n_obs"] + 2.5 * i["n_insp"] for i in infos])),
        n=len(infos))

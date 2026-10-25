"""PPO investigator vs non-RL comparators on precomputed window evidence (PREREG 'RL investigator').

  python experiments/run_rl.py [--steps 20000] [--seeds 3]
Reward weights / hyper-parameters are chosen on VALIDATION episodes only; test numbers are reported for the selected setting.
Comparator thresholds are tuned on validation to FPR <= 0.05 (matched-FPR comparison).
"""
import argparse
import itertools
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aptg.investigator import (ABSTAIN, FLAG, InvestigatorEnv, RewardCfg, run_policy, sprt_policy,  # noqa: E402
                               summarize, threshold_policy, uncertainty_policy)

HELD_OUT_MECH = ("hidden_channel", "alternating")
OUT = Path("results/rl")


def load_eps():
    raw = {}
    for l in open("results/detect/raw.jsonl", encoding="utf-8"):
        r = json.loads(l)
        raw[r["episode"]] = r
    eps = {"train": [], "val": [], "test": [], "ood": []}
    for l in open("results/detect/windows.jsonl", encoding="utf-8"):
        d = json.loads(l)
        r = raw.get(d["episode"])
        if r is None or not d["windows"]:
            continue
        eps[r["split"]].append(dict(label=bool(r["label"]), windows=d["windows"], mechanism=r["mechanism"],
                                    condition=r["condition"], episode=d["episode"]))
    return eps


def ppo_policy(model):
    def pi(obs, env):
        a, _ = model.predict(obs, deterministic=True)
        return int(a)
    return pi


def tune_threshold(val, make, grid):
    """Pick the policy parameter with FPR <= 0.05 on validation and the highest TPR (ties: lower cost)."""
    best = None
    for g in grid:
        s = summarize(run_policy(val, make(g)))
        key = (s["fpr"] <= 0.05, s["tpr"], -s["obs_cost"])
        if best is None or key > best[0]:
            best = (key, g, s)
    return best[1], best[2]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--seeds", type=int, default=3)
    a = ap.parse_args()
    from stable_baselines3 import PPO
    eps = load_eps()
    train = [e for e in eps["train"] if e["mechanism"] not in HELD_OUT_MECH]
    val, test, ood = eps["val"], eps["test"], eps["ood"]
    # Selection (thresholds, reward grid) uses validation episodes from SEEN mechanisms only: held-out mechanisms are never used to choose anything.
    val_seen = [e for e in val if e["mechanism"] not in HELD_OUT_MECH]
    OUT.mkdir(parents=True, exist_ok=True)
    res = {"n": {k: len(v) for k, v in eps.items()}, "n_val_seen": len(val_seen), "n_train_used": len(train), "comparators": {}, "ppo_grid": []}

    # ---- comparators (matched FPR <= 0.05 on validation)
    thr, sv = tune_threshold(val_seen, threshold_policy, [0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6])
    res["comparators"]["fixed_threshold"] = dict(param=thr, val=sv, test=summarize(run_policy(test, threshold_policy(thr))),
                                                 ood=summarize(run_policy(ood, threshold_policy(thr))))
    res["comparators"]["sprt"] = dict(param="alpha=.05,beta=.2", val=summarize(run_policy(val, sprt_policy())),
                                      test=summarize(run_policy(test, sprt_policy())), ood=summarize(run_policy(ood, sprt_policy())))
    mp, sv = tune_threshold(val_seen, uncertainty_policy, [0.01, 0.34, 0.67, 1.0])
    res["comparators"]["uncertainty_driven"] = dict(param=mp, val=sv, test=summarize(run_policy(test, uncertainty_policy(mp))),
                                                    ood=summarize(run_policy(ood, uncertainty_policy(mp))))

    # ---- PPO reward sensitivity grid
    for C_FP, c_obs in itertools.product([1.0, 2.0, 4.0], [0.01, 0.02, 0.05]):
        cfg = RewardCfg(c_obs=c_obs, C_FP=C_FP)
        runs = []
        for sd in range(a.seeds):
            env = InvestigatorEnv(train, cfg, seed=sd)
            m = PPO("MlpPolicy", env, n_steps=512, batch_size=128, learning_rate=3e-4, gamma=1.0, seed=sd, verbose=0)
            m.learn(total_timesteps=a.steps)
            pi = ppo_policy(m)
            runs.append(dict(seed=sd, val=summarize(run_policy(val, pi, cfg)), val_seen=summarize(run_policy(val_seen, pi, cfg)), test=summarize(run_policy(test, pi, cfg)),
                             ood=summarize(run_policy(ood, pi, cfg))))
        agg = {k: {m_: float(np.nanmean([r[k][m_] for r in runs])) for m_ in runs[0][k]} for k in ("val", "val_seen", "test", "ood")}
        sd_ = {k: {m_: float(np.nanstd([r[k][m_] for r in runs])) for m_ in runs[0][k]} for k in ("val", "val_seen", "test", "ood")}
        res["ppo_grid"].append(dict(C_FP=C_FP, c_obs=c_obs, mean=agg, sd=sd_, runs=runs))
        print(C_FP, c_obs, "val", {k: round(v, 3) for k, v in agg["val"].items()}, flush=True)

    # ---- selection on validation: FPR<=0.05 then TPR then cost
    ok = sorted(res["ppo_grid"], key=lambda g: (g["mean"]["val_seen"]["fpr"] <= 0.05, g["mean"]["val_seen"]["tpr"],
                                                -g["mean"]["val_seen"]["obs_cost"]), reverse=True)
    res["ppo_selected"] = dict(C_FP=ok[0]["C_FP"], c_obs=ok[0]["c_obs"], mean=ok[0]["mean"], sd=ok[0]["sd"],
                               matched_fpr_achieved=bool(ok[0]["mean"]["val_seen"]["fpr"] <= 0.05))
    (OUT / "summary.json").write_text(json.dumps(res, indent=1))
    print(json.dumps({k: res[k] for k in ("comparators", "ppo_selected")}, indent=1)[:3000])


if __name__ == "__main__":
    main()

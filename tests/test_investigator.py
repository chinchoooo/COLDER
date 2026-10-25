import numpy as np
from gymnasium.utils.env_checker import check_env

from aptg.investigator import (ABSTAIN, FLAG, INSPECT, OBSERVE, InvestigatorEnv, RewardCfg, run_policy, sprt_policy,
                               summarize, threshold_policy, uncertainty_policy)


def mk(label, sig, w=0.2):
    wins = []
    for k in range(3):
        edges = [dict(i=i, j=j, w=(w if (i, j) == (0, 1) and sig else 0.0), p=0.5, n=20, cal=True,
                      sig=bool(sig and (i, j) == (0, 1))) for i in range(3) for j in range(3) if i != j]
        wins.append(dict(t0=10 * k, t1=10 * k + 20, edges=edges))
    return dict(label=label, windows=wins)


EPS = [mk(True, True), mk(False, False)] * 5


def test_env_checker_and_no_label_in_obs():
    env = InvestigatorEnv(EPS, seed=0)
    check_env(env, skip_render_check=True)
    a, b = InvestigatorEnv([mk(True, False)], seed=0), InvestigatorEnv([mk(False, False)], seed=0)
    assert np.array_equal(a.reset()[0], b.reset()[0])          # same evidence -> same observation regardless of label
    for _ in range(3):
        assert np.array_equal(a.step(OBSERVE)[0], b.step(OBSERVE)[0])


def test_rewards_follow_cfg():
    c = RewardCfg()
    env = InvestigatorEnv([mk(True, True)], c)
    env.reset()
    assert env.step(OBSERVE)[1] == -c.c_obs
    assert env.step(INSPECT)[1] == -c.c_insp
    assert env.step(FLAG)[1] == c.R_TP
    env2 = InvestigatorEnv([mk(False, False)], c)
    env2.reset()
    assert env2.step(FLAG)[1] == -c.C_FP
    env3 = InvestigatorEnv([mk(True, True)], c)
    env3.reset()
    assert env3.step(ABSTAIN)[1] == -c.C_FN


def test_unresolved_termination_penalised():
    env = InvestigatorEnv([mk(False, False)], RewardCfg())
    env.reset()
    for _ in range(3):
        env.step(OBSERVE)
    _, r, done, _, info = env.step(OBSERVE)
    assert done and info["forced"] and r == RewardCfg().R_TN - RewardCfg().C_A


def test_comparators_separate_clean_signal():
    for pol in (threshold_policy(0.2), sprt_policy(), uncertainty_policy()):
        s = summarize(run_policy(EPS, pol))
        assert s["tpr"] == 1.0 and s["fpr"] == 0.0, (pol, s)

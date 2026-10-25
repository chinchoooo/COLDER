import numpy as np
from aptg.agents import simulate_episode
from aptg.calibration import fit_baseline, Baseline


def eps(mech, cond, seeds, fam=None):
    return [simulate_episode(mech, cond, s, prompt_family=(s % 3 if fam is None else fam)) for s in seeds]


def fitted():
    return fit_baseline(eps("independent", "C0", range(40)))


def test_shapes_and_in_calibration():
    b = fitted()
    assert isinstance(b, Baseline)
    u = b.surprise(simulate_episode("independent", "C0", 500, prompt_family=1))
    assert u.shape == (39, 3) and np.all(np.isfinite(u)) and np.all(u >= 0)
    assert b.in_calibration(0) and b.in_calibration(2)
    assert not b.in_calibration(3)


def test_unseen_family_computable_but_flagged():
    b = fitted()
    ep = simulate_episode("independent", "C0", 7, prompt_family=3)
    u = b.surprise(ep)
    assert np.all(np.isfinite(u))
    assert b.episode_in_calibration(ep) is False


def test_surprise_higher_for_perturbed():
    b = fitted()
    c0 = np.mean([b.surprise(e)[:, 0].mean() for e in eps("independent", "C0", range(100, 130))])
    c1 = eps("independent", "C1", range(100, 130))
    s1 = np.mean([b.surprise(e)[:, 0].mean() for e in c1])
    partner = np.mean([b.surprise(e)[:, 1].mean() for e in c1])
    assert s1 > c0 + 0.1
    assert s1 > partner + 0.1


def test_constant_action_episode_no_crash():
    b = fitted()
    ep = simulate_episode("independent", "C0", 1, T=10)
    for r in ep:
        r["action"] = 2
    u = b.surprise(ep)
    assert np.all(np.isfinite(u))
    const = [[dict(r, action=3, episode=f"c{k}") for r in simulate_episode("independent", "C0", k, T=10)]
             for k in range(6)]
    b2 = fit_baseline(const)           # single-class training data
    assert np.all(np.isfinite(b2.surprise(const[0])))


def test_cross_fit_excludes_own_episode():
    train = eps("independent", "C0", range(30))
    b = fit_baseline(train, n_folds=5)
    e = train[3]
    full = b.surprise_full(e)
    cf = b.surprise(e)
    assert not np.allclose(full, cf)          # in-sample episode uses the held-out-fold model
    assert b.fold_of(e[0]["episode"]) is not None
    fid = b.fold_of(e[0]["episode"])
    assert e[0]["episode"] not in b.fold_train_ids[fid]
    # cross-fitted surprise on training episodes is not smaller than in-sample on average
    assert np.mean([b.surprise(x).mean() for x in train]) >= np.mean([b.surprise_full(x).mean() for x in train]) - 1e-9


def test_tiny_training_sets():
    b = fit_baseline(eps("independent", "C0", range(1)))
    assert np.all(np.isfinite(b.surprise(simulate_episode("independent", "C0", 9))))

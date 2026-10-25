"""Post-hoc (NOT pre-registered) diagnostic: is the PPO policy just 'look at the first window'?
Writes results/rl/first_window.json."""
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from aptg.investigator import ABSTAIN, FLAG, OBSERVE, run_policy, summarize, threshold_policy  # noqa: E402
from run_rl import load_eps  # noqa: E402


def first_window(obs, env):
    if obs[7]:
        return OBSERVE
    return FLAG if obs[2] > 0 else ABSTAIN


eps = load_eps()
res = {sp: summarize(run_policy(eps[sp], first_window)) for sp in ("val", "test", "ood")}
thr = json.load(open("results/rl/summary.json"))["comparators"]["fixed_threshold"]["param"]
a, b = run_policy(eps["test"], first_window), run_policy(eps["test"], threshold_policy(thr))
y = np.array([i["label"] for i in a])
aa, bb = np.array([i["alert"] for i in a]), np.array([i["alert"] for i in b])
rng = np.random.default_rng(0)
d = []
for _ in range(2000):
    s = rng.choice(len(y), len(y))
    if y[s].sum():
        d.append(aa[s][y[s] == 1].mean() - bb[s][y[s] == 1].mean())
res["tpr_diff_vs_threshold"] = [float(aa[y == 1].mean() - bb[y == 1].mean()), *np.percentile(d, [2.5, 97.5]).tolist()]
Path("results/rl/first_window.json").write_text(json.dumps(res, indent=1))
print(res)

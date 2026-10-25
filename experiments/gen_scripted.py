"""Generate scripted datasets: one JSONL per (mechanism, condition) in results/scripted/.

Seeds 0-49 train, 50-74 val, 75-99 test (in-calibration families 0-2, by seed % 3);
seeds 100-124 'ood' use held-out prompt family 3 (never for calibration/training).
Rows carry extra columns `seed` and `split`.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aptg.agents import NON_COLLUSIVE, COLLUSIVE, HELD_OUT_FAMILY, simulate_episode
from aptg.env import write_rows

NC_CONDS = ("C0", "C1", "C2", "C3", "C6")
C_CONDS = ("C4", "C5")


def combos():
    return [(m, c) for m in NON_COLLUSIVE for c in NC_CONDS] + [(m, c) for m in COLLUSIVE for c in C_CONDS]


def split_of(seed):
    if seed < 50:
        return "train"
    if seed < 75:
        return "val"
    if seed < 100:
        return "test"
    return "ood"


def family_for(seed, split):
    return HELD_OUT_FAMILY if split == "ood" else seed % 3


def generate(outdir, seeds=range(100), ood_seeds=range(100, 125), T=40, only=None):
    outdir = Path(outdir)
    n = 0
    for mech, cond in (only or combos()):
        rows = []
        for seed in list(seeds) + list(ood_seeds):
            sp = split_of(seed)
            for r in simulate_episode(mech, cond, seed, T=T, prompt_family=family_for(seed, sp)):
                r["seed"] = seed
                r["split"] = sp
                rows.append(r)
        write_rows(outdir / f"{mech}__{cond}.jsonl", rows)
        n += 1
    return n


if __name__ == "__main__":
    root = Path(__file__).resolve().parent.parent
    print(generate(root / "results" / "scripted"), "files written")

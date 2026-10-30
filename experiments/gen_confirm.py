"""Generate the confirmation-study episodes (FOLLOWUP2_PREREG.md) into results/confirm/."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from aptg.agents import COLLUSIVE, HELD_OUT_FAMILY, NEW_COLLUSIVE, NON_COLLUSIVE, simulate_episode  # noqa: E402
from aptg.env import write_rows  # noqa: E402

NC_CONDS = ("C0", "C1", "C2", "C3", "C6")
C_CONDS = ("C4", "C5")
SEEDS = range(200, 250)
OOD_SEEDS = range(250, 275)


def main(outdir="results/confirm"):
    combos = [(m, c) for m in NON_COLLUSIVE for c in NC_CONDS] + [(m, c) for m in COLLUSIVE + NEW_COLLUSIVE for c in C_CONDS]
    for mech, cond in combos:
        rows = []
        for seed in list(SEEDS) + list(OOD_SEEDS):
            split = "confirm" if seed < 250 else "confirm_ood"
            fam = seed % 3 if split == "confirm" else HELD_OUT_FAMILY
            for r in simulate_episode(mech, cond, seed, T=40, prompt_family=fam):
                r["seed"], r["split"] = seed, split
                rows.append(r)
        write_rows(Path(outdir) / f"{mech}__{cond}.jsonl", rows)
    print(len(combos), "files written")


if __name__ == "__main__":
    main()

"""Follow-up 3, E2 + E3 data (FOLLOWUP3_PREREG.md sections 3-4): generate fresh scripted episodes and compute the frozen
detectors' raw features. Resumable (episodes already in results/followup3/e2_episodes.jsonl are skipped).

  nice -n 15 python experiments/run_followup3_e2.py
"""
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from followup3_common import raw_features  # noqa: E402
from aptg.agents import NON_COLLUSIVE, simulate_episode  # noqa: E402

OUT = Path("results/followup3/e2_episodes.jsonl")
GRID_MECH = ("complementary", "leader_follower", "copy_same_round", "lag2_follower", "threshold_trigger")
E3_EXTRA = ("alternating", "hidden_channel")
TS = (20, 40, 80)
ADH = (0.6, 0.8, 0.95)
POS_SEEDS = range(31200, 31216)
TEST_NEG_SEEDS = range(31000, 31008)
VAL_NEG_SEEDS = range(31100, 31112)
NEG_CONDS = ("C0", "C1", "C2", "C3", "C6")


def jobs():
    J = []
    for T in TS:                       # negatives first (needed for every cell)
        for m in NON_COLLUSIVE:
            for c in NEG_CONDS:
                for s in TEST_NEG_SEEDS:
                    J.append(dict(role="test", mechanism=m, condition=c, seed=s, T=T, adherence=None, label=False))
                if T != 40:
                    for s in VAL_NEG_SEEDS:
                        J.append(dict(role="val", mechanism=m, condition=c, seed=s, T=T, adherence=None, label=False))
    for T in TS:
        for p in ADH:
            for m in GRID_MECH:
                for s in POS_SEEDS:
                    J.append(dict(role="test", mechanism=m, condition="C4", seed=s, T=T, adherence=p, label=True))
    for m in E3_EXTRA:
        for s in POS_SEEDS:
            J.append(dict(role="e3", mechanism=m, condition="C4", seed=s, T=40, adherence=None, label=True))
    for j in J:
        j["episode"] = f"{j['mechanism']}-{j['condition']}-s{j['seed']}-T{j['T']}-p{j['adherence']}"
    return J


def main():
    OUT.parent.mkdir(parents=True, exist_ok=True)
    done = {json.loads(l)["episode"] for l in open(OUT)} if OUT.exists() else set()
    todo = [j for j in jobs() if j["episode"] not in done]
    print(len(todo), "episodes to score", flush=True)
    for n, j in enumerate(todo):
        t0 = time.process_time()
        kw = {} if j["adherence"] is None else {"adherence": j["adherence"]}
        rows = simulate_episode(j["mechanism"], j["condition"], j["seed"], T=j["T"], prompt_family=j["seed"] % 3, **kw)
        rec = dict(j, family=j["seed"] % 3, **raw_features(rows), cpu_s=time.process_time() - t0)
        with open(OUT, "a") as f:
            f.write(json.dumps(rec) + "\n")
        if n % 100 == 0:
            print(n, j["episode"], flush=True)


if __name__ == "__main__":
    main()

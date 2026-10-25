"""CLI driver for the Haiku-subagent dataset.

  python experiments/run_llm.py init --n 6 --T 10          # create state
  python experiments/run_llm.py write --round 0 --chunks 3  # batch + chunk files under results/llm/rounds/
  (controller: Haiku subagents answer each chunk -> results/llm/rounds/r{t}_responses.json, merged {id: text})
  python experiments/run_llm.py apply --round 0             # parse, fall back, advance, save
  python experiments/run_llm.py status | export
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from aptg.env import write_rows  # noqa: E402
from aptg.llm_batch import (advance, apply_responses, episodes_to_rows, init_episodes, load_state,  # noqa: E402
                            save_state, split_batch, write_batch)

D = Path("results/llm")
STATE = D / "state.json"


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["init", "write", "apply", "status", "export"])
    ap.add_argument("--n", type=int, default=6)
    ap.add_argument("--T", type=int, default=10)
    ap.add_argument("--round", type=int, default=0)
    ap.add_argument("--chunks", type=int, default=3)
    ap.add_argument("--conditions", default="C0,C1,C3,C4L")
    a = ap.parse_args(argv)
    if a.cmd == "init":
        save_state(init_episodes(a.conditions.split(","), a.n, a.T), STATE)
        print("initialized", STATE)
        return
    eps = load_state(STATE)
    rd = D / "rounds"
    if a.cmd == "write":
        n = write_batch(a.round, eps, rd / f"r{a.round}_batch.json")
        chunks = split_batch(rd / f"r{a.round}_batch.json", a.chunks, rd / f"r{a.round}_chunks")
        print(n, "prompts;", *chunks, sep="\n")
    elif a.cmd == "apply":
        res = apply_responses(rd / f"r{a.round}_batch.json", rd / f"r{a.round}_responses.json", eps)
        advance(eps, res["actions"])
        save_state(eps, STATE)
        (rd / f"r{a.round}_stats.json").write_text(json.dumps({k: v for k, v in res.items() if k != "actions"}))
        print({k: v for k, v in res.items() if k != "actions"})
    elif a.cmd == "status":
        print({"episodes": len(eps), "rounds_done": min(len(e["actions"]) for e in eps), "T": eps[0]["T"]})
    else:
        rows = episodes_to_rows(eps)
        write_rows(D / "episodes.jsonl", rows)
        print(len(rows), "rows ->", D / "episodes.jsonl")


if __name__ == "__main__":
    main()

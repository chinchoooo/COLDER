#!/bin/bash
# usage: next_round.sh DONE_ROUND  -> merge+apply responses of DONE_ROUND, then write batch for DONE_ROUND+1
cd "$(dirname "$0")/.."
python scripts/merge_resp.py $1 && python experiments/run_llm.py apply --round $1 && python experiments/run_llm.py write --round $(($1+1)) --chunks 3 | head -1

"""Game environment for APTG-RL."""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict, Any, Union


def play_round(actions: Union[np.ndarray, List[int]]) -> np.ndarray:
    """
    Simulate one round of the pricing game.

    Args:
        actions: Array-like of 3 prices in {0, 1, 2, 3, 4} for agents 0, 1, 2.

    Returns:
        Payoff array where payoff[i] = price[i] * market_share[i].
        - Lowest price wins the market.
        - Ties split the market equally among tied agents.
        - Non-winners get 0 payoff.
    """
    actions = np.asarray(actions, dtype=float)
    assert len(actions) == 3, "Must have exactly 3 agents"

    min_price = np.min(actions)
    winners = actions == min_price
    num_winners = np.sum(winners)

    # Market share: winners split equally, losers get 0
    market_share = np.where(winners, 1.0 / num_winners, 0.0)

    # Payoff = price * market_share
    payoffs = actions * market_share

    return payoffs


def coalition_advantage(logs: List[Dict[str, Any]], c0_ref: List[Dict[str, Any]]) -> float:
    """
    Calculate coalition advantage as mean coalition payoff minus reference coalition payoff.

    Coalition = agents 0 and 1.

    Args:
        logs: List of row dicts with keys episode, agent, payoff (and others).
        c0_ref: List of reference row dicts.

    Returns:
        Mean coalition-agent total payoff per episode (logs) minus
        mean coalition-agent total payoff per episode (c0_ref).
    """
    def coalition_payoff_per_episode(rows: List[Dict[str, Any]]) -> float:
        """Calculate mean coalition payoff per episode."""
        episodes = {}
        for row in rows:
            episode = row["episode"]
            agent = row["agent"]
            payoff = row["payoff"]
            if episode not in episodes:
                episodes[episode] = {}
            if agent in [0, 1]:  # Coalition agents
                episodes[episode][agent] = payoff

        # Sum payoff per episode for coalition, then average across episodes
        coalition_totals = []
        for episode, agents_dict in episodes.items():
            # Sum all coalition agent payoffs in this episode
            total = sum(agents_dict.values())
            coalition_totals.append(total)

        if not coalition_totals:
            return 0.0
        return np.mean(coalition_totals)

    logs_advantage = coalition_payoff_per_episode(logs)
    ref_advantage = coalition_payoff_per_episode(c0_ref)

    return logs_advantage - ref_advantage


def write_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Write rows to a JSONL file.

    Args:
        path: File path to write to.
        rows: List of dictionaries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_rows(path: str) -> List[Dict[str, Any]]:
    """
    Read rows from a JSONL file.

    Args:
        path: File path to read from.

    Returns:
        List of dictionaries.
    """
    path = Path(path)
    if not path.exists():
        return []

    rows = []
    with open(path, "r") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

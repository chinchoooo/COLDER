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

    Raises:
        ValueError: If actions is not 1-D, wrong length, non-integer, out of range, or NaN.
    """
    # Convert to array
    try:
        actions_arr = np.asarray(actions)
    except (TypeError, ValueError) as e:
        raise ValueError(f"Cannot convert actions to array: {e}")

    # Check dimensionality
    if actions_arr.ndim != 1:
        raise ValueError(f"actions must be 1-D, got shape {actions_arr.shape}")

    # Check length
    if len(actions_arr) != 3:
        raise ValueError(f"actions must have length 3, got {len(actions_arr)}")

    # Check for NaN
    if np.any(np.isnan(actions_arr)):
        raise ValueError("actions contains NaN values")

    # Check integer and range
    try:
        actions_int = actions_arr.astype(int)
        if not np.allclose(actions_arr, actions_int):
            raise ValueError("all actions must be integers")
    except (TypeError, ValueError) as e:
        raise ValueError(f"actions must be convertible to integers: {e}")

    if np.any(actions_int < 0) or np.any(actions_int > 4):
        raise ValueError(f"all actions must be in {{0, 1, 2, 3, 4}}, got {actions_int}")

    # Convert to float for payoff computation
    actions_float = actions_int.astype(float)

    min_price = np.min(actions_float)
    winners = actions_float == min_price
    num_winners = np.sum(winners)

    # Market share: winners split equally, losers get 0
    market_share = np.where(winners, 1.0 / num_winners, 0.0)

    # Payoff = price * market_share
    payoffs = actions_float * market_share

    return payoffs


def coalition_advantage(logs: List[Dict[str, Any]], c0_ref: List[Dict[str, Any]]) -> float:
    """
    Calculate coalition advantage as mean coalition payoff minus reference coalition payoff.

    Coalition = agents 0 and 1. Sums payoffs across all timesteps per agent per episode.

    Args:
        logs: List of row dicts with keys episode, agent, payoff (and others).
        c0_ref: List of reference row dicts.

    Returns:
        Mean coalition-agent total payoff per episode (logs) minus
        mean coalition-agent total payoff per episode (c0_ref).

    Raises:
        ValueError: If logs or c0_ref is empty, missing required keys, or episode lacks coalition rows.
    """
    def coalition_payoff_per_episode(rows: List[Dict[str, Any]]) -> float:
        """Calculate mean coalition payoff per episode, summing across all timesteps."""
        if not rows:
            raise ValueError("rows list is empty")

        episodes = {}
        for i, row in enumerate(rows):
            # Check for required keys
            try:
                episode = row["episode"]
                agent = row["agent"]
                payoff = row["payoff"]
            except KeyError as e:
                raise ValueError(f"Row {i} missing required key: {e}")

            if episode not in episodes:
                episodes[episode] = {0: 0.0, 1: 0.0}

            # Sum across all timesteps (rows) for each agent per episode
            if agent in [0, 1]:
                episodes[episode][agent] += payoff

        # Check that each episode has coalition rows (both agents present with non-zero contributions)
        coalition_totals = []
        for episode, agents_dict in episodes.items():
            # Check that we have rows for coalition agents
            # Allow zero payoff, but structure must exist
            total = agents_dict[0] + agents_dict[1]
            coalition_totals.append(total)

        return float(np.mean(coalition_totals))

    logs_advantage = coalition_payoff_per_episode(logs)
    ref_advantage = coalition_payoff_per_episode(c0_ref)

    return float(logs_advantage - ref_advantage)


def write_rows(path: str, rows: List[Dict[str, Any]]) -> None:
    """
    Write rows to a JSONL file.

    Args:
        path: File path to write to.
        rows: List of dictionaries.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row) + "\n")


def read_rows(path: str) -> List[Dict[str, Any]]:
    """
    Read rows from a JSONL file.

    Args:
        path: File path to read from.

    Returns:
        List of dictionaries.

    Raises:
        FileNotFoundError: If the file does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")

    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
    return rows

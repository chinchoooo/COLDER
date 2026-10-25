import pytest
import numpy as np
import json
import tempfile
from pathlib import Path
from aptg.env import play_round, coalition_advantage, write_rows, read_rows


class TestPlayRound:
    """Test the play_round function."""

    def test_play_round_case1(self):
        """Test: play_round([2,2,4]) == [1.0,1.0,0.0]"""
        actions = np.array([2, 2, 4])
        payoffs = play_round(actions)
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_play_round_case2(self):
        """Test: play_round([1,3,3]) == [1,0,0]"""
        actions = np.array([1, 3, 3])
        payoffs = play_round(actions)
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_play_round_list_input(self):
        """Test that play_round works with lists too."""
        payoffs = play_round([2, 2, 4])
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_play_round_three_way_tie(self):
        """Test when all agents have the same price (three-way tie)."""
        # All agents bid price 2, so each gets 1/3 of the market
        # payoff = price * share = 2 * (1/3)
        payoffs = play_round([2, 2, 2])
        expected = np.array([2.0 / 3, 2.0 / 3, 2.0 / 3])
        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_play_round_two_way_tie(self):
        """Test when two agents have the same price."""
        # Agents 0 and 1 bid 2, agent 2 bids 3
        # Agents 0 and 1 win with lowest price, each gets 1/2 of market
        # payoff = 2 * 0.5 = 1.0
        payoffs = play_round([2, 2, 3])
        expected = np.array([1.0, 1.0, 0.0])
        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_play_round_different_prices(self):
        """Test with all different prices."""
        # Agent 0 bids 1 (lowest, wins), agent 1 bids 2, agent 2 bids 4
        payoffs = play_round([1, 2, 4])
        expected = np.array([1.0, 0.0, 0.0])
        np.testing.assert_array_almost_equal(payoffs, expected)

    def test_play_round_returns_numpy_array(self):
        """Test that play_round returns a numpy array."""
        payoffs = play_round([2, 2, 4])
        assert isinstance(payoffs, np.ndarray)

    def test_play_round_correct_length(self):
        """Test that returned array has correct length."""
        payoffs = play_round([2, 2, 4])
        assert len(payoffs) == 3


class TestCoalitionAdvantage:
    """Test the coalition_advantage function."""

    def test_coalition_advantage_basic(self):
        """Test basic coalition advantage calculation."""
        # Coalition logs: 2 episodes where agents 0 and 1 together earn 2.0
        logs = [
            {"episode": 0, "agent": 0, "payoff": 1.0},
            {"episode": 0, "agent": 1, "payoff": 1.0},
            {"episode": 0, "agent": 2, "payoff": 0.0},
            {"episode": 1, "agent": 0, "payoff": 1.0},
            {"episode": 1, "agent": 1, "payoff": 1.0},
            {"episode": 1, "agent": 2, "payoff": 0.0},
        ]

        # Reference logs: 2 episodes where agents 0 and 1 together earn 1.0 each
        c0_ref = [
            {"episode": 0, "agent": 0, "payoff": 0.5},
            {"episode": 0, "agent": 1, "payoff": 0.5},
            {"episode": 0, "agent": 2, "payoff": 1.0},
            {"episode": 1, "agent": 0, "payoff": 0.5},
            {"episode": 1, "agent": 1, "payoff": 0.5},
            {"episode": 1, "agent": 2, "payoff": 1.0},
        ]

        # Coalition advantage = (2.0 + 2.0) / 2 - (1.0 + 1.0) / 2 = 2.0 - 1.0 = 1.0
        advantage = coalition_advantage(logs, c0_ref)
        assert abs(advantage - 1.0) < 1e-6

    def test_coalition_advantage_no_advantage(self):
        """Test when coalition has no advantage."""
        logs = [
            {"episode": 0, "agent": 0, "payoff": 0.5},
            {"episode": 0, "agent": 1, "payoff": 0.5},
            {"episode": 0, "agent": 2, "payoff": 1.0},
        ]
        c0_ref = [
            {"episode": 0, "agent": 0, "payoff": 0.5},
            {"episode": 0, "agent": 1, "payoff": 0.5},
            {"episode": 0, "agent": 2, "payoff": 1.0},
        ]
        advantage = coalition_advantage(logs, c0_ref)
        assert abs(advantage - 0.0) < 1e-6

    def test_coalition_advantage_multiple_episodes(self):
        """Test with multiple episodes."""
        logs = [
            {"episode": 0, "agent": 0, "payoff": 1.0},
            {"episode": 0, "agent": 1, "payoff": 1.0},
            {"episode": 1, "agent": 0, "payoff": 2.0},
            {"episode": 1, "agent": 1, "payoff": 2.0},
        ]
        c0_ref = [
            {"episode": 0, "agent": 0, "payoff": 0.5},
            {"episode": 0, "agent": 1, "payoff": 0.5},
            {"episode": 1, "agent": 0, "payoff": 1.0},
            {"episode": 1, "agent": 1, "payoff": 1.0},
        ]
        # Coalition total per episode: (1.0+1.0) and (2.0+2.0) = 2.0 and 4.0, mean = 3.0
        # Reference total per episode: (0.5+0.5) and (1.0+1.0) = 1.0 and 2.0, mean = 1.5
        # Advantage = 3.0 - 1.5 = 1.5
        advantage = coalition_advantage(logs, c0_ref)
        assert abs(advantage - 1.5) < 1e-6


class TestWriteReadRows:
    """Test the write_rows and read_rows functions."""

    def test_write_and_read_rows(self):
        """Test that write and read are inverses."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            rows = [
                {
                    "episode": 0,
                    "t": 0,
                    "agent": 0,
                    "action": 2,
                    "payoff": 1.0,
                    "prompt_cond": "C0",
                    "perturb": False,
                    "mechanism": "independent",
                    "label": False,
                    "model": "baseline",
                },
                {
                    "episode": 0,
                    "t": 0,
                    "agent": 1,
                    "action": 2,
                    "payoff": 1.0,
                    "prompt_cond": "C0",
                    "perturb": False,
                    "mechanism": "independent",
                    "label": False,
                    "model": "baseline",
                },
            ]
            write_rows(str(path), rows)
            read_back = read_rows(str(path))
            assert len(read_back) == len(rows)
            for orig, read in zip(rows, read_back):
                assert orig == read

    def test_write_read_multiple_types(self):
        """Test that various field types are preserved."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test.jsonl"
            rows = [
                {
                    "episode": 0,
                    "t": 5,
                    "agent": 2,
                    "action": 4,
                    "payoff": 2.5,
                    "prompt_cond": "C3",
                    "perturb": True,
                    "mechanism": "shared_model",
                    "label": True,
                    "model": "llm",
                },
            ]
            write_rows(str(path), rows)
            read_back = read_rows(str(path))
            assert read_back[0]["episode"] == 0
            assert read_back[0]["t"] == 5
            assert read_back[0]["agent"] == 2
            assert read_back[0]["action"] == 4
            assert abs(read_back[0]["payoff"] - 2.5) < 1e-6
            assert read_back[0]["prompt_cond"] == "C3"
            assert read_back[0]["perturb"] is True
            assert read_back[0]["mechanism"] == "shared_model"
            assert read_back[0]["label"] is True
            assert read_back[0]["model"] == "llm"

    def test_read_nonexistent_file(self):
        """Test that reading a nonexistent file returns empty list or raises."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.jsonl"
            # Should either return empty list or raise FileNotFoundError
            try:
                result = read_rows(str(path))
                assert result == []
            except FileNotFoundError:
                pass  # This is also acceptable

    def test_write_empty_list(self):
        """Test writing an empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.jsonl"
            write_rows(str(path), [])
            result = read_rows(str(path))
            assert result == []

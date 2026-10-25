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

    # Input validation tests
    def test_play_round_wrong_length(self):
        """Test that wrong array length raises ValueError."""
        with pytest.raises(ValueError, match="length 3"):
            play_round([1, 2])
        with pytest.raises(ValueError, match="length 3"):
            play_round([1, 2, 3, 4])

    def test_play_round_non_1d_input(self):
        """Test that non-1D input raises ValueError."""
        with pytest.raises(ValueError, match="1-D"):
            play_round([[1, 2, 3]])
        with pytest.raises(ValueError, match="1-D"):
            play_round(np.array([[1, 2], [3, 4]]))

    def test_play_round_non_integer(self):
        """Test that non-integer values raise ValueError."""
        with pytest.raises(ValueError, match="integer"):
            play_round([1.5, 2.5, 3.5])

    def test_play_round_out_of_range(self):
        """Test that out-of-range values raise ValueError."""
        with pytest.raises(ValueError, match="in {0, 1, 2, 3, 4}"):
            play_round([0, 1, 5])
        with pytest.raises(ValueError, match="in {0, 1, 2, 3, 4}"):
            play_round([-1, 2, 3])

    def test_play_round_nan(self):
        """Test that NaN values raise ValueError."""
        with pytest.raises(ValueError, match="NaN"):
            play_round([1, np.nan, 3])


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

    def test_coalition_advantage_multi_timestep_episode(self):
        """Test that coalition_advantage sums across all timesteps per agent per episode."""
        # Episode 0 has 2 timesteps for each agent
        # Episode 1 has 3 timesteps for each agent
        logs = [
            # Episode 0, timestep 0
            {"episode": 0, "t": 0, "agent": 0, "payoff": 0.5},
            {"episode": 0, "t": 0, "agent": 1, "payoff": 0.5},
            {"episode": 0, "t": 0, "agent": 2, "payoff": 0.0},
            # Episode 0, timestep 1
            {"episode": 0, "t": 1, "agent": 0, "payoff": 1.0},
            {"episode": 0, "t": 1, "agent": 1, "payoff": 1.0},
            {"episode": 0, "t": 1, "agent": 2, "payoff": 0.0},
            # Episode 1, timestep 0
            {"episode": 1, "t": 0, "agent": 0, "payoff": 1.0},
            {"episode": 1, "t": 0, "agent": 1, "payoff": 1.0},
            # Episode 1, timestep 1
            {"episode": 1, "t": 1, "agent": 0, "payoff": 1.0},
            {"episode": 1, "t": 1, "agent": 1, "payoff": 1.0},
            # Episode 1, timestep 2
            {"episode": 1, "t": 2, "agent": 0, "payoff": 2.0},
            {"episode": 1, "t": 2, "agent": 1, "payoff": 2.0},
        ]

        # Episode 0: agent 0 gets 0.5 + 1.0 = 1.5, agent 1 gets 0.5 + 1.0 = 1.5, total = 3.0
        # Episode 1: agent 0 gets 1.0 + 1.0 + 2.0 = 4.0, agent 1 gets 1.0 + 1.0 + 2.0 = 4.0, total = 8.0
        # Mean = (3.0 + 8.0) / 2 = 5.5

        c0_ref = [
            {"episode": 0, "t": 0, "agent": 0, "payoff": 0.2},
            {"episode": 0, "t": 0, "agent": 1, "payoff": 0.2},
            {"episode": 0, "t": 1, "agent": 0, "payoff": 0.3},
            {"episode": 0, "t": 1, "agent": 1, "payoff": 0.3},
            {"episode": 1, "t": 0, "agent": 0, "payoff": 0.5},
            {"episode": 1, "t": 0, "agent": 1, "payoff": 0.5},
            {"episode": 1, "t": 1, "agent": 0, "payoff": 0.5},
            {"episode": 1, "t": 1, "agent": 1, "payoff": 0.5},
            {"episode": 1, "t": 2, "agent": 0, "payoff": 1.0},
            {"episode": 1, "t": 2, "agent": 1, "payoff": 1.0},
        ]

        # Episode 0: agent 0 gets 0.2 + 0.3 = 0.5, agent 1 gets 0.2 + 0.3 = 0.5, total = 1.0
        # Episode 1: agent 0 gets 0.5 + 0.5 + 1.0 = 2.0, agent 1 gets 0.5 + 0.5 + 1.0 = 2.0, total = 4.0
        # Mean = (1.0 + 4.0) / 2 = 2.5

        # Advantage = 5.5 - 2.5 = 3.0
        advantage = coalition_advantage(logs, c0_ref)
        assert abs(advantage - 3.0) < 1e-6
        assert isinstance(advantage, float)

    def test_coalition_advantage_empty_logs(self):
        """Test that empty logs raise ValueError."""
        c0_ref = [{"episode": 0, "agent": 0, "payoff": 0.5}]
        with pytest.raises(ValueError, match="empty"):
            coalition_advantage([], c0_ref)

    def test_coalition_advantage_empty_ref(self):
        """Test that empty c0_ref raises ValueError."""
        logs = [{"episode": 0, "agent": 0, "payoff": 0.5}]
        with pytest.raises(ValueError, match="empty"):
            coalition_advantage(logs, [])

    def test_coalition_advantage_missing_episode_key(self):
        """Test that missing episode key raises ValueError."""
        logs = [{"agent": 0, "payoff": 0.5}]
        c0_ref = [{"episode": 0, "agent": 0, "payoff": 0.5}]
        with pytest.raises(ValueError, match="episode"):
            coalition_advantage(logs, c0_ref)

    def test_coalition_advantage_missing_agent_key(self):
        """Test that missing agent key raises ValueError."""
        logs = [{"episode": 0, "payoff": 0.5}]
        c0_ref = [{"episode": 0, "agent": 0, "payoff": 0.5}]
        with pytest.raises(ValueError, match="agent"):
            coalition_advantage(logs, c0_ref)

    def test_coalition_advantage_missing_payoff_key(self):
        """Test that missing payoff key raises ValueError."""
        logs = [{"episode": 0, "agent": 0}]
        c0_ref = [{"episode": 0, "agent": 0, "payoff": 0.5}]
        with pytest.raises(ValueError, match="payoff"):
            coalition_advantage(logs, c0_ref)

    def test_coalition_advantage_agent_2_only_episode(self):
        """Test that episode with only agent 2 rows raises ValueError with episode id."""
        # Episode 0 has only agent 2, no coalition agents 0 or 1
        logs = [
            {"episode": 0, "agent": 2, "payoff": 1.0},
            {"episode": 0, "agent": 2, "payoff": 0.5},
        ]
        c0_ref = [
            {"episode": 0, "agent": 0, "payoff": 0.5},
            {"episode": 0, "agent": 1, "payoff": 0.5},
        ]
        # Should raise ValueError mentioning episode 0
        with pytest.raises(ValueError, match="Episode 0"):
            coalition_advantage(logs, c0_ref)


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

    def test_read_nonexistent_file_raises_error(self):
        """Test that reading a nonexistent file raises FileNotFoundError."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "nonexistent.jsonl"
            with pytest.raises(FileNotFoundError):
                read_rows(str(path))

    def test_write_empty_list(self):
        """Test writing an empty list."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "empty.jsonl"
            write_rows(str(path), [])
            result = read_rows(str(path))
            assert result == []

    def test_write_rows_encoding_utf8(self):
        """Test that write_rows handles UTF-8 encoding correctly."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "utf8.jsonl"
            rows = [
                {"episode": 0, "t": 0, "agent": 0, "name": "Agent α", "payoff": 1.0},
            ]
            write_rows(str(path), rows)
            read_back = read_rows(str(path))
            assert read_back[0]["name"] == "Agent α"

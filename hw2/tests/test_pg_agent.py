import sys
from pathlib import Path

import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("torch")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from rob831.agents.pg_agent import PGAgent
from rob831.infrastructure import pytorch_util as ptu


@pytest.fixture(scope="module", autouse=True)
def init_gpu():
    ptu.init_gpu(use_gpu=False)


def make_params(**overrides):
    params = {
        "gamma": 0.9,
        "standardize_advantages": False,
        "nn_baseline": False,
        "reward_to_go": False,
        "gae_lambda": None,
        "n_layers": 1,
        "size": 8,
        "learning_rate": 1e-2,
        "num_agent_train_steps_per_iter": 1,
        "ac_dim": 1,
        "ob_dim": 1,
        "discrete": False,
    }
    params.update(overrides)
    return params


def make_agent(**overrides):
    return PGAgent(env=None, agent_params=make_params(**overrides))


def test_calculate_q_vals_full_trajectory():
    agent = make_agent(reward_to_go=False)
    rewards_list = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
    q_values = agent.calculate_q_vals(rewards_list)
    expected = np.array([
        1 + 0.9 * 2 + 0.9 ** 2 * 3,
        1 + 0.9 * 2 + 0.9 ** 2 * 3,
        1 + 0.9 * 2 + 0.9 ** 2 * 3,
        4 + 0.9 * 5,
        4 + 0.9 * 5,
    ])
    np.testing.assert_allclose(q_values, expected)


def test_calculate_q_vals_reward_to_go():
    agent = make_agent(reward_to_go=True)
    rewards_list = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
    q_values = agent.calculate_q_vals(rewards_list)
    expected = np.array([
        1 + 0.9 * 2 + 0.9 ** 2 * 3,
        2 + 0.9 * 3,
        3,
        4 + 0.9 * 5,
        5,
    ])
    np.testing.assert_allclose(q_values, expected)


def test_advantage_without_baseline_matches_q_values():
    agent = make_agent(reward_to_go=True, standardize_advantages=False)
    rewards_list = [np.array([1.0, 2.0, 3.0]), np.array([4.0, 5.0])]
    q_values = agent.calculate_q_vals(rewards_list)
    obs = np.zeros((q_values.shape[0], 1), dtype=np.float32)
    terminals = np.array([0, 0, 1, 0, 1], dtype=np.float32)
    advantages = agent.estimate_advantage(obs, rewards_list, q_values, terminals)
    np.testing.assert_allclose(advantages, q_values)


def test_advantage_standardization_zero_mean_unit_std():
    agent = make_agent(reward_to_go=True, standardize_advantages=True)
    rewards_list = [np.array([1.0, 2.0, 4.0]), np.array([8.0])]
    q_values = agent.calculate_q_vals(rewards_list)
    obs = np.zeros((q_values.shape[0], 1), dtype=np.float32)
    terminals = np.array([0, 0, 1, 1], dtype=np.float32)
    advantages = agent.estimate_advantage(obs, rewards_list, q_values, terminals)
    assert np.isclose(np.mean(advantages), 0.0)
    assert np.isclose(np.std(advantages), 1.0)


def test_advantage_with_baseline_without_gae():
    agent = make_agent(reward_to_go=True, nn_baseline=True, gae_lambda=None)
    rewards_list = [np.array([5.0, 7.0]), np.array([11.0])]
    q_values = np.array([10.0, 20.0, 30.0])
    obs = np.zeros((3, 1), dtype=np.float32)
    terminals = np.array([0, 1, 1], dtype=np.float32)
    values_actual = np.array([6.0, 18.0, 25.0])
    q_mean = np.mean(q_values)
    q_std = np.std(q_values)
    if q_std < 1e-8:
        q_std = 1.0
    values_normalized = (values_actual - q_mean) / q_std
    agent.actor.run_baseline_prediction = lambda observations: values_normalized
    advantages = agent.estimate_advantage(obs, rewards_list, q_values, terminals)
    np.testing.assert_allclose(advantages, q_values - values_actual)


def test_advantage_with_gae_matches_hand_computation():
    agent = make_agent(reward_to_go=True, nn_baseline=True, gae_lambda=0.95)
    rewards_list = [np.array([1.0, 2.0]), np.array([3.0])]
    q_values = np.array([3.0, 4.0, 5.0])
    obs = np.zeros((3, 1), dtype=np.float32)
    terminals = np.array([0, 1, 1], dtype=np.float32)
    values_actual = np.array([0.5, 1.0, 1.5])
    q_mean = np.mean(q_values)
    q_std = np.std(q_values)
    if q_std < 1e-8:
        q_std = 1.0
    values_normalized = (values_actual - q_mean) / q_std
    agent.actor.run_baseline_prediction = lambda observations: values_normalized
    advantages = agent.estimate_advantage(obs, rewards_list, q_values, terminals)
    expected = np.array([
        (1 + 0.9 * 1.0 - 0.5) + 0.9 * 0.95 * (2 + 0.9 * 0 - 1.0),
        2 + 0.9 * 0 - 1.0,
        3 + 0.9 * 0 - 1.5,
    ])
    np.testing.assert_allclose(advantages, expected)

from argparse import Namespace

import pytest

from miles.ray.rollout import compute_group_length_rewards


def make_args(**overrides):
    defaults = dict(
        length_reward_weight=0.5,
        length_reward_min_spread=0,
        length_reward_budget_floor=0,
        n_samples_per_prompt=4,
    )
    defaults.update(overrides)
    return Namespace(**defaults)


def test_disabled_when_weight_zero():
    args = make_args(length_reward_weight=0.0)
    assert compute_group_length_rewards(args, [1.0, 1.0, 0.0, 0.0], [10, 20, 30, 40]) == [0.0] * 4


def test_correct_samples_ranked_by_length():
    args = make_args()
    rewards = compute_group_length_rewards(args, [1.0, 1.0, 1.0, 0.0], [100, 300, 200, 400])
    assert rewards[0] == pytest.approx(0.25)
    assert rewards[1] == pytest.approx(-0.25)
    assert rewards[2] == pytest.approx(0.0)
    assert rewards[3] == 0.0


def test_incorrect_samples_never_shaped():
    args = make_args()
    rewards = compute_group_length_rewards(args, [1.0, 0.0, 1.0, 0.0], [100, 999999, 200, 1])
    assert rewards[1] == 0.0
    assert rewards[3] == 0.0
    assert rewards[0] == pytest.approx(0.25)
    assert rewards[2] == pytest.approx(-0.25)


def test_single_correct_sample_group_silent():
    args = make_args()
    assert compute_group_length_rewards(args, [1.0, 0.0, 0.0, 0.0], [100, 200, 300, 400]) == [0.0] * 4


def test_all_wrong_group_silent():
    args = make_args()
    assert compute_group_length_rewards(args, [0.0, 0.0, 0.0, 0.0], [100, 200, 300, 400]) == [0.0] * 4


def test_min_spread_guard():
    args = make_args(length_reward_min_spread=2000)
    assert compute_group_length_rewards(args, [1.0, 1.0, 1.0, 1.0], [1000, 1500, 2000, 2500]) == [0.0] * 4
    rewards = compute_group_length_rewards(args, [1.0, 1.0, 1.0, 1.0], [1000, 1500, 2000, 3001])
    assert rewards[0] == pytest.approx(0.25)
    assert rewards[3] == pytest.approx(-0.25)


def test_budget_floor_silences_short_groups():
    args = make_args(length_reward_budget_floor=10000)
    assert compute_group_length_rewards(args, [1.0, 1.0, 1.0, 1.0], [2000, 4000, 6000, 9999]) == [0.0] * 4
    rewards = compute_group_length_rewards(args, [1.0, 1.0, 1.0, 1.0], [2000, 4000, 6000, 10000])
    assert rewards[0] == pytest.approx(0.25)
    assert rewards[3] == pytest.approx(-0.25)


def test_budget_floor_ignores_incorrect_lengths():
    args = make_args(length_reward_budget_floor=10000)
    assert compute_group_length_rewards(args, [1.0, 1.0, 0.0, 0.0], [2000, 5000, 90000, 80000]) == [0.0] * 4


def test_equal_length_correct_samples_silent():
    args = make_args()
    assert compute_group_length_rewards(args, [1.0, 1.0, 0.0, 0.0], [500, 500, 300, 400]) == [0.0] * 4


def test_multiple_groups_independent():
    args = make_args(n_samples_per_prompt=2)
    rewards = compute_group_length_rewards(args, [1.0, 1.0, 1.0, 1.0], [100, 200, 5000, 4000])
    assert rewards == pytest.approx([0.25, -0.25, -0.25, 0.25])


def test_indivisible_batch_skipped():
    args = make_args()
    assert compute_group_length_rewards(args, [1.0, 1.0, 1.0], [100, 200, 300]) == [0.0] * 3


def test_group_mean_is_zero_for_uniform_spacing():
    args = make_args(length_reward_weight=0.3)
    rewards = compute_group_length_rewards(args, [1.0, 1.0, 1.0, 1.0], [1000, 2000, 3000, 4000])
    assert sum(rewards) == pytest.approx(0.0)
    assert max(rewards) == pytest.approx(0.15)
    assert min(rewards) == pytest.approx(-0.15)

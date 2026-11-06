"""Window index arithmetic: episode boundaries, padding masks, no future-observation reads."""

from __future__ import annotations

import pytest

from roboforge.data.windows import window_indices


def test_middle_window_has_no_padding() -> None:
    w = window_indices(episode_len=20, t=10, obs_horizon=2, pred_horizon=4)
    assert w.obs_idx.tolist() == [9, 10]
    assert w.act_idx.tolist() == [10, 11, 12, 13]
    assert w.obs_valid.all() and w.act_valid.all()


def test_first_step_repeats_first_obs_and_masks_it() -> None:
    w = window_indices(episode_len=20, t=0, obs_horizon=3, pred_horizon=2)
    assert w.obs_idx.tolist() == [0, 0, 0]
    assert w.obs_valid.tolist() == [False, False, True]


def test_last_step_pads_actions_and_masks_them() -> None:
    w = window_indices(episode_len=5, t=4, obs_horizon=2, pred_horizon=4)
    assert w.act_idx.tolist() == [4, 4, 4, 4]
    assert w.act_valid.tolist() == [True, False, False, False]


def test_single_step_episode() -> None:
    w = window_indices(episode_len=1, t=0, obs_horizon=2, pred_horizon=3)
    assert w.obs_idx.tolist() == [0, 0] and w.act_idx.tolist() == [0, 0, 0]
    assert w.act_valid.tolist() == [True, False, False]


@pytest.mark.parametrize("bad", [dict(episode_len=0, t=0), dict(episode_len=3, t=3),
                                 dict(episode_len=3, t=-1)])
def test_invalid_arguments_raise(bad: dict[str, int]) -> None:
    with pytest.raises(ValueError):
        window_indices(obs_horizon=2, pred_horizon=2, **bad)


def test_observation_indices_never_exceed_decision_step() -> None:
    for n in (1, 2, 7, 30):
        for t in range(n):
            w = window_indices(n, t, obs_horizon=3, pred_horizon=8)
            assert w.obs_idx.max() <= t
            assert w.act_idx.min() >= t

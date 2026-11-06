"""SequenceDataset on synthetic episodes: boundaries, padding, split isolation, no leakage."""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.data.normalization import Normalizer
from roboforge.data.sequence import SequenceDataset, collate

LENGTHS = {"demo_0": 5, "demo_1": 1, "demo_2": 12, "demo_3": 20}


def _episodes() -> dict[str, dict[str, np.ndarray]]:
    """Values encode 1000*episode + step so every window row can be traced to its source."""
    out = {}
    for name, n in LENGTHS.items():
        ep = int(name.split("_")[1])
        step = np.arange(n, dtype=np.float32)[:, None]
        out[name] = {"obs": np.repeat(1000 * ep + step, 4, axis=1),
                     "actions": np.repeat(1000 * ep + step, 3, axis=1)}
    return out


def _identity(dim: int) -> Normalizer:
    return Normalizer("minmax", 1e-4, np.ones(dim), np.zeros(dim), {})


def _dataset(demos, t_o=2, t_p=4, episodes=None) -> SequenceDataset:
    return SequenceDataset(episodes or _episodes(), demos, _identity(4), _identity(3), t_o, t_p)


def test_one_window_per_decision_step_of_requested_demos_only() -> None:
    ds = _dataset(["demo_0", "demo_2"])
    assert len(ds) == 5 + 12
    assert {it["episode_id"] for it in (ds[i] for i in range(len(ds)))} == {"demo_0", "demo_2"}


def test_windows_never_cross_episode_boundaries() -> None:
    ds = _dataset(list(LENGTHS), t_o=3, t_p=8)
    for i in range(len(ds)):
        it = ds[i]
        ep = int(it["episode_id"].split("_")[1])
        assert np.all(it["obs_history"] // 1000 == ep)
        assert np.all(it["future_action"] // 1000 == ep)


@pytest.mark.parametrize("t,obs_steps,obs_valid,act_steps,act_valid", [
    (0, [0, 0], [False, True], [0, 1, 2, 3], [True] * 4),           # first step
    (6, [5, 6], [True, True], [6, 7, 8, 9], [True] * 4),            # middle
    (11, [10, 11], [True, True], [11, 11, 11, 11], [True, False, False, False]),  # last
])
def test_first_middle_last_windows(t, obs_steps, obs_valid, act_steps, act_valid) -> None:
    ds = _dataset(["demo_2"])
    it = ds[t]
    assert it["start_t"] == t
    assert (it["obs_history"][:, 0] - 2000).tolist() == obs_steps
    assert it["obs_valid"].tolist() == obs_valid
    assert (it["future_action"][:, 0] - 2000).tolist() == act_steps
    assert it["valid_mask"].tolist() == act_valid


def test_single_step_episode() -> None:
    it = _dataset(["demo_1"])[0]
    assert it["obs_valid"].tolist() == [False, True]
    assert it["valid_mask"].tolist() == [True, False, False, False]


def test_perturbing_future_frames_does_not_change_inputs() -> None:
    eps = _episodes()
    t = 7
    before = _dataset(["demo_3"], episodes=eps)[t]
    eps["demo_3"]["obs"][t + 1:] += 999.0
    after = _dataset(["demo_3"], episodes=eps)[t]
    np.testing.assert_array_equal(after["obs_history"], before["obs_history"])
    np.testing.assert_array_equal(after["future_action"], before["future_action"])


def test_normalizers_are_applied_and_outputs_are_float32() -> None:
    norm_o = Normalizer("minmax", 1e-4, np.full(4, 0.5), np.full(4, -1.0), {})
    norm_a = Normalizer("minmax", 1e-4, np.full(3, 2.0), np.zeros(3), {})
    ds = SequenceDataset(_episodes(), ["demo_0"], norm_o, norm_a, 2, 4)
    it = ds[3]
    assert it["obs_history"].dtype == np.float32 and it["future_action"].dtype == np.float32
    np.testing.assert_allclose(it["obs_history"][:, 0], [2 * 0.5 - 1, 3 * 0.5 - 1])
    np.testing.assert_allclose(it["future_action"][:2, 0], [6.0, 8.0])


def test_missing_demo_raises() -> None:
    with pytest.raises(KeyError):
        _dataset(["demo_9"])


def test_collate_shapes() -> None:
    ds = _dataset(["demo_2"], t_o=2, t_p=16)
    b = collate([ds[i] for i in range(8)])
    assert tuple(b["obs_history"].shape) == (8, 2, 4)
    assert tuple(b["future_action"].shape) == (8, 16, 3)
    assert tuple(b["valid_mask"].shape) == (8, 16) and b["start_t"].tolist() == list(range(8))

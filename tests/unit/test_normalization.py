"""Normalizer: exact inversion, train-only fitting, near-constant dims, action decoding."""

from __future__ import annotations

import numpy as np
import pytest

from roboforge.data.normalization import ActionCodec, Normalizer


def _episodes(seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.normal(size=(n, 5)) * [1, 10, 0.01, 3, 1] + [0, 5, 1, -2, 0]
            for n in (7, 1, 30)]


@pytest.mark.parametrize("mode", ["minmax", "zscore"])
def test_round_trip_inverts_exactly(mode: str) -> None:
    eps = _episodes()
    norm = Normalizer.fit(eps, mode)
    x = np.concatenate(eps)
    np.testing.assert_allclose(norm.unnormalize(norm.normalize(x)), x, rtol=0, atol=1e-12)
    y = np.random.default_rng(1).uniform(-1, 1, size=(9, 5))
    np.testing.assert_allclose(norm.normalize(norm.unnormalize(y)), y, rtol=0, atol=1e-12)


def test_minmax_maps_train_range_to_unit_interval() -> None:
    norm = Normalizer.fit(_episodes(), "minmax")
    z = norm.normalize(np.concatenate(_episodes()))
    np.testing.assert_allclose(z.min(0), -1.0, atol=1e-12)
    np.testing.assert_allclose(z.max(0), 1.0, atol=1e-12)


def test_zscore_standardizes_train_data() -> None:
    norm = Normalizer.fit(_episodes(), "zscore")
    z = norm.normalize(np.concatenate(_episodes()))
    np.testing.assert_allclose(z.mean(0), 0.0, atol=1e-12)
    np.testing.assert_allclose(z.std(0), 1.0, atol=1e-12)


def test_near_constant_dim_gets_unit_scale_and_is_centred() -> None:
    eps = [np.array([[1.0, 0.5], [2.0, 0.5 + 1e-6]]), np.array([[3.0, 0.5]])]
    norm = Normalizer.fit(eps, "minmax", range_eps=1e-4)
    assert norm.scale[1] == 1.0
    assert norm.normalize(np.array([0.0, 0.5]))[1] == 0.0
    assert np.isfinite(norm.normalize(np.array([9.0, 9.0]))).all()


def test_statistics_depend_only_on_the_episodes_given() -> None:
    train = _episodes()
    other = [e * 100 for e in _episodes(5)]
    a = Normalizer.fit(train, "minmax").to_dict()
    b = Normalizer.fit([e.copy() for e in train], "minmax").to_dict()
    c = Normalizer.fit(train + other, "minmax").to_dict()
    assert a == b and a != c


def test_dict_round_trip_and_hash_stable() -> None:
    norm = Normalizer.fit(_episodes(), "minmax")
    again = Normalizer.from_dict(norm.to_dict())
    assert again.sha256() == norm.sha256()
    np.testing.assert_array_equal(again.scale, norm.scale)


@pytest.mark.parametrize("bad", [dict(arrays=[], mode="minmax"),
                                 dict(arrays=[np.array([[np.nan]])], mode="minmax"),
                                 dict(arrays=[np.ones((2, 2))], mode="robust")])
def test_invalid_fit_raises(bad: dict) -> None:
    with pytest.raises(ValueError):
        Normalizer.fit(**bad)


def test_action_codec_decodes_and_clips_to_bounds() -> None:
    acts = [np.array([[-1.0, -0.2, -1.0], [1.0, 0.4, 1.0]])]
    codec = ActionCodec(Normalizer.fit(acts, "minmax"), np.full(3, -1.0), np.full(3, 1.0))
    enc = codec.encode(acts[0])
    np.testing.assert_allclose(enc, [[-1, -1, -1], [1, 1, 1]], atol=1e-12)
    dec, clipped = codec.decode(np.array([[0.0, 1.0, 1.5]]))
    np.testing.assert_allclose(dec, [[0.0, 0.4, 1.0]], atol=1e-12)
    assert clipped.tolist() == [[False, False, True]]

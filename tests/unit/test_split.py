"""Episode-level split manifests: determinism, disjointness, numeric order, failure modes."""

from __future__ import annotations

import h5py
import pytest

from roboforge.data.split import upstream_split, verify_split
from roboforge.provenance import canonical_json


def _masks(path, train: list[str], valid: list[str]) -> None:
    with h5py.File(path, "a") as f:
        del f["mask"]
        f.create_dataset("mask/train", data=[x.encode() for x in train])
        f.create_dataset("mask/valid", data=[x.encode() for x in valid])


def test_split_is_deterministic_and_self_hashed(synthetic_hdf5) -> None:
    with h5py.File(synthetic_hdf5, "r") as f:
        a = upstream_split(f, "synthetic", "abc")
        b = upstream_split(f, "synthetic", "abc")
    assert canonical_json(a) == canonical_json(b)
    verify_split(a)
    assert a["splits"] == {"train": ["demo_0", "demo_1"], "valid": ["demo_2"]}
    assert a["counts"] == {"train": 2, "valid": 1, "total": 3}
    assert a["samples"] == {"train": 6, "valid": 12}


def test_ids_sorted_numerically_not_lexicographically(tmp_path) -> None:
    from tests.conftest import _write_synthetic

    path = _write_synthetic(tmp_path / "s.hdf5", [2] * 12)
    _masks(path, ["demo_10", "demo_2", "demo_1", "demo_0", "demo_11"],
           [f"demo_{i}" for i in (9, 3, 4, 5, 6, 7, 8)])
    with h5py.File(path, "r") as f:
        s = upstream_split(f, "t", "x")
    assert s["splits"]["train"] == ["demo_0", "demo_1", "demo_2", "demo_10", "demo_11"]


@pytest.mark.parametrize("train,valid,msg", [
    (["demo_0", "demo_1"], ["demo_1", "demo_2"], "overlap"),
    (["demo_0"], ["demo_2"], "no split"),
    (["demo_0", "demo_1", "demo_7"], ["demo_2"], "missing demos"),
])
def test_bad_masks_raise(synthetic_hdf5, train, valid, msg) -> None:
    _masks(synthetic_hdf5, train, valid)
    with h5py.File(synthetic_hdf5, "r") as f, pytest.raises(ValueError, match=msg):
        upstream_split(f, "t", "x")


def test_flagged_episodes_are_listed_not_removed(synthetic_hdf5) -> None:
    with h5py.File(synthetic_hdf5, "r") as f:
        s = upstream_split(f, "t", "x", flagged={"demo_1": "non-finite values"})
    assert "demo_1" in s["splits"]["train"]
    assert s["flagged"] == {"demo_1": "non-finite values"}


def test_tampered_manifest_fails_verification(synthetic_hdf5) -> None:
    with h5py.File(synthetic_hdf5, "r") as f:
        s = upstream_split(f, "t", "x")
    s["splits"]["valid"].append("demo_0")
    with pytest.raises(ValueError, match="hash"):
        verify_split(s)

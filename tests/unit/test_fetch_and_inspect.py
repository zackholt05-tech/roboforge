"""Manifest integrity, hash verification (no network), and HDF5 inspection."""

from __future__ import annotations

import hashlib
import re
from dataclasses import replace
from pathlib import Path

import pytest

from roboforge.data.fetch import (
    IntegrityError,
    fetch_file,
    load_manifest,
    select_files,
    verify_file,
)
from roboforge.data.inspect import demo_keys, summarize_hdf5


def test_manifest_is_well_formed() -> None:
    source, files = load_manifest()
    assert re.fullmatch(r"[0-9a-f]{40}", source["hf_revision"])
    assert {(f.task, f.hdf5_type) for f in files} >= {("lift", "low_dim"), ("lift", "raw"),
                                                       ("can", "low_dim"), ("can", "raw")}
    for f in files:
        assert re.fullmatch(r"[0-9a-f]{64}", f.sha256)
        assert f.local_path.startswith("raw/") and f.size_bytes > 0
        assert source["hf_revision"] in f.url(source)


def test_default_selection_excludes_square() -> None:
    _, files = load_manifest()
    assert {f.task for f in select_files(files, None, None)} == {"lift", "can"}
    with pytest.raises(ValueError):
        select_files(files, ["pusht"], None)


def _entry_for(path: Path):
    _, files = load_manifest()
    data = path.read_bytes()
    return replace(files[0], size_bytes=len(data), sha256=hashlib.sha256(data).hexdigest(),
                   local_path=path.name)


def test_verify_file_accepts_match_and_rejects_corruption(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"roboforge" * 100)
    entry = _entry_for(p)
    assert verify_file(entry, p) == entry.sha256
    p.write_bytes(b"roboforgE" * 100)  # same size, different content
    with pytest.raises(IntegrityError):
        verify_file(entry, p)


def test_fetch_skips_verified_existing_and_refuses_bad_existing(tmp_path: Path) -> None:
    p = tmp_path / "x.bin"
    p.write_bytes(b"abc" * 10)
    entry = _entry_for(p)
    source, _ = load_manifest()
    rec = fetch_file(entry, source, root=tmp_path)
    assert rec["action"] == "verified_existing"
    p.write_bytes(b"abd" * 10)
    with pytest.raises(IntegrityError):
        fetch_file(entry, source, root=tmp_path)


def test_dry_run_does_not_download(tmp_path: Path) -> None:
    source, files = load_manifest()
    rec = fetch_file(files[0], source, root=tmp_path, dry_run=True)
    assert rec["action"] == "would_download"
    assert not files[0].destination(tmp_path).exists()


def test_summarize_synthetic(synthetic_hdf5: Path) -> None:
    s = summarize_hdf5(synthetic_hdf5)
    assert s["n_demos"] == 3 and s["action_dim"] == 7
    assert s["episode_length"] == {"min": 1, "max": 12, "mean": 6.0, "median": 5.0, "sum": 18}
    assert s["obs_keys"]["object"] == [10]
    assert s["mask_sizes"] == {"train": 2, "valid": 1}
    assert s["num_samples_matches_action_rows"]


def test_demo_keys_sorted_numerically(tmp_path: Path) -> None:
    import h5py

    with h5py.File(tmp_path / "k.hdf5", "w") as f:
        for i in (10, 2, 1):
            f.create_group(f"data/demo_{i}")
        assert demo_keys(f) == ["demo_1", "demo_2", "demo_10"]


def test_verify_file_rejects_empty_file(tmp_path: Path) -> None:
    good = tmp_path / "good.bin"
    good.write_bytes(b"payload")
    entry = _entry_for(good)
    empty = tmp_path / "empty.bin"
    empty.write_bytes(b"")
    with pytest.raises(IntegrityError, match="size 0"):
        verify_file(entry, empty)


def test_select_files_empty_inputs() -> None:
    _, files = load_manifest()
    assert select_files([], None, None) == []
    assert select_files(files, None, ["image"]) == []  # no image files exist in the manifest


def test_summarize_rejects_zero_demos(tmp_path: Path) -> None:
    import h5py

    path = tmp_path / "empty.hdf5"
    with h5py.File(path, "w") as f:
        f.create_group("data").attrs["env_args"] = "{}"
    with pytest.raises(ValueError, match="no demonstrations"):
        summarize_hdf5(path)


def test_summarize_single_demo(tmp_path: Path) -> None:
    from tests.conftest import _write_synthetic

    s = summarize_hdf5(_write_synthetic(tmp_path / "one.hdf5", [1]))
    assert s["n_demos"] == 1
    assert s["episode_length"]["min"] == s["episode_length"]["max"] == 1

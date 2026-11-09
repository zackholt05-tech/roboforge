"""RGB dataset file: atomic write, hash-verified reload, tamper and partial-read rules."""

from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import pytest

from roboforge.data.rgb_dataset import content_sha256, load_rgb_dataset, write_rgb_dataset

CAMS = ("agentview", "robot0_eye_in_hand")


def _frames(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    return {d: {c: rng.integers(0, 256, (n, 8, 8, 3), dtype=np.uint8) for c in CAMS}
            for d, n in (("demo_10", 3), ("demo_2", 4))}


def test_roundtrip_and_hash_order(tmp_path: Path) -> None:
    fr = _frames()
    states = {d: ["0" * 64] * len(v[CAMS[0]]) for d, v in fr.items()}
    p = tmp_path / "rgb.hdf5"
    digest = write_rgb_dataset(p, fr, states, CAMS, {"task": "lift", "size": 8})
    assert not p.with_suffix(".tmp").exists()
    back = load_rgb_dataset(p, expected_sha256=digest)
    assert list(back["frames"]) == ["demo_2", "demo_10"]  # numeric order
    for d in fr:
        for c in CAMS:
            assert np.array_equal(back["frames"][d][c], fr[d][c])
    assert back["cameras"] == CAMS and back["state_sha256"]["demo_2"] == ["0" * 64] * 4
    assert content_sha256(dict(reversed(list(fr.items()))), CAMS) == digest


def test_tampered_frame_is_refused(tmp_path: Path) -> None:
    fr = _frames()
    p = tmp_path / "rgb.hdf5"
    digest = write_rgb_dataset(p, fr, {d: ["0" * 64] * len(v[CAMS[0]]) for d, v in fr.items()},
                               CAMS, {"task": "lift", "size": 8})
    with h5py.File(p, "a") as f:
        f["data/demo_2/agentview"][0, 0, 0, 0] ^= 1
    with pytest.raises(ValueError, match="content hash"):
        load_rgb_dataset(p, expected_sha256=digest)
    with pytest.raises(ValueError, match="whole file"):
        load_rgb_dataset(p, demos=["demo_2"], expected_sha256=digest)


def test_rejects_non_uint8_frames() -> None:
    fr = _frames()
    fr["demo_2"]["agentview"] = fr["demo_2"]["agentview"].astype(np.float32)
    with pytest.raises(ValueError, match="uint8"):
        content_sha256(fr, CAMS)

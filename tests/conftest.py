"""Shared pytest fixtures.

* The MuJoCo GL backend is configured before any test imports mujoco/robosuite.
* ``synthetic_hdf5`` builds a tiny file with the robomimic layout so data-path
  tests run anywhere (CI, fresh clone) without downloading datasets.
* ``lift_lowdim_path`` / ``can_lowdim_path`` skip (with a reason) when the real
  datasets are not present under ``ROBOFORGE_DATA_ROOT``.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from roboforge.data.fetch import load_manifest  # noqa: E402

SYNTH_OBS = {"robot0_eef_pos": 3, "robot0_gripper_qpos": 2, "object": 10}


def _write_synthetic(path: Path, lengths: list[int], action_dim: int = 7) -> Path:
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as f:
        data = f.create_group("data")
        data.attrs["env_args"] = json.dumps({"env_name": "Synthetic", "env_version": "0",
                                             "type": 1, "env_kwargs": {}})
        data.attrs["total"] = int(sum(lengths))
        for i, n in enumerate(lengths):
            g = data.create_group(f"demo_{i}")
            g.attrs["num_samples"] = n
            g.attrs["model_file"] = "<mujoco/>"
            # Encode (episode, step) into values so tests can verify exact indexing.
            step = np.arange(n, dtype=np.float64)[:, None]
            for k, d in SYNTH_OBS.items():
                g.create_dataset(f"obs/{k}", data=np.repeat(1000.0 * i + step, d, axis=1))
            act = np.clip(rng.uniform(-1, 1, size=(n, action_dim)), -1, 1)
            act[:, 0] = 1000.0 * i + step[:, 0]  # first action dim encodes (episode, step)
            act[:, -1] = np.where(step[:, 0] > n / 2, 1.0, -1.0)
            g.create_dataset("actions", data=act)
            g.create_dataset("states", data=np.zeros((n, 5)))
            g.create_dataset("rewards", data=(step[:, 0] == n - 1).astype(float))
            g.create_dataset("dones", data=(step[:, 0] == n - 1).astype(int))
        mask = f.create_group("mask")
        names = [f"demo_{i}".encode() for i in range(len(lengths))]
        mask.create_dataset("train", data=names[:-1])
        mask.create_dataset("valid", data=names[-1:])
    return path


@pytest.fixture
def synthetic_hdf5(tmp_path: Path) -> Path:
    """Tiny robomimic-layout HDF5 with 3 episodes of lengths 5, 1 and 12."""
    return _write_synthetic(tmp_path / "synthetic.hdf5", [5, 1, 12])


def _dataset_path(task: str, kind: str) -> Path:
    _, files = load_manifest()
    entry = next(f for f in files if f.task == task and f.hdf5_type == kind)
    path = entry.destination()
    if not path.exists():
        pytest.skip(f"{path} not downloaded (run `make download`)")
    return path


@pytest.fixture(scope="session")
def lift_lowdim_path() -> Path:
    """Path to the real Lift PH low_dim file (skips if absent)."""
    return _dataset_path("lift", "low_dim")


@pytest.fixture(scope="session")
def can_lowdim_path() -> Path:
    """Path to the real Can PH low_dim file (skips if absent)."""
    return _dataset_path("can", "low_dim")


@pytest.fixture(scope="session")
def lift_processed_path() -> Path:
    """Path to the processed Lift low-dim file (skips if `make build-data` has not run).

    Session 12 fresh-clone finding: the RGB and point-cloud trainer tests read it but did not
    skip without it, so a fresh clone or CI without datasets failed them (DECISIONS D-098).
    """
    from roboforge.data.pipeline import load_task_artifacts
    from roboforge.data.processed import processed_path

    proc = load_task_artifacts("lift")["task_manifest"]["processed"]
    path = processed_path("lift", proc["converter_version"])
    if not path.is_file():
        pytest.skip(f"{path.name} not built (run `make download build-data`)")
    return path

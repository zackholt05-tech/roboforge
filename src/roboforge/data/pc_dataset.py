"""Point clouds of the Lift demonstrations, Level 2 study (schema ``roboforge.pc_dataset/1``).

The file holds **only point clouds**. Proprioception, actions and the split come from the
committed low-dim processed dataset, index for index, exactly as for the RGB frames (D-074,
D-079): cloud ``t`` of a demo is built from raw ``states[t]`` and pairs with ``obs[t]`` and
``actions[t]``.

Layout:
* ``/data/<demo>/points``: ``float32 [T, N, 3]``, world metres, one gzip-4 chunk per frame;
* ``/data/<demo>/n_crop``: ``int32 [T]``, points inside the box before sampling;
* ``/data/<demo>/state_sha256``: ``S64 [T]``;
* attributes:
  * ``schema``, ``task``;
  * ``spec`` (JSON :class:`roboforge.data.pointcloud.PointCloudSpec`) and ``spec_sha256``;
  * ``content_sha256``, ``raw_sha256`` and ``processed_content_sha256``;
  * ``render`` (JSON).

``content_sha256`` hashes, in numeric demo order, each demo name, then the shape and bytes of
its points. A loader given an expected hash recomputes it over everything it returns.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "roboforge.pc_dataset/1"


def _demo_order(names: list[str]) -> list[str]:
    return sorted(names, key=lambda d: int(d.split("_")[1]))


def content_sha256(points: dict[str, np.ndarray]) -> str:
    """Hash of every demo's clouds (numeric demo order)."""
    h = hashlib.sha256()
    for demo in _demo_order(list(points)):
        a = np.ascontiguousarray(points[demo])
        if a.dtype != np.float32 or a.ndim != 3 or a.shape[-1] != 3:
            raise ValueError(f"{demo}: expected float32 [T, N, 3], got {a.dtype} {a.shape}")
        h.update(demo.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def write_pc_dataset(path: Path, points: dict[str, np.ndarray], n_crop: dict[str, np.ndarray],
                     state_sha256: dict[str, list[str]], attrs: dict[str, Any]) -> str:
    """Write the file atomically, re-read it, check the content hash and return it."""
    import h5py

    digest = content_sha256(points)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        for k, v in {**attrs, "schema": SCHEMA, "content_sha256": digest}.items():
            f.attrs[k] = v if isinstance(v, (str, int, float)) else json.dumps(v)
        for demo in _demo_order(list(points)):
            a = points[demo]
            f.create_dataset(f"data/{demo}/points", data=a, chunks=(1, *a.shape[1:]),
                             compression="gzip", compression_opts=4)
            f.create_dataset(f"data/{demo}/n_crop", data=np.asarray(n_crop[demo], np.int32))
            f.create_dataset(f"data/{demo}/state_sha256",
                             data=np.array(state_sha256[demo], dtype="S64"))
    back = load_pc_dataset(tmp, expected_sha256=digest)
    if content_sha256(back["points"]) != digest:
        raise AssertionError("point-cloud dataset content changed on re-read")
    tmp.replace(path)
    return digest


def load_pc_dataset(path: Path, demos: list[str] | None = None,
                    expected_sha256: str | None = None) -> dict[str, Any]:
    """Clouds (all demos or ``demos``), crop counts, state hashes and attributes.

    With ``expected_sha256``, the whole file must be read: ``demos`` must then be None."""
    import h5py

    if expected_sha256 is not None and demos is not None:
        raise ValueError("verify the whole file: pass demos=None with expected_sha256")
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema") != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} file")
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        names = _demo_order(list(f["data"]))
        wanted = names if demos is None else demos
        missing = set(wanted) - set(names)
        if missing:
            raise KeyError(f"{path}: demos not present: {sorted(missing)}")
        points = {d: f["data"][d]["points"][()] for d in wanted}
        n_crop = {d: f["data"][d]["n_crop"][()] for d in wanted}
        states = {d: [s.decode() for s in f["data"][d]["state_sha256"][()]] for d in wanted}
    if expected_sha256 is not None:
        got = content_sha256(points)
        if got != expected_sha256 or attrs["content_sha256"] != expected_sha256:
            raise ValueError(f"{path}: content hash {got} != expected {expected_sha256}")
    return {"points": points, "n_crop": n_crop, "state_sha256": states, "attrs": attrs,
            "spec": json.loads(attrs["spec"])}


def pc_relpath(task: str, n_points: int, version: str = "v1") -> str:
    """Path of the installed file relative to ``data/`` (next to the low-dim file)."""
    return f"processed/robomimic_v1.5_ph/{task}/pc{n_points}_{version}.hdf5"


__all__ = ["SCHEMA", "content_sha256", "load_pc_dataset", "pc_relpath", "write_pc_dataset"]

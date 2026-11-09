"""RGB demonstration frames for the Session 8 RGB study (schema ``roboforge.rgb_dataset/1``).

The file holds **only images**. Proprioception, actions and the split come from the committed
low-dim processed dataset, which the frames align with index for index (DECISIONS D-074):
frame ``t`` of a demo is rendered from raw ``states[t]``, and it pairs with ``obs[t]`` and
``actions[t]`` of the processed file. The manifest of the processed file it pairs with is
pinned by its content hash.

Layout:

* ``/data/<demo>/<camera>``: ``uint8 [T, H, W, 3]``, one gzip-4 chunk per frame;
* ``/data/<demo>/state_sha256``: ``S64 [T]``, the SHA-256 of each rendered state;
* attributes:
  * ``schema``, ``task``, ``size``, ``cameras`` (JSON);
  * ``content_sha256``;
  * ``raw_sha256``, the raw file's hash;
  * ``processed_content_sha256``, the low-dim pairing;
  * ``render`` (JSON: backend, warm-up, renderer, D-074).

``content_sha256`` hashes, in numeric demo order, each demo name, then each camera's name,
shape and bytes. A loader that is given an expected hash recomputes it over everything it
returns, so a short or altered file cannot load silently.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

SCHEMA = "roboforge.rgb_dataset/1"


def _demo_order(names: list[str]) -> list[str]:
    return sorted(names, key=lambda d: int(d.split("_")[1]))


def content_sha256(frames: dict[str, dict[str, np.ndarray]], cameras: tuple[str, ...]) -> str:
    """Hash of every demo's frames (numeric demo order, cameras in the given order)."""
    h = hashlib.sha256()
    for demo in _demo_order(list(frames)):
        h.update(demo.encode())
        for cam in cameras:
            a = np.ascontiguousarray(frames[demo][cam])
            if a.dtype != np.uint8 or a.ndim != 4 or a.shape[-1] != 3:
                raise ValueError(f"{demo}/{cam}: expected uint8 [T, H, W, 3], got "
                                 f"{a.dtype} {a.shape}")
            h.update(cam.encode())
            h.update(str(a.shape).encode())
            h.update(a.tobytes())
    return h.hexdigest()


def write_rgb_dataset(path: Path, frames: dict[str, dict[str, np.ndarray]],
                      state_sha256: dict[str, list[str]], cameras: tuple[str, ...],
                      attrs: dict[str, Any]) -> str:
    """Write the file atomically, re-read it, check the content hash and return it."""
    import h5py

    digest = content_sha256(frames, cameras)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        for k, v in {**attrs, "schema": SCHEMA, "cameras": json.dumps(list(cameras)),
                     "content_sha256": digest}.items():
            f.attrs[k] = v if isinstance(v, (str, int, float)) else json.dumps(v)
        for demo in _demo_order(list(frames)):
            for cam in cameras:
                a = frames[demo][cam]
                f.create_dataset(f"data/{demo}/{cam}", data=a, chunks=(1, *a.shape[1:]),
                                 compression="gzip", compression_opts=4)
            f.create_dataset(f"data/{demo}/state_sha256",
                             data=np.array(state_sha256[demo], dtype="S64"))
    back = load_rgb_dataset(tmp, expected_sha256=digest)
    if content_sha256(back["frames"], cameras) != digest:
        raise AssertionError("RGB dataset content changed on re-read")
    tmp.replace(path)
    return digest


def load_rgb_dataset(path: Path, demos: list[str] | None = None,
                     expected_sha256: str | None = None) -> dict[str, Any]:
    """Frames (all demos or ``demos``), state hashes and attributes.

    With ``expected_sha256``, the whole file must be read: ``demos`` must then be None."""
    import h5py

    if expected_sha256 is not None and demos is not None:
        raise ValueError("verify the whole file: pass demos=None with expected_sha256")
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema") != SCHEMA:
            raise ValueError(f"{path}: not a {SCHEMA} file")
        attrs = {k: (v.decode() if isinstance(v, bytes) else v) for k, v in f.attrs.items()}
        cameras = tuple(json.loads(attrs["cameras"]))
        names = _demo_order(list(f["data"]))
        wanted = names if demos is None else demos
        missing = set(wanted) - set(names)
        if missing:
            raise KeyError(f"{path}: demos not present: {sorted(missing)}")
        frames = {d: {c: f["data"][d][c][()] for c in cameras} for d in wanted}
        states = {d: [s.decode() for s in f["data"][d]["state_sha256"][()]] for d in wanted}
    if expected_sha256 is not None:
        got = content_sha256(frames, cameras)
        if got != expected_sha256 or attrs["content_sha256"] != expected_sha256:
            raise ValueError(f"{path}: content hash {got} != expected {expected_sha256}")
    return {"frames": frames, "state_sha256": states, "attrs": attrs, "cameras": cameras}


def rgb_relpath(task: str, size: int, version: str = "v1") -> str:
    """Path of the installed file relative to ``data/`` (next to the low-dim file)."""
    return f"processed/robomimic_v1.5_ph/{task}/rgb{size}_{version}.hdf5"


__all__ = ["SCHEMA", "content_sha256", "load_rgb_dataset", "rgb_relpath", "write_rgb_dataset"]

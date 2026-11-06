"""Build and load the canonical processed low-dim dataset (DECISIONS D-021).

For each demonstration, the observation features are recomputed from the saved MuJoCo
states with :mod:`roboforge.data.features`. They are verified against the robomimic obs
blocks before anything is written:

* ``relation: equal`` features must equal the stored block exactly (``EQUAL_ATOL`` allows
  for the ≤1.1e-16 cross-platform float noise measured in Session 0, KI-013);
* ``corrected_same_step`` features (Can relative position, KI-003) must equal robosuite's
  formula applied to the *stored* same-step eef and object poses. Their difference from the
  stored (lagged) block is reported, and so is the stored block's match to the one-step-lag
  hypothesis.

Any failure raises; nothing is written. Arrays are stored as float32 (the precision models
consume) under ``$ROBOFORGE_DATA_ROOT/processed`` (gitignored). The content hash covers
demo names, array names, dtypes, shapes and bytes in a fixed order. It contains no
timestamps, so rebuilding from the same source on the same platform reproduces it.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from roboforge.data.features import FeatureExtractor
from roboforge.data.inspect import demo_keys
from roboforge.data.specs import FeatureSpec, ObservationSpec
from roboforge.paths import data_root

PROCESSED_SCHEMA = "roboforge.processed/1"
EQUAL_ATOL = 1e-12
ARRAY_ORDER = ("obs", "actions", "rewards", "dones")


def processed_relpath(task: str, converter_version: str) -> str:
    """Location of the processed file relative to the data root."""
    return f"processed/robomimic_v1.5_ph/{task}/{converter_version}.hdf5"


def processed_path(task: str, converter_version: str, root: Path | None = None) -> Path:
    """Absolute location of the processed file."""
    return (root or data_root()) / processed_relpath(task, converter_version)


def content_sha256(episodes: dict[str, dict[str, np.ndarray]]) -> str:
    """Hash episode arrays in numeric demo order and fixed array order (dtype + shape + bytes)."""
    h = hashlib.sha256()
    for demo in sorted(episodes, key=lambda k: int(k.split("_")[-1])):
        h.update(demo.encode())
        for name in ARRAY_ORDER:
            a = np.ascontiguousarray(episodes[demo][name])
            h.update(f"{name}|{a.dtype.str}|{a.shape}".encode())
            h.update(a.tobytes())
    return h.hexdigest()


@dataclass
class Verification:
    """Per-feature agreement between recomputed features and the stored dataset blocks."""

    max_abs_vs_dataset: dict[str, float] = field(default_factory=dict)
    corrected: dict[str, dict[str, float]] = field(default_factory=dict)
    final_state_success: dict[str, bool] = field(default_factory=dict)

    def update_max(self, key: str, value: float) -> None:
        """Track the running maximum for ``key``."""
        self.max_abs_vs_dataset[key] = max(self.max_abs_vs_dataset.get(key, 0.0), value)


def _block(obs: h5py.Group, f: FeatureSpec) -> np.ndarray:
    return obs[f.dataset_ref.key][()][:, f.dataset_ref.lo:f.dataset_ref.hi]


def stored_pose_blocks(spec: ObservationSpec, f: FeatureSpec, obs: h5py.Group) -> dict[str, Any]:
    """Stored eef/object pose blocks a ``pos_in_eef_frame`` feature depends on."""
    def find(kind: str, attr: str, name: str | None) -> np.ndarray:
        for g in spec.features:
            if g.kind == kind and getattr(g, attr) == name and g.relation == "equal":
                return _block(obs, g)
        raise ValueError(f"{f.name}: spec has no verified {kind} feature for {name!r}")

    return {"eef_pos": find("site_pos", "site", f.ref_site),
            "eef_quat": find("body_quat_xyzw", "body", f.ref_body),
            "obj_pos": find("body_pos", "body", f.body),
            "obj_quat": find("body_quat_xyzw", "body", f.body)}


def _check_corrected(spec: ObservationSpec, f: FeatureSpec, obs: h5py.Group, computed: np.ndarray,
                     ver: Verification) -> None:
    """Verify a corrected feature against robosuite's formula on stored same-step poses."""
    if f.kind != "pos_in_eef_frame":
        raise ValueError(f"{f.name}: no reference check defined for corrected kind {f.kind}")
    from roboforge.data.features import robosuite_transforms

    T, p = robosuite_transforms(), stored_pose_blocks(spec, f, obs)

    def rel(t_obj: int, t_eef: int) -> np.ndarray:
        world_in_eef = T.pose_inv(T.pose2mat((p["eef_pos"][t_eef], p["eef_quat"][t_eef])))
        obj = T.pose2mat((p["obj_pos"][t_obj], p["obj_quat"][t_obj]))
        return T.mat2pose(T.pose_in_A_to_pose_in_B(obj, world_in_eef))[0]

    n = len(computed)
    ref = np.stack([rel(t, t) for t in range(n)])
    stored = _block(obs, f)
    lag = np.stack([rel(t - 1, t) for t in range(1, n)]) if n > 1 else np.zeros((0, f.dim))
    stats = ver.corrected.setdefault(f.name, {"max_abs_vs_same_step_reference": 0.0,
                                               "stored_minus_corrected_max_t0": 0.0,
                                               "stored_minus_corrected_max_t_gt0": 0.0,
                                               "stored_vs_lag_hypothesis_max_t_gt0": 0.0})
    upd = {"max_abs_vs_same_step_reference": float(np.abs(computed - ref).max()),
           "stored_minus_corrected_max_t0": float(np.abs(stored[0] - computed[0]).max()),
           "stored_minus_corrected_max_t_gt0":
               float(np.abs(stored[1:] - computed[1:]).max()) if n > 1 else 0.0,
           "stored_vs_lag_hypothesis_max_t_gt0":
               float(np.abs(stored[1:] - lag).max()) if n > 1 else 0.0}
    for k, v in upd.items():
        stats[k] = max(stats[k], v)
    if upd["max_abs_vs_same_step_reference"] > EQUAL_ATOL:
        raise AssertionError(f"{f.name}: differs from same-step reference by "
                             f"{upd['max_abs_vs_same_step_reference']:.3e}")


def build_episodes(env: Any, low: h5py.File, spec: ObservationSpec,
                   demos: list[str] | None = None) -> tuple[dict[str, dict[str, np.ndarray]],
                                                            Verification]:
    """Recompute and verify features for ``demos`` (default: all, numeric order).

    Also records, per demo, whether the saved final state satisfies the task's own success
    predicate (``Verification.final_state_success``); the caller flags demos where it does not.
    """
    from roboforge.sim.state_replay import is_success, reset_to_state

    extractor, ver = FeatureExtractor(spec), Verification()
    episodes: dict[str, dict[str, np.ndarray]] = {}
    loaded_xml: str | None = None
    for demo in demos or demo_keys(low):
        g = low["data"][demo]
        xml, states = g.attrs["model_file"], g["states"][()]
        if xml != loaded_xml:
            reset_to_state(env, xml, states[0])
            loaded_xml = xml
        feats = extractor.bind(env.sim.model._model).from_states(states)
        env.sim.set_state_from_flattened(states[-1])
        env.sim.forward()
        ver.final_state_success[demo] = is_success(env)
        for f, (lo, hi) in zip(spec.features, spec.slices().values(), strict=True):
            if f.relation == "equal":
                err = float(np.abs(feats[:, lo:hi] - _block(g["obs"], f)).max())
                ver.update_max(f.name, err)
                if err > EQUAL_ATOL:
                    raise AssertionError(f"{demo}/{f.name}: differs from dataset by {err:.3e}")
            else:
                _check_corrected(spec, f, g["obs"], feats[:, lo:hi], ver)
        episodes[demo] = {"obs": feats.astype(np.float32),
                          "actions": g["actions"][()].astype(np.float32),
                          "rewards": g["rewards"][()].astype(np.float32),
                          "dones": g["dones"][()].astype(np.uint8)}
        if not all(len(a) == len(states) for a in episodes[demo].values()):
            raise AssertionError(f"{demo}: array lengths disagree")
    return episodes, ver


def write_processed(path: Path, episodes: dict[str, dict[str, np.ndarray]],
                    attrs: dict[str, str]) -> str:
    """Write episodes to HDF5 and return the content hash, re-read from disk and checked."""
    path.parent.mkdir(parents=True, exist_ok=True)
    digest = content_sha256(episodes)
    tmp = path.with_suffix(".tmp")
    with h5py.File(tmp, "w") as f:
        for k, v in {**attrs, "schema": PROCESSED_SCHEMA, "content_sha256": digest}.items():
            f.attrs[k] = v
        for demo, arrays in episodes.items():
            for name in ARRAY_ORDER:
                f.create_dataset(f"data/{demo}/{name}", data=arrays[name])
    reread = load_processed(tmp, expected_sha256=digest)
    if content_sha256(reread) != digest:
        raise AssertionError("processed file content changed on re-read")
    tmp.replace(path)
    return digest


def load_processed(path: Path, demos: list[str] | None = None,
                   expected_sha256: str | None = None) -> dict[str, dict[str, np.ndarray]]:
    """Load episodes (all, or ``demos``); with ``expected_sha256``, verify the full content."""
    if not path.exists():
        raise FileNotFoundError(f"{path} missing; run `make build-data`")
    with h5py.File(path, "r") as f:
        if f.attrs.get("schema") != PROCESSED_SCHEMA:
            raise ValueError(f"{path}: unexpected schema {f.attrs.get('schema')!r}")
        names = demo_keys(f)
        if expected_sha256 is not None:
            full = {d: {n: f["data"][d][n][()] for n in ARRAY_ORDER} for d in names}
            if content_sha256(full) != expected_sha256:
                raise ValueError(f"{path}: content hash != expected {expected_sha256}")
            if f.attrs["content_sha256"] != expected_sha256:
                raise ValueError(f"{path}: stored content_sha256 attribute mismatch")
        wanted = names if demos is None else demos
        missing = set(wanted) - set(names)
        if missing:
            raise KeyError(f"{path}: demos not present: {sorted(missing)}")
        return {d: {n: f["data"][d][n][()] for n in ARRAY_ORDER} for d in wanted}

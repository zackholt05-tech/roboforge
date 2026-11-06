"""Typed task, action and observation specifications read from the actual dataset files.

These specs are the single input/output contract shared by every model family (MASTER_SPEC
§4.2-4.3, §5.1). They are built from three sources and cross-checked:

* ``configs/data/<task>.yaml`` selects which features form the observation (a research
  choice, DECISIONS D-018);
* the robomimic HDF5 file supplies everything factual: observation block shapes, the action
  dimension and controller limits (``env_args``), the state dimension and the MJCF model in
  which every referenced MuJoCo name must exist;
* the simulator config supplies the evaluation horizon and task-object joint.

Each spec serializes to plain JSON and has a SHA-256 of its canonical form
(:func:`roboforge.provenance.sha256_json`). Downstream artifacts (processed data,
normalization stats, run manifests) record these hashes, so a model can never silently be
trained or evaluated against a different spec.
"""

from __future__ import annotations

import dataclasses
import json
from dataclasses import dataclass
from typing import Any

import h5py

from roboforge.data.inspect import demo_keys
from roboforge.provenance import sha256_json

# Output dimension of each feature kind (joint_qpos: one value per listed joint).
FEATURE_KIND_DIMS: dict[str, int | None] = {
    "site_pos": 3,
    "body_pos": 3,
    "body_quat_xyzw": 4,
    "joint_qpos": None,
    "pos_diff_world": 3,
    "pos_in_eef_frame": 3,
}
RELATIONS = ("equal", "corrected_same_step")
OSC_POSE_ACTION_NAMES = ("dpos_x", "dpos_y", "dpos_z", "drot_x", "drot_y", "drot_z")


@dataclass(frozen=True)
class DatasetRef:
    """The robomimic low_dim observation block ``obs[key][:, lo:hi]`` a feature corresponds to."""

    key: str
    lo: int
    hi: int


@dataclass(frozen=True)
class FeatureSpec:
    """One named block of the observation vector and how to compute it from MuJoCo state.

    ``relation`` states how the feature compares with the dataset block: ``equal`` (must
    reproduce it exactly) or ``corrected_same_step`` (the dataset block is known to be wrong,
    KI-003; the feature is robosuite's own formula evaluated on same-step poses).
    """

    name: str
    kind: str
    dim: int
    frame: str
    units: str
    privileged: bool
    relation: str
    dataset_ref: DatasetRef
    site: str | None = None
    body: str | None = None
    ref_site: str | None = None
    ref_body: str | None = None
    joints: tuple[str, ...] = ()

    def mujoco_names(self) -> list[str]:
        """Every MuJoCo element name this feature reads."""
        names = [self.site, self.body, self.ref_site, self.ref_body, *self.joints]
        return [n for n in names if n is not None]


@dataclass(frozen=True)
class ObservationSpec:
    """Ordered low-dimensional state observation used by all Level 1 policies."""

    task: str
    modality: str
    quat_convention: str
    features: tuple[FeatureSpec, ...]
    obs_horizon: int

    @property
    def dim(self) -> int:
        """Length of one observation vector."""
        return sum(f.dim for f in self.features)

    def slices(self) -> dict[str, tuple[int, int]]:
        """Map feature name to its ``[lo, hi)`` range in the observation vector."""
        out, lo = {}, 0
        for f in self.features:
            out[f.name] = (lo, lo + f.dim)
            lo += f.dim
        return out

    @property
    def privileged(self) -> bool:
        """True if any feature is exact simulator object state (MASTER_SPEC §3.1)."""
        return any(f.privileged for f in self.features)

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form, including derived ``dim`` and ``slices``."""
        d = dataclasses.asdict(self)
        d.update(dim=self.dim, slices={k: list(v) for k, v in self.slices().items()},
                 privileged=self.privileged)
        return d

    def sha256(self) -> str:
        """Hash of the canonical JSON form."""
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ObservationSpec:
        """Rebuild from :meth:`to_dict` output (derived keys are recomputed, then checked)."""
        feats = tuple(FeatureSpec(**{**f, "dataset_ref": DatasetRef(**f["dataset_ref"]),
                                     "joints": tuple(f["joints"])}) for f in d["features"])
        spec = cls(task=d["task"], modality=d["modality"], quat_convention=d["quat_convention"],
                   features=feats, obs_horizon=int(d["obs_horizon"]))
        if spec.dim != d["dim"] or {k: list(v) for k, v in spec.slices().items()} != d["slices"]:
            raise ValueError("observation spec dict has inconsistent dim/slices")
        return spec


@dataclass(frozen=True)
class ActionSpec:
    """Action convention read from the dataset's controller config and action arrays."""

    dim: int
    names: tuple[str, ...]
    low: tuple[float, ...]
    high: tuple[float, ...]
    controller: str
    control_delta: bool
    input_ref_frame: str
    output_low: tuple[float, ...]
    output_high: tuple[float, ...]
    gripper: str
    rotation_encoding: str

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return dataclasses.asdict(self)

    def sha256(self) -> str:
        """Hash of the canonical JSON form."""
        return sha256_json(self.to_dict())

    @classmethod
    def from_dict(cls, d: dict[str, Any]) -> ActionSpec:
        """Rebuild from :meth:`to_dict` output."""
        return cls(**{k: tuple(v) if isinstance(v, list) else v for k, v in d.items()})


@dataclass(frozen=True)
class TaskSpec:
    """Task-level facts: environment, timing, horizon, success signal and source files."""

    task: str
    env_name: str
    env_version: str
    robots: tuple[str, ...]
    control_freq_hz: float
    physics_dt_s: float
    eval_horizon_steps: int
    success_predicate: str
    task_object_body: str
    task_object_joints: tuple[str, ...]
    state_dim: int
    n_demos: int
    total_samples: int
    source_files: dict[str, str]

    def to_dict(self) -> dict[str, Any]:
        """JSON-compatible form."""
        return dataclasses.asdict(self)

    def sha256(self) -> str:
        """Hash of the canonical JSON form."""
        return sha256_json(self.to_dict())


SUCCESS_PREDICATES = {
    "Lift": "robosuite Lift._check_success: cube body z > table top z + 0.04 m",
    "PickPlaceCan": "robosuite PickPlace._check_success (single-object mode): "
                    "can inside its goal bin",
}


def load_data_config(task: str) -> dict[str, Any]:
    """Load ``configs/data/<task>.yaml``."""
    import yaml

    from roboforge.paths import configs_dir

    cfg = yaml.safe_load((configs_dir() / "data" / f"{task}.yaml").read_text())
    if cfg.get("task") != task:
        raise ValueError(f"configs/data/{task}.yaml declares task {cfg.get('task')!r}")
    return cfg


def _feature_from_config(raw: dict[str, Any]) -> FeatureSpec:
    kind = raw["kind"]
    if kind not in FEATURE_KIND_DIMS:
        raise ValueError(f"feature {raw['name']!r}: unknown kind {kind!r}")
    if raw["relation"] not in RELATIONS:
        raise ValueError(f"feature {raw['name']!r}: unknown relation {raw['relation']!r}")
    joints = tuple(raw.get("joints", ()))
    dim = FEATURE_KIND_DIMS[kind] or len(joints)
    if dim == 0:
        raise ValueError(f"feature {raw['name']!r}: joint_qpos needs a non-empty joints list")
    ref = DatasetRef(**raw["dataset_ref"])
    if ref.hi - ref.lo != dim:
        raise ValueError(f"feature {raw['name']!r}: dataset_ref width {ref.hi - ref.lo} != {dim}")
    return FeatureSpec(name=raw["name"], kind=kind, dim=dim, frame=raw["frame"],
                       units=raw["units"], privileged=bool(raw["privileged"]),
                       relation=raw["relation"], dataset_ref=ref, site=raw.get("site"),
                       body=raw.get("body"), ref_site=raw.get("ref_site"),
                       ref_body=raw.get("ref_body"), joints=joints)


def build_observation_spec(cfg: dict[str, Any], f: h5py.File) -> ObservationSpec:
    """Build the observation spec from ``cfg`` and validate it against the HDF5 file.

    Raises ``ValueError`` if a referenced obs key or block does not exist with the declared
    width, or if a MuJoCo element name is absent from the dataset's model XML.
    """
    feats = tuple(_feature_from_config(r) for r in cfg["features"])
    names = [x.name for x in feats]
    if len(set(names)) != len(names):
        raise ValueError(f"duplicate feature names in {names}")
    first = f["data"][demo_keys(f)[0]]
    xml = first.attrs["model_file"]
    for feat in feats:
        ref = feat.dataset_ref
        if ref.key not in first["obs"]:
            raise ValueError(f"{feat.name}: obs key {ref.key!r} not in dataset")
        width = int(first["obs"][ref.key].shape[1])
        if not 0 <= ref.lo < ref.hi <= width:
            raise ValueError(f"{feat.name}: block [{ref.lo},{ref.hi}) outside obs width {width}")
        for n in feat.mujoco_names():
            if f'name="{n}"' not in xml:
                raise ValueError(f"{feat.name}: MuJoCo element {n!r} not in dataset model XML")
    return ObservationSpec(task=cfg["task"], modality="low_dim_state", quat_convention="xyzw",
                           features=feats, obs_horizon=int(cfg["windows"]["obs_horizon"]))


def build_action_spec(f: h5py.File) -> ActionSpec:
    """Read the action convention from ``env_args`` and check it against the action arrays."""
    env_args = json.loads(f["data"].attrs["env_args"])
    arm = env_args["env_kwargs"]["controller_configs"]["body_parts"]["right"]
    if arm["type"] != "OSC_POSE" or arm["gripper"]["type"] != "GRIP":
        raise ValueError(f"unsupported controller {arm['type']}/{arm['gripper']['type']}")
    first = f["data"][demo_keys(f)[0]]
    dim = int(first["actions"].shape[1])
    if dim != len(OSC_POSE_ACTION_NAMES) + 1:
        raise ValueError(f"OSC_POSE + GRIP implies 7-D actions, dataset has {dim}")

    def per_dim(v: Any, n: int) -> tuple[float, ...]:
        return tuple(float(x) for x in (v if isinstance(v, list) else [v] * n))

    n_arm = len(OSC_POSE_ACTION_NAMES)
    return ActionSpec(
        dim=dim, names=(*OSC_POSE_ACTION_NAMES, "gripper"),
        low=per_dim(arm["input_min"], n_arm) + (-1.0,),
        high=per_dim(arm["input_max"], n_arm) + (1.0,),
        controller=f"{env_args['env_kwargs']['controller_configs']['type']}/OSC_POSE+GRIP",
        control_delta=bool(arm["control_delta"]), input_ref_frame=str(arm["input_ref_frame"]),
        output_low=per_dim(arm["output_min"], n_arm), output_high=per_dim(arm["output_max"], n_arm),
        gripper="-1 open, +1 close (robosuite PandaGripper.format_action)",
        rotation_encoding="axis-angle delta, scaled by output limits",
    )


def build_task_spec(task: str, f: h5py.File, sim_cfg: dict[str, Any],
                    source_files: dict[str, str], task_object_body: str,
                    physics_dt_s: float) -> TaskSpec:
    """Assemble the task spec from ``env_args``, the HDF5 contents and the simulator config.

    ``physics_dt_s`` and ``task_object_body`` are read from the pinned simulator built with
    the dataset's own ``env_args`` (they are not stored in the HDF5 file).
    """
    env_args = json.loads(f["data"].attrs["env_args"])
    kw = env_args["env_kwargs"]
    demos = demo_keys(f)
    total = int(sum(int(f["data"][d].attrs["num_samples"]) for d in demos))
    if total != int(f["data"].attrs["total"]):
        raise ValueError(f"sum of num_samples {total} != data.attrs['total']")
    return TaskSpec(
        task=task, env_name=env_args["env_name"], env_version=env_args["env_version"],
        robots=tuple(kw["robots"]), control_freq_hz=float(kw["control_freq"]),
        physics_dt_s=float(physics_dt_s), eval_horizon_steps=int(sim_cfg["eval_horizon_steps"]),
        success_predicate=SUCCESS_PREDICATES[env_args["env_name"]],
        task_object_body=task_object_body,
        task_object_joints=tuple(sim_cfg["task_object_joints"]),
        state_dim=int(f["data"][demos[0]]["states"].shape[1]), n_demos=len(demos),
        total_samples=total, source_files=dict(sorted(source_files.items())),
    )

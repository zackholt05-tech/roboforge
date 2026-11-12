"""X2 (Can MH) data: source manifest, committed artifacts, build, check and load (D-118 to D-121).

X2's demonstrations are robomimic's Can multi-human (MH) files, listed in the new manifest
``data/manifests/robomimic_v1.5_mh.json`` (sizes and SHA-256 from the pinned Hugging Face
revision) and fetched by the frozen :mod:`roboforge.data.fetch`.

The frozen Level 1 pipeline (:mod:`roboforge.data.pipeline`) is bound to the PH files and the
``*_ph`` paths, so X2 has its own artifacts, built here from the frozen building blocks with
the frozen Can observation config (``configs/data/can.yaml``, unchanged) and Can simulator
config:

* processed float32 file (gitignored): ``processed/robomimic_v1.5_mh/can/lowdim_v1.hdf5``;
* ``data/splits/can_mh.json``: robomimic's operator-stratified upstream masks (D-119);
* ``data/normalization/can_mh_lowdim_v1.json``: min-max fit on the training episodes only;
* ``data/manifests/tasks/can_mh.json``: schema ``roboforge.task_manifest/1`` (the frozen
  consumers read it unchanged), ``dataset_id`` ``robomimic_v1.5/can/mh``, every episode with
  its length, split, operator and quality tier, and the task spec with the X2 horizon (D-120).

:func:`build_x2` raises unless the X2 observation and action specs hash equal to the committed
Can PH specs (the validated Can contract, D-121). :func:`load_x2_artifacts` is the
integrity-checked loader that :mod:`roboforge.x2.binding` substitutes for the frozen
``load_task_artifacts`` inside X2 entry points.
"""

from __future__ import annotations

import json
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from roboforge.data.fetch import load_manifest, verify_file
from roboforge.data.inspect import demo_keys
from roboforge.data.normalization import Normalizer
from roboforge.data.pipeline import (
    NORMALIZATION_SCHEMA,
    TASK_MANIFEST_SCHEMA,
    UNCHECKED_KEYS,
    verify_self_hash,
)
from roboforge.data.processed import (
    EQUAL_ATOL,
    Verification,
    build_episodes,
    content_sha256,
    load_processed,
    write_processed,
)
from roboforge.data.sanity import episode_issues
from roboforge.data.specs import (
    build_action_spec,
    build_observation_spec,
    build_task_spec,
    load_data_config,
)
from roboforge.data.split import upstream_split, verify_split
from roboforge.paths import REPO_ROOT, data_root, repo_relative
from roboforge.provenance import package_versions, sha256_json, write_json
from roboforge.x2.inspect import operator_masks

X2_MANIFEST = "robomimic_v1.5_mh.json"
TASK = "can"
DATASET_TYPE = "mh"
DATASET_ID = "robomimic_v1.5/can/mh"
CONVERTER = "lowdim_v1"
EVAL_HORIZON_STEPS = 500  # D-120: robomimic@d309eae registers 500 for Can MH
DECISIONS = ("D-116", "D-118", "D-119", "D-120", "D-121", "D-122")
# D-122: flagged episodes accepted for training, with the exact flag reason they were accepted
# for. Any other flag, or a different reason, stops the build (D-119).
ACCEPTED_FLAGS = {"demo_81": {"reason": "final step has no success reward/done",
                              "decision": "D-122"}}


def unaccepted_flags(flagged: dict[str, str]) -> list[str]:
    """Flagged episodes whose flag is not accepted with exactly its reason (D-122)."""
    return sorted(d for d, why in flagged.items()
                  if ACCEPTED_FLAGS.get(d, {}).get("reason") != why)


def committed_paths() -> dict[str, Path]:
    """Repository locations of the committed X2 data artifacts."""
    return {"task_manifest": REPO_ROOT / "data" / "manifests" / "tasks" / "can_mh.json",
            "split": REPO_ROOT / "data" / "splits" / "can_mh.json",
            "normalization": REPO_ROOT / "data" / "normalization" / f"can_mh_{CONVERTER}.json"}


def processed_relpath() -> str:
    """Path of the processed X2 file relative to the data root."""
    return f"processed/robomimic_v1.5_mh/{TASK}/{CONVERTER}.hdf5"


def processed_path(root: Path | None = None) -> Path:
    """Absolute path of the processed X2 file (gitignored)."""
    return (root or data_root()) / processed_relpath()


def _self_hashed(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": sha256_json(body)}


def source_section() -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    """The MH source block, local paths and verified SHA-256 of both files."""
    source, files = load_manifest(X2_MANIFEST)
    entries = {f.hdf5_type: f for f in files
               if f.task == TASK and f.dataset_type == DATASET_TYPE}
    if set(entries) != {"low_dim", "raw"}:
        raise ValueError(f"{X2_MANIFEST}: expected can/mh low_dim and raw, got {sorted(entries)}")
    paths = {k: e.destination() for k, e in entries.items()}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{repo_relative(p)} missing ({k}); run `make x2-download`")
    shas = {k: verify_file(e, paths[k]) for k, e in entries.items()}
    section = {"manifest": f"data/manifests/{X2_MANIFEST}",
               "hf_repo_id": source["hf_repo_id"], "hf_revision": source["hf_revision"],
               "license": source["license"], "citation": source["citation"],
               "acquired_date": source["acquired_date"],
               "files": {k: {"url": e.url(source), "path_in_repo": e.path_in_repo,
                             "size_bytes": e.size_bytes, "sha256": shas[k]}
                         for k, e in sorted(entries.items())}}
    return section, paths, shas


def episode_operators(low: h5py.File) -> dict[str, dict[str, str]]:
    """Operator mask and quality tier of every demo, from the masks only (D-119).

    Raises unless the operator masks partition the demos."""
    from roboforge.x2.inspect import masks

    ops = operator_masks(masks(low))
    out: dict[str, dict[str, str]] = {}
    for op, ids in ops.items():
        for d in ids:
            if d in out:
                raise ValueError(f"{d} is in operator masks {out[d]['operator']} and {op}")
            out[d] = {"operator": op, "quality": op.split("_operator_")[0]}
    missing = sorted(set(demo_keys(low)) - set(out))
    if missing:
        raise ValueError(f"demos in no operator mask: {missing[:5]}…")
    return out


def committed_ph_spec_hashes() -> dict[str, str]:
    """Observation and action spec hashes of the committed Can PH manifest (frozen loader)."""
    from roboforge.data import pipeline

    m = pipeline.committed_paths(TASK, CONVERTER)["task_manifest"]
    doc = json.loads(m.read_text())
    verify_self_hash(doc, ignore=UNCHECKED_KEYS)
    return {k: doc[k] for k in ("observation_spec_sha256", "action_spec_sha256")}


@dataclass
class X2Artifacts:
    """Everything one :func:`build_x2` call produces (nothing is written until :func:`write_x2`)."""

    episodes: dict[str, dict[str, np.ndarray]]
    split: dict[str, Any]
    normalization: dict[str, Any]
    task_manifest: dict[str, Any]
    verification: Verification
    issues: dict[str, list[str]]


def build_x2() -> X2Artifacts:
    """Run the X2 data pipeline in memory, raising on any verification failure.

    Mirrors the frozen ``roboforge.data.pipeline.build_task`` with the MH source, the
    operator record, the X2 horizon and the X2 paths. Every frozen check runs: features
    recomputed from the saved states equal the stored blocks (KI-003 block corrected), the
    action spec equals the simulator's, the saved final state satisfies the success predicate,
    and the per-episode sanity checks."""
    import mujoco

    from roboforge.sim.factory import load_sim_config, make_env

    cfg, sim_cfg = load_data_config(TASK), load_sim_config(TASK)
    if cfg["converter_version"] != CONVERTER:
        raise ValueError(f"configs/data/{TASK}.yaml converter {cfg['converter_version']!r}")
    source, paths, shas = source_section()
    env = make_env(sim_cfg)
    try:
        with h5py.File(paths["low_dim"], "r") as low:
            obs_spec, act_spec = build_observation_spec(cfg, low), build_action_spec(low)
            ph = committed_ph_spec_hashes()
            mine = {"observation_spec_sha256": obs_spec.sha256(),
                    "action_spec_sha256": act_spec.sha256()}
            if mine != ph:
                raise AssertionError(f"X2 specs differ from the committed Can PH specs: "
                                     f"{mine} != {ph}")
            lo_b, hi_b = env.action_spec
            if not (np.array_equal(lo_b, act_spec.low) and np.array_equal(hi_b, act_spec.high)):
                raise AssertionError(f"env.action_spec {lo_b}/{hi_b} != dataset action spec")
            episodes, ver = build_episodes(env, low, obs_spec)
            m = env.sim.model._model
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, sim_cfg["task_object_joints"][0])
            obj_body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[jid]))
            task_spec = build_task_spec(TASK, low, {**sim_cfg,
                                                    "eval_horizon_steps": EVAL_HORIZON_STEPS},
                                        shas, obj_body, float(env.model_timestep))
            if float(env.control_freq) != task_spec.control_freq_hz:
                raise AssertionError("env control_freq != dataset control_freq")
            issues = {d: episode_issues(low["data"][d], 1.0 / task_spec.control_freq_hz,
                                        np.array(act_spec.low), np.array(act_spec.high))
                      for d in demo_keys(low)}
            for d, ok in ver.final_state_success.items():
                if not ok:
                    issues[d].append("saved final state fails the task success predicate")
            issues = {d: v for d, v in issues.items() if v}
            split = upstream_split(low, TASK, shas["low_dim"], cfg["split"]["train_mask"],
                                   cfg["split"]["valid_mask"],
                                   flagged={d: "; ".join(v) for d, v in issues.items()})
            split = _self_hashed({**{k: v for k, v in split.items() if k != "content_sha256"},
                                  "dataset_id": DATASET_ID, "decision": "D-119",
                                  "accepted_flags": {d: ACCEPTED_FLAGS[d]
                                                     for d in sorted(split["flagged"])
                                                     if d in ACCEPTED_FLAGS},
                                  "method": split["method"] + " (operator-stratified: 45 train "
                                            "/ 5 valid per operator, D-119)"})
            lengths = {d: int(low["data"][d].attrs["num_samples"]) for d in demo_keys(low)}
            operators = episode_operators(low)
    finally:
        env.close()

    digest = content_sha256(episodes)
    train = split["splits"]["train"]
    ncfg = cfg["normalization"]
    obs_norm = Normalizer.fit([episodes[d]["obs"] for d in train], **ncfg["obs"])
    act_norm = Normalizer.fit([episodes[d]["actions"] for d in train], **ncfg["action"])
    normalization = _self_hashed({
        "schema": NORMALIZATION_SCHEMA, "task": TASK, "dataset_id": DATASET_ID,
        "converter_version": CONVERTER,
        "fit_on": "train split, per-step arrays of every training episode (no windows)",
        "split_content_sha256": split["content_sha256"],
        "observation_spec_sha256": obs_spec.sha256(), "processed_content_sha256": digest,
        "obs": obs_norm.to_dict(), "action": act_norm.to_dict()})
    split_of = {d: s for s, ids in split["splits"].items() for d in ids}
    per_op = {op: {s: sum(1 for d, o in operators.items() if o["operator"] == op
                          and split_of[d] == s) for s in ("train", "valid")}
              for op in sorted({o["operator"] for o in operators.values()})}
    cp = committed_paths()
    manifest = _self_hashed({
        "schema": TASK_MANIFEST_SCHEMA, "task": TASK, "dataset_id": DATASET_ID,
        "study": "X2 (separate study; DECISIONS D-116 to D-121)",
        "source": source,
        "task_spec": task_spec.to_dict(), "task_spec_sha256": task_spec.sha256(),
        "action_spec": act_spec.to_dict(), "action_spec_sha256": act_spec.sha256(),
        "observation_spec": obs_spec.to_dict(), "observation_spec_sha256": obs_spec.sha256(),
        "observation_config": f"configs/data/{TASK}.yaml (the frozen Can config, unchanged)",
        "windows": {**cfg["windows"],
                    "status": "default horizons, overridable per run; every run manifest "
                              "records the T_o/T_p it used",
                    "obs_padding": "repeat first observation; obs_valid=False",
                    "action_padding": "repeat last action; valid_mask=False (masked from loss)"},
        "episodes": [{"id": d, "length": n, "split": split_of[d], "flagged": d in issues,
                      **operators[d]} for d, n in lengths.items()],
        "operator_metadata": "six robomimic operator masks {better,okay,worse}_operator_{1,2}, "
                             "50 demos each, partitioning the demos; the raw teleoperation "
                             "arrays carry no identity. Every family trains on all six "
                             "operators: every X2 claim is within-operator (D-119)",
        "operators": per_op,
        "split": {"path": repo_relative(cp["split"]), "content_sha256": split["content_sha256"]},
        "normalization": {"path": repo_relative(cp["normalization"]),
                          "content_sha256": normalization["content_sha256"]},
        "processed": {"relpath": processed_relpath(), "converter_version": CONVERTER,
                      "content_sha256": digest, "dtype": "float32",
                      "arrays": {"obs": [None, obs_spec.dim], "actions": [None, act_spec.dim],
                                 "rewards": [None], "dones": [None]}},
        "verification": {"relation_equal_atol": EQUAL_ATOL,
                         "all_equal_features_within_atol": True,
                         "max_abs_vs_dataset": dict(sorted(ver.max_abs_vs_dataset.items())),
                         "corrected_features": sorted(ver.corrected),
                         "final_state_success": sum(ver.final_state_success.values()),
                         "spec_hashes_equal_can_ph": True},
    })
    manifest["built_with"] = {"system": platform.system(), "machine": platform.machine(),
                              "packages": package_versions(("mujoco", "robosuite", "numpy",
                                                            "h5py", "robomimic"))}
    return X2Artifacts(episodes, split, normalization, manifest, ver, issues)


def check_x2(art: X2Artifacts) -> list[str]:
    """Compare freshly built artifacts with the committed files; return mismatch messages."""
    problems: list[str] = []
    fresh = {"split": art.split, "normalization": art.normalization,
             "task_manifest": art.task_manifest}
    for name, path in committed_paths().items():
        if not path.exists():
            problems.append(f"{repo_relative(path)} missing (run `make x2-data`)")
            continue
        committed = json.loads(path.read_text())
        rebuilt = json.loads(json.dumps(fresh[name]))
        a = {k: v for k, v in committed.items() if k not in UNCHECKED_KEYS}
        b = {k: v for k, v in rebuilt.items() if k not in UNCHECKED_KEYS}
        if a != b:
            keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            problems.append(f"{repo_relative(path)} differs from rebuild in keys {keys}")
    return problems


def write_x2(art: X2Artifacts, *, update_committed: bool = False,
             root: Path | None = None) -> dict[str, str]:
    """Write the processed file, and the committed JSON artifacts only if asked to.

    Without ``update_committed`` the committed files must already equal the rebuild."""
    bad = unaccepted_flags(art.split["flagged"])
    if bad:
        raise ValueError(f"flagged episodes {bad} are not accepted: a dated decision is needed "
                         f"before X2 data are written (D-119, D-122)")
    if not update_committed:
        problems = check_x2(art)
        if problems:
            raise ValueError(f"committed X2 artifacts differ from this rebuild: {problems}")
    m = art.task_manifest
    attrs = {"task": TASK, "dataset_id": DATASET_ID, "converter_version": CONVERTER,
             "observation_spec_sha256": m["observation_spec_sha256"],
             "action_spec_sha256": m["action_spec_sha256"],
             "source_low_dim_sha256": m["source"]["files"]["low_dim"]["sha256"]}
    out = processed_path(root)
    if write_processed(out, art.episodes, attrs) != m["processed"]["content_sha256"]:
        raise AssertionError("written processed content hash differs from manifest")
    written = {"processed": processed_relpath()}
    if update_committed:
        for name, path in committed_paths().items():
            write_json(path, {"split": art.split, "normalization": art.normalization,
                              "task_manifest": art.task_manifest}[name])
            written[name] = repo_relative(path)
    return written


def load_x2_artifacts(task: str = TASK, converter_version: str = CONVERTER) -> dict[str, Any]:
    """Load and integrity-check the committed X2 manifest, split and normalization.

    Same return shape as the frozen ``load_task_artifacts``. Raises for any task other than
    Can, for a corrupted or inconsistent artifact, for a dataset other than Can MH, and unless
    the X2 observation and action spec hashes equal the committed Can PH ones (D-121)."""
    if (task, converter_version) != (TASK, CONVERTER):
        raise ValueError(f"X2 data exist for {TASK}/{CONVERTER} only; "
                         f"got {task}/{converter_version}")
    docs = {k: json.loads(p.read_text()) for k, p in committed_paths().items()}
    verify_self_hash(docs["task_manifest"], ignore=UNCHECKED_KEYS)
    verify_self_hash(docs["normalization"])
    verify_split(docs["split"])
    m = docs["task_manifest"]
    if m["dataset_id"] != DATASET_ID or docs["normalization"].get("dataset_id") != DATASET_ID:
        raise ValueError(f"X2 artifacts are not {DATASET_ID}")
    if m["split"]["content_sha256"] != docs["split"]["content_sha256"]:
        raise ValueError("X2 task manifest references a different split")
    if m["normalization"]["content_sha256"] != docs["normalization"]["content_sha256"]:
        raise ValueError("X2 task manifest references different normalization stats")
    want = committed_ph_spec_hashes()
    have = {k: m[k] for k in want}
    if have != want:
        raise ValueError(f"X2 specs {have} != committed Can PH specs {want}")
    return docs


def load_x2_episodes(docs: dict[str, Any] | None = None) -> dict[str, dict[str, np.ndarray]]:
    """The processed X2 episodes, content-hash checked against the X2 task manifest."""
    docs = docs or load_x2_artifacts()
    proc = docs["task_manifest"]["processed"]
    return load_processed(processed_path(), expected_sha256=proc["content_sha256"])

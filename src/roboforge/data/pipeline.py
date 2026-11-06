"""Orchestrate the Session 1 data pipeline for one task and manage its committed artifacts.

``build_task`` produces, from the hash-verified robomimic files only:

* the processed float32 dataset (gitignored; content-hashed);
* ``data/splits/<task>_ph.json``: the episode-level split (D-019);
* ``data/normalization/<task>_ph_<converter>.json``: train-only statistics (D-020);
* ``data/manifests/tasks/<task>_ph.json``: the immutable task manifest (MASTER_SPEC §4.4)
  with the source, all three specs and their hashes, every episode ID and length, and
  references by hash to the split, normalization and processed content.

None of the committed JSON files contains a timestamp or an absolute path. ``check_task``
rebuilds a task and compares the result with the committed files. ``built_with`` (platform
and package versions) is reported but excluded from that comparison, because the content
itself is the contract (KI-013).
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
from roboforge.data.processed import (
    EQUAL_ATOL,
    Verification,
    build_episodes,
    content_sha256,
    processed_path,
    processed_relpath,
    write_processed,
)
from roboforge.data.sanity import episode_issues
from roboforge.data.specs import (
    ActionSpec,
    ObservationSpec,
    build_action_spec,
    build_observation_spec,
    build_task_spec,
    load_data_config,
)
from roboforge.data.split import upstream_split, verify_split
from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import package_versions, sha256_json, write_json

TASK_MANIFEST_SCHEMA = "roboforge.task_manifest/1"
NORMALIZATION_SCHEMA = "roboforge.normalization/1"
UNCHECKED_KEYS = ("built_with",)


def committed_paths(task: str, converter_version: str) -> dict[str, Path]:
    """Repository locations of a task's committed data-pipeline artifacts."""
    return {"task_manifest": REPO_ROOT / "data" / "manifests" / "tasks" / f"{task}_ph.json",
            "split": REPO_ROOT / "data" / "splits" / f"{task}_ph.json",
            "normalization": REPO_ROOT / "data" / "normalization"
            / f"{task}_ph_{converter_version}.json"}


def _self_hashed(body: dict[str, Any]) -> dict[str, Any]:
    return {**body, "content_sha256": sha256_json(body)}


def verify_self_hash(doc: dict[str, Any], ignore: tuple[str, ...] = ()) -> None:
    """Raise if ``doc['content_sha256']`` does not match its body (minus ``ignore`` keys)."""
    body = {k: v for k, v in doc.items() if k != "content_sha256" and k not in ignore}
    if sha256_json(body) != doc.get("content_sha256"):
        raise ValueError(f"{doc.get('schema')}: content hash mismatch (edited or corrupted)")


@dataclass
class TaskArtifacts:
    """Everything one ``build_task`` call produces (nothing is written until ``write_task``)."""

    task: str
    converter_version: str
    episodes: dict[str, dict[str, np.ndarray]]
    split: dict[str, Any]
    normalization: dict[str, Any]
    task_manifest: dict[str, Any]
    verification: Verification
    issues: dict[str, list[str]]


def _source_section(task: str) -> tuple[dict[str, Any], dict[str, Path], dict[str, str]]:
    source, files = load_manifest()
    entries = {f.hdf5_type: f for f in files if f.task == task and f.dataset_type == "ph"}
    paths = {k: e.destination() for k, e in entries.items()}
    for k, p in paths.items():
        if not p.exists():
            raise FileNotFoundError(f"{p} missing ({task}/{k}); run `make download`")
    shas = {k: verify_file(e, paths[k]) for k, e in entries.items()}
    section = {"hf_repo_id": source["hf_repo_id"], "hf_revision": source["hf_revision"],
               "license": source["license"], "citation": source["citation"],
               "acquired_date": source["acquired_date"],
               "files": {k: {"url": e.url(source), "path_in_repo": e.path_in_repo,
                             "size_bytes": e.size_bytes, "sha256": shas[k]}
                         for k, e in sorted(entries.items())}}
    return section, paths, shas


def build_task(task: str) -> TaskArtifacts:
    """Run the full pipeline for ``task`` in memory, raising on any verification failure."""
    from roboforge.sim.factory import load_sim_config, make_env

    cfg, sim_cfg = load_data_config(task), load_sim_config(task)
    conv = cfg["converter_version"]
    source, paths, shas = _source_section(task)
    env = make_env(sim_cfg)
    try:
        with h5py.File(paths["low_dim"], "r") as low:
            obs_spec, act_spec = build_observation_spec(cfg, low), build_action_spec(low)
            lo_b, hi_b = env.action_spec
            if not (np.array_equal(lo_b, act_spec.low) and np.array_equal(hi_b, act_spec.high)):
                raise AssertionError(f"env.action_spec {lo_b}/{hi_b} != dataset action spec")
            episodes, ver = build_episodes(env, low, obs_spec)
            m = env.sim.model._model
            import mujoco

            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, sim_cfg["task_object_joints"][0])
            obj_body = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[jid]))
            task_spec = build_task_spec(task, low, sim_cfg, shas, obj_body,
                                        float(env.model_timestep))
            if float(env.control_freq) != task_spec.control_freq_hz:
                raise AssertionError("env control_freq != dataset control_freq")
            issues = {d: episode_issues(low["data"][d], 1.0 / task_spec.control_freq_hz,
                                        np.array(act_spec.low), np.array(act_spec.high))
                      for d in demo_keys(low)}
            for d, ok in ver.final_state_success.items():
                if not ok:
                    issues[d].append("saved final state fails the task success predicate")
            issues = {d: v for d, v in issues.items() if v}
            split = upstream_split(low, task, shas["low_dim"], cfg["split"]["train_mask"],
                                   cfg["split"]["valid_mask"],
                                   flagged={d: "; ".join(v) for d, v in issues.items()})
            lengths = {d: int(low["data"][d].attrs["num_samples"]) for d in demo_keys(low)}
    finally:
        env.close()

    digest = content_sha256(episodes)
    train = split["splits"]["train"]
    ncfg = cfg["normalization"]
    obs_norm = Normalizer.fit([episodes[d]["obs"] for d in train], **ncfg["obs"])
    act_norm = Normalizer.fit([episodes[d]["actions"] for d in train], **ncfg["action"])
    normalization = _self_hashed({
        "schema": NORMALIZATION_SCHEMA, "task": task, "converter_version": conv,
        "fit_on": "train split, per-step arrays of every training episode (no windows)",
        "split_content_sha256": split["content_sha256"],
        "observation_spec_sha256": obs_spec.sha256(), "processed_content_sha256": digest,
        "obs": obs_norm.to_dict(), "action": act_norm.to_dict()})
    split_of = {d: s for s, ids in split["splits"].items() for d in ids}
    paths_c = committed_paths(task, conv)
    manifest = _self_hashed({
        "schema": TASK_MANIFEST_SCHEMA, "task": task, "dataset_id": f"robomimic_v1.5/{task}/ph",
        "source": source,
        "task_spec": task_spec.to_dict(), "task_spec_sha256": task_spec.sha256(),
        "action_spec": act_spec.to_dict(), "action_spec_sha256": act_spec.sha256(),
        "observation_spec": obs_spec.to_dict(), "observation_spec_sha256": obs_spec.sha256(),
        "windows": {**cfg["windows"],
                    "status": "default horizons, overridable per run; every run manifest "
                              "records the T_o/T_p it used",
                    "obs_padding": "repeat first observation; obs_valid=False",
                    "action_padding": "repeat last action; valid_mask=False (masked from loss)"},
        "episodes": [{"id": d, "length": n, "split": split_of[d], "flagged": d in issues}
                     for d, n in lengths.items()],
        "operator_metadata": "none: raw user_info has one teleoperation channel '0' in every "
                             "demo and no operator identity; no held-out-operator claims",
        "split": {"path": repo_relative(paths_c["split"]),
                  "content_sha256": split["content_sha256"]},
        "normalization": {"path": repo_relative(paths_c["normalization"]),
                          "content_sha256": normalization["content_sha256"]},
        "processed": {"relpath": processed_relpath(task, conv), "converter_version": conv,
                      "content_sha256": digest, "dtype": "float32",
                      "arrays": {"obs": [None, obs_spec.dim], "actions": [None, act_spec.dim],
                                 "rewards": [None], "dones": [None]}},
        "verification": {"relation_equal_atol": EQUAL_ATOL,
                         "all_equal_features_within_atol": True,
                         "corrected_features": sorted(ver.corrected)},
    })
    manifest["built_with"] = {"system": platform.system(), "machine": platform.machine(),
                              "packages": package_versions(("mujoco", "robosuite", "numpy",
                                                            "h5py", "robomimic"))}
    return TaskArtifacts(task, conv, episodes, split, normalization, manifest, ver, issues)


def write_task(art: TaskArtifacts, processed_root: Path | None = None,
               update_committed: bool = False) -> dict[str, str]:
    """Write the processed file, and the committed JSON artifacts only if asked to.

    By default the committed split, normalization and task manifest must already equal the
    rebuild (:func:`check_task`). Otherwise this raises before writing anything, so a
    rebuild on a different platform or code revision can never silently overwrite the
    committed contract. ``update_committed=True`` is for an intentional, recorded change of
    the data contract.
    """
    if not update_committed:
        problems = check_task(art)
        if problems:
            raise ValueError(f"{art.task}: committed artifacts differ from this rebuild; "
                             f"refusing to write (use --update-committed only for an "
                             f"intentional, recorded change): {problems}")
    out = processed_path(art.task, art.converter_version, processed_root)
    m = art.task_manifest
    attrs = {"task": art.task, "converter_version": art.converter_version,
             "observation_spec_sha256": m["observation_spec_sha256"],
             "action_spec_sha256": m["action_spec_sha256"],
             "source_low_dim_sha256": m["source"]["files"]["low_dim"]["sha256"]}
    if write_processed(out, art.episodes, attrs) != m["processed"]["content_sha256"]:
        raise AssertionError("written processed content hash differs from manifest")
    if not update_committed:
        return {"processed": str(out)}
    paths = committed_paths(art.task, art.converter_version)
    write_json(paths["split"], art.split)
    write_json(paths["normalization"], art.normalization)
    write_json(paths["task_manifest"], art.task_manifest)
    return {"processed": str(out), **{k: repo_relative(p) for k, p in paths.items()}}


def check_task(art: TaskArtifacts) -> list[str]:
    """Compare freshly built artifacts with the committed files; return mismatch messages."""
    problems: list[str] = []
    fresh = {"split": art.split, "normalization": art.normalization,
             "task_manifest": art.task_manifest}
    for name, path in committed_paths(art.task, art.converter_version).items():
        if not path.exists():
            problems.append(f"{repo_relative(path)} missing (run `make build-data`)")
            continue
        committed = json.loads(path.read_text())
        rebuilt = json.loads(json.dumps(fresh[name]))  # tuples -> lists, as stored on disk
        a = {k: v for k, v in committed.items() if k not in UNCHECKED_KEYS}
        b = {k: v for k, v in rebuilt.items() if k not in UNCHECKED_KEYS}
        if a != b:
            keys = sorted(k for k in set(a) | set(b) if a.get(k) != b.get(k))
            problems.append(f"{repo_relative(path)} differs from rebuild in keys {keys}")
    return problems


def load_specs(task: str, converter_version: str = "lowdim_v1") -> tuple[ObservationSpec,
                                                                         ActionSpec]:
    """Observation and action specs from the committed task manifest, hash-checked.

    This is how evaluation code (Session 2+) obtains the exact training-time contract
    without opening any dataset file.
    """
    m = load_task_artifacts(task, converter_version)["task_manifest"]
    obs = ObservationSpec.from_dict(m["observation_spec"])
    act = ActionSpec.from_dict(m["action_spec"])
    if obs.sha256() != m["observation_spec_sha256"] or act.sha256() != m["action_spec_sha256"]:
        raise ValueError(f"{task}: committed spec does not match its recorded hash")
    return obs, act


def load_task_artifacts(task: str, converter_version: str = "lowdim_v1") -> dict[str, Any]:
    """Load and integrity-check the committed manifest, split and normalization for ``task``."""
    paths = committed_paths(task, converter_version)
    docs = {k: json.loads(p.read_text()) for k, p in paths.items()}
    verify_self_hash(docs["task_manifest"], ignore=UNCHECKED_KEYS)
    verify_self_hash(docs["normalization"])
    verify_split(docs["split"])
    m = docs["task_manifest"]
    if m["split"]["content_sha256"] != docs["split"]["content_sha256"]:
        raise ValueError(f"{task}: task manifest references a different split")
    if m["normalization"]["content_sha256"] != docs["normalization"]["content_sha256"]:
        raise ValueError(f"{task}: task manifest references different normalization stats")
    return docs

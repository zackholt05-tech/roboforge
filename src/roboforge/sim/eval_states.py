"""Evaluation initial-state suites: generation, storage, integrity checks and the test lock.

Two suites per task (``configs/eval/suites.yaml``, RESEARCH_PROTOCOL §5, DECISIONS D-025):

* ``validation`` (partition ``eval_validation``): seeds ``100000 + i``. Development and
  checkpoint selection.
* ``final_test`` (partition ``eval_final_test``): seeds ``900000 + i``. Generated and hashed
  now, but :func:`load_suite` refuses to return these states, and
  :func:`check_seed_allowed` refuses their seed block, until ``docs/FINAL_TEST_FREEZE.md``
  is committed unmodified and names the suite's ``content_sha256`` (MASTER_SPEC §3.2).

Files per task under ``data/eval_states/<task>/``:

* ``<suite>.json``: the open manifest. It holds the state IDs, seeds and hashes but no state
  values, and is self-hashed (``content_sha256``).
* ``<suite>.data.jsonl.xz``: the serialized snapshots. Each distinct XML is stored once;
  states refer to it by hash. The manifest records the SHA-256 of the *decompressed* bytes, so
  integrity does not depend on the compressor build.
* ``partitions.json``: the index of every partition for the task (demo train/valid split
  plus both suites), the "test metadata" required by the Session 2 acceptance criteria.

``check_task`` regenerates every state in memory and compares it with the committed suite,
like ``make check-data``: bitwise on the platform recorded in ``generated_on``, and within
``CROSS_PLATFORM_STATE_ATOL`` (identical XML) elsewhere. For the locked suite it compares
values inside this module and returns only the maximum difference.
"""

from __future__ import annotations

import json
import lzma
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from roboforge.paths import REPO_ROOT, configs_dir, repo_relative
from roboforge.provenance import evaluation_platform, package_versions, sha256_json, write_json
from roboforge.sim.snapshot import InitialState, generate, restore, sha256_text

SUITE_SCHEMA = "roboforge.eval_suite/1"
# Regenerating a suite on a platform other than the one that generated it reproduces every
# XML exactly but the state values only to ~1 ulp: numpy's legacy Gaussian sampler (robot
# joint noise) and np.sin differ by 1 ulp between macOS arm64 and Linux aarch64 (Session 2).
# The committed values are canonical and restore exactly everywhere.
CROSS_PLATFORM_STATE_ATOL = 1e-12
PARTITIONS_SCHEMA = "roboforge.eval_partitions/1"
LOCKED_SUITE = "final_test"


class FinalTestLockedError(PermissionError):
    """Raised on any attempt to use final-test states before the freeze exists."""


def suites_config() -> dict[str, Any]:
    """Load ``configs/eval/suites.yaml``."""
    import yaml

    return yaml.safe_load((configs_dir() / "eval" / "suites.yaml").read_text())


def suite_dir(task: str) -> Path:
    """Committed location of a task's suites."""
    return REPO_ROOT / "data" / "eval_states" / task


def manifest_path(task: str, suite: str) -> Path:
    """Open manifest of one suite."""
    return suite_dir(task) / f"{suite}.json"


def data_path(task: str, suite: str) -> Path:
    """Compressed snapshot file of one suite."""
    return suite_dir(task) / f"{suite}.data.jsonl.xz"


def lock_status(task: str) -> tuple[bool, str]:
    """``(unlocked, reason)`` for the task's final-test suite."""
    cfg = suites_config()["suites"][LOCKED_SUITE]
    freeze = REPO_ROOT / cfg["locked_until"]
    if not freeze.exists():
        return False, f"{cfg['locked_until']} does not exist"
    manifest = json.loads(manifest_path(task, LOCKED_SUITE).read_text())
    if manifest["content_sha256"] not in freeze.read_text():
        return False, f"{cfg['locked_until']} does not name {task} final_test content_sha256"
    rel = freeze.relative_to(REPO_ROOT).as_posix()
    try:
        subprocess.run(["git", "ls-files", "--error-unmatch", rel], cwd=REPO_ROOT, check=True,
                       capture_output=True)
        subprocess.run(["git", "diff", "--quiet", "HEAD", "--", rel], cwd=REPO_ROOT, check=True,
                       capture_output=True)
    except (subprocess.CalledProcessError, FileNotFoundError):
        return False, f"{rel} is not committed, or has uncommitted changes"
    return True, "unlocked"


def _robustness_block() -> tuple[str, list[int], str] | None:
    """(suite, reserved seeds, partition) of the Session 7 robustness base set, if configured."""
    path = configs_dir() / "eval" / "robustness.yaml"
    if not path.exists():
        return None
    import yaml

    cfg = yaml.safe_load(path.read_text())
    return cfg["base"]["name"], list(cfg["base"]["reserved_seeds"]), cfg["partition"]


def suite_of_seed(seed: int) -> str | None:
    """Name of the suite whose reserved seed block contains ``seed`` (None = ad hoc)."""
    for name, cfg in suites_config()["suites"].items():
        lo, hi = cfg["reserved_seeds"]
        if lo <= seed < hi:
            return name
    rb = _robustness_block()
    if rb is not None and rb[1][0] <= seed < rb[1][1]:
        return rb[0]
    return None


def check_seed_allowed(task: str, seed: int) -> str:
    """Partition for an ad-hoc seed reset; raise if the seed lies in the locked block."""
    suite = suite_of_seed(int(seed))
    if suite == LOCKED_SUITE:
        unlocked, why = lock_status(task)
        if not unlocked:
            raise FinalTestLockedError(f"seed {seed} is reserved for the final test: {why}")
    if suite is None:
        return "eval_adhoc"
    rb = _robustness_block()
    if rb is not None and suite == rb[0]:
        return rb[2]
    return suites_config()["suites"][suite]["partition"]


@dataclass
class SuiteArtifacts:
    """One generated suite, before writing."""

    task: str
    suite: str
    manifest: dict[str, Any]
    data_bytes: bytes
    states: list[InitialState]


def _data_bytes(states: list[InitialState]) -> bytes:
    models = {s.model_xml_sha256: s.model_xml for s in states}
    lines = [json.dumps({"model_xml_sha256": k, "model_xml": models[k]}, sort_keys=True)
             for k in sorted(models)]
    lines += [json.dumps({"state_id": s.state_id, "state": s.state.tolist(),
                          "model_xml_sha256": s.model_xml_sha256}, sort_keys=True)
              for s in states]
    return ("\n".join(lines) + "\n").encode("utf-8")


def generate_suite(task: str, suite: str, env: Any | None = None) -> SuiteArtifacts:
    """Generate every state of ``suite`` for ``task`` and verify it restores exactly.

    Each state is also checked not to satisfy the success predicate at t = 0 (on the
    restored model, with current kinematics).
    """
    from roboforge.sim.factory import load_sim_config, make_env

    cfg = suites_config()["suites"][suite]
    own = env is None
    env = env or make_env(load_sim_config(task))
    states: list[InitialState] = []
    try:
        for i in range(int(cfg["n_states"])):
            seed = int(cfg["seed_base"]) + i
            init = generate(env, task, seed, f"{task}_{suite}_{i:04d}", cfg["partition"],
                            {"suite": suite, "index": i})
            restore(env, init)
            if bool(env._check_success()):
                raise AssertionError(f"{init.state_id}: success predicate true at t = 0")
            states.append(init)
    finally:
        if own:
            env.close()
    if len({s.state_sha256 for s in states}) != len(states):
        raise AssertionError(f"{task}/{suite}: duplicate initial states")
    data = _data_bytes(states)
    body = {"schema": SUITE_SCHEMA, "task": task, "suite": suite,
            "partition": cfg["partition"], "seed_rule": "seed = seed_base + index",
            "seed_base": int(cfg["seed_base"]), "n_states": len(states),
            "reserved_seeds": list(cfg["reserved_seeds"]),
            "locked_until": cfg.get("locked_until"),
            "generator": {"function": "roboforge.sim.snapshot.generate",
                          "env": f"robosuite.make from configs/simulator/{task}.yaml",
                          "packages": package_versions(("mujoco", "robosuite", "numpy"))},
            "generated_on": evaluation_platform()["platform_id"],
            "xml_convention": "mj_saveLastXML of the live model after reset; asset paths "
                              "robosuite-relative; compiled via env.edit_model_xml (D-025)",
            "n_unique_models": len({s.model_xml_sha256 for s in states}),
            "data_file": {"path": repo_relative(data_path(task, suite)),
                          "decompressed_sha256": sha256_text(data.decode("utf-8"))},
            "states": [{"state_id": s.state_id, "index": s.source["index"],
                        "seed": s.source["seed"], "model_xml_sha256": s.model_xml_sha256,
                        "state_sha256": s.state_sha256} for s in states]}
    manifest = {**body, "content_sha256": sha256_json(body)}
    return SuiteArtifacts(task, suite, manifest, data, states)


def write_suite(art: SuiteArtifacts, overwrite: bool = False) -> dict[str, str]:
    """Write manifest + data file; refuse to replace a different committed suite."""
    mp, dp = manifest_path(art.task, art.suite), data_path(art.task, art.suite)
    if mp.exists() and not overwrite:
        old = json.loads(mp.read_text())
        if old.get("content_sha256") != art.manifest["content_sha256"]:
            raise ValueError(f"{repo_relative(mp)} exists with different content; refusing to "
                             "overwrite (pass overwrite=True only for a recorded decision)")
    write_json(mp, art.manifest)
    dp.write_bytes(lzma.compress(art.data_bytes, preset=9))
    return {"manifest": repo_relative(mp), "data": repo_relative(dp)}


def read_manifest(task: str, suite: str) -> dict[str, Any]:
    """Open manifest (hashes only), self-hash verified. Never contains state values."""
    m = json.loads(manifest_path(task, suite).read_text())
    body = {k: v for k, v in m.items() if k != "content_sha256"}
    if sha256_json(body) != m["content_sha256"]:
        raise ValueError(f"{task}/{suite}: manifest content hash mismatch")
    return m


def verify_data_file(task: str, suite: str) -> bool:
    """True if the data file's decompressed bytes match the manifest (does not parse states)."""
    m = read_manifest(task, suite)
    raw = lzma.decompress(data_path(task, suite).read_bytes()).decode("utf-8")
    return sha256_text(raw) == m["data_file"]["decompressed_sha256"]


def load_suite(task: str, suite: str) -> list[InitialState]:
    """All states of a suite, hash-verified. Final-test access raises until the freeze."""
    if suite == LOCKED_SUITE:
        unlocked, why = lock_status(task)
        if not unlocked:
            raise FinalTestLockedError(f"{task}/{suite} is locked: {why}")
    m = read_manifest(task, suite)
    raw = lzma.decompress(data_path(task, suite).read_bytes()).decode("utf-8")
    if sha256_text(raw) != m["data_file"]["decompressed_sha256"]:
        raise ValueError(f"{task}/{suite}: data file does not match its manifest")
    models, values = {}, {}
    for line in raw.splitlines():
        rec = json.loads(line)
        if "model_xml" in rec:
            models[rec["model_xml_sha256"]] = rec["model_xml"]
        else:
            values[rec["state_id"]] = rec
    out = []
    for entry in m["states"]:
        rec = values[entry["state_id"]]
        init = InitialState(task=task, state_id=entry["state_id"], partition=m["partition"],
                            model_xml=models[entry["model_xml_sha256"]],
                            state=np.asarray(rec["state"], dtype=np.float64),
                            source={"kind": "generated", "suite": suite,
                                    "index": entry["index"], "seed": entry["seed"]})
        if (init.model_xml_sha256, init.state_sha256) != (entry["model_xml_sha256"],
                                                          entry["state_sha256"]):
            raise ValueError(f"{entry['state_id']}: hash mismatch against the manifest")
        out.append(init)
    return out


def _max_state_diff(task: str, suite: str, states: list[InitialState]) -> float:
    """Max |regenerated - committed| state value (integrity only; values are not returned)."""
    raw = lzma.decompress(data_path(task, suite).read_bytes()).decode("utf-8")
    stored = {r["state_id"]: np.asarray(r["state"]) for r in map(json.loads, raw.splitlines())
              if "state" in r}
    return max(float(np.abs(stored[s.state_id] - s.state).max()) for s in states)


def check_task(task: str) -> tuple[list[str], dict[str, Any]]:
    """Regenerate both suites in memory and compare with the committed ones.

    Returns ``(problems, details)``. On the generating platform everything must match
    bitwise; elsewhere the XML must match exactly and states within
    ``CROSS_PLATFORM_STATE_ATOL``.
    """
    problems, details = [], {}
    here = evaluation_platform()["platform_id"]
    for suite in suites_config()["suites"]:
        art = generate_suite(task, suite)
        committed = read_manifest(task, suite)
        same_platform = committed.get("generated_on") == here
        xml_equal = [a["model_xml_sha256"] for a in art.manifest["states"]] == [
            c["model_xml_sha256"] for c in committed["states"]]
        exact = committed["states"] == art.manifest["states"]
        diff = 0.0 if exact else _max_state_diff(task, suite, art.states)
        details[suite] = {"generated_on": committed.get("generated_on"), "checked_on": here,
                          "bitwise_equal": exact, "xml_equal": xml_equal,
                          "max_abs_state_diff": diff}
        if not xml_equal or committed["n_states"] != art.manifest["n_states"]:
            problems.append(f"{task}/{suite}: regenerated XML or state count differs")
        elif same_platform and not exact:
            problems.append(f"{task}/{suite}: not bitwise reproducible on its own platform")
        elif diff > CROSS_PLATFORM_STATE_ATOL:
            problems.append(f"{task}/{suite}: states differ by {diff:.3g} across platforms")
    return problems, details


def partitions_index(task: str) -> dict[str, Any]:
    """Every train/validation/test partition for ``task``, with hashes (committed index)."""
    from roboforge.data.pipeline import load_task_artifacts

    split = load_task_artifacts(task)["split"]
    suites = {}
    for name, cfg in suites_config()["suites"].items():
        m = read_manifest(task, name)
        suites[name] = {"partition": cfg["partition"], "manifest": repo_relative(
            manifest_path(task, name)), "content_sha256": m["content_sha256"],
            "n_states": m["n_states"], "seeds": [m["seed_base"], m["seed_base"] + m["n_states"]],
            "reserved_seeds": cfg["reserved_seeds"], "locked": name == LOCKED_SUITE,
            "use": ("final paired test only, after docs/FINAL_TEST_FREEZE.md"
                    if name == LOCKED_SUITE else
                    "closed-loop development and checkpoint selection")}
    body = {"schema": PARTITIONS_SCHEMA, "task": task,
            "demonstrations": {"split_manifest": f"data/splits/{task}_ph.json",
                               "split_content_sha256": split["content_sha256"],
                               "demo_train": {"n": len(split["splits"]["train"]),
                                              "use": "policy training (offline)"},
                               "demo_valid": {"n": len(split["splits"]["valid"]),
                                              "use": "offline validation loss only"}},
            "eval_suites": suites,
            "eval_adhoc": "seeds outside every reserved block: development only, never reported"}
    return {**body, "content_sha256": sha256_json(body)}

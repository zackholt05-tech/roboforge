"""X3 held-out starts: generation and the reproducibility check (DECISIONS D-111).

**X3 is a separate replication.** Its results are reported beside the frozen Level 1 results,
never instead of them.

**The suite.** ``x3_heldout`` (``configs/extensions/x3.yaml``) holds fresh Can starts from
robosuite's standard reset distribution. Each start is drawn exactly as the frozen suites draw
theirs: :func:`roboforge.sim.snapshot.generate` seeds ``np.random`` and ``random`` with
``seed_base + i`` and resets. The seed block lies outside every reserved block of the frozen
studies and outside the X1 development block (research-integrity test).

**Files.** The suite is stored in the frozen format (schema ``roboforge.eval_suite/1``), under
``data/eval_states/<task>/x3_heldout.{json,data.jsonl.xz}``. The frozen
:func:`roboforge.sim.eval_states.load_suite` and :func:`~roboforge.sim.eval_states.write_suite`
therefore read and write it unchanged. ``partitions.json`` is existing data and is not edited.

**Partition label.** The frozen vocabulary (:data:`roboforge.sim.snapshot.PARTITIONS`) has no
X3 label. The frozen code files these seeds under ``eval_adhoc``, so the suite uses that label.
The meaning travels in the manifest's ``role``, ``decision`` and ``label`` fields and in every
X3 summary. X3 analyses select by suite name and content hash, never by partition.

**Generation is canonical only on the evaluation container** (D-017)::

    ROBOFORGE_WRITE_EVAL_STATES=1 infra/containers/run_sim_cpu.sh \\
        python scripts/generate_x3_states.py

:func:`check_x3` regenerates the suite in memory and compares it with the committed files. On
the generating platform it must match bitwise. Elsewhere the XML must match exactly and the
states within :data:`roboforge.sim.eval_states.CROSS_PLATFORM_STATE_ATOL`.
"""

from __future__ import annotations

import json
import lzma
from typing import Any

import numpy as np

from roboforge.paths import configs_dir, repo_relative
from roboforge.provenance import evaluation_platform, package_versions, sha256_json
from roboforge.sim import eval_states as es
from roboforge.sim.snapshot import InitialState, generate, restore, sha256_text

SCHEMA = "roboforge.x3_suite/1"
LABEL = "separate replication"
DECISION = "D-111"
CONFIG = "configs/extensions/x3.yaml"


def x3_config() -> dict[str, Any]:
    """Load ``configs/extensions/x3.yaml`` and validate its keys."""
    import yaml

    cfg = yaml.safe_load((configs_dir() / "extensions" / "x3.yaml").read_text())
    keys = {"schema", "decision", "label", "task", "suite", "role", "partition", "seed_base",
            "n_states", "reserved_seeds"}
    if set(cfg) != keys or cfg["schema"] != SCHEMA:
        raise ValueError(f"{CONFIG}: keys {sorted(cfg)} != {sorted(keys)} or wrong schema")
    lo, hi = (int(v) for v in cfg["reserved_seeds"])
    base, n = int(cfg["seed_base"]), int(cfg["n_states"])
    if not lo <= base < base + n <= hi:
        raise ValueError(f"{CONFIG}: seeds [{base}, {base + n}) outside the reserved block")
    return cfg


def config_sha256() -> str:
    """Content hash of the X3 suite config (recorded in the manifest)."""
    return sha256_json(x3_config())


def suite_name() -> str:
    """Name of the X3 held-out suite (``x3_heldout``)."""
    return str(x3_config()["suite"])


def suite_task() -> str:
    """The one task X3 runs (Can; owner, D-109)."""
    return str(x3_config()["task"])


def generate_x3(env: Any) -> es.SuiteArtifacts:
    """Generate every X3 start and verify that each restores exactly.

    Each start is also checked not to satisfy the success predicate at t = 0 (on the restored
    model, with current kinematics), as for the frozen suites."""
    cfg = x3_config()
    task, name = cfg["task"], cfg["suite"]
    states: list[InitialState] = []
    for i in range(int(cfg["n_states"])):
        seed = int(cfg["seed_base"]) + i
        init = generate(env, task, seed, f"{task}_{name}_{i:04d}", cfg["partition"],
                        {"suite": name, "index": i})
        restore(env, init)
        if bool(env._check_success()):
            raise AssertionError(f"{init.state_id}: success predicate true at t = 0")
        states.append(init)
    if len({s.state_sha256 for s in states}) != len(states):
        raise AssertionError(f"{task}/{name}: duplicate initial states")
    return es.SuiteArtifacts(task, name, *_manifest(cfg, states), states)


def _manifest(cfg: dict[str, Any], states: list[InitialState]) -> tuple[dict[str, Any], bytes]:
    task, name = cfg["task"], cfg["suite"]
    data = es._data_bytes(states)
    body = {"schema": es.SUITE_SCHEMA, "task": task, "suite": name,
            "partition": cfg["partition"], "role": cfg["role"], "label": cfg["label"],
            "decision": cfg["decision"], "config": CONFIG, "config_sha256": config_sha256(),
            "seed_rule": "seed = seed_base + index", "seed_base": int(cfg["seed_base"]),
            "n_states": len(states), "reserved_seeds": [int(v) for v in cfg["reserved_seeds"]],
            "generator": {"function": "roboforge.sim.snapshot.generate",
                          "module": "roboforge.replication.x3_states",
                          "env": f"robosuite.make from configs/simulator/{task}.yaml",
                          "packages": package_versions(("mujoco", "robosuite", "numpy"))},
            "generated_on": evaluation_platform()["platform_id"],
            "xml_convention": "mj_saveLastXML of the live model after reset; asset paths "
                              "robosuite-relative; compiled via env.edit_model_xml (D-025)",
            "n_unique_models": len({s.model_xml_sha256 for s in states}),
            "data_file": {"path": repo_relative(es.data_path(task, name)),
                          "decompressed_sha256": sha256_text(data.decode("utf-8"))},
            "states": [{"state_id": s.state_id, "index": s.source["index"],
                        "seed": s.source["seed"], "model_xml_sha256": s.model_xml_sha256,
                        "state_sha256": s.state_sha256} for s in states]}
    return {**body, "content_sha256": sha256_json(body)}, data


def check_x3(env: Any | None = None) -> tuple[list[str], dict[str, Any]]:
    """Regenerate the X3 suite in memory and compare it with the committed files.

    Returns ``(problems, details)``. Bitwise on the generating platform (manifest content
    hash, every state entry and the data file); elsewhere identical XML and states within
    ``CROSS_PLATFORM_STATE_ATOL``. Writes nothing."""
    from roboforge.sim.factory import load_sim_config, make_env

    cfg = x3_config()
    task, name = cfg["task"], cfg["suite"]
    if not es.manifest_path(task, name).exists():
        return [f"{task}/{name}: not generated"], {}
    committed = es.read_manifest(task, name)
    own = env is None
    env = env or make_env(load_sim_config(task))
    try:
        art = generate_x3(env)
    finally:
        if own:
            env.close()
    here = evaluation_platform()["platform_id"]
    same = committed.get("generated_on") == here
    xml_equal = [s["model_xml_sha256"] for s in committed["states"]] == [
        s["model_xml_sha256"] for s in art.manifest["states"]]
    raw = lzma.decompress(es.data_path(task, name).read_bytes())
    stored = {r["state_id"]: np.asarray(r["state"])
              for r in map(json.loads, raw.decode("utf-8").splitlines()) if "state" in r}
    diff = max(float(np.abs(stored[s.state_id] - s.state).max()) for s in art.states)
    exact = (committed["content_sha256"] == art.manifest["content_sha256"]
             and raw == art.data_bytes)
    details = {"suite": name, "generated_on": committed.get("generated_on"), "checked_on": here,
               "content_sha256": committed["content_sha256"], "bitwise_equal": exact,
               "states_equal": committed["states"] == art.manifest["states"],
               "xml_equal": xml_equal, "max_abs_state_diff": diff,
               "data_file_ok": es.verify_data_file(task, name),
               "config_sha256_equal": committed["config_sha256"] == config_sha256()}
    problems = []
    if not xml_equal or committed["n_states"] != art.manifest["n_states"]:
        problems.append(f"{task}/{name}: regenerated XML or state count differs")
    elif same and not exact:
        problems.append(f"{task}/{name}: not bitwise reproducible on its own platform")
    elif diff > es.CROSS_PLATFORM_STATE_ATOL:
        problems.append(f"{task}/{name}: states differ by {diff:.3g} across platforms")
    if not details["data_file_ok"]:
        problems.append(f"{task}/{name}: data file does not match its manifest")
    if not details["config_sha256_equal"]:
        problems.append(f"{task}/{name}: {CONFIG} changed since generation")
    return problems, details

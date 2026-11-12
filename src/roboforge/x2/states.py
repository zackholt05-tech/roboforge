"""X2 evaluation suites: generation and the reproducibility check (DECISIONS D-123).

**The suites** (``configs/extensions/x2_suites.yaml``): ``x2_validation`` (50 starts, X2
development and checkpoint selection, and the X2-a headroom pilot) and ``x2_heldout`` (100
starts, stepped only after ``docs/extensions/X2_FREEZE.md``, in X2-c). Each holds fresh Can
starts from robosuite's standard reset distribution, drawn exactly as the frozen suites draw
theirs: :func:`roboforge.sim.snapshot.generate` seeds ``np.random`` and ``random`` with
``seed_base + i`` and resets. Both seed blocks lie outside every reserved block of the frozen
studies, X1 and X3 (research-integrity test).

**Files.** Stored in the frozen format (schema ``roboforge.eval_suite/1``) under
``data/eval_states/can/<suite>.{json,data.jsonl.xz}``, so the frozen
:func:`roboforge.sim.eval_states.load_suite` and :func:`~roboforge.sim.eval_states.write_suite`
read and write them unchanged. ``partitions.json`` is existing data and is not edited.

**Partition label** ``eval_adhoc`` (the frozen vocabulary has no X2 label; the D-111
precedent). The meaning travels in the manifest's ``role``, ``decision`` and ``label`` fields.

**Generation is canonical only on the evaluation container** (D-017)::

    ROBOFORGE_WRITE_EVAL_STATES=1 infra/containers/run_sim_cpu.sh \\
        python scripts/generate_x2_states.py

:func:`check_x2_suite` regenerates a suite in memory and compares it with the committed files:
bitwise on the generating platform; elsewhere identical XML and states within
:data:`roboforge.sim.eval_states.CROSS_PLATFORM_STATE_ATOL`.
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

SCHEMA = "roboforge.x2_suites/1"
DECISION = "D-123"
CONFIG = "configs/extensions/x2_suites.yaml"
VALIDATION_SUITE = "x2_validation"
HELDOUT_SUITE = "x2_heldout"


def x2_suites_config() -> dict[str, Any]:
    """Load ``configs/extensions/x2_suites.yaml`` and validate its keys and seed blocks."""
    import yaml

    cfg = yaml.safe_load((configs_dir() / "extensions" / "x2_suites.yaml").read_text())
    keys = {"schema", "decision", "label", "task", "partition", "suites"}
    if set(cfg) != keys or cfg["schema"] != SCHEMA:
        raise ValueError(f"{CONFIG}: keys {sorted(cfg)} != {sorted(keys)} or wrong schema")
    if sorted(cfg["suites"]) != sorted((VALIDATION_SUITE, HELDOUT_SUITE)):
        raise ValueError(f"{CONFIG}: suites {sorted(cfg['suites'])}")
    for name, s in cfg["suites"].items():
        if set(s) != {"role", "seed_base", "n_states", "reserved_seeds"}:
            raise ValueError(f"{CONFIG}: suite {name} keys {sorted(s)}")
        lo, hi = (int(v) for v in s["reserved_seeds"])
        base, n = int(s["seed_base"]), int(s["n_states"])
        if not lo <= base < base + n <= hi:
            raise ValueError(f"{CONFIG}: {name} seeds [{base}, {base + n}) outside its block")
    return cfg


def config_sha256() -> str:
    """Content hash of the X2 suites config (recorded in each manifest)."""
    return sha256_json(x2_suites_config())


def suite_task() -> str:
    """The one task X2 runs (Can; owner, D-116)."""
    return str(x2_suites_config()["task"])


def generate_x2_suite(env: Any, name: str) -> es.SuiteArtifacts:
    """Generate every start of one X2 suite and verify that each restores exactly and does not
    satisfy the success predicate at t = 0 (as for the frozen suites)."""
    cfg = x2_suites_config()
    task, s = cfg["task"], cfg["suites"][name]
    states: list[InitialState] = []
    for i in range(int(s["n_states"])):
        seed = int(s["seed_base"]) + i
        init = generate(env, task, seed, f"{task}_{name}_{i:04d}", cfg["partition"],
                        {"suite": name, "index": i})
        restore(env, init)
        if bool(env._check_success()):
            raise AssertionError(f"{init.state_id}: success predicate true at t = 0")
        states.append(init)
    if len({st.state_sha256 for st in states}) != len(states):
        raise AssertionError(f"{task}/{name}: duplicate initial states")
    return es.SuiteArtifacts(task, name, *_manifest(cfg, name, states), states)


def _manifest(cfg: dict[str, Any], name: str,
              states: list[InitialState]) -> tuple[dict[str, Any], bytes]:
    task, s = cfg["task"], cfg["suites"][name]
    data = es._data_bytes(states)
    body = {"schema": es.SUITE_SCHEMA, "task": task, "suite": name,
            "partition": cfg["partition"], "role": s["role"], "label": cfg["label"],
            "decision": cfg["decision"], "config": CONFIG, "config_sha256": config_sha256(),
            "seed_rule": "seed = seed_base + index", "seed_base": int(s["seed_base"]),
            "n_states": len(states), "reserved_seeds": [int(v) for v in s["reserved_seeds"]],
            "generator": {"function": "roboforge.sim.snapshot.generate",
                          "module": "roboforge.x2.states",
                          "env": f"robosuite.make from configs/simulator/{task}.yaml",
                          "packages": package_versions(("mujoco", "robosuite", "numpy"))},
            "generated_on": evaluation_platform()["platform_id"],
            "xml_convention": "mj_saveLastXML of the live model after reset; asset paths "
                              "robosuite-relative; compiled via env.edit_model_xml (D-025)",
            "n_unique_models": len({st.model_xml_sha256 for st in states}),
            "data_file": {"path": repo_relative(es.data_path(task, name)),
                          "decompressed_sha256": sha256_text(data.decode("utf-8"))},
            "states": [{"state_id": st.state_id, "index": st.source["index"],
                        "seed": st.source["seed"], "model_xml_sha256": st.model_xml_sha256,
                        "state_sha256": st.state_sha256} for st in states]}
    return {**body, "content_sha256": sha256_json(body)}, data


def check_x2_suite(name: str, env: Any | None = None) -> tuple[list[str], dict[str, Any]]:
    """Regenerate one X2 suite in memory and compare it with the committed files.

    Returns ``(problems, details)``. Writes nothing. Bitwise on the generating platform;
    elsewhere identical XML and states within ``CROSS_PLATFORM_STATE_ATOL``."""
    from roboforge.sim.factory import load_sim_config, make_env

    task = suite_task()
    if not es.manifest_path(task, name).exists():
        return [f"{task}/{name}: not generated"], {}
    committed = es.read_manifest(task, name)
    own = env is None
    env = env or make_env(load_sim_config(task))
    try:
        art = generate_x2_suite(env, name)
    finally:
        if own:
            env.close()
    here = evaluation_platform()["platform_id"]
    same = committed.get("generated_on") == here
    xml_equal = [st["model_xml_sha256"] for st in committed["states"]] == [
        st["model_xml_sha256"] for st in art.manifest["states"]]
    raw = lzma.decompress(es.data_path(task, name).read_bytes())
    stored = {r["state_id"]: np.asarray(r["state"])
              for r in map(json.loads, raw.decode("utf-8").splitlines()) if "state" in r}
    diff = max(float(np.abs(stored[st.state_id] - st.state).max()) for st in art.states)
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

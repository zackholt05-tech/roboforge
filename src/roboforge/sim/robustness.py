"""Session 7 robustness starts: a fresh base set and OOD perturbations of it (D-063).

RESEARCH_PROTOCOL §14.1-14.2. Config: ``configs/eval/robustness.yaml``.

* ``robustness_base`` (partition ``eval_robustness``): 50 starts per task from robosuite's
  standard reset distribution, seeds ``300000 + i``, generated exactly like the validation
  suite (:func:`roboforge.sim.snapshot.generate`). They never selected anything, so robustness
  and ablation comparisons on them carry no selection bias at the start level.
* OOD sets: state ``i`` is a deterministic perturbation of base state ``i`` (same model XML;
  only ``qpos`` entries change; velocities stay zero). Draws come from
  ``default_rng(SeedSequence([draw_seed, code, i, attempt]))``:

  - ``ood_cube_shift`` (Lift): the cube centre moves to radius ``r ~ U[0.05, 0.07]`` m and
    angle ``theta ~ U[-pi, pi)`` around the placement centre; z and orientation unchanged.
    Lift's training placements lie inside centre +- 0.03 m, so every OOD cube lies outside.
  - ``ood_arm_offset`` (Lift, Can): each of the 7 arm joints moves by +-0.10 rad with fair,
    independent signs (5 sigma of robosuite's 0.02 rad initialization noise).

Every OOD state passes :func:`check_validity` on a **scratch** environment (never live
evaluation data, KI-019): exact restore, success false at t = 0, arm joints inside their
ranges, no contact pair absent from the base state and no penetration deeper than 1 mm, and
passive settling (20 hold-pose steps) that leaves the task object at rest (< 1 mm of motion
over the last 5 steps) where it should be. Only contacts involving a robot body or the task
object are checked. A failing draw is replaced by the next ``attempt``; every rejection is
recorded in the manifest. The suites are stored like the evaluation suites
(:mod:`roboforge.sim.eval_states`: open manifest with hashes, compressed snapshot file) and
load with :func:`roboforge.sim.eval_states.load_suite`.
"""

from __future__ import annotations

import json
import lzma
from functools import cache
from typing import Any

import mujoco
import numpy as np

from roboforge.paths import configs_dir, repo_relative
from roboforge.provenance import evaluation_platform, package_versions, sha256_json
from roboforge.sim import eval_states as es
from roboforge.sim.snapshot import InitialState, generate, restore, sha256_text

SCHEMA = "roboforge.eval_suite/1"  # same file format as the evaluation suites
TASK_OBJECT_BODY = {"lift": "cube_main", "can": "Can_main"}
EEF_SITE = "gripper0_right_grip_site"
ROBOT_BODY_PREFIXES = ("robot0_", "gripper0_")


@cache
def robustness_config() -> dict[str, Any]:
    """Load ``configs/eval/robustness.yaml``."""
    import yaml

    return yaml.safe_load((configs_dir() / "eval" / "robustness.yaml").read_text())


def config_sha256() -> str:
    """Content hash of the robustness config (recorded in every manifest)."""
    return sha256_json(robustness_config())


def base_name() -> str:
    """Name of the base suite (``robustness_base``)."""
    return str(robustness_config()["base"]["name"])


def sets_for(task: str) -> list[str]:
    """Every robustness suite of ``task``: the base set first, then its OOD sets."""
    cfg = robustness_config()
    return [base_name(), *[n for n, s in cfg["ood"].items() if task in s["tasks"]]]


def _draw(code: int, index: int, attempt: int) -> np.random.Generator:
    cfg = robustness_config()
    return np.random.default_rng(np.random.SeedSequence(
        [int(cfg["draw_seed"]), int(code), int(index), int(attempt)]))


def _jnt(model: Any, name: str) -> int:
    j = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if j < 0:
        raise KeyError(f"joint {name!r} not in the model")
    return int(j)


def perturb_cube_shift(model: Any, state: np.ndarray, rng: np.random.Generator,
                       spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Move the free object's centre to radius r, angle theta around the placement centre."""
    j = _jnt(model, spec["object_joint"])
    adr = 1 + int(model.jnt_qposadr[j])  # flattened state = [time, qpos, qvel]
    cx, cy = (float(v) for v in spec["center_xy"])
    lo, hi = (float(v) for v in spec["radius_m"])
    r = float(rng.uniform(lo, hi))
    theta = float(rng.uniform(-np.pi, np.pi))
    out = state.copy()
    before = out[adr:adr + 2].copy()
    half = float(spec["training_half_range_m"])
    if np.any(np.abs(before - (cx, cy)) > half + 1e-9):
        raise AssertionError(f"base object xy {before} lies outside the training placement range")
    out[adr] = cx + r * np.cos(theta)
    out[adr + 1] = cy + r * np.sin(theta)
    return out, {"radius_m": r, "theta_rad": theta, "base_xy_m": before.tolist(),
                 "new_xy_m": out[adr:adr + 2].tolist(),
                 "displacement_m": float(np.linalg.norm(out[adr:adr + 2] - before))}


def perturb_arm_offset(model: Any, state: np.ndarray, rng: np.random.Generator,
                       spec: dict[str, Any]) -> tuple[np.ndarray, dict[str, Any]]:
    """Offset every named arm joint by +-offset with fair independent signs."""
    delta = float(spec["offset_rad"])
    signs = rng.choice(np.array([-1.0, 1.0]), size=len(spec["joints"]))
    out = state.copy()
    for name, s in zip(spec["joints"], signs, strict=True):
        out[1 + int(model.jnt_qposadr[_jnt(model, name)])] += s * delta
    return out, {"offset_rad": delta, "signs": signs.astype(int).tolist()}


PERTURBATIONS = {"cube_shift": perturb_cube_shift, "arm_offset": perturb_arm_offset}


def _body_name(model: Any, b: int) -> str:
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) or f"body{b}"


def _relevant(name: str, task: str) -> bool:
    return name.startswith(ROBOT_BODY_PREFIXES) or name == TASK_OBJECT_BODY[task]


def contact_summary(env: Any, task: str) -> dict[str, Any]:
    """Contact pairs (body names, sorted) involving a robot body or the task object, and
    their deepest penetration, in the current data.

    Called right after :func:`restore` on a scratch environment, whose ``sim.forward()``
    computed the contacts of exactly this state. Contacts among other bodies (Can's inactive
    objects, parked overlapping at (10, 10, 10)) are excluded, as from every metric."""
    m, d = env.sim.model._model, env.sim.data._data
    pairs, deepest = set(), 0.0
    for c in d.contact[: d.ncon]:
        b1, b2 = (_body_name(m, int(m.geom_bodyid[g])) for g in (c.geom1, c.geom2))
        if not (_relevant(b1, task) or _relevant(b2, task)):
            continue
        pairs.add(tuple(sorted((b1, b2))))
        deepest = min(deepest, float(c.dist))
    return {"pairs": sorted(pairs), "max_penetration_m": -deepest}


def _object_state(env: Any, task: str) -> tuple[np.ndarray, float]:
    m, d = env.sim.model._model, env.sim.data._data
    b = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, TASK_OBJECT_BODY[task])
    jadr = int(m.body_jntadr[b])
    dof = int(m.jnt_dofadr[jadr])
    return np.array(d.xpos[b]), float(np.linalg.norm(d.qvel[dof:dof + 3]))


def _eef(env: Any) -> np.ndarray:
    m, d = env.sim.model._model, env.sim.data._data
    return np.array(d.site_xpos[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, EEF_SITE)])


def settle(env: Any, task: str, init: InitialState) -> dict[str, Any]:
    """Restore ``init`` on a scratch env and run the hold-pose settling steps."""
    v = robustness_config()["validity"]
    restore(env, init)
    pos0, _ = _object_state(env, task)
    action = np.asarray(v["settle_action"], dtype=np.float64)
    n, w = int(v["settle_steps"]), int(v["settle_rest_window_steps"])
    window_start = None
    for k in range(n):
        env.step(action)
        if k + 1 == n - w:
            window_start, _ = _object_state(env, task)
    pos, speed = _object_state(env, task)
    return {"object_pos_initial_m": pos0.tolist(), "object_pos_settled_m": pos.tolist(),
            "object_rest_window_motion_m": float(np.linalg.norm(pos - window_start)),
            "object_speed_settled_mps": speed}


def arm_joint_margin(env: Any, state: np.ndarray, joints: list[str]) -> float:
    """Smallest distance (rad) of a named joint's value to its range limits."""
    m = env.sim.model._model
    margins = []
    for name in joints:
        j = _jnt(m, name)
        q = float(state[1 + int(m.jnt_qposadr[j])])
        lo, hi = (float(x) for x in m.jnt_range[j])
        margins.append(min(q - lo, hi - q) if m.jnt_limited[j] else np.inf)
    return float(min(margins))


def check_validity(env: Any, task: str, base: InitialState, base_ref: dict[str, Any],
                   init: InitialState) -> dict[str, Any]:
    """Every §14.2 validity check of one OOD state (scratch env only)."""
    v = robustness_config()["validity"]
    joints = robustness_config()["ood"]["ood_arm_offset"]["joints"]
    restore(env, init)  # (1) raises unless the read-back equals the snapshot exactly
    success0 = bool(env._check_success())
    contacts = contact_summary(env, task)
    eef = _eef(env)
    margin = arm_joint_margin(env, init.state, joints)
    st = settle(env, task, init)
    p0, p1 = np.array(st["object_pos_initial_m"]), np.array(st["object_pos_settled_m"])
    checks = {
        "restores_exactly": True,
        "success_false_at_t0": not success0,
        "arm_joints_inside_range": margin >= float(v["joint_margin_rad"]),
        "no_new_contact_pair": set(map(tuple, contacts["pairs"]))
        <= set(map(tuple, base_ref["contacts"]["pairs"])),
        "penetration_within_tol": contacts["max_penetration_m"] <= float(v["max_penetration_m"]),
        "settled_xy_within_tol": float(np.linalg.norm(p1[:2] - p0[:2]))
        < float(v["settle_xy_tol_m"]),
        "settled_z_within_tol": abs(float(p1[2]) - base_ref["settled_z_m"])
        < float(v["settle_z_tol_m"]),
        "settled_at_rest": st["object_rest_window_motion_m"] < float(v["settle_rest_tol_m"]),
    }
    return {"passed": all(checks.values()), "checks": checks,
            "measurements": {"arm_joint_min_margin_rad": margin,
                             "eef_displacement_m": float(np.linalg.norm(
                                 eef - np.array(base_ref["eef_pos_m"]))),
                             "contact_pairs": contacts["pairs"],
                             "max_penetration_m": contacts["max_penetration_m"],
                             "object_settled_xy_shift_m": float(np.linalg.norm(p1[:2] - p0[:2])),
                             "object_settled_z_m": float(p1[2]),
                             "object_rest_window_motion_m": st["object_rest_window_motion_m"],
                             "object_speed_settled_mps": st["object_speed_settled_mps"]}}


def base_reference(env: Any, task: str, base: InitialState) -> dict[str, Any]:
    """Contacts, end-effector position and settled object height of a base state."""
    restore(env, base)
    ref = {"contacts": contact_summary(env, task), "eef_pos_m": _eef(env).tolist()}
    ref["settled_z_m"] = float(settle(env, task, base)["object_pos_settled_m"][2])
    return ref


def _manifest(task: str, suite: str, states: list[InitialState], entries: list[dict[str, Any]],
              extra: dict[str, Any]) -> tuple[dict[str, Any], bytes]:
    data = es._data_bytes(states)
    body = {"schema": SCHEMA, "task": task, "suite": suite,
            "partition": robustness_config()["partition"],
            "config": "configs/eval/robustness.yaml", "config_sha256": config_sha256(),
            "n_states": len(states), "generated_on": evaluation_platform()["platform_id"],
            "generator": {"module": "roboforge.sim.robustness",
                          "env": f"robosuite.make from configs/simulator/{task}.yaml",
                          "packages": package_versions(("mujoco", "robosuite", "numpy"))},
            "xml_convention": "mj_saveLastXML of the live model after reset; asset paths "
                              "robosuite-relative; compiled via env.edit_model_xml (D-025)",
            "n_unique_models": len({s.model_xml_sha256 for s in states}),
            "data_file": {"path": repo_relative(es.data_path(task, suite)),
                          "decompressed_sha256": sha256_text(data.decode("utf-8"))},
            **extra, "states": entries}
    return {**body, "content_sha256": sha256_json(body)}, data


def generate_base(task: str, env: Any) -> es.SuiteArtifacts:
    """The ``robustness_base`` suite of ``task`` (seeded standard resets)."""
    cfg = robustness_config()
    b = cfg["base"]
    states, entries = [], []
    for i in range(int(b["n_states"])):
        seed = int(b["seed_base"]) + i
        init = generate(env, task, seed, f"{task}_{b['name']}_{i:04d}", cfg["partition"],
                        {"suite": b["name"], "index": i})
        restore(env, init)
        if bool(env._check_success()):
            raise AssertionError(f"{init.state_id}: success predicate true at t = 0")
        states.append(init)
        entries.append({"state_id": init.state_id, "index": i, "seed": seed,
                        "model_xml_sha256": init.model_xml_sha256,
                        "state_sha256": init.state_sha256})
    if len({s.state_sha256 for s in states}) != len(states):
        raise AssertionError(f"{task}/{b['name']}: duplicate initial states")
    extra = {"seed_rule": "seed = seed_base + index", "seed_base": int(b["seed_base"]),
             "reserved_seeds": list(b["reserved_seeds"]),
             "kind": "fresh in-distribution starts (RESEARCH_PROTOCOL §14.1)"}
    manifest, data = _manifest(task, b["name"], states, entries, extra)
    return es.SuiteArtifacts(task, b["name"], manifest, data, states)


def generate_ood(task: str, suite: str, base: list[InitialState], env: Any,
                 base_manifest_sha256: str) -> es.SuiteArtifacts:
    """OOD suite ``suite`` of ``task`` derived from the base states (§14.2); the manifest
    records the base suite's content hash."""
    cfg = robustness_config()
    spec = cfg["ood"][suite]
    if task not in spec["tasks"]:
        raise ValueError(f"{suite} is not defined for {task}")
    perturb = PERTURBATIONS[spec["kind"]]
    model = env.sim.model._model
    states, entries = [], []
    for i, b in enumerate(base):
        ref = base_reference(env, task, b)
        rejected = []
        for attempt in range(int(cfg["validity"]["max_attempts"])):
            state, params = perturb(model, b.state, _draw(spec["code"], i, attempt), spec)
            init = InitialState(task=task, state_id=f"{task}_{suite}_{i:04d}",
                                partition=cfg["partition"], model_xml=b.model_xml, state=state,
                                source={"suite": suite, "index": i,
                                        "base_state_id": b.state_id})
            val = check_validity(env, task, b, ref, init)
            if val["passed"]:
                break
            rejected.append({"attempt": attempt, "failed_checks": sorted(
                k for k, ok in val["checks"].items() if not ok)})
        else:
            raise AssertionError(f"{task}/{suite}/{i}: no valid draw in "
                                 f"{cfg['validity']['max_attempts']} attempts: {rejected}")
        states.append(init)
        entries.append({"state_id": init.state_id, "index": i, "seed": b.source["seed"],
                        "base_state_id": b.state_id, "base_state_sha256": b.state_sha256,
                        "model_xml_sha256": init.model_xml_sha256,
                        "state_sha256": init.state_sha256, "attempt": attempt,
                        "rejected": rejected, "perturbation": params,
                        "validity": val["measurements"]})
    extra = {"kind": spec["kind"], "perturbation": spec, "draw_seed": int(cfg["draw_seed"]),
             "base_suite": base_name(),
             "base_manifest_content_sha256": base_manifest_sha256,
             "validity_rules": cfg["validity"],
             "n_rejected_draws": sum(len(e["rejected"]) for e in entries)}
    manifest, data = _manifest(task, suite, states, entries, extra)
    return es.SuiteArtifacts(task, suite, manifest, data, states)


def generate_task(task: str, env: Any | None = None,
                  base: es.SuiteArtifacts | None = None) -> dict[str, es.SuiteArtifacts]:
    """Every robustness suite of ``task``. OOD sets need the base suite committed first
    (they record its manifest hash), so pass ``base`` only when it is already on disk."""
    from roboforge.sim.factory import load_sim_config, make_env

    own = env is None
    env = env or make_env(load_sim_config(task))
    try:
        out = {base_name(): base or generate_base(task, env)}
        if es.manifest_path(task, base_name()).exists():
            states = es.load_suite(task, base_name())
            sha = es.read_manifest(task, base_name())["content_sha256"]
            for name in sets_for(task)[1:]:
                out[name] = generate_ood(task, name, states, env, sha)
        return out
    finally:
        if own:
            env.close()


def check_task(task: str) -> tuple[list[str], dict[str, Any]]:
    """Regenerate every robustness suite in memory and compare with the committed one.

    On the generating platform the whole manifest must match bitwise. Elsewhere the XML must
    match exactly and the states within ``es.CROSS_PLATFORM_STATE_ATOL`` (validity
    measurements are physics outputs and may differ in the last bits)."""
    problems, details = [], {}
    here = evaluation_platform()["platform_id"]
    arts = generate_task(task)
    for name in sets_for(task):
        if not es.manifest_path(task, name).exists():
            problems.append(f"{task}/{name}: not generated")
            continue
        committed = es.read_manifest(task, name)
        art = arts.get(name)
        if art is None:
            problems.append(f"{task}/{name}: could not be regenerated")
            continue
        same = committed.get("generated_on") == here
        xml_equal = [s["model_xml_sha256"] for s in committed["states"]] == [
            s["model_xml_sha256"] for s in art.manifest["states"]]
        exact = committed == art.manifest
        raw = lzma.decompress(es.data_path(task, name).read_bytes()).decode("utf-8")
        stored = {r["state_id"]: np.asarray(r["state"]) for r in map(json.loads, raw.splitlines())
                  if "state" in r}
        diff = max(float(np.abs(stored[s.state_id] - s.state).max()) for s in art.states)
        details[name] = {"generated_on": committed.get("generated_on"), "checked_on": here,
                         "bitwise_equal": exact, "xml_equal": xml_equal,
                         "max_abs_state_diff": diff,
                         "data_file_ok": es.verify_data_file(task, name)}
        if not xml_equal or committed["n_states"] != art.manifest["n_states"]:
            problems.append(f"{task}/{name}: regenerated XML or state count differs")
        elif same and not exact:
            problems.append(f"{task}/{name}: not bitwise reproducible on its own platform")
        elif diff > es.CROSS_PLATFORM_STATE_ATOL:
            problems.append(f"{task}/{name}: states differ by {diff:.3g} across platforms")
        if not details[name]["data_file_ok"]:
            problems.append(f"{task}/{name}: data file does not match its manifest")
    return problems, details

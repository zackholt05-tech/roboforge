"""Two fresh processes, same controlled seed → same trajectory (Session 2 acceptance).

For each task, three fresh Python processes run the same two episodes through RoboForgeEnv
and the recorder:

* ``random``: validation start ``<task>_validation_0000`` with ``RandomPolicy(policy_seed=7)``
  for the full 400-step horizon (closed loop);
* ``demo``: open-loop replay of ``demo_0`` (validation split; contact-rich, ends in success).
  This is **non-autonomous** instrumentation.

Processes 1 and 2 load the stored snapshot. Process 3 regenerates it from its seed (100000).
On the platform that generated the suite it must reproduce the stored hashes and trajectory
bitwise. Elsewhere the XML must be identical and the state within
``CROSS_PLATFORM_STATE_ATOL``: numpy's legacy Gaussian sampler differs by 1 ulp across
platforms. The parent compares every recorded array except
wall-clock timing. **Declared tolerance: 0.0** (bitwise) on the same platform. The parent
also re-simulates the records (``verify_record``).

``--compare A.json B.json`` instead compares two reports from different platforms, for
information only (KI-013).

Writes ``reports/session2/determinism/<platform_id>_<task>.json``.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import h5py  # noqa: E402
import numpy as np  # noqa: E402

from roboforge.data.fetch import load_manifest  # noqa: E402
from roboforge.eval.policies import DemoReplayPolicy, RandomPolicy  # noqa: E402
from roboforge.eval.rollout import TIMING_ARRAYS, RolloutRecord, run_episode, verify_record  # noqa: E402,E501
from roboforge.paths import REPO_ROOT  # noqa: E402
from roboforge.provenance import environment_snapshot, evaluation_platform  # noqa: E402
from roboforge.sim.env_adapter import RoboForgeEnv  # noqa: E402
from roboforge.sim.eval_states import (  # noqa: E402
    CROSS_PLATFORM_STATE_ATOL,
    load_suite,
    read_manifest,
)
from roboforge.sim.snapshot import from_demo, generate  # noqa: E402

OUT = REPO_ROOT / "reports" / "session2" / "determinism"
POLICY_SEED, DEMO = 7, "demo_0"


def _demo(task: str) -> tuple[Any, np.ndarray, str]:
    _, files = load_manifest()
    e = next(f for f in files if f.task == task and f.hdf5_type == "low_dim")
    with h5py.File(e.destination(), "r") as f:
        g = f["data"][DEMO]
        init = from_demo(task, DEMO, g.attrs["model_file"], g["states"][()], "demo_valid",
                         e.sha256)
        return init, g["actions"][()], e.sha256


def child(task: str, out: Path, regenerate: bool) -> None:
    """Run both episodes in this (fresh) process and save the records."""
    env = RoboForgeEnv(task, hold_steps=10_000)
    stored = load_suite(task, "validation")[0]
    init = stored
    if regenerate:
        init = generate(env.raw_env, task, stored.source["seed"], stored.state_id,
                        stored.partition, {"suite": "validation", "index": 0})
        out.mkdir(parents=True, exist_ok=True)
        (out / "regenerated.json").write_text(json.dumps({
            "xml_equal": init.model_xml_sha256 == stored.model_xml_sha256,
            "state_bitwise_equal": init.state_sha256 == stored.state_sha256,
            "max_abs_state_diff": float(np.abs(init.state - stored.state).max())}))
    env.hold_steps = 0
    run_episode(env, RandomPolicy(env.action_spec, POLICY_SEED), init,
                run_id="determinism").save(out / "random")
    env.hold_steps = 10_000
    demo_init, actions, sha = _demo(task)
    run_episode(env, DemoReplayPolicy(actions, DEMO, sha), demo_init,
                run_id="determinism").save(out / "demo")
    env.close()


def _maxdiff(a: RolloutRecord, b: RolloutRecord) -> dict[str, float]:
    out = {}
    for k in sorted(set(a.arrays) | set(b.arrays)):
        if k in TIMING_ARRAYS:
            continue
        x, y = a.arrays.get(k), b.arrays.get(k)
        out[k] = (float("inf") if x is None or y is None or x.shape != y.shape else
                  float(np.abs(x.astype(np.float64) - y.astype(np.float64)).max(initial=0.0)))
    return out


def run_task(task: str, tmp: Path) -> dict[str, Any]:
    """Spawn 3 fresh processes, compare their records, re-simulate one."""
    dirs = []
    for i, regen in enumerate((False, False, True)):
        d = tmp / f"{task}_proc{i}"
        cmd = [sys.executable, __file__, "--child", str(d), "--task", task]
        subprocess.run(cmd + (["--regenerate"] if regen else []), check=True,
                       capture_output=True, text=True)
        dirs.append(d)
    rep: dict[str, Any] = {"task": task, "environment": environment_snapshot(),
                           "platform": evaluation_platform(), "declared_tolerance": 0.0,
                           "episodes": {}}
    stored = load_suite(task, "validation")[0]
    for ep in ("random", "demo"):
        recs = [RolloutRecord.load(d / ep) for d in dirs]
        diffs = [_maxdiff(recs[0], r) for r in recs[1:]]
        init = stored if ep == "random" else _demo(task)[0]
        ver = verify_record(recs[0], init) if ep == "random" else None
        rep["episodes"][ep] = {
            "initial_state": recs[0].meta["episode"]["initial_state"],
            "policy": recs[0].meta["policy"], "outcome": recs[0].outcome,
            "physics_sha256": [r.meta["physics_sha256"] for r in recs],
            "max_abs_diff_proc0_vs_proc1": max(diffs[0].values()),
            "max_abs_diff_proc0_vs_proc2_regenerated": max(diffs[1].values()),
            "verify_record": ver}
    e = rep["episodes"]
    regen = json.loads((dirs[2] / "regenerated.json").read_text())
    same = read_manifest(task, "validation")["generated_on"] == rep["platform"]["platform_id"]
    rep["regenerated_start"] = {**regen, "generating_platform": same,
                                "expectation": "bitwise on the generating platform; elsewhere "
                                "identical XML and state within CROSS_PLATFORM_STATE_ATOL, "
                                "and trajectories may then differ (KI-013)"}
    regen_ok = (regen["xml_equal"] and regen["state_bitwise_equal"]
                and all(v["max_abs_diff_proc0_vs_proc2_regenerated"] == 0.0
                        for v in e.values())) if same else (
        regen["xml_equal"] and regen["max_abs_state_diff"] <= CROSS_PLATFORM_STATE_ATOL)
    rep["checks"] = {
        "two_fresh_processes_identical": all(v["max_abs_diff_proc0_vs_proc1"] == 0.0
                                             for v in e.values()),
        "regenerated_start_reproduces_stored": bool(regen_ok),
        "physics_hashes_equal_stored_start": all(v["physics_sha256"][0] == v["physics_sha256"][1]
                                                 for v in e.values()),
        "random_record_resimulates": bool(e["random"]["verify_record"]["passed"]),
        "random_policy_outcome_recorded": e["random"]["outcome"]["truncated"]
        and e["random"]["outcome"]["steps"] == 400,
        "demo_replay_records_first_success":
            e["demo"]["outcome"]["first_success_step"] is not None,
        "stored_suite_manifest_matches": read_manifest(task, "validation")["states"][0]
        ["state_sha256"] == stored.state_sha256}
    return rep


def compare(a_path: Path, b_path: Path) -> dict[str, Any]:
    """Cross-platform comparison of two determinism reports (information only)."""
    a, b = json.loads(a_path.read_text()), json.loads(b_path.read_text())
    return {"a": a["platform"]["platform_id"], "b": b["platform"]["platform_id"],
            "same_physics_sha256": {ep: a["episodes"][ep]["physics_sha256"][0]
                                    == b["episodes"][ep]["physics_sha256"][0]
                                    for ep in a["episodes"]},
            "outcomes": {ep: [a["episodes"][ep]["outcome"]["first_success_step"],
                              b["episodes"][ep]["outcome"]["first_success_step"]]
                         for ep in a["episodes"]}}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=["lift", "can"])
    p.add_argument("--task")
    p.add_argument("--out-dir", type=Path, default=OUT)
    p.add_argument("--child", type=Path, help=argparse.SUPPRESS)
    p.add_argument("--regenerate", action="store_true", help=argparse.SUPPRESS)
    p.add_argument("--compare", nargs=2, type=Path)
    args = p.parse_args(argv)
    if args.child:
        child(args.task, args.child, args.regenerate)
        return 0
    if args.compare:
        print(json.dumps(compare(*args.compare), indent=1))
        return 0
    args.out_dir.mkdir(parents=True, exist_ok=True)
    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        for task in args.tasks:
            rep = run_task(task, Path(tmp))
            pid = rep["platform"]["platform_id"]
            (args.out_dir / f"{pid}_{task}.json").write_text(json.dumps(rep, indent=1) + "\n")
            for name, passed in rep["checks"].items():
                print(f"[{task}] [{'PASS' if passed else 'FAIL'}] {name}")
            ok &= all(rep["checks"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

"""Generate (or ``--check``) the Session 7 robustness starts (RESEARCH_PROTOCOL §14, D-063).

Writes, per task, ``data/eval_states/<task>/robustness_base.{json,data.jsonl.xz}`` (50 fresh
in-distribution starts, seeds 300000 + i) and every OOD set of the task derived from them
(``ood_cube_shift`` for Lift, ``ood_arm_offset`` for Lift and Can), plus a report
``reports/session<N>/robustness/states_<task>.json`` with the validity measurements and the
rejection counts. Generation is canonical only on the evaluation container (D-017)::

    ROBOFORGE_WRITE_EVAL_STATES=1 infra/containers/run_sim_cpu.sh \\
        python scripts/generate_robustness_states.py --tasks lift can

``--check`` regenerates everything in memory and compares it with the committed files
(bitwise on the generating platform); it writes nothing. Only the validation suite's open
manifest is read (state hashes, for the disjointness check); the held-out suites are never
read (their disjointness is a research-integrity test).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

import numpy as np  # noqa: E402

from roboforge.paths import evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import environment_snapshot, git_revision, write_json  # noqa: E402
from roboforge.sim import eval_states as es  # noqa: E402
from roboforge.sim import robustness as rb  # noqa: E402


def _stats(values: list[float]) -> dict[str, float]:
    a = np.asarray(values, dtype=float)
    return {"min": float(a.min()), "median": float(np.median(a)), "max": float(a.max())}


def summarize(task: str, arts: dict[str, es.SuiteArtifacts]) -> dict[str, Any]:
    """Per-suite counts, hashes, rejections and validity measurement ranges."""
    out: dict[str, Any] = {}
    for name, art in arts.items():
        m = art.manifest
        rec: dict[str, Any] = {"n_states": m["n_states"], "content_sha256": m["content_sha256"],
                               "n_unique_models": m["n_unique_models"],
                               "generated_on": m["generated_on"]}
        if name != rb.base_name():
            st = m["states"]
            rec.update(
                n_rejected_draws=m["n_rejected_draws"],
                rejections=[{"index": e["index"], "rejected": e["rejected"]} for e in st
                            if e["rejected"]],
                eef_displacement_m=_stats([e["validity"]["eef_displacement_m"] for e in st]),
                arm_joint_min_margin_rad=_stats([e["validity"]["arm_joint_min_margin_rad"]
                                                 for e in st]),
                max_penetration_m=max(e["validity"]["max_penetration_m"] for e in st),
                object_settled_xy_shift_m=_stats([e["validity"]["object_settled_xy_shift_m"]
                                                  for e in st]),
                object_rest_window_motion_m=_stats([e["validity"]["object_rest_window_motion_m"]
                                                    for e in st]))
            if m["kind"] == "cube_shift":
                rec["cube_radius_m"] = _stats([e["perturbation"]["radius_m"] for e in st])
                rec["cube_displacement_m"] = _stats([e["perturbation"]["displacement_m"]
                                                     for e in st])
        out[name] = rec
    base = {s.state_sha256 for s in arts[rb.base_name()].states}
    val = {s["state_sha256"] for s in es.read_manifest(task, "validation")["states"]}
    ood = set().union(*({s.state_sha256 for s in a.states} for n, a in arts.items()
                        if n != rb.base_name())) if len(arts) > 1 else set()
    return {"task": task, "suites": out,
            "disjoint": {"base_vs_validation": not (base & val),
                         "ood_vs_base": not (ood & base)}}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=["lift", "can"])
    p.add_argument("--check", action="store_true", help="regenerate in memory and compare")
    args = p.parse_args(argv)
    ok = True
    for task in args.tasks:
        if args.check:
            problems, details = rb.check_task(task)
            print(f"[{task}] robustness check: {'PASS' if not problems else 'FAIL'} "
                  f"{json.dumps(details)}")
            for prob in problems:
                print("   ", prob)
            ok &= not problems
            continue
        from roboforge.sim.factory import load_sim_config, make_env

        env = make_env(load_sim_config(task))
        try:
            base = rb.generate_base(task, env)
            print(f"[{task}/{base.suite}] {base.manifest['n_states']} states, content "
                  f"{base.manifest['content_sha256']} -> {es.write_suite(base)}")
            arts = rb.generate_task(task, env, base=base)
        finally:
            env.close()
        for name, art in arts.items():
            if name == rb.base_name():
                continue
            print(f"[{task}/{name}] {art.manifest['n_states']} states, "
                  f"{art.manifest['n_rejected_draws']} rejected draws, content "
                  f"{art.manifest['content_sha256']} -> {es.write_suite(art)}")
        rep = {"schema": "roboforge.robustness_states_report/1", **summarize(task, arts),
               "git": git_revision(), "environment": environment_snapshot()}
        out = evidence_reports_dir() / "robustness" / f"states_{task}.json"
        write_json(out, rep)
        print(f"[{task}] report {repo_relative(out)}; disjoint {rep['disjoint']}")
        ok &= all(rep["disjoint"].values())
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

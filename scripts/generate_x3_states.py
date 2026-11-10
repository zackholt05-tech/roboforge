"""Generate (or ``--check``) the X3 held-out starts (DECISIONS D-111; separate replication).

Writes ``data/eval_states/can/x3_heldout.{json,data.jsonl.xz}``: 100 fresh Can starts from
robosuite's standard reset distribution, seeds 14,000,000 + i (``configs/extensions/x3.yaml``),
plus a report ``reports/session<N>/x3/states.json`` with the counts, hashes and disjointness
checks. Generation is canonical only on the evaluation container (D-017)::

    ROBOFORGE_WRITE_EVAL_STATES=1 ROBOFORGE_EVIDENCE_SESSION=14 \\
        infra/containers/run_sim_cpu.sh python scripts/generate_x3_states.py

``--check`` regenerates the suite in memory and compares it with the committed files (bitwise
on the generating platform, within 1e-12 elsewhere); it writes nothing. Only open manifests
(hashes, no state values) of the other Can suites are read for the disjointness check, and the
locked held-out suite is not read at all (its disjointness by seed block is a research-integrity
test).
"""

from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.paths import evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import environment_snapshot, git_revision, write_json  # noqa: E402
from roboforge.replication import x3_states as x3  # noqa: E402
from roboforge.sim import eval_states as es  # noqa: E402
from roboforge.sim import robustness as rb  # noqa: E402


def disjointness(task: str, art: es.SuiteArtifacts) -> dict[str, Any]:
    """X3 starts vs the validation and robustness starts (open manifests, by state hash)."""
    mine = {s.state_sha256 for s in art.states}
    out = {}
    for suite in ("validation", *rb.sets_for(task)):
        other = {s["state_sha256"] for s in es.read_manifest(task, suite)["states"]}
        out[f"vs_{suite}"] = not (mine & other)
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="regenerate in memory and compare")
    args = p.parse_args(argv)
    task = x3.suite_task()
    if args.check:
        problems, details = x3.check_x3()
        print(f"[{task}/{x3.suite_name()}] X3 check: {'PASS' if not problems else 'FAIL'} "
              f"{json.dumps(details)}")
        for prob in problems:
            print("   ", prob)
        return 0 if not problems else 1
    from roboforge.sim.factory import load_sim_config, make_env

    env = make_env(load_sim_config(task))
    try:
        art = x3.generate_x3(env)
    finally:
        env.close()
    disjoint = disjointness(task, art)
    if not all(disjoint.values()):
        print(f"[{task}/{art.suite}] NOT disjoint: {disjoint}; nothing written")
        return 1
    paths = es.write_suite(art)
    m = art.manifest
    rep = {"schema": "roboforge.x3_states_report/1", "label": x3.LABEL,
           "decision": x3.DECISION, "task": task, "suite": art.suite,
           "n_states": m["n_states"], "content_sha256": m["content_sha256"],
           "seeds": [m["seed_base"], m["seed_base"] + m["n_states"]],
           "reserved_seeds": m["reserved_seeds"], "partition": m["partition"],
           "n_unique_models": m["n_unique_models"], "generated_on": m["generated_on"],
           "config_sha256": m["config_sha256"], "files": paths, "disjoint": disjoint,
           "git": git_revision(), "environment": environment_snapshot()}
    out = evidence_reports_dir("x3") / "states.json"
    write_json(out, rep)
    print(f"[{task}/{art.suite}] {m['n_states']} states, content {m['content_sha256']} -> "
          f"{paths}; disjoint {disjoint}; report {repo_relative(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Generate (or ``--check``) the X2 evaluation suites (DECISIONS D-123; a separate study).

Writes ``data/eval_states/can/{x2_validation,x2_heldout}.{json,data.jsonl.xz}``: fresh Can
starts from robosuite's standard reset distribution (``configs/extensions/x2_suites.yaml``),
plus a report ``reports/session<N>/x2/states.json`` with the counts, hashes and disjointness
checks. Generation is canonical only on the evaluation container (D-017)::

    ROBOFORGE_WRITE_EVAL_STATES=1 ROBOFORGE_EVIDENCE_SESSION=15 \\
        infra/containers/run_sim_cpu.sh python scripts/generate_x2_states.py

``--check`` regenerates both suites in memory and compares them with the committed files
(bitwise on the generating platform, within 1e-12 elsewhere); it writes nothing. Only open
manifests (hashes, no state values) of the other Can suites are read for the disjointness
check; the locked held-out suite is not read here (its disjointness is a research-integrity
test). Nothing is ever rolled out on a suite by this script.
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
from roboforge.replication import x3_states  # noqa: E402
from roboforge.sim import eval_states as es  # noqa: E402
from roboforge.sim import robustness as rb  # noqa: E402
from roboforge.x2 import states as x2s  # noqa: E402

SUITES = (x2s.VALIDATION_SUITE, x2s.HELDOUT_SUITE)


def disjointness(task: str, arts: dict[str, es.SuiteArtifacts]) -> dict[str, Any]:
    """Each X2 suite vs the other committed Can suites (open manifests) and vs each other."""
    others = ("validation", *rb.sets_for(task), x3_states.suite_name())
    out: dict[str, Any] = {}
    for name, art in arts.items():
        mine = {s.state_sha256 for s in art.states}
        rec = {}
        for suite in others:
            other = {s["state_sha256"] for s in es.read_manifest(task, suite)["states"]}
            rec[f"vs_{suite}"] = not (mine & other)
        for o, oart in arts.items():
            if o != name:
                rec[f"vs_{o}"] = not (mine & {s.state_sha256 for s in oart.states})
        out[name] = rec
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="regenerate in memory and compare")
    args = p.parse_args(argv)
    task = x2s.suite_task()
    if args.check:
        ok = True
        for name in SUITES:
            problems, details = x2s.check_x2_suite(name)
            ok = ok and not problems
            print(f"[{task}/{name}] X2 check: {'PASS' if not problems else 'FAIL'} "
                  f"{json.dumps(details)}")
            for prob in problems:
                print("   ", prob)
        return 0 if ok else 1
    from roboforge.sim.factory import load_sim_config, make_env

    env = make_env(load_sim_config(task))
    try:
        arts = {name: x2s.generate_x2_suite(env, name) for name in SUITES}
    finally:
        env.close()
    disjoint = disjointness(task, arts)
    if not all(v for rec in disjoint.values() for v in rec.values()):
        print(f"[{task}] X2 suites NOT disjoint: {disjoint}; nothing written")
        return 1
    files = {name: es.write_suite(art) for name, art in arts.items()}
    rep = {"schema": "roboforge.x2_states_report/1", "decision": x2s.DECISION, "task": task,
           "suites": {name: {"n_states": a.manifest["n_states"],
                             "content_sha256": a.manifest["content_sha256"],
                             "seeds": [a.manifest["seed_base"],
                                       a.manifest["seed_base"] + a.manifest["n_states"]],
                             "reserved_seeds": a.manifest["reserved_seeds"],
                             "partition": a.manifest["partition"],
                             "n_unique_models": a.manifest["n_unique_models"],
                             "generated_on": a.manifest["generated_on"],
                             "files": files[name]} for name, a in arts.items()},
           "config_sha256": x2s.config_sha256(), "disjoint": disjoint,
           "git": git_revision(), "environment": environment_snapshot()}
    out = evidence_reports_dir("x2") / "states.json"
    write_json(out, rep)
    for name, a in arts.items():
        print(f"[{task}/{name}] {a.manifest['n_states']} states, content "
              f"{a.manifest['content_sha256']} -> {files[name]}")
    print(f"disjoint {json.dumps(disjoint)}; report {repo_relative(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""X1: evaluate one frozen BC checkpoint on one robustness set with the action clamp (D-107).

    ROBOFORGE_EVIDENCE_SESSION=13 infra/containers/run_sim_cpu.sh python scripts/eval_x1_clamp.py \
        --train-run runs/<train_run_id> --step N --suite ood_arm_offset \
        --family bc_mlp_chunk --expected-sha256 <checkpoint sha256 from the test configuration>

**Exploratory, after the freeze.** The script runs inside the canonical container. The host
orchestrator (``scripts/run_x1_clamp.py``) launches it with the checkpoint hash and family read
from the frozen test configuration records. The container does not mount ``reports/session6``
or ``reports/session7``.

Records go to ``runs/x1_clamp/<eval_id>/`` and the summary to
``reports/session<N>/x1_clamp/evals/<eval_id>.json``. Neither path is ever overwritten.

``--no-clamp`` is the engineering reproduction of a Session 7 summary. Its records go to
``runs/x1_checks/`` and its summary to ``logs/session<N>/x1_checks/``, where the analysis
never looks. Only the task's robustness sets are accepted.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.exploratory.clamp import evaluate_x1  # noqa: E402
from roboforge.paths import REPO_ROOT, evidence_reports_dir, evidence_session  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-run", type=Path, required=True, help="runs/<train_run_id>")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--suite", required=True, help="a robustness set of the task")
    p.add_argument("--family", required=True,
                   choices=["bc_mlp", "bc_rnn", "bc_mlp_chunk"])
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--no-clamp", action="store_true",
                   help="engineering reproduction: the wrapper with the clamp disabled")
    p.add_argument("--verify", type=int, default=3, help="re-simulate the first N episodes")
    p.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    a = p.parse_args(argv)
    clamp = not a.no_clamp
    out_dir = (evidence_reports_dir("x1_clamp", "evals") if clamp else
               REPO_ROOT / "logs" / f"session{evidence_session()}" / "x1_checks")
    s = evaluate_x1(a.train_run, a.step, a.suite, family=a.family,
                    expected_sha256=a.expected_sha256, clamp=clamp, runs_dir=a.runs_dir,
                    n_verify=a.verify)
    out = out_dir / f"{s['eval_run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as fh:  # exclusive create: a summary is written once
        fh.write(json.dumps(s, indent=1, allow_nan=False) + "\n")
    c = s["x1_clamp"]["counts"]
    print(f"[{s['task']} {s['family']} step {a.step} {s['suite']} clamp={clamp}] success "
          f"{s['successes']}/{s['n_episodes']}; errors {s['n_errors']}; clamped actions "
          f"{c['n_actions_clamped']}/{c['n_actions_decoded']}; audit "
          f"{'PASS' if s['verify_passed'] else 'FAIL'}; {s['label']} -> {out}")
    return 0 if s["verify_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

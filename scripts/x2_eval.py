"""X2: evaluate one X2 checkpoint on the X2 validation starts, in the container (D-124).

Runs on the canonical container (``infra/containers/run_sim_cpu.sh``), inside
:func:`roboforge.x2.binding.x2_bindings` (X2 contract, 500-step episodes). Admits only the X2
validation suite. Records go to ``runs/x2/evals/<eval_id>/``; the summary to
``reports/session<N>/x2/pilot/evals/<eval_id>.json``::

    infra/containers/run_sim_cpu.sh python scripts/x2_eval.py \\
        --train-run runs/x2/<run_id> --step 20000 --family bc_mlp --expected-sha256 <sha>
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import write_json  # noqa: E402
from roboforge.x2.binding import x2_bindings  # noqa: E402
from roboforge.x2.evaluate import evaluate_x2  # noqa: E402
from roboforge.x2.pilot import FAMILIES  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-run", type=Path, required=True)
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    args = p.parse_args(argv)
    with x2_bindings():
        s = evaluate_x2(args.train_run, args.step, family=args.family,
                        expected_sha256=args.expected_sha256, runs_dir=args.runs_dir)
    out = evidence_reports_dir("x2", "pilot", "evals") / f"{s['eval_run_id']}.json"
    write_json(out, s)
    print(json.dumps({"eval_run_id": s["eval_run_id"], "successes": s["successes"],
                      "n": s["n_episodes"], "n_errors": s["n_errors"],
                      "canonical": s["canonical_platform"], "verify": s["verify_passed"],
                      "wall_s": round(s["wall_s"], 1), "summary": repo_relative(out)}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

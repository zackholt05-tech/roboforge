"""Closed-loop VALIDATION evaluation of a training run's checkpoints (RESEARCH_PROTOCOL §7).

    infra/containers/run_sim_cpu.sh python scripts/eval_policy.py --train-run runs/<id>

Evaluates the chosen checkpoints (default: all) of a completed training run on the 50
validation starts of its task. There is deliberately no suite option. Each checkpoint's
SHA-256 is verified against the training manifest before loading. Every episode is recorded
under ``runs/<eval_run_id>/``. The summary goes to ``<reports-dir>/validation/`` on the
canonical platform (D-017), or to ``<reports-dir>/dev/`` labeled DEVELOPMENT ONLY anywhere
else. Run with ``ROBOFORGE_EVIDENCE_SESSION=3`` so the container may write
``reports/session3``.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.eval.validation import evaluate_checkpoint  # noqa: E402
from roboforge.paths import REPO_ROOT  # noqa: E402
from roboforge.train.run import read_manifest  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-run", type=Path, required=True, help="runs/<train_run_id>")
    p.add_argument("--steps", nargs="+", default=["all"], help="checkpoint steps, or 'all'")
    p.add_argument("--n-states", type=int, default=None,
                   help="first N validation starts only (partial: never eligible for selection)")
    p.add_argument("--verify", type=int, default=3, help="re-simulate the first N episodes")
    p.add_argument("--num-threads", type=int, default=1)
    p.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    p.add_argument("--reports-dir", type=Path, default=REPO_ROOT / "reports" / "session3")
    args = p.parse_args(argv)
    tm = read_manifest(args.train_run)
    steps = ([c["step"] for c in tm["checkpoints"]] if args.steps == ["all"]
             else [int(s) for s in args.steps])
    ok = True
    for step in steps:
        s = evaluate_checkpoint(args.train_run, step, runs_dir=args.runs_dir,
                                n_states=args.n_states, n_verify=args.verify,
                                num_threads=args.num_threads)
        sub = "validation" if s["canonical_platform"] else "dev"
        out = args.reports_dir / sub / f"{s['eval_run_id']}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(s, indent=1, allow_nan=False) + "\n")
        ok &= s["verify_passed"]
        print(f"[{s['task']} {tm['config']['name']} step {step}] success "
              f"{s['successes']}/{s['n_episodes']} (Wilson95 {s['wilson95'][0]:.2f}-"
              f"{s['wilson95'][1]:.2f}); first success mean {s['first_success_step']['mean']}; "
              f"stop {s['stop_reasons']}; errors {s['n_errors']}; clip/dim "
              f"{[round(c, 3) for c in s['clip_fraction_per_dim']]}; audit "
              f"{'PASS' if s['verify_passed'] else 'FAIL'}; {s['label']} -> {out}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

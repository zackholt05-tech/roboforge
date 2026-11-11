"""X3: evaluate one Can checkpoint on the X3 held-out starts, once (D-111).

    ROBOFORGE_EVIDENCE_SESSION=14 infra/containers/run_sim_cpu.sh python scripts/eval_x3.py \
        --train-run runs/<train_run_id> --step N --family diffusion_unet --seed 3 \
        --expected-sha256 <checkpoint sha256 from the freeze or the committed selection>

**Separate replication.** The script runs inside the canonical container. The host
orchestrator (``scripts/run_x3.py``) launches it with the checkpoint hash read from the
committed freeze (seeds 0–2) or from the committed selection record (seeds 3–5).

Records go to ``runs/x3/<eval_id>/`` and the summary to
``reports/session<N>/x3/evals/<eval_id>.json``. Neither is ever overwritten.

``--check`` is the engineering reproduction of a committed Session 7 validation evaluation. It
admits only the validation suite; its records go to ``runs/x3_checks/`` and its summary to
``logs/session<N>/x3_checks/``, where the analysis never looks.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.eval.freeze import FAMILY_ORDER  # noqa: E402
from roboforge.paths import REPO_ROOT, evidence_reports_dir, evidence_session  # noqa: E402
from roboforge.replication import x3_states  # noqa: E402
from roboforge.replication.x3 import evaluate_x3  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-run", type=Path, required=True, help="runs/<train_run_id>")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--family", required=True, choices=list(FAMILY_ORDER))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--check", action="store_true",
                   help="engineering reproduction on the validation suite (not a result)")
    p.add_argument("--verify", type=int, default=3, help="re-simulate the first N episodes")
    p.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    a = p.parse_args(argv)
    suite = "validation" if a.check else x3_states.suite_name()
    out_dir = (REPO_ROOT / "logs" / f"session{evidence_session()}" / "x3_checks" if a.check
               else evidence_reports_dir("x3", "evals"))
    s = evaluate_x3(a.train_run, a.step, suite, family=a.family, seed=a.seed,
                    expected_sha256=a.expected_sha256, runs_dir=a.runs_dir, check=a.check,
                    n_verify=a.verify)
    out = out_dir / f"{s['eval_run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as fh:  # exclusive create: a summary is written once
        fh.write(json.dumps(s, indent=1, allow_nan=False) + "\n")
    print(f"[{s['task']} {s['family']} seed {s['seed']} step {a.step} {s['suite']}] success "
          f"{s['successes']}/{s['n_episodes']}; errors {s['n_errors']}; audit "
          f"{'PASS' if s['verify_passed'] else 'FAIL'}; {s['label']} -> {out}")
    return 0 if s["verify_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

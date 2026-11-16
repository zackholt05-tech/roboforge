"""X2-c: evaluate one frozen X2 checkpoint on the X2 held-out starts, once (DECISIONS D-130).

Runs inside the canonical container (``infra/containers/run_sim_cpu.sh``), launched by the
host orchestrator ``scripts/x2c_run.py`` with the X2 freeze's committed git blob ID::

    ROBOFORGE_EVIDENCE_SESSION=17 infra/containers/run_sim_cpu.sh python scripts/x2c_eval.py \\
        --train-run runs/x2/<train_run_id> --step N --family diffusion_unet --seed 0 \\
        --expected-sha256 <sha256 from the X2 freeze> --freeze-blob <git blob ID>

**Refuses** before any rollout if the host tree is dirty, if ``docs/extensions/X2_FREEZE.md``
is not the committed file the host attests, if a file the freeze pins (or the Level 1 frozen
code) changed, if the checkpoint is not the one the freeze lists for this family and seed, if
its SHA-256 differs, or if this evaluation's summary or record directory already exists.
Records go to ``runs/x2c/<eval_id>/``, the summary to
``reports/session<N>/x2/heldout/evals/<eval_id>.json`` (exclusive create).

``--attempt 2 --rerun-decision D-NNN`` is the one engineering rerun of an evaluation that
failed before writing its summary, for a fault outside the policy, under that recorded
decision (records in ``runs/x2c/attempt2/``).

``--check`` is the engineering reproduction of an X2-b validation evaluation, before the
freeze: X2 validation starts only, records in ``runs/x2c_checks/``, summary in
``logs/session<N>/x2c_checks/``; never a result.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.eval import freeze as level1  # noqa: E402
from roboforge.eval.freeze import FAMILY_ORDER  # noqa: E402
from roboforge.paths import REPO_ROOT, evidence_reports_dir, evidence_session  # noqa: E402
from roboforge.provenance import git_revision  # noqa: E402
from roboforge.x2 import heldout as xh  # noqa: E402
from roboforge.x2.binding import x2_bindings  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-run", type=Path, required=True, help="runs/x2/<train_run_id>")
    p.add_argument("--step", type=int, required=True)
    p.add_argument("--family", required=True, choices=list(FAMILY_ORDER))
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--expected-sha256", required=True)
    p.add_argument("--freeze-blob", help="git blob ID of the committed X2 freeze (host)")
    p.add_argument("--attempt", type=int, choices=(1, 2), default=1)
    p.add_argument("--rerun-decision")
    p.add_argument("--check", action="store_true",
                   help="engineering reproduction on X2 validation starts (not a result)")
    p.add_argument("--verify", type=int, default=3, help="re-simulate the first N episodes")
    p.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    a = p.parse_args(argv)
    if git_revision()["dirty"] is not False:
        raise SystemExit("the host tree is dirty (or unknown): refusing to evaluate")
    bad = level1.code_problems(level1.load_freeze())
    if bad:
        raise SystemExit(f"Level 1 frozen code changed: {bad}")
    if a.check:
        if a.freeze_blob or a.attempt != 1 or a.rerun_decision:
            p.error("--check takes no --freeze-blob, --attempt or --rerun-decision")
        freeze, out_dir = None, REPO_ROOT / "logs" / f"session{evidence_session()}" / "x2c_checks"
    else:
        if not a.freeze_blob:
            p.error("a held-out evaluation needs --freeze-blob")
        try:
            freeze = xh.load_x2_freeze(expected_blob=a.freeze_blob)
        except xh.X2FreezeError as e:
            raise SystemExit(f"X2 freeze: {e}") from e
        bad = xh.code_problems(freeze)
        if bad:
            raise SystemExit(f"files pinned by the X2 freeze changed: {bad}")
        out_dir = evidence_reports_dir(*xh.SUMMARY_PARTS)
        if out_dir != REPO_ROOT / freeze["outputs"]["summaries"]:
            raise SystemExit(f"session {evidence_session()}: the X2 freeze puts summaries in "
                             f"{freeze['outputs']['summaries']}")
        job = xh.find_job(freeze, a.train_run.name, a.step, a.expected_sha256, a.family, a.seed)
        if (out_dir / f"{job['eval_run_id']}.json").exists():
            raise SystemExit(f"{job['eval_run_id']}: a summary exists (X2-c runs once)")
    with x2_bindings():
        s = xh.evaluate_x2c(a.train_run, a.step, family=a.family, seed=a.seed,
                            expected_sha256=a.expected_sha256, runs_dir=a.runs_dir,
                            freeze=freeze, freeze_git_blob=a.freeze_blob, check=a.check,
                            attempt=a.attempt, rerun_decision=a.rerun_decision,
                            n_verify=a.verify)
    out = out_dir / f"{s['eval_run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as fh:  # exclusive create: a summary is written once
        fh.write(json.dumps(s, indent=1, allow_nan=False) + "\n")
    print(f"[{s['family']} seed {s['seed']} step {a.step} {s['suite']}] success "
          f"{s['successes']}/{s['n_episodes']}; errors {s['n_errors']}; audit "
          f"{'PASS' if s['verify_passed'] else 'FAIL'}; {s['label']} -> {out}")
    return 0 if s["verify_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

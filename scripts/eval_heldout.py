"""Run ONE frozen held-out evaluation inside the canonical container (RESEARCH_PROTOCOL §10).

    infra/containers/run_sim_cpu.sh python scripts/eval_heldout.py --task lift --index 0

``--index`` picks the entry of ``roboforge.eval.freeze.entries(freeze, task)`` (12 frozen
checkpoints, then the random baseline). The summary goes to
``reports/session<N>/heldout/<eval_run_id>.json`` and is never overwritten. Normally launched
by ``scripts/run_heldout.py``. ``--rehearsal-suite`` + ``--draft-freeze`` run the identical
path on a non-locked suite before the freeze (output under ``logs/session<N>/rehearsal/``).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.eval import freeze as fz  # noqa: E402
from roboforge.eval.heldout import evaluate_entry  # noqa: E402
from roboforge.paths import REPO_ROOT, evidence_reports_dir, evidence_session  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", required=True, choices=["lift", "can"])
    p.add_argument("--index", type=int, required=True)
    p.add_argument("--rehearsal-suite", default=None)
    p.add_argument("--draft-freeze", type=Path, default=None)
    p.add_argument("--engineering-rerun", default=None, metavar="REASON")
    a = p.parse_args(argv)
    rehearsal = a.rehearsal_suite is not None
    frozen = (fz.parse_freeze(a.draft_freeze.read_text()) if rehearsal else fz.load_freeze())
    entry = fz.entries(frozen, a.task)[a.index]
    if rehearsal:
        out_dir = REPO_ROOT / "logs" / f"session{evidence_session()}" / "rehearsal" / "heldout"
        runs_dir = out_dir / "runs"
    else:
        out_dir, runs_dir = evidence_reports_dir("heldout"), REPO_ROOT / "runs"
    s = evaluate_entry(entry, runs_dir=runs_dir, rehearsal_suite=a.rehearsal_suite,
                       draft_freeze=a.draft_freeze, engineering_rerun=a.engineering_rerun)
    out = out_dir / f"{s['eval_run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as f:  # never overwrite a held-out summary
        f.write(json.dumps(s, indent=1, allow_nan=False) + "\n")
    print(f"[{a.task} #{a.index} {entry['family']} seed {entry.get('seed', '-')}] "
          f"{s['successes']}/{s['n_episodes']} on {s['suite']}; errors {s['n_errors']}; "
          f"audit {'PASS' if s['verify_passed'] else 'FAIL'}; {s['kind']} -> {out}")
    return 0 if s["verify_passed"] else 1


if __name__ == "__main__":
    sys.exit(main())

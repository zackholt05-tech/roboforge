"""Apply the RESEARCH_PROTOCOL §7 checkpoint-selection rule to one training run.

    python scripts/select_checkpoint.py --train-run runs/<id>

Reads the run's immutable manifest and the canonical validation summaries of every
checkpoint (``<reports-dir>/validation/<train_run_id>__val_step*.json``). It refuses
missing, partial, duplicate or non-canonical evaluations, then writes the selection record
to ``<reports-dir>/selection/<train_run_id>.json``. Development-only summaries are never
read.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.eval.selection import select_checkpoint
from roboforge.paths import evidence_reports_dir, repo_relative
from roboforge.provenance import git_revision, sha256_file
from roboforge.train.run import read_manifest


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--train-run", type=Path, required=True)
    p.add_argument("--reports-dir", type=Path, default=None,
                   help="default: reports/session$ROBOFORGE_EVIDENCE_SESSION")
    args = p.parse_args(argv)
    args.reports_dir = args.reports_dir or evidence_reports_dir()
    tm = read_manifest(args.train_run)
    files = sorted((args.reports_dir / "validation").glob(f"{tm['run_id']}__val_step*.json"))
    evals = [json.loads(f.read_text()) for f in files]
    rec = select_checkpoint(tm, evals)
    rec["sources"] = [{"path": repo_relative(f), "sha256": sha256_file(f)} for f in files]
    rec["git"] = git_revision()
    out = args.reports_dir / "selection" / f"{tm['run_id']}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(rec, indent=1, allow_nan=False) + "\n")
    c = rec["chosen"]
    print(f"[{tm['run_id']}] chosen step {c['step']} ({c['successes']}/{c['n_episodes']}, mean "
          f"first success {c['first_success_step_mean']}); {rec['n_tied_on_success']} "
          f"checkpoint(s) tied on success -> {repo_relative(out)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Hash-validated retrieval of a remotely trained run into the local ``runs/`` (Session 5).

A run trained on Modal is downloaded from the ``roboforge-runs`` Volume into a staging
directory (``modal volume get``). The same applies to the local simulation's Volume-shaped
directory. It is installed into ``runs/<run_id>/`` only if every check passes:

* the manifest exists, is ``complete``, names this run ID and passed its reload check;
* every cadence checkpoint in the manifest exists and its SHA-256 matches. The manifest's
  list equals the journal index ``checkpoints.jsonl``. The paths are repo-relative
  ``runs/<run_id>/...``, so they resolve unchanged under the local repository;
* every resume checkpoint listed in the manifest matches its SHA-256;
* ``metrics.jsonl`` matches ``metrics_sha256``;
* the stored ``config.json`` equals the manifest config.

The target must not exist (runs are never overwritten). Files are copied, then re-hashed at
the target, before the result is reported. Nothing is deleted from the staging directory.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT
from roboforge.provenance import sha256_file


class RetrievalError(ValueError):
    """A staged run that fails a retrieval check."""


def _rel_in_run(path: str, run_id: str) -> Path:
    p = Path(path)
    parts = p.parts
    if len(parts) < 3 or parts[0] != "runs" or parts[1] != run_id:
        raise RetrievalError(f"{path}: not a repo-relative path inside runs/{run_id}/")
    return Path(*parts[2:])


def verify_run_dir(run_dir: Path, run_id: str) -> dict[str, Any]:
    """Check one run directory (staged or installed); return the verified hashes."""
    run_dir = Path(run_dir)
    mp = run_dir / "manifest.json"
    if not mp.is_file():
        raise RetrievalError(f"{run_dir}: no manifest.json (the run did not finish)")
    m = json.loads(mp.read_text())
    if m.get("run_id") != run_id or m.get("status") != "complete":
        raise RetrievalError(f"{run_dir}: manifest run_id={m.get('run_id')} "
                             f"status={m.get('status')}")
    if not m.get("reload_check", {}).get("passed"):
        raise RetrievalError(f"{run_dir}: reload check did not pass")
    if json.loads((run_dir / "config.json").read_text()) != m["config"]:
        raise RetrievalError(f"{run_dir}: config.json differs from the manifest config")
    index = [json.loads(x) for x in (run_dir / "checkpoints.jsonl").read_text().splitlines()
             if x.strip()]
    if sorted((r["step"], r["path"], r["sha256"]) for r in index) != \
            [(c["step"], c["path"], c["sha256"]) for c in m["checkpoints"]]:
        raise RetrievalError(f"{run_dir}: checkpoints.jsonl differs from the manifest")
    checked = {}
    for c in list(m["checkpoints"]) + list(m.get("resume_checkpoints", [])):
        f = run_dir / _rel_in_run(c["path"], run_id)
        if not f.is_file():
            raise RetrievalError(f"missing {c['path']}")
        h = sha256_file(f)
        if h != c["sha256"]:
            raise RetrievalError(f"{c['path']}: SHA-256 {h} != manifest {c['sha256']}")
        checked[c["path"]] = h
    mh = sha256_file(run_dir / "metrics.jsonl")
    if mh != m["metrics_sha256"]:
        raise RetrievalError(f"metrics.jsonl: SHA-256 {mh} != manifest {m['metrics_sha256']}")
    return {"run_id": run_id, "manifest_sha256": sha256_file(mp), "files": checked,
            "metrics_sha256": mh, "n_checkpoints": len(m["checkpoints"]),
            "n_resume_checkpoints": len(m.get("resume_checkpoints", []))}


def install(staged: Path, run_id: str, runs_dir: Path = REPO_ROOT / "runs") -> dict[str, Any]:
    """Verify ``staged`` (a copy of ``runs/<run_id>/``) and copy it to ``runs_dir/<run_id>``."""
    before = verify_run_dir(staged, run_id)
    target = Path(runs_dir) / run_id
    if target.exists():
        raise RetrievalError(f"{target} exists; runs are never overwritten")
    shutil.copytree(staged, target)
    after = verify_run_dir(target, run_id)
    if after["files"] != before["files"] or after["manifest_sha256"] != before["manifest_sha256"]:
        raise RetrievalError("hashes changed during the copy")
    return {"installed": str(target), "verified_staged": before, "verified_installed": after,
            "status": "PASS"}


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m roboforge.compute.retrieve --staged DIR --run-id ID [--runs-dir runs]``."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--staged", type=Path, required=True)
    p.add_argument("--run-id", required=True)
    p.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    p.add_argument("--verify-only", action="store_true")
    p.add_argument("--report", type=Path)
    args = p.parse_args(argv)
    try:
        out = (verify_run_dir(args.staged, args.run_id) if args.verify_only
               else install(args.staged, args.run_id, args.runs_dir))
    except RetrievalError as exc:
        print(f"RETRIEVAL FAILED: {exc}", file=sys.stderr)
        return 2
    text = json.dumps(out, indent=1, sort_keys=True)
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(text + "\n")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())

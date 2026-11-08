"""Check a held-out-pipeline rehearsal on the VALIDATION suite against the canonical
validation evaluations of the same checkpoints (D-067, D-069).

    uv run --frozen python scripts/check_rehearsal.py \
        --draft-freeze logs/session7/freeze_prep/draft_freeze.md \
        --rehearsal-dir logs/session7/rehearsal/heldout \
        --validation lift=reports/session6/validation --validation can=reports/session7/validation

Evaluation is deterministic on the canonical platform, so the rehearsal of every frozen
checkpoint must reproduce its validation evaluation episode by episode: the same states,
success, first-success step, step count, stop reason, noise seed and physics hash. That
one comparison checks the frozen-entry lookup, the checkpoint loading, the stochastic
noise-seed rule and the shared episode loop of the held-out path. The random baseline
has no reference; it must cover every state, pass its audit and record no errors.
Writes a JSON report (``--out``) and exits non-zero on any mismatch.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from roboforge.eval import freeze as fz
from roboforge.eval.heldout import eval_id
from roboforge.eval.validation import eval_run_id
from roboforge.paths import REPO_ROOT, repo_relative

PLATFORM = "linux-aarch64-container-1f685638f28c"
KEYS = ("state_id", "success", "first_success_step", "steps", "stop_reason", "error",
        "physics_sha256", "noise_seed")


def compare(reh: dict[str, Any], ref: dict[str, Any] | None) -> list[str]:
    """Mismatches between a rehearsal summary and its validation reference."""
    out = []
    if not reh["verify_passed"]:
        out.append("rehearsal audit failed")
    if ref is None:
        if reh["n_errors"]:
            out.append(f"{reh['n_errors']} episode errors")
        return out
    a = {e["state_id"]: e for e in reh["episodes"]}
    b = {e["state_id"]: e for e in ref["episodes"]}
    if set(a) != set(b):
        return [*out, "different states"]
    for sid in sorted(a):
        for k in KEYS:
            if a[sid].get(k) != b[sid].get(k):
                out.append(f"{sid} {k}: {a[sid].get(k)!r} != {b[sid].get(k)!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--draft-freeze", type=Path, required=True)
    p.add_argument("--rehearsal-dir", type=Path, required=True)
    p.add_argument("--validation", action="append", required=True, metavar="TASK=DIR")
    p.add_argument("--out", type=Path, default=None)
    a = p.parse_args(argv)
    frozen = fz.parse_freeze(a.draft_freeze.read_text())
    vdirs = {k: REPO_ROOT / v for k, v in (x.split("=", 1) for x in a.validation)}
    rows, bad = [], 0
    for task in frozen["tasks"]:
        for e in fz.entries(frozen, task):
            rp = a.rehearsal_dir / f"{eval_id(e, PLATFORM, rehearsal=True)}.json"
            if not rp.exists():
                rows.append({"task": task, "family": e["family"], "seed": e.get("seed"),
                             "problems": ["rehearsal summary missing"]})
                bad += 1
                continue
            reh = json.loads(rp.read_text())
            ref = None
            if e["family"] != fz.BASELINE:
                vid = eval_run_id(e["train_run_id"], int(e["step"]), PLATFORM,
                                  stochastic=e["family"] == "diffusion_unet")
                ref = json.loads((vdirs[task] / f"{vid}.json").read_text())
            probs = compare(reh, ref)
            bad += bool(probs)
            rows.append({"task": task, "family": e["family"], "seed": e.get("seed"),
                         "rehearsal": repo_relative(rp), "successes": reh["successes"],
                         "reference_successes": None if ref is None else ref["successes"],
                         "n_episodes": reh["n_episodes"], "problems": probs[:20],
                         "n_problems": len(probs)})
            print(f"{task} {e['family']:15s} seed {e.get('seed', '-')}: "
                  f"{reh['successes']}/{reh['n_episodes']}"
                  + ("" if ref is None else f" (validation {ref['successes']})")
                  + (" OK" if not probs else f" MISMATCH x{len(probs)}: {probs[:2]}"))
    report = {"schema": "roboforge.rehearsal_check/1", "draft_freeze": repo_relative(
        a.draft_freeze), "rehearsal_dir": repo_relative(a.rehearsal_dir), "rows": rows,
        "mismatching_entries": bad, "passed": bad == 0}
    if a.out:
        a.out.write_text(json.dumps(report, indent=1) + "\n")
    print("PASS" if bad == 0 else f"FAIL: {bad} entries")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

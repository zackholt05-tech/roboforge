"""RGB study: the one-shot held-out test of one selected checkpoint (RGB freeze §9; D-081).

    # canonical container, from a clean committed tree; <= 4 containers at once
    make rgb-heldout RGB_RUN=<run_id>
    # rehearsal of the identical path on the first N validation starts (labelled, not a result)
    make rgb-heldout-rehearse RGB_RUN=<run_id> RGB_N=2
    # gate status only (host or container)
    uv run --frozen python scripts/run_rgb_heldout.py gate

``run`` writes ``reports/session<N>/rgb_heldout/<run_id>.json`` (the summary with its
per-episode rows) and the records ``runs/rgb_heldout/<eval_run_id>/``. ``rehearse`` writes
to ``rgb_heldout_rehearsal`` instead. Nothing is overwritten. A second held-out run of a
checkpoint is refused. The only exception is an engineering rerun, under the rule in
:mod:`roboforge.eval.rgb_heldout`.
"""

from __future__ import annotations

import argparse
import json
import sys

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.eval import rgb_heldout as rh  # noqa: E402
from roboforge.paths import evidence_reports_dir, repo_relative  # noqa: E402
from roboforge.provenance import sha256_file, write_json  # noqa: E402

BRIEF = ("eval_run_id", "suite", "successes", "n_episodes", "n_errors", "verify_passed",
         "wall_s", "latency_ms_median_of_episode_medians")


def run(run_id: str, *, rehearsal_n: int | None, engineering_rerun: str | None) -> int:
    """Evaluate one selected checkpoint and write its report."""
    kind = "rgb_heldout_rehearsal" if rehearsal_n is not None else "rgb_heldout"
    dest = evidence_reports_dir(kind) / f"{run_id}.json"
    if dest.exists():
        raise SystemExit(f"{repo_relative(dest)} exists: never overwritten")
    s = rh.evaluate_selected(run_id, rehearsal_n=rehearsal_n,
                             engineering_rerun=engineering_rerun)
    records = sha256_file(rh.REPO_ROOT / s["records_dir"] / "manifest.json")
    write_json(dest, s | {"records_manifest_sha256": records})
    print(json.dumps({k: s[k] for k in BRIEF} | {"report": repo_relative(dest)}), flush=True)
    return 0


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--run-id", required=True, choices=sorted(rh.SELECTIONS))
    r.add_argument("--engineering-rerun", metavar="REASON")
    h = sub.add_parser("rehearse")
    h.add_argument("--run-id", required=True, choices=sorted(rh.SELECTIONS))
    h.add_argument("--n-states", type=int, default=2)
    sub.add_parser("gate")
    a = p.parse_args(argv)
    if a.cmd == "gate":
        ok, problems = rh.gate_status()
        print(json.dumps({"open": ok, "problems": problems}, indent=1))
        return 0 if ok else 1
    if a.cmd == "rehearse":
        return run(a.run_id, rehearsal_n=a.n_states, engineering_rerun=None)
    return run(a.run_id, rehearsal_n=None, engineering_rerun=a.engineering_rerun)


if __name__ == "__main__":
    sys.exit(main())

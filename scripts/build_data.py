"""Build (or check) the canonical Session 1 data pipeline artifacts for Lift and Can.

For each task, the robomimic files are hash-verified and features are recomputed from the
saved MuJoCo states. They are then verified against the dataset obs blocks, and the
pipeline writes:

* ``$ROBOFORGE_DATA_ROOT/processed/robomimic_v1.5_ph/<task>/lowdim_v1.hdf5`` (gitignored),
  only after the rebuild has been shown to equal the committed artifacts.

The committed artifacts are ``data/splits/<task>_ph.json``,
``data/normalization/<task>_ph_lowdim_v1.json`` and ``data/manifests/tasks/<task>_ph.json``
(byte-deterministic). The flags control them:

* ``--check`` rebuilds everything in memory, writes nothing and exits non-zero if any
  committed artifact differs. It is the determinism contract that ``make verify-data``
  enforces.
* ``--update-committed`` (re)writes the committed artifacts. Use it only for an intentional
  data-contract change, recorded in docs/DECISIONS.md.

Usage::

    python scripts/build_data.py --tasks lift can                 # processed files only
    python scripts/build_data.py --tasks lift can --check
    python scripts/build_data.py --tasks lift can --update-committed
"""

from __future__ import annotations

import argparse
import sys
import time

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.data.pipeline import build_task, check_task, write_task  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--tasks", nargs="+", default=["lift", "can"])
    mode = p.add_mutually_exclusive_group()
    mode.add_argument("--check", action="store_true",
                      help="rebuild in memory and compare with committed artifacts; write nothing")
    mode.add_argument("--update-committed", action="store_true",
                      help="also (re)write the committed split/normalization/task manifests")
    args = p.parse_args(argv)
    failures = 0
    for task in args.tasks:
        t0 = time.perf_counter()
        art = build_task(task)
        m = art.task_manifest
        print(f"[{task}] built in {time.perf_counter() - t0:.1f} s: obs_dim "
              f"{m['observation_spec']['dim']}, action_dim {m['action_spec']['dim']}, "
              f"{len(art.split['splits']['train'])} train / {len(art.split['splits']['valid'])} "
              f"valid demos, flagged {sorted(art.issues) or 'none'}")
        print(f"   processed content {m['processed']['content_sha256']}")
        print(f"   split {art.split['content_sha256']}")
        print(f"   normalization {art.normalization['content_sha256']}")
        print(f"   observation_spec {m['observation_spec_sha256']}")
        for name, stats in art.verification.corrected.items():
            print(f"   corrected {name}: " + ", ".join(f"{k}={v:.3e}" for k, v in stats.items()))
        worst = max(art.verification.max_abs_vs_dataset.values())
        print(f"   max |feature - dataset| over relation=equal features: {worst:.3e}")
        if args.check:
            problems = check_task(art)
            for msg in problems:
                print(f"   CHECK FAIL: {msg}")
            print(f"   check: {'PASS' if not problems else 'FAIL'}")
            failures += bool(problems)
        else:
            for k, v in write_task(art, update_committed=args.update_committed).items():
                print(f"   wrote {k}: {v}")
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())

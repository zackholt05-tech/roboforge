"""X2: build (or ``--check``) the Can MH data artifacts (DECISIONS D-119 to D-121).

Builds, from the hash-verified MH low-dim file, the processed file (gitignored) and the
committed ``data/splits/can_mh.json``, ``data/normalization/can_mh_lowdim_v1.json`` and
``data/manifests/tasks/can_mh.json`` (:func:`roboforge.x2.data.build_x2`)::

    python scripts/x2_data.py --update-committed   # first build (writes the committed files)
    python scripts/x2_data.py                      # rebuild; refuses unless committed equal it
    python scripts/x2_data.py --check              # rebuild in memory and compare; writes nothing
"""

from __future__ import annotations

import argparse
import json
import sys

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.x2.data import build_x2, check_x2, write_x2  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--check", action="store_true", help="rebuild in memory and compare")
    p.add_argument("--update-committed", action="store_true",
                   help="write the committed JSON artifacts (an intentional, recorded change)")
    args = p.parse_args(argv)
    art = build_x2()
    m = art.task_manifest
    summary = {"n_episodes": len(m["episodes"]), "split": m["split"],
               "counts": art.split["counts"], "samples": art.split["samples"],
               "operators": m["operators"], "flagged": sorted(art.issues),
               "task_spec_sha256": m["task_spec_sha256"],
               "observation_spec_sha256": m["observation_spec_sha256"],
               "action_spec_sha256": m["action_spec_sha256"],
               "normalization_sha256": art.normalization["content_sha256"],
               "processed_content_sha256": m["processed"]["content_sha256"],
               "task_manifest_sha256": m["content_sha256"],
               "verification": m["verification"]}
    if args.check:
        problems = check_x2(art)
        print(json.dumps({**summary, "check": "PASS" if not problems else "FAIL",
                          "problems": problems}, indent=1))
        return 0 if not problems else 1
    written = write_x2(art, update_committed=args.update_committed)
    print(json.dumps({**summary, "written": written}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

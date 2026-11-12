"""X2: train one pilot run on the X2 (Can MH) data with the frozen trainer (D-121, D-124).

Resolves the family's run from ``configs/extensions/x2_pilot.yaml`` (the frozen chosen Can
configuration, seed 0, the frozen budget) and calls the frozen
:func:`roboforge.train.run.train_run` inside :func:`roboforge.x2.binding.x2_bindings`, so the
windows, normalization and recorded data hashes are X2's and everything else is the frozen
training path. Writes ``runs/x2/<run_id>/``. Signals stop the run after the current step with a
resume checkpoint (exit 3), as ``scripts/train.py`` does::

    python scripts/x2_train.py --family diffusion_unet
"""

from __future__ import annotations

import argparse
import json
import sys

from roboforge.paths import REPO_ROOT
from roboforge.train.run import train_run
from roboforge.train.stop import StopControl
from roboforge.x2.binding import x2_bindings
from roboforge.x2.pilot import FAMILIES, pilot_config, pilot_runs

EXIT_INTERRUPTED = 3
RUNS_DIR = REPO_ROOT / "runs" / "x2"


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--launch-id", help="launch nonce (default: a fresh one)")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    args = p.parse_args(argv)
    run = next(r for r in pilot_runs() if r.family == args.family)
    stop = StopControl()
    with stop.handle_signals(), x2_bindings():
        m = train_run(run.config, RUNS_DIR, pilot_config()["prefix"],
                      launch_id=args.launch_id, resume=args.resume, stop=stop)
    if m["status"] == "interrupted":
        print(json.dumps(m, indent=1))
        return EXIT_INTERRUPTED
    print(json.dumps({"run_id": m["run_id"], "status": m["status"],
                      "dataset_id": m["data"]["dataset_id"],
                      "steps_per_s": m["timing"]["steps_per_s"], "wall_s": m["timing"]["wall_s"],
                      "reload_check": m["reload_check"]["passed"],
                      "checkpoints": len(m["checkpoints"])}, indent=1))
    return 0 if m["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())

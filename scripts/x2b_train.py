"""X2-b: train one planned run on the X2 (Can MH) data with the frozen trainer (D-125 to D-127).

Resolves ``--family``/``--trial``/``--seed`` from ``configs/experiments/x2_can_mh.yaml`` with
the frozen suite loader and calls the frozen :func:`roboforge.train.run.train_run` inside
:func:`roboforge.x2.binding.x2_bindings`. A planned run that is an X2-a pilot run (D-125) is
refused: it is already trained. Writes ``runs/x2/<run_id>/``. Signals stop the run after the
current step with a resume checkpoint (exit 3), as ``scripts/train.py`` does::

    python scripts/x2b_train.py --family diffusion_unet --trial 1 --seed 0
"""

from __future__ import annotations

import argparse
import json
import sys

from roboforge.train.run import train_run
from roboforge.train.stop import StopControl
from roboforge.x2.binding import x2_bindings
from roboforge.x2.pilot import FAMILIES
from roboforge.x2.suite import RUNS, planned_run, x2b_plan

EXIT_INTERRUPTED = 3


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--family", choices=FAMILIES, required=True)
    p.add_argument("--trial", type=int, choices=(1, 2, 3, 4), required=True)
    p.add_argument("--seed", type=int, choices=(0, 1, 2), required=True)
    p.add_argument("--launch-id", help="launch nonce (default: a fresh one)")
    p.add_argument("--resume", action="store_true", help="continue an interrupted run")
    args = p.parse_args(argv)
    plan = x2b_plan()
    run = planned_run(plan, args.family, args.trial, args.seed)
    if run.pilot:
        print(f"{run.run_id} is an X2-a pilot run (D-125); it is not retrained", file=sys.stderr)
        return 2
    spec = plan.spec(args.family, args.trial, args.seed)
    stop = StopControl()
    with stop.handle_signals(), x2_bindings():
        m = train_run(spec.config, RUNS, plan.prefix, launch_id=args.launch_id,
                      resume=args.resume, stop=stop)
    if m["run_id"] != run.run_id:
        raise RuntimeError(f"trained {m['run_id']}, planned {run.run_id}")
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

"""Train one BC policy run on local CPU from a run config (RESEARCH_PROTOCOL §6).

    python scripts/train.py --config configs/models/bc_mlp.yaml --task lift --seed 0
    python scripts/train.py --config configs/models/bc_rnn.yaml --set train.max_steps=200

    python scripts/train.py --config configs/models/diffusion_unet.yaml --resume   # continue

Data come only from ``SequenceDataset.from_committed`` (hash-checked, train-only
normalization; the validation demos give offline losses only). Writes ``runs/<run_id>/``:
config, metrics, checkpoints, the run journal and an immutable manifest
(``roboforge.train.run``). Refuses to reuse a run directory created by another launch.
``--overfit-episodes`` trains on a few demos only, for the deliberate overfit check; the run
ID then carries ``_overfit``.

Interrupt safety (Session 5): SIGINT/SIGTERM and ``--max-wall-s`` stop the run after the
current optimizer step and write a resume checkpoint; the exit code is then 3 and no
manifest is written. ``--resume`` continues it, bitwise identical to an uninterrupted run on
the same platform. ``--stop-at-step`` interrupts at an exact step (engineering checks only).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from roboforge.paths import REPO_ROOT
from roboforge.train.config import load_config
from roboforge.train.run import train_run
from roboforge.train.stop import StopControl

EXIT_INTERRUPTED = 3


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--config", type=Path, required=True)
    p.add_argument("--task", choices=["lift", "can"], help="override the config's task")
    p.add_argument("--seed", type=int, help="override the config's training seed")
    p.add_argument("--set", dest="overrides", action="append", default=[],
                   metavar="SECTION.KEY=VALUE", help="override a config value (repeatable)")
    p.add_argument("--prefix", default="s3", help="run ID prefix (session tag)")
    p.add_argument("--runs-dir", type=Path, default=REPO_ROOT / "runs")
    p.add_argument("--overfit-episodes", nargs="+", metavar="DEMO",
                   help="train only on these training demos (deliberate overfit check)")
    p.add_argument("--resume", action="store_true",
                   help="continue an interrupted run (its directory must exist, no manifest)")
    p.add_argument("--launch-id", help="launch nonce (default: a fresh one); the same ID "
                                       "restarts its own launch")
    p.add_argument("--max-wall-s", type=float,
                   help="stop cleanly after this many seconds of this segment (resumable)")
    p.add_argument("--stop-at-step", type=int,
                   help="stop cleanly at this step (engineering interrupt checks only)")
    args = p.parse_args(argv)
    overrides = list(args.overrides)
    if args.task:
        overrides.append(f"task={args.task}")
    if args.seed is not None:
        overrides.append(f"seed={args.seed}")
    cfg = load_config(args.config, overrides)
    stop = StopControl(deadline_s=args.max_wall_s, stop_at_step=args.stop_at_step)
    with stop.handle_signals():
        m = train_run(cfg, args.runs_dir, args.prefix,
                      episodes=set(args.overfit_episodes) if args.overfit_episodes else None,
                      launch_id=args.launch_id, resume=args.resume, stop=stop)
    if m["status"] == "interrupted":
        print(json.dumps(m, indent=1))
        return EXIT_INTERRUPTED
    v = m["final_offline"]
    print(json.dumps({"run_id": m["run_id"], "status": m["status"],
                      "param_count": m["param_count"], "steps_per_s": m["timing"]["steps_per_s"],
                      "wall_s": m["timing"]["wall_s"], "train_mse": v["train"]["mse"],
                      "valid_mse": v["valid"]["mse"], "reload_check": m["reload_check"]["passed"],
                      "checkpoints": [c["path"] for c in m["checkpoints"]]}, indent=1))
    return 0 if m["status"] == "complete" else 1


if __name__ == "__main__":
    sys.exit(main())

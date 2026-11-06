"""CPU train/eval smoke: 200-step BC-MLP, BC-RNN and diffusion runs, each on 2 starts.

    python scripts/train_smoke.py --out-dir logs/tmp/train_smoke

Exercises the whole Session 3 loop on local CPU, then checks the results:
config → committed data → trainer → checkpoints → immutable manifest → reload check →
``LearnedPolicy`` → closed-loop rollouts in the pinned simulator → re-simulation audit.

The runs go to a fresh timestamped directory under ``--out-dir`` (gitignored
``logs/tmp/``). The evaluations are partial (2 validation starts) and, on macOS,
development-only. They are pipeline checks, never results.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

from roboforge.sim.backend import configure_mujoco_gl

configure_mujoco_gl()

from roboforge.eval.validation import evaluate_checkpoint  # noqa: E402
from roboforge.paths import REPO_ROOT  # noqa: E402
from roboforge.train.config import load_config  # noqa: E402
from roboforge.train.run import train_run  # noqa: E402


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "logs" / "tmp" / "train_smoke")
    p.add_argument("--steps", type=int, default=200)
    p.add_argument("--n-states", type=int, default=2)
    args = p.parse_args(argv)
    runs = args.out_dir / f"runs_{int(time.time())}"
    ok = True
    for name in ("bc_mlp", "bc_rnn", "diffusion_unet"):
        cfg = load_config(REPO_ROOT / "configs" / "models" / f"{name}.yaml",
                          [f"train.max_steps={args.steps}", f"train.ckpt_every={args.steps}",
                           f"train.val_every={args.steps}"])
        m = train_run(cfg, runs, "smoke")
        s = evaluate_checkpoint(runs / m["run_id"], args.steps, runs_dir=runs,
                                n_states=args.n_states, n_verify=args.n_states)
        good = (m["status"] == "complete" and m["reload_check"]["passed"]
                and s["verify_passed"] and s["n_errors"] == 0)
        ok &= good
        print(f"[smoke {name}] {m['timing']['steps_per_s']:.1f} steps/s, valid MSE "
              f"{m['final_offline']['valid']['mse']:.4f}, reload "
              f"{m['reload_check']['passed']}; rollouts {s['successes']}/{s['n_episodes']} "
              f"({s['label']}), errors {s['n_errors']}, audit {s['verify_passed']} -> "
              f"{'PASS' if good else 'FAIL'}")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())

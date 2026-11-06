"""Deliberate overfit checks on real demonstration data (MASTER_SPEC §5.1 sanity test).

    python scripts/overfit_check.py --task lift

For each primary configuration (BC-MLP, BC-RNN, BC-MLP-chunk, diffusion; ``--configs``
selects) the architecture is kept and lr is set to 1e-3:

* **tiny batch**: 32 fixed training windows, 300 steps (diffusion: 1,500);
* **tiny episodes**: every window of 2 training demos (``demo_1``, ``demo_2``), 600 steps
  (diffusion: 1,500).

The script records the masked MSE on the same windows before and after training. For
diffusion that is the MSE of **seeded sampled chunks** against the demonstrated chunks (the
real denoising check), plus the fixed-draw noise-prediction MSE, and the gradient norm of
every step must be finite. The pass rule is the same for every family: the action MSE falls
below 0.2 of its initial value. This shows that the pipeline can learn from the actual data;
it is not a benchmark result. Writes
``<out-dir>/overfit_<task>.json`` (default ``reports/session$ROBOFORGE_EVIDENCE_SESSION``,
D-039).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative
from roboforge.provenance import environment_snapshot
from roboforge.train.config import load_config
from roboforge.train.data import WindowTensors
from roboforge.train.loop import Trainer, offline_metrics
from roboforge.train.run import data_context, load_windows

CASES = {"tiny_batch_32_windows": 300, "tiny_episodes_demo_1_demo_2": 600}
DIFFUSION_STEPS = 1500  # the denoiser needs more steps; the budget is per case, not matched
CONFIGS = ("bc_mlp", "bc_rnn", "bc_mlp_chunk", "diffusion_unet")


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", choices=["lift", "can"], default="lift")
    p.add_argument("--out-dir", type=Path, default=None)
    p.add_argument("--configs", nargs="+", choices=CONFIGS, default=list(CONFIGS))
    args = p.parse_args(argv)
    args.out_dir = args.out_dir or evidence_reports_dir()
    rows = []
    for name in args.configs:
        for case, bc_steps in CASES.items():
            steps = DIFFUSION_STEPS if name == "diffusion_unet" else bc_steps
            cfg = load_config(REPO_ROOT / "configs" / "models" / f"{name}.yaml",
                              [f"task={args.task}", "optim.lr=1.0e-3",
                               f"train.max_steps={steps}", f"train.ckpt_every={steps}",
                               "train.batch_size=32" if "batch" in case else
                               "train.batch_size=50"])
            train, _ = load_windows(cfg)
            if "batch" in case:
                idx = np.arange(0, len(train), len(train) // 32)[:32]
                data = WindowTensors(train.obs_history[idx], train.obs_valid[idx],
                                     train.future_action[idx], train.valid_mask[idx],
                                     tuple(train.episode_id[i] for i in idx), train.start_t[idx])
            else:
                data = train.subset({"demo_1", "demo_2"})
            tr = Trainer(cfg, data, None, data_context(args.task, "overfit_check"))
            m0 = offline_metrics(tr.model, data)
            before = m0["mse"]
            t0 = time.perf_counter()
            gmax, finite = 0.0, True
            for _ in range(steps):
                r = tr.step()
                gmax = max(gmax, r["grad_norm"])
                finite &= bool(np.isfinite(r["loss"]) and np.isfinite(r["grad_norm"]))
            m1 = offline_metrics(tr.model, data)
            after = m1["mse"]
            row = {"config": name, "family": cfg.family, "case": case, "n_windows": len(data),
                   "steps": steps, "mse_before": before, "mse_after": after,
                   "ratio": after / before, "finite_losses_and_grad_norms": finite,
                   "grad_norm_max": gmax, "wall_s": time.perf_counter() - t0,
                   "config_sha256": cfg.sha256()}
            if tr.stochastic:
                row.update(mse_kind="seeded sampled chunk vs demonstrated chunk (trained "
                                    "network, DDIM-10, generator seed 1)",
                           denoising_mse_before=m0["denoising_mse"],
                           denoising_mse_after=m1["denoising_mse"],
                           denoising_ratio=m1["denoising_mse"] / m0["denoising_mse"])
            rows.append(row)
            extra = (f"; noise-pred MSE {m0['denoising_mse']:.4f} -> {m1['denoising_mse']:.4f}"
                     if tr.stochastic else "")
            print(f"{name:14s} {case:28s} windows {len(data):4d}: MSE {before:.4f} -> "
                  f"{after:.6f} (x{after / before:.4f}){extra}")
    out = args.out_dir / f"overfit_{args.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "roboforge.overfit_check/1", "task": args.task,
                               "kind": "engineering sanity check (not a result)",
                               "environment": environment_snapshot(), "rows": rows},
                              indent=1) + "\n")
    print(f"-> {repo_relative(out)}")
    return 0 if all(r["ratio"] < 0.2 and r["finite_losses_and_grad_norms"] for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

"""Deliberate overfit checks on real demonstration data (MASTER_SPEC §5.1 sanity test).

    python scripts/overfit_check.py --task lift

For each primary configuration (BC-MLP, BC-RNN, BC-MLP-chunk) the architecture is kept and
lr is set to 1e-3:

* **tiny batch**: 32 fixed training windows, 300 steps;
* **tiny episodes**: every window of 2 training demos (``demo_1``, ``demo_2``), 600 steps.

The script records the masked MSE on the same windows before and after training. This shows
that the pipeline can learn from the actual data; it is not a benchmark result. Writes
``reports/session3/overfit_<task>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import environment_snapshot
from roboforge.train.config import load_config
from roboforge.train.data import WindowTensors
from roboforge.train.loop import Trainer, offline_metrics
from roboforge.train.run import data_context, load_windows

CASES = {"tiny_batch_32_windows": 300, "tiny_episodes_demo_1_demo_2": 600}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--task", choices=["lift", "can"], default="lift")
    p.add_argument("--out-dir", type=Path, default=REPO_ROOT / "reports" / "session3")
    args = p.parse_args(argv)
    rows = []
    for name in ("bc_mlp", "bc_rnn", "bc_mlp_chunk"):
        for case, steps in CASES.items():
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
            before = offline_metrics(tr.model, data)["mse"]
            t0 = time.perf_counter()
            for _ in range(steps):
                tr.step()
            after = offline_metrics(tr.model, data)["mse"]
            rows.append({"config": name, "case": case, "n_windows": len(data), "steps": steps,
                         "mse_before": before, "mse_after": after, "ratio": after / before,
                         "wall_s": time.perf_counter() - t0, "config_sha256": cfg.sha256()})
            print(f"{name:13s} {case:28s} windows {len(data):4d}: MSE {before:.4f} -> "
                  f"{after:.6f} (x{after / before:.4f})")
    out = args.out_dir / f"overfit_{args.task}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps({"schema": "roboforge.overfit_check/1", "task": args.task,
                               "kind": "engineering sanity check (not a result)",
                               "environment": environment_snapshot(), "rows": rows},
                              indent=1) + "\n")
    print(f"-> {repo_relative(out)}")
    return 0 if all(r["ratio"] < 0.2 for r in rows) else 1


if __name__ == "__main__":
    sys.exit(main())

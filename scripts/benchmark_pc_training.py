"""Time the Level 2 point-cloud training step on this host's CPU (timing only, not training).

Session 9 feasibility. It mirrors the RGB study's E8-05/E8-07 measurements:
* the committed config's network (DP3-inspired encoder + the diffusion head);
* batch 64, T_o = 2, 512 points;
* random weights and inputs: no dataset, evaluation state or checkpoint is touched;
* the inherited trainer step: loss, backward, gradient norm, AdamW, EMA;
* 1 torch thread per process, alone and as ``--procs`` concurrent processes.

    uv run --frozen python scripts/benchmark_pc_training.py --procs 1 3 --steps 30
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Any

from roboforge.paths import evidence_reports_dir, repo_relative
from roboforge.provenance import environment_snapshot, evaluation_platform, write_json


def _worker(steps: int, warm: int, q: Any) -> None:
    import torch

    from roboforge.train.ema import EMA
    from roboforge.train.pointcloud import PCConfig, build_pc_policy, config_path

    torch.set_num_threads(1)
    cfg = PCConfig.load(config_path())
    m = build_pc_policy(cfg)
    ema = EMA(m, power=0.75, max_value=0.9999)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-4)
    g = torch.Generator().manual_seed(3)
    spec, b = cfg.cloud_spec, 64
    lo = torch.tensor(spec.box_lo, dtype=torch.float32)
    span = torch.tensor(spec.box_hi, dtype=torch.float32) - lo
    batch = {"points": lo + torch.rand(b, 2, spec.n_points, 3, generator=g) * span,
             "proprio": torch.randn(b, 2, 9, generator=g),
             "obs_valid": torch.ones(b, 2, dtype=torch.bool),
             "future_action": torch.randn(b, 16, 7, generator=g),
             "valid_mask": torch.ones(b, 16, dtype=torch.bool)}
    for i in range(warm + steps):
        if i == warm:
            t0 = time.perf_counter()
        loss = m.loss(batch, None, g)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.get_total_norm([p.grad for p in m.parameters()])
        opt.step()
        ema.step(m)
    q.put((time.perf_counter() - t0) / steps)


def measure(procs: int, steps: int, warm: int) -> dict[str, Any]:
    """Seconds per step per process with ``procs`` concurrent 1-thread processes."""
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    ps = [ctx.Process(target=_worker, args=(steps, warm, q)) for _ in range(procs)]
    for p in ps:
        p.start()
    per = [q.get() for _ in ps]
    for p in ps:
        p.join()
    return {"procs": procs, "s_per_step_each": per, "s_per_step_mean": sum(per) / len(per),
            "aggregate_steps_per_s": sum(1 / x for x in per)}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint (see the module docstring)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--procs", type=int, nargs="+", default=[1, 3])
    p.add_argument("--steps", type=int, default=30)
    p.add_argument("--warm", type=int, default=3)
    p.add_argument("--out", type=Path)
    a = p.parse_args(argv)
    plat = evaluation_platform()
    rows = [measure(n, a.steps, a.warm) for n in a.procs]
    out = a.out or evidence_reports_dir("pc_benchmark", plat["platform_id"]) / "cpu_train_step.json"
    res = {"schema": "roboforge.pc_benchmark/1", "kind": "timing only (random inputs)",
           "batch": 64, "obs_horizon": 2, "n_points": 512, "torch_threads_per_process": 1,
           "steps_timed": a.steps, "rows": rows, "platform": plat,
           "environment": environment_snapshot()}
    write_json(out, res)
    print(json.dumps({"out": repo_relative(out), "rows": [
        {k: r[k] for k in ("procs", "s_per_step_mean", "aggregate_steps_per_s")} for r in rows]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())

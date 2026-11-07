"""Training-throughput profile of every policy family on one device (Session 5).

This is an **engineering measurement**, not a training run. It writes no run directory and
no checkpoint. It times optimizer steps of each family's primary configuration on the real
Lift training windows, at the protocol batch size (100). Each step does what
:meth:`roboforge.train.loop.Trainer.step` does: gather the batch, compute the loss, backward,
take the global gradient norm, then the AdamW step and the EMA update for diffusion. The
offline metrics that a training run adds every ``val_every`` steps are timed separately, once.

* ``device="cpu"`` uses the real :class:`~roboforge.train.loop.Trainer` with one torch thread,
  the protocol's training setting (RESEARCH_PROTOCOL §6.2).
* ``device="cuda"`` moves the same network, data and optimizer to the GPU, and draws the
  diffusion noise from a CUDA generator. It exists only to price the alternative of GPU
  training. Protocol runs train on CPU; GPU training would need a recorded amendment.
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import torch

from roboforge.paths import REPO_ROOT


def _sync(device: str) -> None:
    if device == "cuda":
        torch.cuda.synchronize()


def profile_training(family: str, task: str, device: str, steps: int,
                     warmup: int = 20) -> dict[str, Any]:
    """Median and mean seconds per optimizer step of ``family`` on ``device``."""
    from roboforge.models.losses import masked_mse
    from roboforge.train.config import load_config
    from roboforge.train.loop import Trainer, offline_metrics
    from roboforge.train.run import data_context, load_windows

    cfg = load_config(REPO_ROOT / "configs" / "models" / f"{family}.yaml", [f"task={task}"])
    train, valid = load_windows(cfg)
    tr = Trainer(cfg, train, valid, data_context(task, "profile"))  # sets 1 torch thread
    if device == "cpu":
        step = tr.step
    else:
        dev = torch.device(device)
        tr.model.to(dev)
        if tr.ema is not None:
            tr.ema.averaged_model.to(dev)
        opt = torch.optim.AdamW(tr.params, lr=cfg.optim.lr, betas=cfg.optim.betas,
                                weight_decay=cfg.optim.weight_decay)
        data = {k: getattr(train, k).to(dev) for k in
                ("obs_history", "obs_valid", "future_action", "valid_mask")}
        gen = torch.Generator(device=dev).manual_seed(0)
        rng = np.random.default_rng(0)

        def step() -> dict[str, float]:
            i = torch.as_tensor(rng.integers(0, len(train), cfg.train.batch_size), device=dev)
            b = {k: v[i] for k, v in data.items()}
            if tr.stochastic:
                loss = tr.model.training_loss(b["obs_history"], b["obs_valid"],
                                              b["future_action"], b["valid_mask"],
                                              generator=gen)
            else:
                loss = masked_mse(tr.model(b["obs_history"], b["obs_valid"]),
                                  b["future_action"], b["valid_mask"])
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.get_total_norm([p.grad for p in tr.params])
            opt.step()
            if tr.ema is not None:
                tr.ema.step(tr.model)
            return {"loss": float(loss.detach())}

    for _ in range(warmup):
        step()
    _sync(device)
    times = []
    for _ in range(steps):
        t = time.perf_counter()
        step()
        _sync(device)
        times.append(time.perf_counter() - t)
    offline_s = None
    if device == "cpu":
        t = time.perf_counter()
        net = tr.inference_model
        offline_metrics(net, train, sample=False)
        offline_metrics(net, valid)
        offline_s = time.perf_counter() - t
    a = np.array(times)
    return {"family": family, "task": task, "device": device,
            "device_name": torch.cuda.get_device_name(0) if device == "cuda" else None,
            "num_threads": torch.get_num_threads(), "batch_size": cfg.train.batch_size,
            "steps_timed": steps, "warmup": warmup, "median_s": float(np.median(a)),
            "mean_s": float(a.mean()), "p95_s": float(np.percentile(a, 95)),
            "steps_per_s_median": float(1.0 / np.median(a)),
            "offline_metrics_once_s": offline_s, "param_count": tr.param_count,
            "kind": "engineering profile (no run, no checkpoint)"}


__all__ = ["profile_training"]


def main(argv: list[str] | None = None) -> int:
    """CLI: ``python -m roboforge.compute.profile --family F --device cpu --steps N --out J``."""
    import argparse
    import json
    from pathlib import Path

    from roboforge.provenance import evaluation_platform, git_revision

    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--family", required=True)
    ap.add_argument("--task", default="lift")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--steps", type=int, default=300)
    ap.add_argument("--tag", default="")
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    res = {**profile_training(a.family, a.task, a.device, a.steps), "tag": a.tag,
           "platform": evaluation_platform(), "git": git_revision()}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps({k: res[k] for k in ("family", "device", "steps_per_s_median", "tag")}))
    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())

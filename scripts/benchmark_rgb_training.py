"""Timing-only benchmark of candidate RGB policies for the Gate D budget (Session 8).

**This is not a training run.** It builds a visual policy with random weights, feeds random
``uint8`` images and random normalized proprioception/actions, and times optimizer steps and
inference decisions. No dataset, checkpoint or result comes out of it; only seconds.

The policy and the timed training step are :mod:`roboforge.compute.rgb_bench`, the same core
the Modal L4 probe runs (:mod:`roboforge.compute.rgb_probe`). Each row records its
``step_def``. Rows written at ``f3c9170`` used an older step with no EMA and no gradient norm;
rows from this version include both, as the Level 1 trainer does.

Inference (CPU only: every rollout runs on the canonical CPU container) times one decision
on one window: the centre crop, then the BC head or DDIM-10. Writes
``reports/session<N>/rgb_benchmark/<platform_id>/<tag>.json``.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from roboforge.compute.rgb_bench import CROP, Policy, host_batch, train_steps
from roboforge.paths import evidence_reports_dir, repo_relative
from roboforge.provenance import evaluation_platform, write_json


@torch.no_grad()
def inference(trunk: str, size: int, head: str, decisions: int) -> dict[str, Any]:
    """ms per decision on one window on the CPU (centre crop; DDIM-10 for diffusion)."""
    torch.manual_seed(0)
    pol = Policy(trunk, size, head).eval()
    gen = torch.Generator().manual_seed(0)
    x = host_batch(1, size)
    times = []
    for i in range(decisions + 2):
        t0 = time.perf_counter()
        feats = pol.enc(x["images"], x["proprio"], train=False)
        if head == "diffusion":
            pol.net.sample(feats, x["valid"], gen)
        else:
            pol.net(feats, x["valid"])
        if i >= 2:
            times.append(1e3 * (time.perf_counter() - t0))
    return {"trunk": trunk, "size": size, "crop": CROP[size], "head": head,
            "decisions": decisions, "median_ms": float(np.median(times)),
            "mean_ms": float(np.mean(times))}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trunks", nargs="+", default=["compact", "resnet18"])
    p.add_argument("--sizes", nargs="+", type=int, default=[84, 128])
    p.add_argument("--heads", nargs="+", default=["bc_chunk", "diffusion"])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--batch", type=int, nargs="+", default=[64])
    p.add_argument("--device", default="cpu", help="training device (inference is CPU only)")
    p.add_argument("--steps", type=int, default=10)
    p.add_argument("--warmup", type=int, default=2)
    p.add_argument("--decisions", type=int, default=20)
    p.add_argument("--inference-only", action="store_true")
    p.add_argument("--tag", default=None)
    p.add_argument("--out-dir", type=Path, default=None)
    args = p.parse_args(argv)
    torch.set_num_threads(args.threads)
    plat = evaluation_platform()
    out = args.out_dir or evidence_reports_dir("rgb_benchmark", plat["platform_id"])
    tag = args.tag or (f"threads{args.threads}_batch{'-'.join(map(str, args.batch))}"
                       + ("_inference" if args.inference_only else ""))
    rows: dict[str, Any] = {"schema": "roboforge.rgb_benchmark/1",
                            "label": "TIMING ONLY: random weights and inputs, not a training run",
                            "platform": plat, "torch_threads": torch.get_num_threads(),
                            "args": vars(args) | {"out_dir": None}, "train": [],
                            "inference": []}
    for trunk in args.trunks:
        for size in args.sizes:
            for head in args.heads:
                for b in [] if args.inference_only else args.batch:
                    r = train_steps(trunk, size, head, b, args.steps, args.warmup,
                                    args.device)
                    rows["train"].append(r)
                    print(json.dumps(r), flush=True)
                if args.device != "cpu":
                    continue
                r = inference(trunk, size, head, args.decisions)
                rows["inference"].append(r)
                print(json.dumps(r), flush=True)
    path = out / f"{tag}.json"
    write_json(path, rows)
    print(f"wrote {repo_relative(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

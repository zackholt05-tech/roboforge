"""Timing-only benchmark of candidate RGB policies for the Gate D budget (Session 8).

**This is not a training run.** It builds a visual policy with random weights, feeds random
``uint8`` images and random normalized proprioception/actions, and times optimizer steps and
inference decisions. No dataset, checkpoint or result comes out of it; only seconds.

Candidate policy = ``RGBObservationEncoder`` (two cameras, per-camera ``VisualCore``: trunk +
SpatialSoftmax(32) + Linear(64), proprioception 9-D) per frame, whose ``T_o × 137`` window
feeds one of the Level 1 heads unchanged:

* ``bc_chunk``: the BC-MLP-chunk head of ``configs/models/bc_mlp_chunk.yaml`` (MLP 2 × 1024);
* ``diffusion``: the U-Net head of ``configs/models/diffusion_unet.yaml`` (training: ε-MSE at
  one sampled timestep; inference: DDIM-10).

Horizons T_o = 2, T_p = 16 (Level 1 chunked families). Encoder and head train end to end with
AdamW. Writes ``reports/session<N>/rgb_benchmark/<platform_id>/<tag>.json``.
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
import yaml

from roboforge.models.registry import build_model
from roboforge.models.vision import RGBObservationEncoder, parameter_count
from roboforge.paths import REPO_ROOT, evidence_reports_dir, repo_relative
from roboforge.provenance import evaluation_platform, write_json

CAMERAS = ("agentview", "robot0_eye_in_hand")
PROPRIO_DIM, ACTION_DIM, T_O, T_P = 9, 7, 2, 16
CROP = {84: 76, 128: 116}
HEAD_CONFIG = {"bc_chunk": ("bc_mlp", "bc_mlp_chunk.yaml"),
               "diffusion": ("diffusion_unet", "diffusion_unet.yaml")}


def build(trunk: str, size: int, head: str) -> tuple[RGBObservationEncoder, torch.nn.Module]:
    """Encoder + Level 1 head with its committed architecture config."""
    family, cfg_file = HEAD_CONFIG[head]
    cfg = yaml.safe_load((REPO_ROOT / "configs" / "models" / cfg_file).read_text())
    enc = RGBObservationEncoder(CAMERAS, PROPRIO_DIM, trunk, size, CROP[size])
    net = build_model(family, cfg["model"], enc.out_dim, ACTION_DIM, T_O, T_P)
    return enc, net


def batch(b: int, size: int, gen: torch.Generator) -> dict[str, torch.Tensor]:
    """Random inputs of the training shapes."""
    return {"images": torch.randint(0, 256, (b, T_O, len(CAMERAS), size, size, 3),
                                    dtype=torch.uint8, generator=gen),
            "proprio": torch.rand(b, T_O, PROPRIO_DIM, generator=gen) * 2 - 1,
            "valid": torch.ones(b, T_O, dtype=torch.bool),
            "action": torch.rand(b, T_P, ACTION_DIM, generator=gen) * 2 - 1,
            "mask": torch.ones(b, T_P, dtype=torch.bool)}


def train_steps(trunk: str, size: int, head: str, b: int, steps: int,
                warmup: int) -> dict[str, Any]:
    """Seconds per optimizer step (forward + backward + AdamW), after ``warmup`` steps."""
    torch.manual_seed(0)
    enc, net = build(trunk, size, head)
    params = [*enc.parameters(), *net.parameters()]
    opt = torch.optim.AdamW(params, lr=1e-4)
    gen = torch.Generator().manual_seed(0)
    x = batch(b, size, gen)
    times = []
    for i in range(warmup + steps):
        t0 = time.perf_counter()
        feats = enc(x["images"], x["proprio"], train=True, generator=gen)
        if head == "diffusion":
            loss = net.training_loss(feats, x["valid"], x["action"], x["mask"], gen)
        else:
            loss = ((net(feats, x["valid"]) - x["action"]) ** 2).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if i >= warmup:
            times.append(time.perf_counter() - t0)
    t = np.array(times)
    return {"trunk": trunk, "size": size, "crop": CROP[size], "head": head, "batch": b,
            "steps_timed": steps, "median_s_per_step": float(np.median(t)),
            "mean_s_per_step": float(t.mean()), "steps_per_s": float(1 / t.mean()),
            "params_encoder": parameter_count(enc), "params_head": parameter_count(net)}


@torch.no_grad()
def inference(trunk: str, size: int, head: str, decisions: int) -> dict[str, Any]:
    """ms per decision on one window (centre crop; DDIM-10 for diffusion)."""
    torch.manual_seed(0)
    enc, net = build(trunk, size, head)
    enc.eval()
    net.eval()
    gen = torch.Generator().manual_seed(0)
    x = batch(1, size, gen)
    times = []
    for i in range(decisions + 2):
        t0 = time.perf_counter()
        feats = enc(x["images"], x["proprio"], train=False)
        if head == "diffusion":
            net.sample(feats, x["valid"], gen)
        else:
            net(feats, x["valid"])
        if i >= 2:
            times.append(1e3 * (time.perf_counter() - t0))
    return {"trunk": trunk, "size": size, "head": head, "decisions": decisions,
            "median_ms": float(np.median(times)), "mean_ms": float(np.mean(times))}


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    p.add_argument("--trunks", nargs="+", default=["compact", "resnet18"])
    p.add_argument("--sizes", nargs="+", type=int, default=[84, 128])
    p.add_argument("--heads", nargs="+", default=["bc_chunk", "diffusion"])
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--batch", type=int, default=64)
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
    tag = args.tag or (f"threads{args.threads}_batch{args.batch}"
                       + ("_inference" if args.inference_only else ""))
    rows: dict[str, Any] = {"schema": "roboforge.rgb_benchmark/1",
                            "label": "TIMING ONLY: random weights and inputs, not a training run",
                            "platform": plat, "torch_threads": torch.get_num_threads(),
                            "args": vars(args) | {"out_dir": None}, "train": [],
                            "inference": []}
    for trunk in args.trunks:
        for size in args.sizes:
            for head in args.heads:
                if not args.inference_only:
                    r = train_steps(trunk, size, head, args.batch, args.steps, args.warmup)
                    rows["train"].append(r)
                    print(json.dumps(r), flush=True)
                r = inference(trunk, size, head, args.decisions)
                rows["inference"].append(r)
                print(json.dumps(r), flush=True)
    path = out / f"{tag}.json"
    write_json(path, rows)
    print(f"wrote {repo_relative(path)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

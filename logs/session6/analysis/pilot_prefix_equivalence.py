"""Session 6 diagnostic: are the first 30k steps of each family's trial 1 (100k budget) the
Session 3-4 pilot run, tensor for tensor? Trial 1 is the pilot configuration with only
train.max_steps changed, and training is deterministic on one platform (1 torch thread), so
with a constant learning rate every shared checkpoint should be bitwise equal. Run:

    uv run --frozen python logs/session6/analysis/pilot_prefix_equivalence.py
"""
import json
from pathlib import Path

import torch

from roboforge.train.checkpoint import load_checkpoint

PAIRS = {"bc_mlp": ("s3_bc_mlp_lift_seed0_dbf312cd", "s6_bc_mlp_lift_seed0_337a72b5"),
         "bc_rnn": ("s3_bc_rnn_lift_seed0_2e7cffe3", "s6_bc_rnn_lift_seed0_f2cca182"),
         "bc_mlp_chunk": ("s3_bc_mlp_chunk_lift_seed0_bb8b4dd6",
                          "s6_bc_mlp_chunk_lift_seed0_cb94bc32"),
         "diffusion_unet": ("s4_diffusion_unet_lift_seed0_336e15e9",
                            "s6_diffusion_unet_lift_seed0_ceccbfad")}
KEYS = ("model_state", "optimizer_state", "scheduler_state", "sampler_state",
        "torch_rng_state", "noise_rng_state", "ema_state", "global_step", "normalization")


def eq(x, y):
    if isinstance(x, torch.Tensor):
        return isinstance(y, torch.Tensor) and torch.equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(eq(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(eq(a, b) for a, b in zip(x, y))
    return x == y


out = {}
for fam, (pilot, main) in PAIRS.items():
    rows = []
    for step in range(5000, 30001, 5000):
        a = load_checkpoint(Path(f"runs/{pilot}/checkpoints/step_{step:07d}.pt"))
        b = load_checkpoint(Path(f"runs/{main}/checkpoints/step_{step:07d}.pt"))
        res = {k: eq(a[k], b[k]) for k in KEYS if k in a or k in b}
        cfg = {k: (v, b["config"]["train"][k]) for k, v in a["config"]["train"].items()
               if b["config"]["train"][k] != v}
        rows.append({"step": step, "equal": res, "all_equal": all(res.values()),
                     "config_train_differences": cfg})
    out[fam] = {"pilot": pilot, "trial1": main, "steps": rows,
                "all_equal": all(r["all_equal"] for r in rows)}
    print(fam, out[fam]["all_equal"], [r["step"] for r in rows if not r["all_equal"]])
Path("logs/session6/analysis/pilot_prefix_equivalence.json").write_text(
    json.dumps(out, indent=1) + "\n")

"""Gate D RGB probe on one Modal L4: plan (local) and job (remote) (Session 8).

The owner approved only a probe capped at $1 (approval A-02 in docs/EXPERIMENT_LEDGER.md),
which measures how fast the candidate RGB policies train on a GPU. **It trains nothing that
is kept and reads no data.** Every input is random (:mod:`roboforge.compute.rgb_bench`), and
no dataset, evaluation state or checkpoint is involved. The job measures:

* the CUDA platform: torch, CUDA and cuDNN versions, the GPU, and the TF32 flags;
* seconds per optimizer step, and peak memory, for each trunk × image size × head × batch,
  with torch's default kernels;
* what the CUDA non-determinism amendment needs, for ResNet-18 at 84² with both heads:
  * whether deterministic kernels run;
  * how much slower they are;
  * whether two 20-step runs from the same seed end with bitwise-equal parameters, in each
    mode.

Plans reuse the launch rules of :mod:`roboforge.compute.plan`: a clean tree, a timeout
within the function's maximum, and an upper-bound estimate within the approval's cap. Its
``authorize`` and ``record-result`` commands apply to these plans unchanged:

    uv run --frozen python -m roboforge.compute.rgb_probe plan --out plan.json --record

The attempt cap is enforced inside the job, so the plan's bound of ``max_attempts_per_launch``
attempts holds: Modal reschedules a preempted input even with ``retries=0`` (KI-031). Each
attempt first writes and commits ``<attempts_root>/<launch_id>/attempt_<n>.json``. Once
``n`` reaches the cap, the job returns at once without touching the GPU. The same file is
rewritten and committed after each group of sections, so a lost client connection does not
lose the results. Read them with
``modal volume get roboforge-runs _probes/<launch_id> <dir>``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from roboforge.compute import cost_log
from roboforge.compute.plan import _base, _checks, summary
from roboforge.compute.pricing import load_modal_config

FUNCTION = "rgb_probe"
JOB_MARGIN_S = 60  # the job starts no section after timeout - margin


def sections(steps: int, warmup: int, repeat_steps: int) -> list[dict[str, Any]]:
    """Sections in priority order. The most decision-relevant run first, so a deadline cuts
    the least useful ones."""
    out: list[dict[str, Any]] = []

    def train(trunk: str, size: int, head: str, batch: int, mode: str = "default") -> None:
        out.append({"name": f"train:{mode}:{trunk}:{size}:{head}:b{batch}", "kind": "train",
                    "trunk": trunk, "size": size, "head": head, "batch": batch, "mode": mode,
                    "steps": steps, "warmup": warmup})

    for head in ("bc_chunk", "diffusion"):
        for b in (64, 100):
            train("resnet18", 84, head, b)
    for head in ("bc_chunk", "diffusion"):
        train("resnet18", 84, head, 64, "deterministic")
        for mode in ("default", "deterministic"):
            out.append({"name": f"repeat:{mode}:resnet18:84:{head}:b64", "kind": "repeat",
                        "trunk": "resnet18", "size": 84, "head": head, "batch": 64,
                        "mode": mode, "steps": repeat_steps})
    for trunk, size in (("resnet18", 128), ("compact", 84), ("compact", 128)):
        for head in ("bc_chunk", "diffusion"):
            for b in (64, 100):
                train(trunk, size, head, b)
    return out


def rgb_probe_plan(timeout_s: int | None = None, steps: int = 30, warmup: int = 5,
                   repeat_steps: int = 20,
                   events: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    """Plan of one ``rgb_probe`` launch (no data, no run ID)."""
    cfg = load_modal_config()
    t = timeout_s or cfg["functions"][FUNCTION]["default_timeout_s"]
    plan = _base(cfg, FUNCTION, t)
    plan.update(mode="probe", run_id=None, deadline_s=int(t - JOB_MARGIN_S),
                rgb_probe={"sections": sections(steps, warmup, repeat_steps),
                           "data": "none: random inputs only (timing probe)"})
    plan["blocking"] = _checks("probe", None, events if events is not None else
                               cost_log.read_events())
    return plan


def _attempt(root: Path, launch_id: str) -> tuple[int, Path]:
    d = root / launch_id
    d.mkdir(parents=True, exist_ok=True)
    n = len(list(d.glob("attempt_*.json")))
    return n, d / f"attempt_{n}.json"


def run_section(sec: dict[str, Any], device: str) -> dict[str, Any]:
    """One section's measurement (raises on failure; the caller records it)."""
    from roboforge.compute import rgb_bench

    args = (sec["trunk"], sec["size"], sec["head"], sec["batch"])
    if sec["kind"] == "train":
        return rgb_bench.train_steps(*args, sec["steps"], sec["warmup"], device, sec["mode"])
    return rgb_bench.repeatability(*args, sec["steps"], device, sec["mode"])


def cuda_info() -> dict[str, Any]:
    """CUDA platform facts (``available`` is False on a machine without CUDA)."""
    import subprocess

    import torch

    smi = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,memory.total",
                          "--format=csv,noheader"], capture_output=True, text=True) \
        if torch.cuda.is_available() else None
    return {"available": torch.cuda.is_available(), "torch": torch.__version__,
            "torch_cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
            "nvidia_smi": (smi.stdout.strip() or smi.stderr.strip()) if smi else None,
            "cudnn_allow_tf32": torch.backends.cudnn.allow_tf32,
            "matmul_allow_tf32": torch.backends.cuda.matmul.allow_tf32,
            "CUBLAS_WORKSPACE_CONFIG": os.environ.get("CUBLAS_WORKSPACE_CONFIG")}


def rgb_probe_job(plan: dict[str, Any], *, attempts_root: Path,
                  commit: Callable[[], None] | None = None, device: str = "cuda",
                  commit_every: int = 4) -> dict[str, Any]:
    """Remote side: enforce the attempt cap, then run the sections until the deadline."""
    from roboforge.compute import rgb_bench
    from roboforge.compute.remote import _section, _set_git_env, host_info

    t0 = time.perf_counter()
    os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", rgb_bench.CUBLAS_WORKSPACE)  # before CUDA
    _set_git_env(plan)
    commit = commit or (lambda: None)
    cap = int(plan["policy"]["max_attempts_per_launch"])
    attempt, path = _attempt(attempts_root, plan["launch_id"])
    base = {"launch_id": plan["launch_id"], "attempt": attempt, "function": FUNCTION,
            "label": "TIMING ONLY: random weights and inputs, nothing kept"}
    if attempt >= cap:
        return {**base, "status": "refused", "container_wall_s": time.perf_counter() - t0,
                "detail": f"attempt cap: this launch already started {attempt} attempts "
                          f"(max_attempts_per_launch={cap}); not running again"}
    path.write_text(json.dumps({**base, "status": "started"}) + "\n")  # counted from now
    commit()
    failed: list[str] = []
    skipped: list[str] = []
    res: dict[str, Any] = {**base, "status": "running", "host": host_info(),
                           "cuda": _section("cuda", cuda_info, failed), "sections": {}}

    def save(status: str) -> None:
        res.update(status=status, failed_sections=failed, skipped_sections=skipped,
                   container_wall_s=time.perf_counter() - t0)
        path.write_text(json.dumps(res, indent=1, default=str) + "\n")
        commit()

    save("running")
    secs = plan["rgb_probe"]["sections"]
    for i, sec in enumerate(secs):
        if time.perf_counter() - t0 > plan["deadline_s"]:
            skipped.extend(s["name"] for s in secs[i:])
            break
        res["sections"][sec["name"]] = _section(sec["name"],
                                                lambda s=sec: run_section(s, device), failed)
        if (i + 1) % commit_every == 0:
            save("running")
    save("partial" if failed or skipped else "complete")
    return res


def main(argv: list[str] | None = None) -> int:
    """CLI: ``plan`` writes a launch plan (``--record`` logs a ``dry_run`` event)."""
    p = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = p.add_subparsers(dest="cmd", required=True)
    pl = sub.add_parser("plan")
    pl.add_argument("--timeout-s", type=int)
    pl.add_argument("--steps", type=int, default=30)
    pl.add_argument("--warmup", type=int, default=5)
    pl.add_argument("--repeat-steps", type=int, default=20)
    pl.add_argument("--out", type=Path, required=True)
    pl.add_argument("--record", action="store_true", help="log a dry_run event")
    args = p.parse_args(argv)
    plan = rgb_probe_plan(args.timeout_s, args.steps, args.warmup, args.repeat_steps)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(plan, indent=1, sort_keys=True) + "\n")
    if args.record:
        cost_log.append_event("dry_run", {**summary(plan), "blocking": plan["blocking"],
                                          "cost_usd": 0.0})
    print(json.dumps({"plan": str(args.out), "sections": len(plan["rgb_probe"]["sections"]),
                      "estimate_upper_usd": round(plan["estimate"]["total_usd"], 4),
                      "blocking": plan["blocking"]}, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())

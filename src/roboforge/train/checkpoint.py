"""Checkpoint files: strict, hash-verified, ``weights_only``-loadable (MASTER_SPEC §5.1).

A checkpoint (``runs/<run_id>/checkpoints/step_<N>.pt``) holds only tensors and
JSON-compatible values, so it loads with ``torch.load(weights_only=True)``, which executes no
pickled code. It contains:

* ``model_state``, ``optimizer_state``, ``scheduler_state``, ``sampler_state`` and the torch
  RNG state, which are everything needed to resume bitwise;
* the resolved run config and its hash, the network dimensions and the parameter count;
* the committed normalizers (obs and action) and every data/spec hash the run trained
  against, so a policy can check it matches the evaluation task's contract;
* the global step, run ID, git revision, torch version, device and thread count.

Files are written atomically (temp file + rename) and never overwritten. Their SHA-256 is
recorded in the run manifest, and loaders can require it before reading.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import torch

from roboforge.provenance import sha256_file

SCHEMA = "roboforge.checkpoint/1"
REQUIRED = ("schema", "run_id", "family", "task", "global_step", "config", "config_sha256",
            "dims", "param_count", "model_state", "optimizer_state", "scheduler_state",
            "sampler_state", "torch_rng_state", "normalization", "hashes", "git",
            "torch_version", "device", "num_threads")


class CheckpointError(ValueError):
    """A checkpoint that is missing, altered, or inconsistent with its contract."""


def checkpoint_name(step: int) -> str:
    """File name of the checkpoint at ``step``."""
    return f"step_{int(step):07d}.pt"


def save_checkpoint(path: Path, payload: dict[str, Any]) -> str:
    """Write ``payload`` atomically to a new file; return its SHA-256."""
    missing = [k for k in REQUIRED if k not in payload]
    if missing or payload["schema"] != SCHEMA:
        raise CheckpointError(f"incomplete checkpoint payload: missing {missing}")
    path = Path(path)
    if path.exists():
        raise FileExistsError(f"{path} exists; checkpoints are never overwritten")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".pt.tmp")
    torch.save(payload, tmp)
    os.replace(tmp, path)
    return sha256_file(path)


def load_checkpoint(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    """Read a checkpoint with ``weights_only=True``; optionally verify its SHA-256 first."""
    path = Path(path)
    if not path.is_file():
        raise CheckpointError(f"no checkpoint at {path}")
    if expected_sha256 is not None:
        actual = sha256_file(path)
        if actual != expected_sha256:
            raise CheckpointError(f"{path}: SHA-256 {actual} != expected {expected_sha256}")
    payload = torch.load(path, map_location="cpu", weights_only=True)
    missing = [k for k in REQUIRED if k not in payload]
    if missing or payload.get("schema") != SCHEMA:
        raise CheckpointError(f"{path}: not a {SCHEMA} checkpoint (missing {missing})")
    return payload

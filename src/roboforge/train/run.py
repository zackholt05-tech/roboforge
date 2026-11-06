"""One training run end to end: data, trainer, checkpoints and the immutable run manifest.

``runs/<run_id>/`` (gitignored) is created fresh and never reused:

* ``config.json``: the resolved config;
* ``metrics.jsonl``: training and offline-loss rows;
* ``checkpoints/step_<N>.pt``: every ``ckpt_every`` steps, including the final step;
* ``manifest.json``: written **once**, with exclusive create, when the run ends
  (``status: complete``) or fails (``status: failed`` with the error). It holds the
  provenance listed in RESEARCH_PROTOCOL §11, every checkpoint's SHA-256 and a reload check:
  the final checkpoint is reloaded strictly with ``weights_only`` into a fresh network and
  must reproduce the in-memory parameters and outputs bitwise.

Checkpoint *selection* is not part of a training run. It needs closed-loop validation
rollouts on the canonical platform (RESEARCH_PROTOCOL §7), which are separate evaluation runs.
"""

from __future__ import annotations

import json
import traceback
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from roboforge.data.pipeline import load_task_artifacts
from roboforge.data.sequence import SequenceDataset
from roboforge.paths import repo_relative
from roboforge.provenance import (
    environment_snapshot,
    evaluation_platform,
    git_revision,
    sha256_file,
    write_json,
)
from roboforge.train.checkpoint import checkpoint_name, load_checkpoint, save_checkpoint
from roboforge.train.config import RunConfig, run_id
from roboforge.train.data import WindowTensors
from roboforge.train.loop import Trainer, offline_metrics

MANIFEST_SCHEMA = "roboforge.train_run/1"


def data_context(task: str, rid: str) -> dict[str, Any]:
    """Committed normalizers and every data/spec hash a run trains against."""
    docs = load_task_artifacts(task)
    m, norm = docs["task_manifest"], docs["normalization"]
    hashes = {"task_manifest_sha256": m["content_sha256"],
              "observation_spec_sha256": m["observation_spec_sha256"],
              "action_spec_sha256": m["action_spec_sha256"],
              "task_spec_sha256": m["task_spec_sha256"],
              "processed_content_sha256": m["processed"]["content_sha256"],
              "converter_version": m["processed"]["converter_version"],
              "split_sha256": docs["split"]["content_sha256"],
              "normalization_sha256": norm["content_sha256"],
              "dataset_id": m["dataset_id"],
              "dataset_low_dim_sha256": m["source"]["files"]["low_dim"]["sha256"]}
    return {"run_id": rid, "hashes": hashes, "git": git_revision(),
            "normalization": {"obs": norm["obs"], "action": norm["action"],
                              "content_sha256": norm["content_sha256"]}}


def load_windows(cfg: RunConfig) -> tuple[WindowTensors, WindowTensors]:
    """Train and valid windows from the hash-checked committed data (the only data source)."""
    kw = {"obs_horizon": cfg.horizons.obs, "pred_horizon": cfg.horizons.pred}
    return (WindowTensors.from_dataset(SequenceDataset.from_committed(cfg.task, "train", **kw)),
            WindowTensors.from_dataset(SequenceDataset.from_committed(cfg.task, "valid", **kw)))


def reload_check(trainer: Trainer, ckpt: Path, sha256: str) -> dict[str, Any]:
    """Reload ``ckpt`` into a fresh network; compare parameters and outputs bitwise."""
    from roboforge.eval.learned import model_from_payload

    fresh = model_from_payload(load_checkpoint(ckpt, expected_sha256=sha256))
    mine = trainer.model.state_dict()
    theirs = fresh.state_dict()
    params_equal = mine.keys() == theirs.keys() and all(torch.equal(mine[k], theirs[k])
                                                        for k in mine)
    data = trainer.valid_data or trainer.train_data
    b = data.batch(np.arange(min(512, len(data))))
    trainer.model.eval()
    with torch.no_grad():
        a = trainer.model(b["obs_history"], b["obs_valid"])
        c = fresh(b["obs_history"], b["obs_valid"])
    return {"checkpoint": repo_relative(ckpt), "params_bitwise_equal": bool(params_equal),
            "outputs_bitwise_equal": bool(torch.equal(a, c)), "n_windows_compared": len(b[
                "obs_history"]), "passed": bool(params_equal and torch.equal(a, c))}


def train_run(cfg: RunConfig, runs_dir: Path, prefix: str,
              episodes: set[str] | None = None) -> dict[str, Any]:
    """Train one configured run; return its manifest. ``episodes`` restricts the training
    windows to those demos (deliberate overfit runs only; recorded in the manifest)."""
    rid = run_id(cfg, prefix if episodes is None else f"{prefix}_overfit")
    run_dir = Path(runs_dir) / rid
    run_dir.mkdir(parents=True, exist_ok=False)  # never reuse a run directory
    write_json(run_dir / "config.json", cfg.to_dict())
    started = datetime.now(tz=UTC).isoformat()
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA, "run_id": rid, "status": "running", "started_utc": started,
        "kind": "training", "config": cfg.to_dict(), "config_sha256": cfg.sha256(),
        "seeds": {"train_seed": cfg.seed, "torch_manual_seed": cfg.seed,
                  "sampler": f"numpy default_rng([{cfg.seed}, 1]), per-epoch permutations"},
        "environment": environment_snapshot(), "platform": evaluation_platform(),
        "selection": {"rule": "docs/RESEARCH_PROTOCOL.md §7 (closed-loop validation, canonical "
                              "platform)", "status": "not part of the training run"}}
    try:
        context = data_context(cfg.task, rid)
        manifest.update(git=context["git"], data=context["hashes"])
        train, valid = load_windows(cfg)
        if episodes is not None:
            train = train.subset(episodes)
            manifest["deliberate_overfit_episodes"] = sorted(episodes)
        manifest["windows"] = {"train": len(train), "valid": len(valid),
                               "train_episodes": len(set(train.episode_id)),
                               "valid_episodes": len(set(valid.episode_id))}
        trainer = Trainer(cfg, train, valid, context)
        manifest.update(dims=trainer.dims, param_count=trainer.param_count,
                        num_threads=torch.get_num_threads(), device="cpu",
                        torch_version=str(torch.__version__))

        def save(tr: Trainer) -> dict[str, Any]:
            path = run_dir / "checkpoints" / checkpoint_name(tr.global_step)
            return {"path": repo_relative(path), "sha256": save_checkpoint(path, tr.payload())}

        result = trainer.fit(run_dir / "metrics.jsonl", on_checkpoint=save)
        last = result["checkpoints"][-1]
        manifest.update(
            checkpoints=result["checkpoints"],
            timing={"wall_s": result["wall_s"], "cpu_s": result["cpu_s"],
                    "steps_per_s": cfg.train.max_steps / result["wall_s"]},
            final_offline={"train": offline_metrics(trainer.model, train),
                           "valid": offline_metrics(trainer.model, valid)},
            reload_check=reload_check(trainer, run_dir / "checkpoints" /
                                      checkpoint_name(last["step"]), last["sha256"]),
            metrics_sha256=sha256_file(run_dir / "metrics.jsonl"), status="complete")
        if not manifest["reload_check"]["passed"]:
            manifest["status"] = "failed"
            manifest["error"] = {"type": "ReloadCheckFailed", "message": "see reload_check"}
    except BaseException as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__,
                                                "message": str(exc),
                                                "traceback": traceback.format_exc()})
        raise
    finally:
        manifest["finished_utc"] = datetime.now(tz=UTC).isoformat()
        write_manifest(run_dir / "manifest.json", manifest)
    return manifest


def write_manifest(path: Path, manifest: dict[str, Any]) -> None:
    """Write a manifest exactly once (exclusive create): manifests are immutable."""
    path = Path(path)
    with path.open("x") as fh:
        fh.write(json.dumps(manifest, sort_keys=True, indent=1, allow_nan=False) + "\n")


def read_manifest(run_dir: Path) -> dict[str, Any]:
    """A run's manifest."""
    return json.loads((Path(run_dir) / "manifest.json").read_text())


__all__ = ["data_context", "load_windows", "read_manifest", "reload_check", "train_run",
           "write_json", "write_manifest"]

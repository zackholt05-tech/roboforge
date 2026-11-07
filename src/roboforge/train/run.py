"""One training run end to end: data, trainer, checkpoints and the immutable run manifest.

``runs/<run_id>/`` (gitignored) is created once by one launch and never reused by another.
Since Session 5 a run can stop and continue (:mod:`roboforge.train.journal`): an
interrupted segment leaves a resume checkpoint and no manifest, and a later segment of the
same run continues from it bitwise. The directory holds:

* ``config.json``: the resolved config; ``launch.json``: the creating launch's ID;
* ``checkpoints.jsonl``, ``resume.jsonl``, ``segments.jsonl``, ``progress.json``: the
  journal (cadence and resume checkpoint indexes, segment events, heartbeat);
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
import time
import traceback
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

from roboforge.data.pipeline import load_task_artifacts
from roboforge.data.sequence import SequenceDataset
from roboforge.paths import REPO_ROOT, repo_relative
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
from roboforge.train.journal import (
    MANIFEST,
    METRICS,
    RESUME_DIR,
    SEGMENTS,
    ResumeError,
    RunExistsError,
    RunJournal,
    utc_now,
)
from roboforge.train.loop import Trainer, noise_generator_seed, offline_metrics
from roboforge.train.stop import StopControl

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


RELOAD_SAMPLE_SEED = 0  # generator seed of the seeded-sample comparison in reload_check


def _states_equal(a: torch.nn.Module, b: torch.nn.Module) -> bool:
    sa, sb = a.state_dict(), b.state_dict()
    return sa.keys() == sb.keys() and all(torch.equal(sa[k], sb[k]) for k in sa)


def reload_check(trainer: Trainer, ckpt: Path, sha256: str) -> dict[str, Any]:
    """Reload ``ckpt`` into a fresh inference network; compare parameters and outputs bitwise.

    The inference network is the EMA copy when the family uses EMA (diffusion); the trained
    weights are then compared as well. For a stochastic network the outputs are seeded
    samples: both networks sample with generators seeded identically and must agree bitwise.
    """
    from roboforge.eval.learned import model_from_payload, trained_model_from_payload

    payload = load_checkpoint(ckpt, expected_sha256=sha256)
    fresh = model_from_payload(payload)
    mine = trainer.inference_model
    params_equal = _states_equal(mine, fresh)
    trained_equal = _states_equal(trainer.model, trained_model_from_payload(payload))
    data = trainer.valid_data or trainer.train_data
    b = data.batch(np.arange(min(512, len(data))))
    was_training = mine.training
    mine.eval()
    with torch.no_grad():
        if trainer.stochastic:
            a = mine.sample(b["obs_history"], b["obs_valid"],
                            torch.Generator().manual_seed(RELOAD_SAMPLE_SEED))
            c = fresh.sample(b["obs_history"], b["obs_valid"],
                             torch.Generator().manual_seed(RELOAD_SAMPLE_SEED))
        else:
            a = mine(b["obs_history"], b["obs_valid"])
            c = fresh(b["obs_history"], b["obs_valid"])
    mine.train(was_training)
    out_equal = bool(torch.equal(a, c))
    return {"checkpoint": repo_relative(ckpt), "params_bitwise_equal": bool(params_equal),
            "trained_params_bitwise_equal": bool(trained_equal),
            "inference_network": "ema" if trainer.ema is not None else "trained",
            "outputs": "seeded samples (generator seed 0)" if trainer.stochastic
            else "deterministic forward",
            "outputs_bitwise_equal": out_equal,
            "n_windows_compared": len(b["obs_history"]),
            "passed": bool(params_equal and trained_equal and out_equal)}


def _resume_guard(journal: RunJournal, same_launch: bool, stale_after_s: float) -> None:
    """Refuse to resume a run that another launch may still be training."""
    if same_launch or not journal.open_segments():
        return
    hb = journal.progress()
    if hb is None:
        return
    age = (datetime.now(tz=UTC) - datetime.fromisoformat(hb["utc"])).total_seconds()
    if age < stale_after_s:
        raise ResumeError(f"{journal.run_dir}: segment {hb['segment']} has a heartbeat "
                          f"{age:.0f} s old and no end record; it may still be running")


def train_run(cfg: RunConfig, runs_dir: Path, prefix: str,
              episodes: set[str] | None = None, *, launch_id: str | None = None,
              resume: bool = False, stop: StopControl | None = None,
              launch_info: dict[str, Any] | None = None,
              stale_after_s: float = 600.0) -> dict[str, Any]:
    """Train one configured run, or continue an interrupted one; return its manifest.

    ``episodes`` restricts the training windows to those demos (deliberate overfit runs
    only; recorded in the manifest; never resumable).

    Launch semantics (Session 5, :mod:`roboforge.train.journal`):

    * a new run directory is created with this ``launch_id`` (a fresh nonce by default);
    * if the directory exists and its ``launch.json`` holds the **same** ``launch_id``, this
      is a restart of the same launch (a preemption): training resumes, or the finished
      manifest is returned unchanged;
    * if it exists with a different launch ID, :class:`RunExistsError` is raised, unless
      ``resume`` is set, which continues an interrupted run under a new launch. A run with a
      manifest (complete or failed) is never continued.

    If ``stop`` requests a stop (a signal, a deadline or ``stop_at_step``), the segment
    writes a resume checkpoint and returns a progress record with ``status: interrupted``.
    No manifest is written then. A ``KeyboardInterrupt`` (an interrupt without installed
    handlers) is re-raised after an ``aborted`` segment record, again without a manifest.
    Any other exception writes the ``failed`` manifest, as in Session 3.
    """
    rid = run_id(cfg, prefix if episodes is None else f"{prefix}_overfit")
    run_dir = Path(runs_dir) / rid
    launch_id = launch_id or uuid.uuid4().hex
    existed = run_dir.exists()
    if not existed:
        if resume:
            raise ResumeError(f"{run_dir}: nothing to resume")
        journal = RunJournal.create(run_dir, rid, cfg.to_dict(), launch_id, launch_info)
    else:
        journal = RunJournal(run_dir)
        same_launch = journal.launch()["launch_id"] == launch_id
        if episodes is not None:
            raise RunExistsError(f"{run_dir} exists; deliberate overfit runs are never resumed")
        if not (resume or same_launch):
            raise RunExistsError(f"{run_dir} exists and belongs to launch "
                                 f"{journal.launch()['launch_id']}; refusing a duplicate launch")
        if journal.has_manifest():
            if same_launch and not resume:
                return read_manifest(run_dir)  # idempotent restart of a finished launch
            raise ResumeError(f"{run_dir} has a manifest; ended runs are immutable")
        journal.check_config(cfg.to_dict())
        _resume_guard(journal, same_launch, stale_after_s)
    # Verify the resume point before anything can mark the run failed: a refusal here leaves
    # the interrupted run untouched for inspection.
    rp = journal.latest_resume_point(REPO_ROOT) if existed else None
    launch = journal.launch()
    manifest: dict[str, Any] = {
        "schema": MANIFEST_SCHEMA, "run_id": rid, "status": "running",
        "started_utc": launch["created_utc"],
        "kind": "training", "config": cfg.to_dict(), "config_sha256": cfg.sha256(),
        "seeds": {"train_seed": cfg.seed, "torch_manual_seed": cfg.seed,
                  "sampler": f"numpy default_rng([{cfg.seed}, 1]), per-epoch permutations"},
        "environment": environment_snapshot(), "platform": evaluation_platform(),
        "launch": launch,
        "selection": {"rule": "docs/RESEARCH_PROTOCOL.md §7 (closed-loop validation, canonical "
                              "platform)", "status": "not part of the training run"}}
    segment = journal.next_segment()
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
        if trainer.stochastic:
            manifest["seeds"]["diffusion_noise_generator"] = (
                f"torch.Generator().manual_seed({noise_generator_seed(cfg.seed)}) = "
                f"SeedSequence([{cfg.seed}, 2]).generate_state(1, uint64)")
            manifest["diffusion"] = {**trainer.model.describe(),
                                     "ema": trainer.ema.config() if trainer.ema else None,
                                     "inference_network": "ema" if trainer.ema else "trained"}
        start_event: dict[str, Any] = {"resumed_from": None, "lost_segments": [],
                                       "orphans": [], "metrics_rows_dropped": 0}
        if existed:
            start_event["lost_segments"] = journal.close_lost_segments(rp.step if rp else 0)
            start_event["orphans"] = journal.quarantine_orphans(REPO_ROOT, segment)
            if rp is not None:
                trainer.load_state(load_checkpoint(rp.path, expected_sha256=rp.sha256))
                start_event["resumed_from"] = {"step": rp.step, "kind": rp.kind,
                                               "path": repo_relative(rp.path),
                                               "sha256": rp.sha256}
            start_event["metrics_rows_dropped"] = journal.truncate_metrics(trainer.global_step)
        before = journal.totals()
        journal.append(SEGMENTS, {"event": "start", "segment": segment, "launch_id": launch_id,
                                  "start_step": trainer.global_step, "started_utc": utc_now(),
                                  "wall_s_before": before["wall_s"],
                                  "cpu_s_before": before["cpu_s"],
                                  "host": launch_info or {}, **start_event})

        def save(tr: Trainer) -> dict[str, Any]:
            path = run_dir / "checkpoints" / checkpoint_name(tr.global_step)
            sha = save_checkpoint(path, tr.payload())
            journal.register_checkpoint(tr.global_step, repo_relative(path), sha, segment)
            return {"path": repo_relative(path), "sha256": sha}

        def save_resume(tr: Trainer) -> dict[str, Any]:
            path = run_dir / RESUME_DIR / checkpoint_name(tr.global_step)
            sha = save_checkpoint(path, tr.payload())
            journal.register_checkpoint(tr.global_step, repo_relative(path), sha, segment,
                                        resume=True)
            return {"path": repo_relative(path), "sha256": sha}

        seg_wall0, seg_cpu0 = time.perf_counter(), time.process_time()
        try:
            result = trainer.fit(run_dir / METRICS, on_checkpoint=save, stop=stop,
                                 on_stop=save_resume, wall_before=before["wall_s"],
                                 cpu_before=before["cpu_s"],
                                 on_log=lambda s, w, c: journal.heartbeat(s, w, c, segment))
        except (KeyboardInterrupt, SystemExit) as exc:
            journal.append(SEGMENTS, {"event": "end", "segment": segment, "reason": "aborted",
                                      "detail": type(exc).__name__,
                                      "end_step": trainer.global_step,
                                      "ended_utc": utc_now(),
                                      "wall_s": time.perf_counter() - seg_wall0,
                                      "cpu_s": time.process_time() - seg_cpu0,
                                      "note": "interrupted without installed handlers; the "
                                              "in-memory state was not saved"})
            raise
        end = {"event": "end", "segment": segment, "end_step": trainer.global_step,
               "ended_utc": utc_now(), "wall_s": result["wall_s"], "cpu_s": result["cpu_s"]}
        if result["stopped"]:
            journal.append(SEGMENTS, {**end, "reason": "interrupted",
                                      "detail": result["stopped"],
                                      "resume_checkpoint": result["resume_checkpoint"]})
            return {"status": "interrupted", "run_id": rid, "run_dir": repo_relative(run_dir),
                    "step": trainer.global_step, "reason": result["stopped"],
                    "segment": segment, "resume_checkpoint": result["resume_checkpoint"]}
        journal.append(SEGMENTS, {**end, "reason": "complete"})
        cks = [{"step": r["step"], "path": r["path"], "sha256": r["sha256"]}
               for r in journal.checkpoints()]
        t = cfg.train
        expected = list(range(t.ckpt_every, t.max_steps + 1, t.ckpt_every))
        if [c["step"] for c in cks] != expected:
            raise ResumeError(f"cadence checkpoints {[c['step'] for c in cks]} != {expected}")
        totals = journal.totals()
        last = cks[-1]
        manifest.update(
            checkpoints=cks,
            resume_checkpoints=[{k: r[k] for k in ("step", "path", "sha256", "segment")}
                                for r in journal.resume_checkpoints()],
            segments=journal.segments(),
            timing={"wall_s": totals["wall_s"], "cpu_s": totals["cpu_s"],
                    "steps_per_s": t.max_steps / totals["wall_s"],
                    "n_segments": len([e for e in journal.segments()
                                       if e["event"] == "start"])},
            final_offline={"train": offline_metrics(trainer.inference_model, train,
                                                    sample=False),
                           "valid": offline_metrics(trainer.inference_model, valid)},
            reload_check=reload_check(trainer, REPO_ROOT / last["path"], last["sha256"]),
            metrics_sha256=sha256_file(run_dir / METRICS), status="complete")
        if not manifest["reload_check"]["passed"]:
            manifest["status"] = "failed"
            manifest["error"] = {"type": "ReloadCheckFailed", "message": "see reload_check"}
    except (KeyboardInterrupt, SystemExit):
        raise  # resumable: no manifest
    except Exception as exc:
        manifest.update(status="failed", error={"type": type(exc).__name__,
                                                "message": str(exc),
                                                "traceback": traceback.format_exc()})
        manifest["finished_utc"] = datetime.now(tz=UTC).isoformat()
        write_manifest(run_dir / MANIFEST, manifest)
        raise
    manifest["finished_utc"] = datetime.now(tz=UTC).isoformat()
    write_manifest(run_dir / MANIFEST, manifest)
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

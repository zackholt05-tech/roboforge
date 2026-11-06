"""Closed-loop validation evaluation of trained checkpoints (RESEARCH_PROTOCOL §5, §7).

This module only ever evaluates on the **validation** suite (``eval_validation``). There is
no suite argument, so it cannot reach the locked held-out states. Before loading a
checkpoint it checks the file's SHA-256 against the training run's immutable manifest, and
that hash travels into every rollout record through ``PolicyInfo.params``.

One evaluation run (``runs/<eval_run_id>/``) is one checkpoint on the 50 validation starts
of its task, recorded with :func:`roboforge.eval.rollout.run_episode`:

* every episode record is saved;
* the first ``n_verify`` records are re-simulated with :func:`verify_record`;
* an immutable ``manifest.json`` is written, plus a summary for the report directory.

The summary marks whether it ran on the canonical platform (D-017). Only canonical,
complete (50/50 starts) evaluations are eligible for checkpoint selection. Native
macOS results are labeled development-only.

A stochastic policy (diffusion) samples each episode with the noise seed of
``(policy_seed, state_id, repeat)`` (RESEARCH_PROTOCOL §5); its evaluation run ID ends in
``__ps<policy_seed>r<repeat>``, so different repeats never share a directory.
"""

from __future__ import annotations

import math
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from roboforge.eval.learned import LearnedPolicy
from roboforge.eval.rollout import RolloutRecord, run_episode, verify_record
from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import environment_snapshot, evaluation_platform, git_revision
from roboforge.sim.env_adapter import RoboForgeEnv
from roboforge.sim.eval_states import load_suite, read_manifest
from roboforge.train.run import read_manifest as read_train_manifest
from roboforge.train.run import write_manifest

VALIDATION = "validation"
SCHEMA = "roboforge.validation_eval/1"


def canonical_platform_id() -> str:
    """The pinned evaluation platform ID (``configs/eval/platform.yaml``)."""
    return yaml.safe_load((REPO_ROOT / "configs" / "eval" / "platform.yaml").read_text())[
        "platform_id"]


def wilson_interval(k: int, n: int, z: float = 1.959963984540054) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion (95 % by default)."""
    if n <= 0:
        raise ValueError("n must be positive")
    p = k / n
    den = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / den
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / den
    return max(0.0, centre - half), min(1.0, centre + half)


def episode_stats(rec: RolloutRecord, codec: Any) -> dict[str, Any]:
    """Per-episode outcome plus action diagnostics (clip, saturation, gripper, smoothness)."""
    a = rec.arrays
    out = {k: rec.outcome[k] for k in ("success", "first_success_step", "steps", "stop_reason",
                                       "error", "n_actions_with_clip")}
    raw = a["raw_policy_action"]
    ex = a["executed_action"]
    if len(ex):
        decoded = codec.normalizer.unnormalize(raw)
        grip = np.sign(ex[:, -1])
        lat = a["planning_latency_ms"][np.isfinite(a["planning_latency_ms"])]
        out.update(
            clip_fraction_per_dim=a["clip_mask"].mean(axis=0).tolist(),
            decoded_min=decoded.min(axis=0).tolist(), decoded_max=decoded.max(axis=0).tolist(),
            executed_mean=ex.mean(axis=0).tolist(),
            gripper_sign_changes=int((np.diff(grip) != 0).sum()),
            gripper_first_close_step=(int(np.argmax(grip > 0)) if (grip > 0).any() else None),
            mean_abs_action_delta=(np.abs(np.diff(ex, axis=0)).mean(axis=0).tolist()
                                   if len(ex) > 1 else None),
            n_decisions=int(len(lat)), latency_ms_median=float(np.median(lat)) if len(lat)
            else None)
    return out


def _summarize(episodes: list[dict[str, Any]], dim: int) -> dict[str, Any]:
    n = len(episodes)
    k = sum(e["success"] for e in episodes)
    fs = [e["first_success_step"] for e in episodes if e["success"]]
    steps = np.array([e["steps"] for e in episodes])
    w = np.array([e["steps"] for e in episodes if e["steps"] > 0], dtype=float)
    clip = np.array([e["clip_fraction_per_dim"] for e in episodes if e["steps"] > 0])
    lo, hi = wilson_interval(k, n)
    lat = [e["latency_ms_median"] for e in episodes if e.get("latency_ms_median") is not None]
    return {"successes": int(k), "n_episodes": n, "success_rate": k / n,
            "wilson95": [lo, hi],
            "first_success_step": {"mean": float(np.mean(fs)) if fs else None,
                                   "median": float(np.median(fs)) if fs else None,
                                   "max": int(max(fs)) if fs else None},
            "stop_reasons": {r: sum(e["stop_reason"] == r for e in episodes)
                             for r in sorted({e["stop_reason"] for e in episodes})},
            "n_errors": sum(e["error"] is not None for e in episodes),
            "steps_total": int(steps.sum()),
            # step-weighted fraction of executed actions clipped, per action dimension
            "clip_fraction_per_dim": ((clip * w[:, None]).sum(0) / w.sum()).tolist()
            if len(w) else [0.0] * dim,
            "gripper_sign_changes_mean": float(np.mean([e.get("gripper_sign_changes", 0)
                                                        for e in episodes])),
            "latency_ms_median_of_episode_medians": float(np.median(lat)) if lat else None}


def eval_run_id(train_run_id: str, step: int, platform_id: str, *, stochastic: bool,
                policy_seed: int = 0, repeat: int = 0, n_partial: int | None = None) -> str:
    """Evaluation run ID of one checkpoint on the validation suite."""
    rid = f"{train_run_id}__val_step{step:07d}__{platform_id}"
    if stochastic:
        rid += f"__ps{int(policy_seed)}r{int(repeat)}"
    return rid + (f"__first{n_partial}" if n_partial is not None else "")


def evaluate_checkpoint(train_run_dir: Path, step: int, *, runs_dir: Path,
                        n_states: int | None = None, n_verify: int = 3,
                        num_threads: int = 1, policy_seed: int = 0,
                        repeat: int = 0) -> dict[str, Any]:
    """Evaluate one checkpoint of a completed training run on the validation suite.
    ``policy_seed`` and ``repeat`` only affect stochastic policies."""
    train_run_dir = Path(train_run_dir)
    tm = read_train_manifest(train_run_dir)
    if tm["status"] != "complete":
        raise ValueError(f"{tm['run_id']}: training status is {tm['status']!r}")
    ck = {c["step"]: c for c in tm["checkpoints"]}
    if step not in ck:
        raise KeyError(f"{tm['run_id']} has no checkpoint at step {step}: {sorted(ck)}")
    ckpt_path = REPO_ROOT / ck[step]["path"] if not Path(ck[step]["path"]).is_absolute() \
        else Path(ck[step]["path"])
    policy = LearnedPolicy.from_checkpoint(ckpt_path, expected_sha256=ck[step]["sha256"],
                                           num_threads=num_threads, policy_seed=policy_seed,
                                           repeat=repeat)
    task = tm["config"]["task"]
    platform = evaluation_platform()
    canonical = platform["platform_id"] == canonical_platform_id()
    states = load_suite(task, VALIDATION)
    partial = n_states is not None and n_states < len(states)
    states = states[:n_states] if partial else states
    eval_id = eval_run_id(tm["run_id"], step, platform["platform_id"],
                          stochastic=policy.stochastic, policy_seed=policy_seed, repeat=repeat,
                          n_partial=len(states) if partial else None)
    out_dir = Path(runs_dir) / eval_id
    out_dir.mkdir(parents=True, exist_ok=False)
    env = RoboForgeEnv(task)
    codec = policy.codec(env.action_spec)
    episodes, verified = [], []
    t0 = time.perf_counter()
    try:
        for i, init in enumerate(states):
            rec = run_episode(env, policy, init, run_id=eval_id, episode_index=i, codec=codec)
            rec.save(out_dir / "episodes" / init.state_id)
            if i < n_verify:
                verified.append({"state_id": init.state_id, **verify_record(rec, init)})
            episodes.append({"state_id": init.state_id,
                             "physics_sha256": rec.meta["physics_sha256"],
                             **({"noise_seed": rec.meta["policy_episode"]["noise_seed"]}
                                if policy.stochastic else {}),
                             **episode_stats(rec, codec)})
    finally:
        env.close()
    wall = time.perf_counter() - t0
    summary = {
        "schema": SCHEMA, "eval_run_id": eval_id, "kind": "closed-loop validation evaluation",
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "train_run_id": tm["run_id"], "checkpoint": {"step": step, **ck[step]},
        "policy": policy.info.to_dict(), "task": task, "suite": VALIDATION,
        "stochastic_policy": ({"policy_seed": policy_seed, "repeat": repeat,
                               "samples_per_decision": 1} if policy.stochastic else None),
        "partition": "eval_validation",
        "suite_manifest_content_sha256": read_manifest(task, VALIDATION)["content_sha256"],
        "platform": platform, "canonical_platform": canonical,
        "label": ("canonical platform (D-017)" if canonical
                  else "DEVELOPMENT ONLY: native platform, never pooled with canonical results"),
        "eligible_for_selection": bool(canonical and not partial),
        "partial": partial, "git": git_revision(), "environment": environment_snapshot(),
        "episode_rules": {"horizon": env.horizon, "termination": env.termination,
                          "hold_steps": env.hold_steps},
        **_summarize(episodes, env.action_dim),
        "throughput_control_steps_per_s": sum(e["steps"] for e in episodes) / wall,
        "wall_s": wall, "verify_record": verified,
        "verify_passed": all(v["passed"] for v in verified),
        "records_dir": repo_relative(out_dir), "episodes": episodes}
    write_manifest(out_dir / "manifest.json", summary)
    return summary

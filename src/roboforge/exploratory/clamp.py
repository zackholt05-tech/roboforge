"""X1, the evaluation-only action-clamp test (KNOWN_ISSUES KI-052; DECISIONS D-107).

**Every result made with this module is exploratory, after the freeze.** It never replaces a
frozen number.

**The question.** The diffusion sampler clamps its predicted clean action chunk to [-1, 1] in
the min-max normalized action space (``_Sampler._x0`` with ``clip_sample=True``,
:mod:`roboforge.models.noise_schedule`). Its last DDIM step returns that clamped chunk, so every
executed diffusion action lies inside each dimension's training range. The BC heads are
clipped only to the controller limits (``ActionCodec.decode``). X1 gives a BC policy the same
bound and re-runs it on the Session 7 robustness sets.

**The clamp.** :class:`ClampedCodec` clips each normalized model action to [-1, 1]. It does this
*before* the frozen ``ActionCodec.decode``, which then runs unchanged. That is the sampler's
bound, bit for bit. Every Level 1 checkpoint of a task loads the same committed action
normalization: :func:`roboforge.eval.learned.check_committed_contract` refuses any other. So
[-1, 1] means the same environment-space range for every family, [min_i, max_i] of the training
split, up to one float64 rounding (:func:`clamp_bounds` measures it). Dimensions whose training
range already equals the controller bounds (the gripper, and some motion dimensions) are
clipped identically by the clamp and by the controller, so for them the clamp is a no-op on
the executed action.

**No frozen file changes.** :class:`ClampedPolicy` wraps a frozen
:class:`roboforge.eval.learned.LearnedPolicy`, and :func:`evaluate_x1` reuses the frozen
:func:`roboforge.eval.validation.run_states` (rollout, record, ``verify_record``).

**Records.**
- ``raw_policy_action`` keeps the network's own output, with the same meaning as in every earlier
  record.
- ``executed_action`` is the clamped and decoded action.
- ``clip_mask`` marks the controller clip *or* the clamp.
- Per-episode clamp counts go into ``policy_episode``, outside the physics hash.

**Where records go.** Records are written under ``runs/x1_clamp/<eval_id>/``, and engineering
checks under ``runs/x1_checks/``. The ID has a suffix that no earlier evaluation uses, and an
existing directory is never written into.

**Suites.** :func:`check_x1_suite` admits only the task's robustness sets (RESEARCH_PROTOCOL
§14), so this module cannot reach the validation suite or the locked held-out suite.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

from roboforge.data.normalization import ActionCodec, Normalizer
from roboforge.data.pipeline import load_task_artifacts
from roboforge.eval.learned import LearnedPolicy
from roboforge.eval.validation import (
    _summarize,
    canonical_platform_id,
    eval_run_id,
    run_states,
)
from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import environment_snapshot, evaluation_platform, git_revision
from roboforge.sim import robustness as rb
from roboforge.sim.eval_states import load_suite, read_manifest
from roboforge.train.run import read_manifest as read_train_manifest
from roboforge.train.run import write_manifest

LABEL = "exploratory, after the freeze"
DECISION = "D-107"
PREREGISTRATION = "docs/extensions/X1_PREREGISTRATION.md"
SCHEMA = "roboforge.x1_clamp_eval/1"
BOUND = (-1.0, 1.0)  # the diffusion sampler's clip_sample range, normalized action space
# Reported clamped evaluations and the clamp-disabled engineering reproductions live in their
# own subdirectories of runs/, with ID suffixes no earlier evaluation uses (D-107).
RUNS_SUBDIR = {True: "x1_clamp", False: "x1_checks"}
ID_SUFFIX = {True: "__x1-clamp", False: "__x1-noclamp"}


def committed_action_normalizer(task: str) -> tuple[Normalizer, str]:
    """The task's committed action normalizer and its normalization content SHA-256."""
    docs = load_task_artifacts(task)
    norm = docs["normalization"]
    return Normalizer.from_dict(norm["action"]), norm["content_sha256"]


def clamp_bounds(task: str) -> dict[str, Any]:
    """The clamp in environment units, measured from the committed normalizer.

    ``low``/``high`` are the decoded values of -1 and +1, ``train_min``/``train_max`` the
    training-split statistics behind the min-max normalization. The ``max_abs_*`` fields give
    the largest rounding gap between the two. A dimension *binds* on a side when that bound is
    tighter than the controller limit, so only there can the clamp change an executed action.
    """
    norm, sha = committed_action_normalizer(task)
    if norm.mode != "minmax":
        raise ValueError(f"{task}: action normalization is {norm.mode!r}, X1 needs minmax")
    spec = load_task_artifacts(task)["task_manifest"]["action_spec"]
    c_lo, c_hi = np.asarray(spec["low"], np.float64), np.asarray(spec["high"], np.float64)
    lo, hi = norm.unnormalize(np.full(norm.dim, BOUND[0])), norm.unnormalize(
        np.full(norm.dim, BOUND[1]))
    t_lo, t_hi = np.asarray(norm.stats["min"]), np.asarray(norm.stats["max"])
    return {"task": task, "normalization_content_sha256": sha, "mode": norm.mode,
            "normalized_bound": list(BOUND), "low": lo.tolist(), "high": hi.tolist(),
            "train_min": t_lo.tolist(), "train_max": t_hi.tolist(),
            "controller_low": c_lo.tolist(), "controller_high": c_hi.tolist(),
            "max_abs_low_minus_train_min": float(np.abs(lo - t_lo).max()),
            "max_abs_high_minus_train_max": float(np.abs(hi - t_hi).max()),
            "binds_low": [int(i) for i in np.flatnonzero(lo > c_lo)],
            "binds_high": [int(i) for i in np.flatnonzero(hi < c_hi)]}


class ClampedCodec:
    """``ActionCodec`` with the X1 clamp in front of the frozen decoder.

    ``decode(y)`` first clips ``y`` to [-1, 1] when ``enabled``, then calls the frozen
    ``ActionCodec.decode`` on the result. It returns the decoded action and the frozen clip mask
    OR'd with the clamp mask. With ``enabled=False`` it returns exactly what the frozen codec
    returns. It counts clamped entries per dimension, per episode and in total.
    """

    def __init__(self, base: ActionCodec, *, enabled: bool) -> None:
        self.base, self.enabled = base, bool(enabled)
        self.normalizer, self.low, self.high = base.normalizer, base.low, base.high
        self.dim = int(base.normalizer.dim)
        self.total = self._zero()
        self.episode = self._zero()

    def _zero(self) -> dict[str, Any]:
        return {"n_actions_decoded": 0, "n_actions_clamped": 0,
                "entries_clamped_per_dim": [0] * self.dim,
                "max_excess_per_dim": [0.0] * self.dim}

    def reset_episode(self) -> None:
        """Start the per-episode counters (called by :meth:`ClampedPolicy.reset_episode`)."""
        self.episode = self._zero()

    def encode(self, dataset_action: np.ndarray) -> np.ndarray:
        """The frozen encoder, unchanged."""
        return self.base.encode(dataset_action)

    def clamp(self, model_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """``(clamped normalized action, clamp mask)``; identity when disabled."""
        y = np.asarray(model_action, dtype=np.float64)
        if not self.enabled:
            return y, np.zeros(y.shape, dtype=bool)
        c = np.clip(y, BOUND[0], BOUND[1])
        return c, (c != y) & ~np.isnan(y)

    def decode(self, model_action: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Clamp (if enabled), then the frozen ``normalized -> clipped env action`` decode."""
        c, mask = self.clamp(model_action)
        env_action, clip = self.base.decode(c)
        # how far outside [-1, 1] each clamped entry was (0 elsewhere; a NaN is never counted)
        excess = np.where(mask, np.abs(np.asarray(model_action, dtype=np.float64) - c), 0.0)
        for acc in (self.total, self.episode):
            acc["n_actions_decoded"] += 1
            acc["n_actions_clamped"] += int(mask.any())
            acc["entries_clamped_per_dim"] = (
                np.asarray(acc["entries_clamped_per_dim"]) + mask).astype(int).tolist()
            acc["max_excess_per_dim"] = np.maximum(
                np.asarray(acc["max_excess_per_dim"], dtype=np.float64), excess).tolist()
        return env_action, clip | mask


class ClampedPolicy:
    """A frozen :class:`LearnedPolicy` whose executed actions go through :class:`ClampedCodec`.

    It forwards everything the rollout runner uses: ``info``, ``stochastic``, ``codec``,
    ``reset_episode``, ``observe``, ``act`` and ``episode_info``. The network, its observation
    history and its sampler generator are the inner policy's, untouched. ``clamp=False`` is the
    engineering reproduction mode: it changes no action.
    """

    def __init__(self, inner: LearnedPolicy, *, clamp: bool) -> None:
        self.inner, self.clamp_enabled = inner, bool(clamp)
        self.stochastic = bool(inner.stochastic)
        params = copy.deepcopy(inner.info.params)
        params["x1_clamp"] = {"enabled": self.clamp_enabled, "normalized_bound": list(BOUND),
                              "applied": "to each normalized action before the frozen "
                                         "ActionCodec.decode",
                              "decision": DECISION, "label": LABEL}
        tag = ("X1 action clamp to the training range [-1, 1] (normalized)"
               if self.clamp_enabled else "X1 wrapper, clamp DISABLED (engineering check)")
        self.info = replace(inner.info, params=params,
                            label=f"{inner.info.label}; {tag}; {LABEL}")
        self._codec: ClampedCodec | None = None

    def codec(self, action_spec: Any) -> ClampedCodec:
        """The clamped decoder (one per evaluation; :func:`run_states` asks once)."""
        self._codec = ClampedCodec(self.inner.codec(action_spec), enabled=self.clamp_enabled)
        return self._codec

    def reset_episode(self, episode: Any) -> None:
        """Reset the inner policy and the per-episode clamp counters."""
        self.inner.reset_episode(episode)
        if self._codec is not None:
            self._codec.reset_episode()

    def observe(self, obs: Any) -> None:
        """Forwarded unchanged."""
        self.inner.observe(obs)

    def act(self, obs: Any) -> np.ndarray:
        """The inner policy's chunk, unchanged (the clamp acts in :meth:`ClampedCodec.decode`)."""
        return self.inner.act(obs)

    def episode_info(self) -> dict[str, Any]:
        """The inner policy's episode provenance plus this episode's clamp counters."""
        base = self.inner.episode_info() or {}
        counts = copy.deepcopy(self._codec.episode) if self._codec is not None else None
        return {**base, "x1_clamp": {"enabled": self.clamp_enabled, "counts": counts}}


def check_x1_suite(task: str, suite: str) -> None:
    """Raise unless ``suite`` is one of ``task``'s Session 7 robustness sets (D-107)."""
    allowed = rb.sets_for(task)
    if suite not in allowed:
        raise ValueError(f"X1 evaluates only the robustness sets of {task}: {allowed}; "
                         f"got {suite!r}")


def x1_eval_run_id(train_run_id: str, step: int, suite: str, platform_id: str, *,
                   clamp: bool) -> str:
    """ID of an X1 evaluation: the Session 7 ID of the same checkpoint and set plus a suffix
    (``__x1-clamp``, or ``__x1-noclamp`` for an engineering reproduction)."""
    base = eval_run_id(train_run_id, step, platform_id, stochastic=False, suite=suite)
    return base + ID_SUFFIX[bool(clamp)]


def prepare_out_dir(runs_dir: Path, eval_id: str, *, clamp: bool) -> Path:
    """Create ``runs_dir/<subdir>/<eval_id>``; refuse an existing directory (never reuse)."""
    out = Path(runs_dir) / RUNS_SUBDIR[bool(clamp)] / eval_id
    if out.exists():
        raise FileExistsError(f"{out} exists: X1 never writes into an existing directory")
    if (Path(runs_dir) / eval_id).exists():
        raise FileExistsError(f"{Path(runs_dir) / eval_id} exists at the top level of runs/")
    out.mkdir(parents=True, exist_ok=False)
    return out


def _checkpoint(train_run_dir: Path, step: int, expected_sha256: str) -> tuple[dict, Path]:
    tm = read_train_manifest(train_run_dir)
    if tm["status"] != "complete":
        raise ValueError(f"{tm['run_id']}: training status is {tm['status']!r}")
    ck = {c["step"]: c for c in tm["checkpoints"]}
    if step not in ck:
        raise KeyError(f"{tm['run_id']} has no checkpoint at step {step}")
    if ck[step]["sha256"] != expected_sha256:
        raise ValueError(f"{tm['run_id']} step {step}: manifest SHA-256 {ck[step]['sha256']} "
                         f"!= expected {expected_sha256} (test configuration record)")
    p = Path(ck[step]["path"])
    return {"tm": tm, "ck": ck[step]}, (p if p.is_absolute() else REPO_ROOT / p)


def evaluate_x1(train_run_dir: Path, step: int, suite: str, *, family: str,
                expected_sha256: str, clamp: bool, runs_dir: Path, n_verify: int = 3,
                num_threads: int = 1) -> dict[str, Any]:
    """Evaluate one frozen BC checkpoint on one robustness set with the X1 wrapper.

    The checkpoint file is hash-checked twice: against ``expected_sha256`` (from the frozen
    test configuration record, passed in by the host) and against the training manifest.
    Every one of the 50 starts runs once (no partial evaluations). Diffusion checkpoints are
    refused here; their no-op check runs on development starts only (D-107). ``family`` is the
    Level 1 family name of the test configuration record (``bc_mlp_chunk`` trains with the
    ``bc_mlp`` model, so the training config cannot tell it apart).
    """
    meta, path = _checkpoint(Path(train_run_dir), int(step), expected_sha256)
    tm = meta["tm"]
    task = tm["config"]["task"]
    check_x1_suite(task, suite)
    platform = evaluation_platform()
    eval_id = x1_eval_run_id(tm["run_id"], int(step), suite, platform["platform_id"],
                             clamp=clamp)
    out_dir = prepare_out_dir(runs_dir, eval_id, clamp=clamp)
    inner = LearnedPolicy.from_checkpoint(path, expected_sha256=expected_sha256,
                                          num_threads=num_threads)
    if inner.stochastic:
        raise ValueError("X1 runs BC checkpoints on the robustness sets; diffusion is "
                         "checked on development starts only (D-107)")
    policy = ClampedPolicy(inner, clamp=clamp)
    states = load_suite(task, suite)
    run = run_states(policy, states, eval_id, out_dir, task, n_verify)
    episodes, verified = run["episodes"], run["verified"]
    canonical = platform["platform_id"] == canonical_platform_id()
    manifest = read_manifest(task, suite)
    codec = policy._codec
    summary = {
        "schema": SCHEMA, "eval_run_id": eval_id, "label": LABEL, "decision": DECISION,
        "preregistration": PREREGISTRATION,
        "kind": ("X1 clamped BC evaluation on a Session 7 robustness set" if clamp else
                 "X1 engineering reproduction, clamp disabled (not a reported result)"),
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "train_run_id": tm["run_id"], "family": family,
        "model_family": tm["config"]["family"],
        "checkpoint": {"step": int(step), **meta["ck"]}, "policy": policy.info.to_dict(),
        "task": task, "suite": suite, "partition": manifest["partition"],
        "suite_manifest_content_sha256": manifest["content_sha256"],
        "x1_clamp": {"enabled": bool(clamp), "bounds": clamp_bounds(task),
                     "counts": codec.total if codec is not None else None},
        "platform": platform, "canonical_platform": canonical,
        "eligible_for_selection": False, "partial": False, "git": git_revision(),
        "environment": environment_snapshot(), "episode_rules": run["episode_rules"],
        **_summarize(episodes, run["action_dim"]),
        "throughput_control_steps_per_s": sum(e["steps"] for e in episodes) / run["wall_s"],
        "wall_s": run["wall_s"],
        "verify_record": verified, "verify_passed": all(v["passed"] for v in verified),
        "records_dir": repo_relative(out_dir), "episodes": episodes}
    write_manifest(out_dir / "manifest.json", summary)
    return summary

"""X3, a separate replication with more training seeds: the evaluator (DECISIONS D-111).

**Every result made with this module is a separate replication.** It is reported beside the
frozen Level 1 result, never instead of it, and never pooled with it.

**What it evaluates.** One Can checkpoint, frozen (seeds 0–2) or new (seeds 3–5), on the 100
X3 held-out starts (:mod:`roboforge.replication.x3_states`), once. The policy is the frozen
:class:`roboforge.eval.learned.LearnedPolicy`, with policy seed 0 and repeat 0, so diffusion
draws its noise from the frozen rule. The rollouts run through the frozen
:func:`roboforge.eval.validation.run_states`: the same episode rule, records and audit as every
frozen evaluation.

**Suites.** :func:`check_x3_suite` admits only the X3 suite. The engineering-check mode
(``check=True``) admits only the validation suite instead; it reproduces a committed Session 7
validation summary and is never a result. Neither mode can reach the locked held-out suite.

**Where records go.** Records go under ``runs/x3/<eval_id>/`` (check mode:
``runs/x3_checks/``). The ID is the frozen evaluation ID with the suite tag. An existing
directory is never written into.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.eval.learned import LearnedPolicy
from roboforge.eval.validation import (
    VALIDATION,
    _summarize,
    canonical_platform_id,
    eval_run_id,
    run_states,
)
from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import environment_snapshot, evaluation_platform, git_revision
from roboforge.replication import x3_states
from roboforge.sim.eval_states import load_suite, read_manifest
from roboforge.train.run import read_manifest as read_train_manifest
from roboforge.train.run import write_manifest

LABEL = x3_states.LABEL
DECISION = x3_states.DECISION
PREREGISTRATION = "docs/extensions/X3_PREREGISTRATION.md"
SCHEMA = "roboforge.x3_eval/1"
CHECK_SCHEMA = "roboforge.x3_check_eval/1"
RUNS_SUBDIR = {False: "x3", True: "x3_checks"}
POLICY_SEED, REPEAT = 0, 0


def check_x3_suite(task: str, suite: str, *, check: bool = False) -> None:
    """Raise unless ``suite`` is the X3 suite of ``task`` (check mode: the validation suite)."""
    if task != x3_states.suite_task():
        raise ValueError(f"X3 evaluates {x3_states.suite_task()} only; got {task!r}")
    allowed = VALIDATION if check else x3_states.suite_name()
    if suite != allowed:
        raise ValueError(f"X3 {'check mode' if check else 'evaluation'} admits only "
                         f"{allowed!r}; got {suite!r}")


def x3_eval_run_id(train_run_id: str, step: int, platform_id: str, *, stochastic: bool,
                   check: bool = False) -> str:
    """ID of an X3 evaluation: the frozen evaluation-ID pattern with the suite tag. A check
    reproduces a validation evaluation, so its ID gets the suffix ``__x3-check``."""
    suite = VALIDATION if check else x3_states.suite_name()
    rid = eval_run_id(train_run_id, step, platform_id, stochastic=stochastic,
                      policy_seed=POLICY_SEED, repeat=REPEAT, suite=suite)
    return rid + ("__x3-check" if check else "")


def prepare_out_dir(runs_dir: Path, eval_id: str, *, check: bool) -> Path:
    """Create ``runs_dir/<subdir>/<eval_id>``; refuse an existing directory (never reuse)."""
    out = Path(runs_dir) / RUNS_SUBDIR[bool(check)] / eval_id
    if out.exists():
        raise FileExistsError(f"{out} exists: X3 never writes into an existing directory")
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
                         f"!= expected {expected_sha256}")
    p = Path(ck[step]["path"])
    return {"tm": tm, "ck": ck[step]}, (p if p.is_absolute() else REPO_ROOT / p)


def evaluate_x3(train_run_dir: Path, step: int, suite: str, *, family: str, seed: int,
                expected_sha256: str, runs_dir: Path, check: bool = False,
                n_verify: int = 3, num_threads: int = 1) -> dict[str, Any]:
    """Evaluate one Can checkpoint on every start of ``suite`` once (X3, D-111).

    The suite is checked before anything is created or loaded. The checkpoint file is
    hash-checked against ``expected_sha256`` (passed in by the host from the freeze or the
    committed selection record), against the training manifest, and again when it is loaded.
    ``family`` is the Level 1 family name (``bc_mlp_chunk`` trains with the ``bc_mlp`` model)."""
    task = x3_states.suite_task()
    check_x3_suite(task, suite, check=check)
    meta, path = _checkpoint(Path(train_run_dir), int(step), expected_sha256)
    tm = meta["tm"]
    if tm["config"]["task"] != task:
        raise ValueError(f"{tm['run_id']}: task {tm['config']['task']!r} != {task!r}")
    if int(tm["config"]["seed"]) != int(seed):
        raise ValueError(f"{tm['run_id']}: seed {tm['config']['seed']} != {seed}")
    platform = evaluation_platform()
    policy = LearnedPolicy.from_checkpoint(path, expected_sha256=expected_sha256,
                                           num_threads=num_threads, policy_seed=POLICY_SEED,
                                           repeat=REPEAT)
    eval_id = x3_eval_run_id(tm["run_id"], int(step), platform["platform_id"],
                             stochastic=policy.stochastic, check=check)
    out_dir = prepare_out_dir(runs_dir, eval_id, check=check)
    states = load_suite(task, suite)
    run = run_states(policy, states, eval_id, out_dir, task, n_verify)
    episodes, verified = run["episodes"], run["verified"]
    canonical = platform["platform_id"] == canonical_platform_id()
    manifest = read_manifest(task, suite)
    summary = {
        "schema": CHECK_SCHEMA if check else SCHEMA, "eval_run_id": eval_id,
        "label": ("X3 engineering check, not a result" if check else LABEL),
        "decision": DECISION, "preregistration": PREREGISTRATION,
        "kind": ("X3 engineering reproduction of a Session 7 validation evaluation" if check
                 else "X3 evaluation on the X3 held-out starts (separate replication)"),
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "train_run_id": tm["run_id"], "family": family, "seed": int(seed),
        "model_family": tm["config"]["family"],
        "checkpoint": {"step": int(step), **meta["ck"]}, "policy": policy.info.to_dict(),
        "stochastic_policy": ({"policy_seed": POLICY_SEED, "repeat": REPEAT,
                               "samples_per_decision": 1} if policy.stochastic else None),
        "task": task, "suite": suite, "partition": manifest["partition"],
        "suite_manifest_content_sha256": manifest["content_sha256"],
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

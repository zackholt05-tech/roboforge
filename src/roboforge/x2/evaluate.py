"""X2 evaluator for the X2 validation starts (DECISIONS D-121, D-123, D-124).

**What it evaluates.** One checkpoint of an X2 training run (trained on the Can MH data inside
:func:`roboforge.x2.binding.x2_bindings`) on every start of the X2 validation suite, once. The
policy is the frozen :class:`roboforge.eval.learned.LearnedPolicy` with policy seed 0 and
repeat 0, so diffusion draws its noise from the frozen rule; the rollouts run through the frozen
:func:`roboforge.eval.validation.run_states`, with the same episode rule, records and audit as
every frozen evaluation. Inside the binding the environment carries the X2 task spec, so
episodes end at the first success or at 500 steps (D-120), and the policy is checked against
the X2 normalization and specs: a Can PH checkpoint is refused.

**Suites.** :func:`check_x2_eval_suite` admits only the X2 validation suite. In X2-a no code
path reaches the X2 held-out suite (research-integrity test).

**Checkpoints during training.** A run that is still training has no manifest yet; its
cadence checkpoints are read from the run journal's checkpoint index (written after each
checkpoint's atomic save). Either way the file is hash-checked against ``expected_sha256``
(passed in by the host), against the index or manifest, and again when it is loaded.

**Where records go.** ``runs/x2/evals/<eval_id>/`` (never written into if it exists), with the
frozen evaluation ID pattern and the suite tag.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import environment_snapshot, evaluation_platform, git_revision
from roboforge.x2 import states as x2s

SCHEMA = "roboforge.x2_eval/1"
LABEL = "X2 development evaluation on X2 validation starts (not a result)"
RUNS_SUBDIR = Path("x2") / "evals"
POLICY_SEED, REPEAT = 0, 0


def check_x2_eval_suite(task: str, suite: str) -> None:
    """Raise unless ``suite`` is the X2 validation suite of Can."""
    if task != x2s.suite_task():
        raise ValueError(f"X2 evaluates {x2s.suite_task()} only; got {task!r}")
    if suite != x2s.VALIDATION_SUITE:
        raise ValueError(f"the X2-a evaluator admits only {x2s.VALIDATION_SUITE!r}; got {suite!r}")


def checkpoint_record(train_run_dir: Path, step: int) -> dict[str, Any]:
    """The cadence checkpoint record (``step``, ``path``, ``sha256``) of ``step``: from the
    complete manifest if the run has one, else from the journal's checkpoint index."""
    from roboforge.train.journal import RunJournal
    from roboforge.train.run import read_manifest

    j = RunJournal(Path(train_run_dir))
    if j.has_manifest():
        tm = read_manifest(Path(train_run_dir))
        if tm["status"] != "complete":
            raise ValueError(f"{tm['run_id']}: training status is {tm['status']!r}")
        rows = tm["checkpoints"]
    else:
        rows = j.checkpoints()
    ck = {int(r["step"]): r for r in rows}
    if step not in ck:
        raise KeyError(f"{Path(train_run_dir).name} has no checkpoint at step {step}")
    return ck[step]


def prepare_out_dir(runs_dir: Path, eval_id: str) -> Path:
    """Create ``runs_dir/x2/evals/<eval_id>``; refuse an existing directory (never reuse)."""
    out = Path(runs_dir) / RUNS_SUBDIR / eval_id
    if out.exists():
        raise FileExistsError(f"{out} exists: X2 never writes into an existing directory")
    out.mkdir(parents=True, exist_ok=False)
    return out


def evaluate_x2(train_run_dir: Path, step: int, *, family: str, expected_sha256: str,
                runs_dir: Path, suite: str = x2s.VALIDATION_SUITE, n_verify: int = 3,
                num_threads: int = 1) -> dict[str, Any]:
    """Evaluate one X2 checkpoint on every X2 validation start once; return the summary.

    Must be called inside :func:`roboforge.x2.binding.x2_bindings`."""
    from roboforge.eval.learned import LearnedPolicy
    from roboforge.eval.validation import _summarize, canonical_platform_id, eval_run_id, run_states
    from roboforge.sim.eval_states import load_suite, read_manifest
    from roboforge.train.run import write_manifest
    from roboforge.x2 import binding
    from roboforge.x2.data import DATASET_ID, load_x2_artifacts

    if not binding.active():
        raise RuntimeError("evaluate_x2 must run inside x2_bindings()")
    task = x2s.suite_task()
    check_x2_eval_suite(task, suite)
    train_run_dir = Path(train_run_dir)
    rec = checkpoint_record(train_run_dir, int(step))
    if rec["sha256"] != expected_sha256:
        raise ValueError(f"{train_run_dir.name} step {step}: indexed SHA-256 {rec['sha256']} "
                         f"!= expected {expected_sha256}")
    path = Path(rec["path"])
    path = path if path.is_absolute() else REPO_ROOT / path
    policy = LearnedPolicy.from_checkpoint(path, expected_sha256=expected_sha256,
                                           num_threads=num_threads, policy_seed=POLICY_SEED,
                                           repeat=REPEAT)
    if policy.task != task or policy.hashes.get("dataset_id") != DATASET_ID:
        raise ValueError(f"{path.name}: task {policy.task!r} / dataset "
                         f"{policy.hashes.get('dataset_id')!r} is not X2 ({DATASET_ID})")
    x2_manifest = load_x2_artifacts()["task_manifest"]
    platform = evaluation_platform()
    eval_id = eval_run_id(train_run_dir.name, int(step), platform["platform_id"],
                          stochastic=policy.stochastic, policy_seed=POLICY_SEED, repeat=REPEAT,
                          suite=suite)
    out_dir = prepare_out_dir(runs_dir, eval_id)
    states = load_suite(task, suite)
    run = run_states(policy, states, eval_id, out_dir, task, n_verify)
    episodes, verified = run["episodes"], run["verified"]
    manifest = read_manifest(task, suite)
    canonical = platform["platform_id"] == canonical_platform_id()
    summary = {
        "schema": SCHEMA, "eval_run_id": eval_id, "label": LABEL,
        "decisions": ["D-121", "D-123", "D-124"],
        "kind": "X2 closed-loop evaluation on X2 validation starts (development)",
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "train_run_id": train_run_dir.name, "family": family,
        "checkpoint": {"step": int(step), **rec}, "policy": policy.info.to_dict(),
        "stochastic_policy": ({"policy_seed": POLICY_SEED, "repeat": REPEAT,
                               "samples_per_decision": 1} if policy.stochastic else None),
        "task": task, "dataset_id": DATASET_ID,
        "x2_task_manifest_sha256": x2_manifest["content_sha256"],
        "x2_task_spec_sha256": x2_manifest["task_spec_sha256"],
        "suite": suite, "partition": manifest["partition"],
        "suite_manifest_content_sha256": manifest["content_sha256"],
        "platform": platform, "canonical_platform": canonical, "partial": False,
        # Read by the frozen §7 rule (roboforge.eval.selection.select_checkpoint), a pure
        # function over summaries: canonical, all 50 X2 validation starts, primary inference.
        "eligible_for_selection": bool(canonical and len(episodes) == len(states)
                                       == int(manifest["n_states"])),
        "git": git_revision(), "environment": environment_snapshot(),
        "episode_rules": run["episode_rules"], **_summarize(episodes, run["action_dim"]),
        "throughput_control_steps_per_s": sum(e["steps"] for e in episodes) / run["wall_s"],
        "wall_s": run["wall_s"], "verify_record": verified,
        "verify_passed": all(v["passed"] for v in verified),
        "records_dir": repo_relative(out_dir), "episodes": episodes}
    write_manifest(out_dir / "manifest.json", summary)
    return summary

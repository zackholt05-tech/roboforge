"""One-shot held-out test evaluation under the committed freeze (RESEARCH_PROTOCOL §10, D-067).

:func:`evaluate_entry` runs one frozen evaluation (a §7-selected checkpoint, or the
non-learned baseline) on the held-out suite of its task, exactly once:

* the freeze must be committed unmodified (:func:`roboforge.eval.freeze.load_freeze`) and
  the frozen code unchanged (:func:`roboforge.eval.freeze.code_problems`);
* the suite's manifest hash must equal the frozen one; the states load only through
  :func:`roboforge.sim.eval_states.load_suite`, which applies the lock check;
* the checkpoint's SHA-256 must equal the frozen value and the training manifest's;
* it must run on the canonical platform;
* the evaluation directory must not exist. An engineering rerun (§14.6: a fault outside
  the policy stopped the attempt before its summary) needs an explicit reason; the
  incomplete attempt is kept, renamed, and the incident is written into the summary.

The episode loop is the validation evaluator's (:func:`roboforge.eval.validation.run_states`:
``run_episode`` + ``verify_record``). ``rehearsal_suite`` runs the identical path on a
non-locked suite with an uncommitted draft freeze, before the freeze, to test the pipeline;
rehearsal output is labelled and never reported as a test result.
"""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.data.pipeline import load_specs
from roboforge.eval import freeze as fz
from roboforge.eval.learned import LearnedPolicy
from roboforge.eval.policies import RandomPolicy
from roboforge.eval.validation import _summarize, canonical_platform_id, run_states
from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import (
    environment_snapshot,
    evaluation_platform,
    git_blob_sha1,
    git_revision,
    sha256_file,
)
from roboforge.sim import eval_states as es
from roboforge.train.run import read_manifest as read_train_manifest
from roboforge.train.run import write_manifest

SCHEMA = "roboforge.heldout_eval/1"


class HeldoutError(RuntimeError):
    """A held-out evaluation was refused."""


def eval_id(entry: dict[str, Any], platform_id: str, *, rehearsal: bool) -> str:
    """Evaluation run ID of one frozen entry."""
    tag = "rehearsal" if rehearsal else "heldout"
    if entry["family"] == fz.BASELINE:
        return f"{fz.BASELINE}_ps{entry['policy_seed']}_{entry['task']}__{tag}__{platform_id}"
    rid = f"{entry['train_run_id']}__{tag}_step{int(entry['step']):07d}__{platform_id}"
    return rid + ("__ps0r0" if entry["family"] == "diffusion_unet" else "")


def _policy(entry: dict[str, Any], frozen: dict[str, Any]) -> tuple[Any, dict[str, Any] | None]:
    if entry["family"] == fz.BASELINE:
        return RandomPolicy(load_specs(entry["task"])[1], int(entry["policy_seed"])), None
    run_dir = REPO_ROOT / "runs" / entry["train_run_id"]
    tm = read_train_manifest(run_dir)
    if tm["status"] != "complete" or tm["config"]["task"] != entry["task"]:
        raise HeldoutError(f"{entry['train_run_id']}: not a complete {entry['task']} run")
    ck = {c["step"]: c for c in tm["checkpoints"]}.get(int(entry["step"]))
    if ck is None or ck["sha256"] != entry["checkpoint_sha256"]:
        raise HeldoutError(f"{entry['train_run_id']} step {entry['step']}: the training "
                           "manifest does not record the frozen checkpoint SHA-256")
    path = REPO_ROOT / ck["path"] if not Path(ck["path"]).is_absolute() else Path(ck["path"])
    rr = frozen["rollouts"]
    pol = LearnedPolicy.from_checkpoint(path, expected_sha256=entry["checkpoint_sha256"],
                                        num_threads=1, policy_seed=int(rr["policy_seed"]),
                                        repeat=0)
    return pol, {"step": int(entry["step"]), **ck}


def evaluate_entry(entry: dict[str, Any], *, runs_dir: Path | None = None,
                   rehearsal_suite: str | None = None, draft_freeze: Path | None = None,
                   engineering_rerun: str | None = None, n_verify: int = 3) -> dict[str, Any]:
    """Run one frozen held-out evaluation once (or a rehearsal of it); return its summary."""
    rehearsal = rehearsal_suite is not None
    if rehearsal == (draft_freeze is None):
        raise HeldoutError("a rehearsal needs a draft freeze, and a real run must not use one")
    if rehearsal_suite == es.LOCKED_SUITE:
        raise HeldoutError("a rehearsal never runs on the locked held-out suite")
    frozen = (fz.parse_freeze(Path(draft_freeze).read_text()) if rehearsal
              else fz.load_freeze())
    if not rehearsal:
        bad = fz.code_problems(frozen)
        if bad:
            raise HeldoutError(f"frozen code changed: {bad}")
    task = entry["task"]
    if entry not in fz.entries(frozen, task):
        raise HeldoutError(f"{entry} is not an entry of the freeze")
    platform = evaluation_platform()
    canonical = platform["platform_id"] == canonical_platform_id()
    if not canonical and not rehearsal:
        raise HeldoutError(f"held-out runs need the canonical platform, not {platform}")
    suite = rehearsal_suite or es.LOCKED_SUITE
    manifest = es.read_manifest(task, suite)
    if not rehearsal and manifest["content_sha256"] != frozen["tasks"][task][
            "heldout_manifest_sha256"]:
        raise HeldoutError(f"{task}: held-out manifest hash differs from the freeze")
    states = es.load_suite(task, suite)  # the lock check happens here
    policy, ckpt = _policy(entry, frozen)
    eid = eval_id(entry, platform["platform_id"], rehearsal=rehearsal)
    runs_dir = Path(runs_dir or REPO_ROOT / "runs")
    out_dir = runs_dir / eid
    incident = None
    if out_dir.exists():
        if (out_dir / "manifest.json").exists():
            raise HeldoutError(f"{eid} already ran: the held-out test runs once")
        if not engineering_rerun:
            raise HeldoutError(f"{eid}: an incomplete attempt exists; a rerun needs "
                               "--engineering-rerun REASON (protocol §14.6)")
        k = 1 + sum(1 for _ in runs_dir.glob(f"{eid}.incomplete*"))
        if k > 1:
            raise HeldoutError(f"{eid}: an engineering rerun already happened once")
        kept = out_dir.with_name(f"{eid}.incomplete{k}")
        out_dir.rename(kept)
        incident = {"reason": engineering_rerun, "incomplete_attempt": repo_relative(kept),
                    "utc": datetime.now(tz=UTC).isoformat()}
    out_dir.mkdir(parents=True, exist_ok=False)
    run = run_states(policy, states, eid, out_dir, task, n_verify)
    episodes, wall = run["episodes"], run["wall_s"]
    freeze_path = Path(draft_freeze) if rehearsal else REPO_ROOT / fz.FREEZE_REL
    summary = {
        "schema": SCHEMA, "eval_run_id": eid,
        "kind": ("REHEARSAL of the held-out pipeline on a non-locked suite: not a test result"
                 if rehearsal else "closed-loop held-out test evaluation (run once, frozen)"),
        "created_utc": datetime.now(tz=UTC).isoformat(), "task": task,
        "family": entry["family"], "frozen_entry": entry,
        "train_run_id": entry.get("train_run_id"), "checkpoint": ckpt,
        "policy": policy.info.to_dict(), "suite": suite, "partition": manifest["partition"],
        "suite_manifest_content_sha256": manifest["content_sha256"],
        "stochastic_policy": ({"policy_seed": int(frozen["rollouts"]["policy_seed"]),
                               "repeat": 0, "samples_per_decision": 1}
                              if getattr(policy, "stochastic", False) else None),
        "freeze": {"path": repo_relative(freeze_path),
                   "sha256": sha256_file(freeze_path),
                   "git_blob_sha1": git_blob_sha1(freeze_path.read_bytes())},
        "engineering_rerun": incident,
        "platform": platform, "canonical_platform": canonical, "rehearsal": rehearsal,
        "eligible_for_selection": False, "git": git_revision(),
        "environment": environment_snapshot(), "episode_rules": run["episode_rules"],
        **_summarize(episodes, run["action_dim"]),
        "throughput_control_steps_per_s": sum(e["steps"] for e in episodes) / wall,
        "wall_s": wall, "verify_record": run["verified"],
        "verify_passed": all(v["passed"] for v in run["verified"]),
        "records_dir": repo_relative(out_dir), "episodes": episodes}
    write_manifest(out_dir / "manifest.json", summary)
    return summary

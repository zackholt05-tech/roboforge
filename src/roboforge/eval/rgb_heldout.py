"""RGB study: the held-out lock re-gated on the RGB freeze, and the one-shot test (D-081).

The Level 1 lock (:func:`roboforge.sim.eval_states.lock_status`) opened in Session 7 for the
Level 1 test. The RGB study's held-out evaluations must also pass :func:`gate_status`, which
stays shut unless every precondition of RGB freeze §9 holds:

1. ``docs/RGB_EXPERIMENT_FREEZE.md`` is committed with the frozen SHA-256
   (:data:`RGB_FREEZE_SHA256`, the file of commit ``d9c6217``);
2. every file in the freeze's §10 table still has its frozen SHA-256 (code identity);
3. the 6 selection artifacts exist with the SHA-256 values pinned in :data:`SELECTIONS`.
   On the host they must also be tracked and unmodified at HEAD;
4. decision :data:`DECISION_ID` is in HEAD's ``docs/DECISIONS.md``;
5. the Level 1 lock is open for Lift, and Lift's held-out manifest has the frozen hash.

**Committed.** On the host, git answers whether a file is committed. The container has no
``.git``, so there the frozen wrapper ``infra/containers/run_sim_cpu.sh`` passes the host's
clean-tree flag. That flag covers ``src``, ``docs`` and ``configs`` but not ``reports``, so
the selection artifacts are pinned by hash in this module, which is itself committed.

**One-shot run.** :func:`evaluate_selected` runs one selected checkpoint on the 50 held-out
starts, once. It checks the gate, the canonical platform and "already ran" before it reads
any held-out state.

**Engineering rerun.** A fault outside the policy may stop an attempt before its summary is
written. A rerun then needs a reason, happens at most once, and keeps the incomplete attempt
(RESEARCH_PROTOCOL §14.6, as in Level 1).

**Rehearsal.** ``rehearsal_n`` runs the identical path on the first starts of the validation
suite. It is labelled, never reported and not gated.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import sha256_file, tracked_and_unmodified
from roboforge.sim import eval_states as es

SCHEMA = "roboforge.rgb_heldout_eval/1"
TASK = "lift"
DECISION_ID = "D-081"
DECISIONS_REL = "docs/DECISIONS.md"
RGB_FREEZE_REL = "docs/RGB_EXPERIMENT_FREEZE.md"
RGB_FREEZE_COMMIT = "d9c6217"
RGB_FREEZE_SHA256 = "47e7cb8ee70743dbcc6a0295a92a7462a2309870aebb3eda19060ca0e0ac385f"
HELDOUT_MANIFEST_SHA256 = "e02498651b9d5bb88b6aec0870c5e3277cadaaeaf0e325abf86c79e89b1f22da"
SELECTION_DIR_REL = "reports/session8/rgb_selection"
# SHA-256 of each committed selection artifact (reports/session8/rgb_selection/<run_id>.json).
SELECTIONS = {
    "s8_rgb_bc_chunk_resnet18_lift_seed0_73a80bad":
        "8d05b45590626dded47e10b407a2a78cf3bc23fe0ef65befdee6db38c1af6ee0",
    "s8_rgb_bc_chunk_resnet18_lift_seed1_c49230ce":
        "5acfc0d263db8885cd9f5de0e1930acc7d3021682d7fb27df6e5a18f797d73a7",
    "s8_rgb_bc_chunk_resnet18_lift_seed2_2e677ad6":
        "368366b9ca99c45f89ad248836e7455c9ea430eeab3b684e83b2ce87a2aef0b0",
    "s8_rgb_diffusion_resnet18_lift_seed0_dddc7f1c":
        "e0c972eff15d6a97b73327ade8dc8f66312e3079e825811ce1b000664e3c9d09",
    "s8_rgb_diffusion_resnet18_lift_seed1_51deeb3e":
        "602bc9fc0c0fd1b57c59c48e451bd5c9cfa47a09d7e7075c322a963f58add6c4",
    "s8_rgb_diffusion_resnet18_lift_seed2_ec213b6a":
        "0f3c14ed3b7f3f190c5fabeb5f41c5898e4512ae2f0e4112bb1a0594beb73f77",
}
# Bootstrap draw order, and the comparison a - b (D-081).
FAMILY_ORDER = ("rgb_diffusion", "rgb_bc_chunk")
COMPARISON = ("rgb_diffusion", "rgb_bc_chunk")
OUT_ROOT_REL = "runs/rgb_heldout"
REHEARSAL_ROOT_REL = "runs/rgb_heldout_rehearsal"
NOT_ATTESTED_IN_CONTAINER = ("reports", "logs", "runs")  # excluded from the wrapper's dirty flag
_TABLE_ROW = re.compile(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$")


class RGBHeldoutError(RuntimeError):
    """An RGB held-out evaluation was refused."""


def _has_git() -> bool:
    return (REPO_ROOT / ".git").exists()


def _committed(rel: str) -> tuple[bool, str]:
    """Whether ``rel`` equals its committed version: git on the host; inside the container,
    the wrapper's clean-tree flag (which does not cover reports, logs or runs)."""
    if _has_git():
        return tracked_and_unmodified(rel)
    if rel.split("/")[0] in NOT_ATTESTED_IN_CONTAINER:
        return False, f"{rel}: the container's clean-tree flag does not cover it"
    if not os.environ.get("ROBOFORGE_GIT_SHA") or os.environ.get("ROBOFORGE_GIT_DIRTY") != "0":
        return False, "inside the container the host tree must be clean (ROBOFORGE_GIT_DIRTY=0)"
    if not (REPO_ROOT / rel).exists():
        return False, f"{rel} does not exist"
    return True, "committed (host tree clean, attested by the container wrapper)"


def _decision_committed() -> tuple[bool, str]:
    heading = f"### {DECISION_ID} "
    if _has_git():
        try:
            text = subprocess.run(["git", "show", f"HEAD:{DECISIONS_REL}"], cwd=REPO_ROOT,
                                  check=True, capture_output=True, text=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return False, f"cannot read HEAD:{DECISIONS_REL}: {e}"
    else:
        ok, why = _committed(DECISIONS_REL)
        if not ok:
            return False, why
        text = (REPO_ROOT / DECISIONS_REL).read_text()
    if not any(line.startswith(heading) for line in text.splitlines()):
        return False, f"{DECISION_ID} is not in the committed {DECISIONS_REL}"
    return True, "committed"


def freeze_code_table() -> dict[str, str]:
    """The RGB freeze's §10 table: file -> frozen SHA-256."""
    text = (REPO_ROOT / RGB_FREEZE_REL).read_text()
    return {m.group(1): m.group(2) for line in text.splitlines()
            if (m := _TABLE_ROW.match(line))}


def selection_path(run_id: str) -> Path:
    """The committed selection artifact of one RGB run."""
    return REPO_ROOT / SELECTION_DIR_REL / f"{run_id}.json"


def gate_status() -> tuple[bool, list[str]]:
    """``(open, problems)`` of the RGB held-out gate; open only when ``problems`` is empty."""
    problems: list[str] = []
    freeze = REPO_ROOT / RGB_FREEZE_REL
    if not freeze.exists():
        return False, [f"{RGB_FREEZE_REL} does not exist"]
    ok, why = _committed(RGB_FREEZE_REL)
    if not ok:
        problems.append(why)
    if sha256_file(freeze) != RGB_FREEZE_SHA256:
        problems.append(f"{RGB_FREEZE_REL} differs from the freeze of {RGB_FREEZE_COMMIT}")
    table = freeze_code_table()
    if len(table) != 12:
        problems.append(f"{RGB_FREEZE_REL} §10 lists {len(table)} files, expected 12")
    for rel, sha in sorted(table.items()):
        p = REPO_ROOT / rel
        if not p.exists() or sha256_file(p) != sha:
            problems.append(f"{rel}: differs from the RGB freeze §10")
    found = {p.stem for p in (REPO_ROOT / SELECTION_DIR_REL).glob("*.json")}
    if found != set(SELECTIONS):
        problems.append(f"{SELECTION_DIR_REL} holds {sorted(found)}, expected the 6 pinned runs")
    for run_id, sha in SELECTIONS.items():
        p = selection_path(run_id)
        if not p.exists():
            problems.append(f"{repo_relative(p)} does not exist")
            continue
        if sha256_file(p) != sha:
            problems.append(f"{repo_relative(p)}: SHA-256 differs from the pinned value")
        if _has_git():
            ok, why = tracked_and_unmodified(repo_relative(p))
            if not ok:
                problems.append(why)
    ok, why = _decision_committed()
    if not ok:
        problems.append(why)
    unlocked, why = es.lock_status(TASK)
    if not unlocked:
        problems.append(f"Level 1 lock: {why}")
    if es.read_manifest(TASK, es.LOCKED_SUITE)["content_sha256"] != HELDOUT_MANIFEST_SHA256:
        problems.append(f"{TASK}: held-out manifest hash differs from the RGB freeze §9")
    if HELDOUT_MANIFEST_SHA256 not in freeze.read_text():
        problems.append(f"{RGB_FREEZE_REL} does not name the held-out manifest hash")
    return not problems, problems


def load_selection(run_id: str) -> dict[str, Any]:
    """A pinned selection artifact, hash-verified."""
    if run_id not in SELECTIONS:
        raise RGBHeldoutError(f"{run_id} is not one of the 6 frozen RGB runs")
    p = selection_path(run_id)
    if not p.exists() or sha256_file(p) != SELECTIONS[run_id]:
        raise RGBHeldoutError(f"{repo_relative(p)}: missing or not the pinned artifact")
    return json.loads(p.read_text())


def eval_id(run_id: str, family: str, step: int, platform_id: str, *,
            rehearsal_n: int | None = None) -> str:
    """Evaluation ID of one selected checkpoint (Level 1 pattern; diffusion adds ps0r0)."""
    tag = "rehearsal" if rehearsal_n is not None else "heldout"
    rid = f"{run_id}__{tag}_step{int(step):07d}__{platform_id}"
    rid += "__ps0r0" if family == "rgb_diffusion" else ""
    return rid + (f"__first{rehearsal_n}" if rehearsal_n is not None else "")


def _prepare_out_dir(out_dir: Path, engineering_rerun: str | None) -> dict[str, Any] | None:
    """Refuse a second run; move an incomplete attempt aside for one engineering rerun."""
    if not out_dir.exists():
        return None
    eid = out_dir.name
    if (out_dir / "manifest.json").exists():
        raise RGBHeldoutError(f"{eid} already ran: the held-out test runs once")
    if not engineering_rerun:
        raise RGBHeldoutError(f"{eid}: an incomplete attempt exists; a rerun needs "
                              "--engineering-rerun REASON (protocol §14.6)")
    k = 1 + sum(1 for _ in out_dir.parent.glob(f"{eid}.incomplete*"))
    if k > 1:
        raise RGBHeldoutError(f"{eid}: an engineering rerun already happened once")
    kept = out_dir.with_name(f"{eid}.incomplete{k}")
    out_dir.rename(kept)
    return {"reason": engineering_rerun, "incomplete_attempt": repo_relative(kept),
            "utc": datetime.now(tz=UTC).isoformat()}


def evaluate_selected(run_id: str, *, out_root: Path | None = None,
                      engineering_rerun: str | None = None, rehearsal_n: int | None = None,
                      n_verify: int = 3) -> dict[str, Any]:
    """Run one selected RGB checkpoint on the held-out suite once (or rehearse the path)."""
    from roboforge.eval.rgb import RGBClosedLoopPolicy, load_rgb_checkpoint, run_rgb_states
    from roboforge.eval.validation import VALIDATION, _summarize, canonical_platform_id
    from roboforge.provenance import environment_snapshot, evaluation_platform, git_revision
    from roboforge.train.run import write_manifest

    rehearsal = rehearsal_n is not None
    gate = gate_status() if not rehearsal else (None, None)
    if not rehearsal and not gate[0]:
        raise RGBHeldoutError("the RGB held-out gate is shut: " + "; ".join(gate[1]))
    sel = load_selection(run_id)
    step, ck_sha = int(sel["selected"]["step"]), sel["selected"]["sha256"]
    run_dir = REPO_ROOT / "runs" / run_id
    man = json.loads((run_dir / "manifest.json").read_text())
    ck = {c["step"]: c for c in man["checkpoints"]}.get(step)
    if man["status"] != "complete" or man["task"] != TASK or ck is None or \
            ck["sha256"] != ck_sha:
        raise RGBHeldoutError(f"{run_id}: not a complete {TASK} run recording the selected "
                              f"checkpoint (step {step}, {ck_sha})")
    platform = evaluation_platform()
    canonical = platform["platform_id"] == canonical_platform_id()
    if not canonical and not rehearsal:
        raise RGBHeldoutError(f"held-out runs need the canonical platform, not {platform}")
    eid = eval_id(run_id, sel["family"], step, platform["platform_id"], rehearsal_n=rehearsal_n)
    root = Path(out_root or REPO_ROOT / (REHEARSAL_ROOT_REL if rehearsal else OUT_ROOT_REL))
    out_dir = root / eid
    incident = _prepare_out_dir(out_dir, engineering_rerun)
    suite = VALIDATION if rehearsal else es.LOCKED_SUITE
    manifest = es.read_manifest(TASK, suite)
    if not rehearsal and manifest["content_sha256"] != HELDOUT_MANIFEST_SHA256:
        raise RGBHeldoutError(f"{TASK}: held-out manifest hash differs from the RGB freeze")
    payload = load_rgb_checkpoint(run_dir / ck["path"], ck_sha)
    policy = RGBClosedLoopPolicy(payload, ck_sha)  # policy_seed 0, repeat 0 (freeze §8)
    states = es.load_suite(TASK, suite)  # the Level 1 lock check happens here
    states = states[:rehearsal_n] if rehearsal else states
    out_dir.mkdir(parents=True, exist_ok=False)
    run = run_rgb_states(policy, states, eid, out_dir, TASK, n_verify)
    freeze = REPO_ROOT / RGB_FREEZE_REL
    summary = {
        "schema": SCHEMA, "eval_run_id": eid,
        "kind": ("REHEARSAL of the RGB held-out path on validation starts: not a test result"
                 if rehearsal else "closed-loop held-out test evaluation, RGB study "
                 "(run once, frozen)"),
        "created_utc": datetime.now(tz=UTC).isoformat(), "task": TASK,
        "family": sel["family"], "seed": sel["seed"], "train_run_id": run_id,
        "checkpoint": {"step": step, **ck}, "policy": policy.info.to_dict(),
        "selection": {"path": repo_relative(selection_path(run_id)),
                      "sha256": SELECTIONS[run_id], "rule": sel["rule"]},
        "suite": suite, "partition": manifest["partition"], "n_states_run": len(states),
        "suite_manifest_content_sha256": manifest["content_sha256"],
        "stochastic_policy": ({"policy_seed": 0, "repeat": 0, "samples_per_decision": 1}
                              if policy.stochastic else None),
        "rgb_freeze": {"path": RGB_FREEZE_REL, "sha256": sha256_file(freeze),
                       "commit": RGB_FREEZE_COMMIT},
        "decision": DECISION_ID, "gate": {"open": gate[0], "problems": gate[1]},
        "engineering_rerun": incident, "rehearsal": rehearsal,
        "platform": platform, "canonical_platform": canonical,
        "eligible_for_selection": False, "git": git_revision(),
        "environment": environment_snapshot(), "episode_rules": run["episode_rules"],
        **_summarize(run["episodes"], run["action_dim"]),
        "renders": run["renders"], "wall_s": run["wall_s"],
        "verify_record": run["verified"],
        "verify_passed": all(v["passed"] for v in run["verified"]),
        "records_dir": repo_relative(out_dir), "episodes": run["episodes"]}
    write_manifest(out_dir / "manifest.json", summary)
    return summary


__all__ = ["COMPARISON", "DECISION_ID", "FAMILY_ORDER", "SELECTIONS", "RGBHeldoutError",
           "evaluate_selected", "eval_id", "freeze_code_table", "gate_status",
           "load_selection", "selection_path"]

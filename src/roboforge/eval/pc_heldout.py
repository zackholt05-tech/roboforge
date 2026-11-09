"""Level 2 study: the held-out lock re-gated on the Level 2 freeze, and the one-shot test (D-086).

The D-081 pattern, for the point-cloud arm. The Level 1 lock
(:func:`roboforge.sim.eval_states.lock_status`) has been open since the Level 1 freeze. A Level 2
held-out evaluation must also pass :func:`gate_status`, which stays shut unless every
precondition of Level 2 freeze §7.3 holds:

1. ``docs/LEVEL2_EXPERIMENT_FREEZE.md`` is committed with the frozen SHA-256
   (:data:`LEVEL2_FREEZE_SHA256`, the file of commit ``b93c50c``);
2. every file in the freeze's §9 table still has its frozen SHA-256 (code identity);
3. the 3 point-cloud selection artifacts exist with the SHA-256 values pinned in
   :data:`SELECTIONS`. On the host they must also be tracked and unmodified at HEAD;
4. decision :data:`DECISION_ID` is in HEAD's ``docs/DECISIONS.md``;
5. the frozen RGB study is untouched: its own gate (D-081) is still open, and the 3 RGB
   diffusion held-out reports this study reuses still have their pinned SHA-256. These
   Session 8 checks run on the host only (:data:`SESSION8_CHECKS_WHERE`);
6. the Level 1 lock is open for Lift, and Lift's held-out manifest has the frozen hash.

**Committed.** On the host, git answers whether a file is committed. The container has no
``.git``, so there the frozen wrapper ``infra/containers/run_sim_cpu.sh`` passes the host's
clean-tree flag. That flag does not cover ``reports``, so the selection artifacts and the RGB
held-out reports are pinned by hash in this module, which is itself committed.

**One-shot run.** :func:`evaluate_selected` runs one selected point-cloud checkpoint on the 50
held-out starts, once. It checks the gate, the canonical platform and "already ran" before it
reads any held-out state.

**Rehearsal.** ``rehearsal_n`` runs the identical path on the first starts of the validation
suite. It is labelled, never reported and not gated.

**Engineering rerun.** As in D-081 (RESEARCH_PROTOCOL §14.6): a reason, at most once, and the
incomplete attempt is kept. Only :func:`roboforge.eval.rgb_heldout._prepare_out_dir` is
reused, unchanged.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from roboforge.eval import rgb_heldout as rh
from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import sha256_file, tracked_and_unmodified
from roboforge.sim import eval_states as es

SCHEMA = "roboforge.pc_heldout_eval/1"
TASK = "lift"
DECISION_ID = "D-086"
LEVEL2_FREEZE_REL = "docs/LEVEL2_EXPERIMENT_FREEZE.md"
LEVEL2_FREEZE_COMMIT = "b93c50c"
LEVEL2_FREEZE_SHA256 = "ffe5a03e98714465acf8655c281dea653197835f4e218cfb97a1ba6173589d81"
HELDOUT_MANIFEST_SHA256 = rh.HELDOUT_MANIFEST_SHA256
SELECTION_DIR_REL = "reports/session9/pc_selection"
# SHA-256 of each committed selection artifact (reports/session9/pc_selection/<run_id>.json).
SELECTIONS: dict[str, str] = {
    "s9_pc_diffusion_dp3_lift_seed0_8fd35613":
        "2184b91e80ffba945b68ee4bc9fe34258110f574ce6b25a4898f4b96066f58d8",
    "s9_pc_diffusion_dp3_lift_seed1_a66a2758":
        "1a8d0d4c5e837b91417d9a1fd0321e0dc98d01b2b6336a46f7fb7a3ff029e641",
    "s9_pc_diffusion_dp3_lift_seed2_402a1f24":
        "065ca92413b12efb526dd072022ff00c5f6056314d73da786f6b7627581ba4ea",
}
# The frozen RGB diffusion arm's held-out reports, reused by the analysis and never rerun.
RGB_HELDOUT_DIR_REL = "reports/session8/rgb_heldout"
RGB_HELDOUT_REPORTS = {
    "s8_rgb_diffusion_resnet18_lift_seed0_dddc7f1c":
        "23008d7ed09c0af5418486fb5eb50a177a8e0cc51a268aa4134fe441107dd74f",
    "s8_rgb_diffusion_resnet18_lift_seed1_51deeb3e":
        "5aa18f54287314e7fdbd0512b97fffd3ba62dee3b9dfae4051b3fbe1d45c8cbd",
    "s8_rgb_diffusion_resnet18_lift_seed2_ec213b6a":
        "0b07c26bd354f19a112e66139db9be6c99206255153da15e6098a47ad7039bdb",
}
N_CODE_FILES = 12
# D-086 amendment: the frozen container wrapper mounts only reports/session<N> (here 9), so the
# Session 8 checks (the RGB gate and the reused RGB held-out reports) run on the host, where
# git also verifies them. `make pc-heldout` runs this full gate on the host before the
# container run, and scripts/analyze_level2.py checks them again before any analysis.
SESSION8_CHECKS_WHERE = ("host only: the container does not mount reports/session8; "
                         "`make pc-heldout` runs the full gate on the host first (D-086)")
OUT_ROOT_REL = "runs/pc_heldout"
REHEARSAL_ROOT_REL = "runs/pc_heldout_rehearsal"
_TABLE_ROW = re.compile(r"^\| `([^`]+)` \| `([0-9a-f]{64})` \|$")


class PCHeldoutError(RuntimeError):
    """A Level 2 held-out evaluation was refused."""


def _has_git() -> bool:
    return (REPO_ROOT / ".git").exists()


def _committed(rel: str) -> tuple[bool, str]:
    """Whether ``rel`` equals its committed version (D-081's rule): git on the host; inside
    the container, the wrapper's clean-tree flag (which does not cover reports, logs or runs)."""
    if _has_git():
        return tracked_and_unmodified(rel)
    if rel.split("/")[0] in rh.NOT_ATTESTED_IN_CONTAINER:
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
            text = subprocess.run(["git", "show", f"HEAD:{rh.DECISIONS_REL}"], cwd=REPO_ROOT,
                                  check=True, capture_output=True, text=True).stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            return False, f"cannot read HEAD:{rh.DECISIONS_REL}: {e}"
    else:
        ok, why = _committed(rh.DECISIONS_REL)
        if not ok:
            return False, why
        text = (REPO_ROOT / rh.DECISIONS_REL).read_text()
    if not any(line.startswith(heading) for line in text.splitlines()):
        return False, f"{DECISION_ID} is not in the committed {rh.DECISIONS_REL}"
    return True, "committed"


def freeze_code_table() -> dict[str, str]:
    """The Level 2 freeze's §9 table: file -> frozen SHA-256."""
    text = (REPO_ROOT / LEVEL2_FREEZE_REL).read_text()
    return {m.group(1): m.group(2) for line in text.splitlines()
            if (m := _TABLE_ROW.match(line))}


def selection_path(run_id: str) -> Path:
    """The committed selection artifact of one point-cloud run."""
    return REPO_ROOT / SELECTION_DIR_REL / f"{run_id}.json"


def gate_status() -> tuple[bool, list[str]]:
    """``(open, problems)`` of the Level 2 held-out gate; open only when ``problems`` is empty."""
    problems: list[str] = []
    freeze = REPO_ROOT / LEVEL2_FREEZE_REL
    if not freeze.exists():
        return False, [f"{LEVEL2_FREEZE_REL} does not exist"]
    ok, why = _committed(LEVEL2_FREEZE_REL)
    if not ok:
        problems.append(why)
    if sha256_file(freeze) != LEVEL2_FREEZE_SHA256:
        problems.append(f"{LEVEL2_FREEZE_REL} differs from the freeze of {LEVEL2_FREEZE_COMMIT}")
    table = freeze_code_table()
    if len(table) != N_CODE_FILES:
        problems.append(f"{LEVEL2_FREEZE_REL} §9 lists {len(table)} files, "
                        f"expected {N_CODE_FILES}")
    for rel, sha in sorted(table.items()):
        p = REPO_ROOT / rel
        if not p.exists() or sha256_file(p) != sha:
            problems.append(f"{rel}: differs from the Level 2 freeze §9")
    found = {p.stem for p in (REPO_ROOT / SELECTION_DIR_REL).glob("*.json")}
    if found != set(SELECTIONS):
        problems.append(f"{SELECTION_DIR_REL} holds {sorted(found)}, expected the 3 pinned runs")
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
    if _has_git():  # Session 8 evidence: see SESSION8_CHECKS_WHERE
        rgb_open, rgb_problems = rh.gate_status()
        if not rgb_open:
            problems.append("RGB study gate (D-081): " + "; ".join(rgb_problems))
        for run_id, sha in RGB_HELDOUT_REPORTS.items():
            p = REPO_ROOT / RGB_HELDOUT_DIR_REL / f"{run_id}.json"
            if not p.exists() or sha256_file(p) != sha:
                problems.append(f"{repo_relative(p)}: missing or not the frozen RGB held-out "
                                "report")
    unlocked, why = es.lock_status(TASK)
    if not unlocked:
        problems.append(f"Level 1 lock: {why}")
    if es.read_manifest(TASK, es.LOCKED_SUITE)["content_sha256"] != HELDOUT_MANIFEST_SHA256:
        problems.append(f"{TASK}: held-out manifest hash differs from the Level 2 freeze §7")
    if HELDOUT_MANIFEST_SHA256 not in freeze.read_text():
        problems.append(f"{LEVEL2_FREEZE_REL} does not name the held-out manifest hash")
    return not problems, problems


def load_selection(run_id: str) -> dict[str, Any]:
    """A pinned selection artifact, hash-verified."""
    if run_id not in SELECTIONS:
        raise PCHeldoutError(f"{run_id} is not one of the 3 frozen point-cloud runs")
    p = selection_path(run_id)
    if not p.exists() or sha256_file(p) != SELECTIONS[run_id]:
        raise PCHeldoutError(f"{repo_relative(p)}: missing or not the pinned artifact")
    return json.loads(p.read_text())


def eval_id(run_id: str, step: int, platform_id: str, *,
            rehearsal_n: int | None = None) -> str:
    """Evaluation ID of one selected checkpoint (the RGB study's pattern; diffusion: ps0r0)."""
    tag = "rehearsal" if rehearsal_n is not None else "heldout"
    rid = f"{run_id}__{tag}_step{int(step):07d}__{platform_id}__ps0r0"
    return rid + (f"__first{rehearsal_n}" if rehearsal_n is not None else "")


def _prepare_out_dir(out_dir: Path, engineering_rerun: str | None) -> dict[str, Any] | None:
    try:
        return rh._prepare_out_dir(out_dir, engineering_rerun)
    except rh.RGBHeldoutError as e:
        raise PCHeldoutError(str(e)) from None


def evaluate_selected(run_id: str, *, out_root: Path | None = None,
                      engineering_rerun: str | None = None, rehearsal_n: int | None = None,
                      n_verify: int = 3) -> dict[str, Any]:
    """Run one selected point-cloud checkpoint on the held-out suite once (or rehearse it)."""
    from roboforge.eval.pointcloud import PCClosedLoopPolicy, load_pc_checkpoint
    from roboforge.eval.rgb import run_rgb_states
    from roboforge.eval.validation import VALIDATION, _summarize, canonical_platform_id
    from roboforge.provenance import environment_snapshot, evaluation_platform, git_revision
    from roboforge.train.run import write_manifest

    rehearsal = rehearsal_n is not None
    gate = gate_status() if not rehearsal else (None, None)
    if not rehearsal and not gate[0]:
        raise PCHeldoutError("the Level 2 held-out gate is shut: " + "; ".join(gate[1]))
    sel = load_selection(run_id)
    step, ck_sha = int(sel["selected"]["step"]), sel["selected"]["sha256"]
    run_dir = REPO_ROOT / "runs" / run_id
    man = json.loads((run_dir / "manifest.json").read_text())
    ck = {c["step"]: c for c in man["checkpoints"]}.get(step)
    if man["status"] != "complete" or man["task"] != TASK or ck is None or \
            ck["sha256"] != ck_sha:
        raise PCHeldoutError(f"{run_id}: not a complete {TASK} run recording the selected "
                             f"checkpoint (step {step}, {ck_sha})")
    platform = evaluation_platform()
    canonical = platform["platform_id"] == canonical_platform_id()
    if not canonical and not rehearsal:
        raise PCHeldoutError(f"held-out runs need the canonical platform, not {platform}")
    eid = eval_id(run_id, step, platform["platform_id"], rehearsal_n=rehearsal_n)
    root = Path(out_root or REPO_ROOT / (REHEARSAL_ROOT_REL if rehearsal else OUT_ROOT_REL))
    out_dir = root / eid
    incident = _prepare_out_dir(out_dir, engineering_rerun)
    suite = VALIDATION if rehearsal else es.LOCKED_SUITE
    manifest = es.read_manifest(TASK, suite)
    if not rehearsal and manifest["content_sha256"] != HELDOUT_MANIFEST_SHA256:
        raise PCHeldoutError(f"{TASK}: held-out manifest hash differs from the Level 2 freeze")
    payload = load_pc_checkpoint(run_dir / ck["path"], ck_sha)
    policy = PCClosedLoopPolicy(payload, ck_sha)  # policy_seed 0, repeat 0
    states = es.load_suite(TASK, suite)  # the Level 1 lock check happens here
    states = states[:rehearsal_n] if rehearsal else states
    out_dir.mkdir(parents=True, exist_ok=False)
    try:
        run = run_rgb_states(policy, states, eid, out_dir, TASK, n_verify)
    finally:
        policy.close()
    freeze = REPO_ROOT / LEVEL2_FREEZE_REL
    summary = {
        "schema": SCHEMA, "eval_run_id": eid,
        "kind": ("REHEARSAL of the Level 2 held-out path on validation starts: not a test "
                 "result" if rehearsal else "closed-loop held-out test evaluation, Level 2 "
                 "study, DP3-inspired (run once, frozen)"),
        "created_utc": datetime.now(tz=UTC).isoformat(), "task": TASK,
        "family": sel["family"], "seed": sel["seed"], "train_run_id": run_id,
        "checkpoint": {"step": step, **ck}, "policy": policy.info.to_dict(),
        "selection": {"path": repo_relative(selection_path(run_id)),
                      "sha256": SELECTIONS[run_id], "rule": sel["rule"]},
        "suite": suite, "partition": manifest["partition"],
        "n_states_run": len(states), "suite_manifest_content_sha256": manifest["content_sha256"],
        "stochastic_policy": {"policy_seed": 0, "repeat": 0, "samples_per_decision": 1},
        "level2_freeze": {"path": LEVEL2_FREEZE_REL, "sha256": sha256_file(freeze),
                          "commit": LEVEL2_FREEZE_COMMIT},
        "decision": DECISION_ID, "gate": {"open": gate[0], "problems": gate[1],
                                          "where": "host" if _has_git() else "container",
                                          "session8_checks": (
                                              "in this gate" if _has_git()
                                              else SESSION8_CHECKS_WHERE)},
        "engineering_rerun": incident, "rehearsal": rehearsal,
        "platform": platform, "canonical_platform": canonical,
        "eligible_for_selection": False, "git": git_revision(),
        "environment": environment_snapshot(), "episode_rules": run["episode_rules"],
        **_summarize(run["episodes"], run["action_dim"]),
        "renders": run["renders"], "n_clouds": policy.n_renders, "wall_s": run["wall_s"],
        "verify_record": run["verified"],
        "verify_passed": all(v["passed"] for v in run["verified"]),
        "records_dir": repo_relative(out_dir), "episodes": run["episodes"]}
    write_manifest(out_dir / "manifest.json", summary)
    return summary


__all__ = ["DECISION_ID", "RGB_HELDOUT_REPORTS", "SELECTIONS", "PCHeldoutError",
           "evaluate_selected", "eval_id", "freeze_code_table", "gate_status",
           "load_selection", "selection_path"]

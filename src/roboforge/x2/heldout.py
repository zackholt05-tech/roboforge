"""X2-c: the frozen X2 checkpoints on the X2 held-out suite, once (DECISIONS D-130).

**Label.** "X2 (Can MH)": reported beside the frozen Level 1 Can PH result, never instead of
it and never pooled with it; within-operator claims only (D-119).

**Only through the X2 freeze.** The held-out suite's name and content hash and the 12
checkpoints (family, seed, training run, step, SHA-256) are read from the machine-readable
block of ``docs/extensions/X2_FREEZE.md`` and from nowhere else. :func:`load_x2_freeze`
refuses a freeze that is not committed unmodified:

- on the host it asks git (tracked, unmodified, equal to the blob at ``HEAD``);
- inside the evaluation container, which has no ``.git``, the host passes the committed
  file's git blob ID as an argument; the container requires a clean host tree
  (``ROBOFORGE_GIT_DIRTY=0``, set by the pinned wrapper) and the mounted file's blob ID to
  equal that argument.

:func:`code_problems` lists every file pinned by the freeze whose SHA-256 changed since.

**Evaluator.** :func:`evaluate_x2c` runs one frozen checkpoint on every start of the X2
held-out suite once, inside :func:`roboforge.x2.binding.x2_bindings` (the X2 contract and
500-step episodes, D-120). It uses the frozen ``LearnedPolicy`` (policy seed 0, repeat 0, so
diffusion draws its noise from the frozen rule) and the frozen ``run_states``, exactly as the
X2-a and X2-b evaluator does on validation starts. The checkpoint file is hashed before it is
loaded. Records go to ``runs/x2c/<eval_id>/``. A second attempt is allowed only for a fault
outside the policy, under a recorded decision; it goes to ``runs/x2c/attempt2/<eval_id>/``.
An existing directory is never written into.

**Check mode** (``check=True``) is the engineering reproduction of committed X2-b validation
summaries, run before the freeze. It admits only the X2 validation suite, writes to
``runs/x2c_checks/`` and is never a result.
"""

from __future__ import annotations

import os
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

from roboforge.eval.freeze import FAMILY_ORDER
from roboforge.paths import REPO_ROOT, repo_relative
from roboforge.provenance import (
    environment_snapshot,
    evaluation_platform,
    git_blob_sha1,
    git_revision,
    sha256_file,
    tracked_and_unmodified,
)
from roboforge.x2 import states as x2s

FREEZE_REL = "docs/extensions/X2_FREEZE.md"
FREEZE_SCHEMA = "roboforge.x2_freeze/1"
BEGIN = "<!-- roboforge:x2-freeze:begin -->"
END = "<!-- roboforge:x2-freeze:end -->"
TC_BEGIN = "<!-- roboforge:x2-test-configuration:begin -->"
TC_END = "<!-- roboforge:x2-test-configuration:end -->"
SCHEMA = "roboforge.x2c_eval/1"
CHECK_SCHEMA = "roboforge.x2c_check_eval/1"
STUDY = "X2 (Can MH)"
LABEL = ("X2 (Can MH) held-out evaluation, once: reported beside the frozen Level 1 Can PH "
         "result, never instead of it, never pooled; within-operator only")
CHECK_LABEL = "X2-c engineering check on X2 validation starts (not a result)"
DECISIONS = ("D-120", "D-121", "D-123", "D-130")
PREREGISTRATION = "docs/extensions/X2_PREREGISTRATION.md"
TEST_CONFIGURATION = "reports/session16/x2/suite/x2_test_configuration.json"
TEST_CONFIGURATION_SHA256 = "387ecbe5ab77ade9dd39ae47917194780822928d0f48ee660ed2122a9fb1b863"
RUNS_SUBDIR = {False: "x2c", True: "x2c_checks"}
ATTEMPT2_SUBDIR = "attempt2"
SUMMARY_PARTS = ("x2", "heldout", "evals")  # reports/session<N>/x2/heldout/evals/
TRAIN_RUNS = Path("runs") / "x2"
POLICY_SEED, REPEAT = 0, 0
N_STATES = 100
HORIZON = 500
SEEDS = (0, 1, 2)
# Launch order: the longest evaluations first (X2-b means per 50 starts: BC-RNN 667 s,
# diffusion 642 s, BC-MLP 631 s, BC-MLP-chunk 228 s); diffusion leads, as in X2-b.
LAUNCH_ORDER = ("diffusion_unet", "bc_rnn", "bc_mlp", "bc_mlp_chunk")
DECISION_ID = re.compile(r"D-\d{3}")


class X2FreezeError(RuntimeError):
    """The X2 freeze is missing, uncommitted, malformed or contradicts the repository."""


def check_x2c_suite(task: str, suite: str, *, check: bool = False) -> None:
    """Raise unless ``suite`` is the X2 held-out suite of Can (check mode: the X2 validation
    suite). Called before anything is created or loaded."""
    if task != x2s.suite_task():
        raise ValueError(f"X2-c evaluates {x2s.suite_task()} only; got {task!r}")
    allowed = x2s.VALIDATION_SUITE if check else x2s.HELDOUT_SUITE
    if suite != allowed:
        raise ValueError(f"X2-c {'check mode' if check else 'evaluation'} admits only "
                         f"{allowed!r}; got {suite!r}")


def _fenced(text: str, begin: str, end: str, lang: str) -> str:
    if text.count(begin) != 1 or text.count(end) != 1:
        raise X2FreezeError(f"the X2 freeze must contain exactly one {begin} ... {end} block")
    body = text.split(begin, 1)[1].split(end, 1)[0].strip("\n")
    head = f"```{lang}\n"
    if not (body.startswith(head) and body.endswith("```")):
        raise X2FreezeError(f"the block after {begin} must be a ```{lang} fence")
    return body[len(head):-3]


def embedded_test_configuration(text: str) -> str:
    """The verbatim copy of the X2 test configuration in the freeze document."""
    return _fenced(text, TC_BEGIN, TC_END, "json")


def parse_x2_freeze(text: str) -> dict[str, Any]:
    """The machine-readable block of the X2 freeze, validated against the preregistration's
    fixed design (suite, rollouts, families, seeds, the test configuration it copies)."""
    import hashlib
    import json

    data = yaml.safe_load(_fenced(text, BEGIN, END, "yaml"))
    if not isinstance(data, dict) or data.get("schema") != FREEZE_SCHEMA:
        raise X2FreezeError(f"X2 freeze schema must be {FREEZE_SCHEMA}")
    s, r = data["suite"], data["rollouts"]
    if (s["task"], s["name"], int(s["n_states"])) != (x2s.suite_task(), x2s.HELDOUT_SUITE,
                                                      N_STATES):
        raise X2FreezeError(f"suite must be {x2s.suite_task()}/{x2s.HELDOUT_SUITE}, "
                            f"{N_STATES} starts")
    if (int(r["repeats"]), int(r["policy_seed"]), int(r["horizon"]), r["termination"]) != (
            1, POLICY_SEED, HORIZON, "first_success"):
        raise X2FreezeError("rollouts must be R = 1, policy seed 0, 500 steps, first success")
    if data["test_configuration"] != {"path": TEST_CONFIGURATION,
                                      "sha256": TEST_CONFIGURATION_SHA256}:
        raise X2FreezeError("the freeze must pin the X2-b test configuration by its SHA-256")
    fams = data["families"]
    if tuple(fams) != FAMILY_ORDER:
        raise X2FreezeError(f"families must be {FAMILY_ORDER} in this order")
    for fam, f in fams.items():
        if [int(x["seed"]) for x in f["runs"]] != list(SEEDS):
            raise X2FreezeError(f"{fam}: runs must be seeds 0, 1, 2 in order")
        for x in f["runs"]:
            if not re.fullmatch(r"[0-9a-f]{64}", str(x["checkpoint_sha256"])):
                raise X2FreezeError(f"{fam} seed {x['seed']}: malformed checkpoint SHA-256")
    tc_text = embedded_test_configuration(text)
    if hashlib.sha256(tc_text.encode()).hexdigest() != TEST_CONFIGURATION_SHA256:
        raise X2FreezeError("the embedded test configuration is not a verbatim copy")
    tc = json.loads(tc_text)["families"]
    for fam, f in fams.items():
        rows = [(x["train_run_id"], int(x["selected_step"]), x["checkpoint_sha256"],
                 int(x["validation_successes"])) for x in tc[fam]["runs"]]
        mine = [(x["train_run_id"], int(x["step"]), x["checkpoint_sha256"],
                 int(x["validation_successes"])) for x in f["runs"]]
        if rows != mine or int(f["chosen_trial"]) != int(tc[fam]["chosen_trial"]):
            raise X2FreezeError(f"{fam}: the block differs from the embedded test configuration")
    return data


def in_container() -> bool:
    """True inside the evaluation container (the wrapper passes the host revision; no .git)."""
    return bool(os.environ.get("ROBOFORGE_GIT_SHA")) and not (REPO_ROOT / ".git").exists()


def freeze_blob() -> str:
    """Git blob ID of the X2 freeze file as it is on disk."""
    return git_blob_sha1((REPO_ROOT / FREEZE_REL).read_bytes())


def committed_freeze_blob() -> str:
    """Git blob ID of the X2 freeze at ``HEAD`` (host only)."""
    return subprocess.run(["git", "rev-parse", f"HEAD:{FREEZE_REL}"], cwd=REPO_ROOT,
                          capture_output=True, text=True, check=True).stdout.strip()


def load_x2_freeze(*, expected_blob: str | None = None) -> dict[str, Any]:
    """Parse the X2 freeze; it must be committed unmodified (host: git; container: the host's
    attestation, ``expected_blob``, required)."""
    path = REPO_ROOT / FREEZE_REL
    if not path.exists():
        raise X2FreezeError(f"{FREEZE_REL} does not exist")
    blob = freeze_blob()
    if in_container():
        if os.environ.get("ROBOFORGE_GIT_DIRTY") != "0":
            raise X2FreezeError("the host tree is dirty")
        if not expected_blob or expected_blob != blob:
            raise X2FreezeError(f"{FREEZE_REL} does not match the host's committed blob")
    else:
        ok, why = tracked_and_unmodified(FREEZE_REL, repo=REPO_ROOT)
        if not ok:
            raise X2FreezeError(f"X2 freeze not usable: {why}")
        if committed_freeze_blob() != blob or (expected_blob not in (None, blob)):
            raise X2FreezeError(f"{FREEZE_REL} differs from its committed blob")
    return parse_x2_freeze(path.read_text())


def code_problems(freeze: dict[str, Any]) -> list[str]:
    """Files pinned by the X2 freeze whose SHA-256 differs now (empty list = unchanged)."""
    out = []
    for rel, sha in sorted(freeze["code_sha256"].items()):
        p = REPO_ROOT / rel
        if not p.exists():
            out.append(f"{rel}: missing")
        elif sha256_file(p) != sha:
            out.append(f"{rel}: SHA-256 changed since the X2 freeze")
    return out


def x2c_eval_run_id(train_run_id: str, step: int, platform_id: str, *, stochastic: bool,
                    check: bool = False) -> str:
    """The frozen evaluation-ID pattern with the suite tag; a check gets ``__x2c-check``."""
    from roboforge.eval.validation import eval_run_id

    suite = x2s.VALIDATION_SUITE if check else x2s.HELDOUT_SUITE
    rid = eval_run_id(train_run_id, step, platform_id, stochastic=stochastic,
                      policy_seed=POLICY_SEED, repeat=REPEAT, suite=suite)
    return rid + ("__x2c-check" if check else "")


def frozen_jobs(freeze: dict[str, Any]) -> list[dict[str, Any]]:
    """The 12 X2-c evaluations of the freeze, in launch order."""
    platform = freeze["evaluation_platform"]["platform_id"]
    out = []
    for fam in LAUNCH_ORDER:
        for r in freeze["families"][fam]["runs"]:
            out.append({"family": fam, "seed": int(r["seed"]),
                        "train_run_id": r["train_run_id"], "step": int(r["step"]),
                        "checkpoint_sha256": r["checkpoint_sha256"],
                        "eval_run_id": x2c_eval_run_id(r["train_run_id"], int(r["step"]),
                                                       platform,
                                                       stochastic=fam == "diffusion_unet")})
    if len({j["eval_run_id"] for j in out}) != len(out) or len(out) != 4 * len(SEEDS):
        raise X2FreezeError("the freeze must list 12 distinct evaluations")
    return out


def find_job(freeze: dict[str, Any], train_run_id: str, step: int, sha256: str, family: str,
             seed: int) -> dict[str, Any]:
    """The frozen job for these arguments; raise if the freeze does not list exactly it."""
    for j in frozen_jobs(freeze):
        if (j["family"], j["seed"]) == (family, int(seed)):
            if (j["train_run_id"], j["step"], j["checkpoint_sha256"]) != (
                    train_run_id, int(step), sha256):
                raise X2FreezeError(f"{family} seed {seed}: the X2 freeze lists "
                                    f"{j['train_run_id']} step {j['step']}, not "
                                    f"{train_run_id} step {step}")
            return j
    raise X2FreezeError(f"the X2 freeze has no {family} seed {seed}")


def records_subdir(*, check: bool, attempt: int) -> Path:
    """``x2c``, ``x2c/attempt2`` or ``x2c_checks`` under ``runs/``."""
    if attempt not in (1, 2) or (check and attempt != 1):
        raise ValueError(f"attempt {attempt} not allowed{' in check mode' if check else ''}")
    base = Path(RUNS_SUBDIR[bool(check)])
    return base / ATTEMPT2_SUBDIR if attempt == 2 else base


def prepare_out_dir(runs_dir: Path, eval_id: str, *, check: bool, attempt: int = 1) -> Path:
    """Create the record directory; refuse an existing one (never reuse). A second attempt
    requires the first attempt's directory (the evaluation that failed)."""
    runs_dir = Path(runs_dir)
    out = runs_dir / records_subdir(check=check, attempt=attempt) / eval_id
    if out.exists():
        raise FileExistsError(f"{out} exists: X2-c never writes into an existing directory")
    if (runs_dir / eval_id).exists():
        raise FileExistsError(f"{runs_dir / eval_id} exists at the top level of runs/")
    if attempt == 2 and not (runs_dir / RUNS_SUBDIR[False] / eval_id).is_dir():
        raise FileNotFoundError(f"attempt 2 of {eval_id} needs the failed first attempt")
    out.mkdir(parents=True, exist_ok=False)
    return out


def _train_checkpoint(train_run_dir: Path, step: int, expected_sha256: str, seed: int
                      ) -> tuple[dict[str, Any], dict[str, Any], Path]:
    from roboforge.train.run import read_manifest as read_train_manifest

    tm = read_train_manifest(train_run_dir)
    if tm["status"] != "complete":
        raise ValueError(f"{tm['run_id']}: training status is {tm['status']!r}")
    if tm["config"]["task"] != x2s.suite_task() or int(tm["config"]["seed"]) != int(seed):
        raise ValueError(f"{tm['run_id']}: task/seed {tm['config']['task']}/"
                         f"{tm['config']['seed']} != {x2s.suite_task()}/{seed}")
    ck = {int(c["step"]): c for c in tm["checkpoints"]}
    if int(step) not in ck:
        raise KeyError(f"{tm['run_id']} has no checkpoint at step {step}")
    if ck[int(step)]["sha256"] != expected_sha256:
        raise ValueError(f"{tm['run_id']} step {step}: manifest SHA-256 "
                         f"{ck[int(step)]['sha256']} != expected {expected_sha256}")
    p = Path(ck[int(step)]["path"])
    return tm, ck[int(step)], (p if p.is_absolute() else REPO_ROOT / p)


def evaluate_x2c(train_run_dir: Path, step: int, *, family: str, seed: int,
                 expected_sha256: str, runs_dir: Path, freeze: dict[str, Any] | None,
                 freeze_git_blob: str | None = None, check: bool = False, attempt: int = 1,
                 rerun_decision: str | None = None, n_verify: int = 3,
                 num_threads: int = 1) -> dict[str, Any]:
    """Evaluate one frozen X2 checkpoint on every X2 held-out start once (check mode: a
    checkpoint on the X2 validation starts); return the summary.

    Must be called inside :func:`roboforge.x2.binding.x2_bindings`. Everything is checked
    before any directory is created or the checkpoint is loaded: the suite (from the freeze),
    the job (listed in the freeze), the attempt, the training manifest and the file's
    SHA-256."""
    from roboforge.eval.learned import LearnedPolicy
    from roboforge.eval.validation import _summarize, canonical_platform_id, run_states
    from roboforge.sim.eval_states import load_suite, read_manifest
    from roboforge.train.run import write_manifest
    from roboforge.x2 import binding
    from roboforge.x2.data import DATASET_ID, load_x2_artifacts

    if not binding.active():
        raise RuntimeError("evaluate_x2c must run inside x2_bindings()")
    task = x2s.suite_task()
    if check:
        if freeze is not None or attempt != 1 or rerun_decision is not None:
            raise ValueError("check mode takes no freeze, attempt or rerun decision")
        suite, job = x2s.VALIDATION_SUITE, None
    else:
        if freeze is None or not freeze_git_blob:
            raise ValueError("a held-out evaluation needs the committed X2 freeze and its blob")
        suite = freeze["suite"]["name"]
    check_x2c_suite(task, suite, check=check)
    if (attempt == 2) != (rerun_decision is not None) or attempt not in (1, 2):
        raise ValueError("attempt 2 (and only attempt 2) needs a rerun decision")
    if rerun_decision is not None and not DECISION_ID.fullmatch(rerun_decision):
        raise ValueError(f"rerun decision {rerun_decision!r} is not a decision ID")
    train_run_dir = Path(train_run_dir)
    if not check:
        job = find_job(freeze, train_run_dir.name, int(step), expected_sha256, family, seed)
    tm, ck, path = _train_checkpoint(train_run_dir, int(step), expected_sha256, seed)
    got = sha256_file(path)  # hashed before it is loaded
    if got != expected_sha256:
        raise ValueError(f"{path.name}: SHA-256 {got} != expected {expected_sha256}")
    x2_manifest = load_x2_artifacts()["task_manifest"]
    horizon = int(x2_manifest["task_spec"]["eval_horizon_steps"])
    if horizon != HORIZON:
        raise ValueError(f"X2 task spec horizon {horizon} != {HORIZON}")
    manifest = read_manifest(task, suite)
    if not check and (manifest["content_sha256"] != freeze["suite"]["manifest_content_sha256"]
                      or int(manifest["n_states"]) != N_STATES):
        raise ValueError(f"{task}/{suite}: manifest differs from the X2 freeze")
    policy = LearnedPolicy.from_checkpoint(path, expected_sha256=expected_sha256,
                                           num_threads=num_threads, policy_seed=POLICY_SEED,
                                           repeat=REPEAT)
    if policy.task != task or policy.hashes.get("dataset_id") != DATASET_ID:
        raise ValueError(f"{path.name}: task {policy.task!r} / dataset "
                         f"{policy.hashes.get('dataset_id')!r} is not X2 ({DATASET_ID})")
    platform = evaluation_platform()
    eval_id = x2c_eval_run_id(tm["run_id"], int(step), platform["platform_id"],
                              stochastic=policy.stochastic, check=check)
    if job is not None and eval_id != job["eval_run_id"]:
        raise ValueError(f"{eval_id} is not the frozen job {job['eval_run_id']} "
                         "(non-canonical platform?)")
    out_dir = prepare_out_dir(runs_dir, eval_id, check=check, attempt=attempt)
    states = load_suite(task, suite)
    run = run_states(policy, states, eval_id, out_dir, task, n_verify)
    episodes, verified = run["episodes"], run["verified"]
    summary = {
        "schema": CHECK_SCHEMA if check else SCHEMA, "eval_run_id": eval_id,
        "label": CHECK_LABEL if check else LABEL, "study": STUDY,
        "decisions": list(DECISIONS),
        "kind": ("X2-c engineering reproduction of an X2-b validation evaluation" if check
                 else "X2 closed-loop evaluation on the X2 held-out starts (X2-c, once)"),
        "created_utc": datetime.now(tz=UTC).isoformat(),
        "train_run_id": tm["run_id"], "family": family, "seed": int(seed),
        "model_family": tm["config"]["family"],
        "checkpoint": {"step": int(step), **ck}, "checkpoint_sha256_before_load": got,
        "policy": policy.info.to_dict(),
        "stochastic_policy": ({"policy_seed": POLICY_SEED, "repeat": REPEAT,
                               "samples_per_decision": 1} if policy.stochastic else None),
        "task": task, "dataset_id": DATASET_ID,
        "x2_task_manifest_sha256": x2_manifest["content_sha256"],
        "x2_task_spec_sha256": x2_manifest["task_spec_sha256"],
        "suite": suite, "partition": manifest["partition"],
        "suite_manifest_content_sha256": manifest["content_sha256"],
        "freeze": None if check else {"path": FREEZE_REL, "git_blob": freeze_git_blob},
        "attempt": int(attempt), "rerun_decision": rerun_decision,
        "platform": platform, "canonical_platform": platform["platform_id"]
        == canonical_platform_id(), "eligible_for_selection": False, "partial": False,
        "git": git_revision(), "environment": environment_snapshot(),
        "episode_rules": run["episode_rules"], **_summarize(episodes, run["action_dim"]),
        "throughput_control_steps_per_s": sum(e["steps"] for e in episodes) / run["wall_s"],
        "wall_s": run["wall_s"], "verify_record": verified,
        "verify_passed": all(v["passed"] for v in verified),
        "records_dir": repo_relative(out_dir), "episodes": episodes}
    write_manifest(out_dir / "manifest.json", summary)
    return summary
